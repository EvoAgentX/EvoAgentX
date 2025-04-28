import json
from typing import Optional, List
from pydantic import Field, PositiveInt 

from ..core.logging import logger
from ..core.module import BaseModule
# from ..core.base_config import Parameter
from ..core.message import Message, MessageType
from ..models.base_model import BaseLLM
from ..agents.agent import Agent
from ..agents.task_planner import TaskPlanner
from ..agents.agent_generator import AgentGenerator
from ..agents.workflow_reviewer import WorkFlowReviewer
from ..actions.task_planning import TaskPlanningOutput
from ..actions.agent_generation import AgentGenerationOutput
from ..workflow.workflow_graph import WorkFlowGraph, WorkFlowNode, WorkFlowEdge


class WorkFlowGenerator(BaseModule):
    """Automated workflow generation system based on high-level goals.
    
    The WorkFlowGenerator is responsible for creating complete workflow graphs
    from high-level goals or task descriptions. It breaks down the goal into
    subtasks, creates the necessary dependency connections between tasks,
    and assigns or generates appropriate agents for each task.
    
    Attributes:
        llm: Language model used for generation and planning
        task_planner: Component responsible for breaking down goals into subtasks
        agent_generator: Component responsible for agent assignment or creation
        workflow_reviewer: Component for reviewing and improving workflows
        num_turns: Number of refinement iterations for the workflow
    """

    llm: Optional[BaseLLM] = None
    task_planner: Optional[TaskPlanner] = Field(default=None, description="Responsible for breaking down the high-level task into manageable sub-tasks.")
    agent_generator: Optional[AgentGenerator] = Field(default=None, description="Assigns or generates the appropriate agent(s) to handle each sub-task.")
    workflow_reviewer: Optional[WorkFlowReviewer] = Field(default=None, description="Provides feedback and reflections to improve the generated workflow.")
    num_turns: Optional[PositiveInt] = Field(default=0, description="Specifies the number of refinement iterations for the generated workflow.")

    def init_module(self):

        if self.task_planner is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `task_planner` is None")
            self.task_planner = TaskPlanner(llm=self.llm)
        
        if self.agent_generator is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `agent_generator` is None")
            self.agent_generator = AgentGenerator(llm=self.llm)
        
        # TODO add WorkFlowReviewer
        # if self.workflow_reviewer is None:
        #     if self.llm is None:
        #         raise ValueError(f"Must provide `llm` when `workflow_reviewer` is None")
        #     self.workflow_reviewer = WorkFlowReviewer(llm=self.llm)

    def generate_workflow(self, goal: str, existing_agents: Optional[List[Agent]] = None, **kwargs) -> WorkFlowGraph:
        """Generate a complete workflow based on a high-level goal.
        
        Orchestrates the entire workflow generation process, including:
        1. Task planning to break down the goal into subtasks
        2. Workflow construction to create a graph with dependencies
        3. Agent generation to assign appropriate agents to each task
        
        Args:
            goal: High-level goal or task description to create a workflow for
            existing_agents: Optional list of pre-existing agents that could be assigned
            **kwargs: Additional parameters for workflow generation
            
        Returns:
            A complete WorkFlowGraph with tasks, dependencies, and assigned agents
        
        Notes:
            The resulting workflow is ready for execution by a workflow manager
        """

        plan_history, plan_suggestion = "", ""
        # generate the initial workflow
        logger.info(f"Generating a workflow for: {goal} ...")
        plan = self.generate_plan(goal=goal, history=plan_history, suggestion=plan_suggestion)
        workflow = self.build_workflow_from_plan(goal=goal, plan=plan)
        logger.info(f"Successfully generate the following workflow:\n{workflow.get_workflow_description()}")
        # generate / assigns the initial agents
        logger.info("Generating agents for the workflow ...")
        workflow = self.generate_agents(goal=goal, workflow=workflow, existing_agents=existing_agents)
        return workflow
    
    def generate_plan(self, goal: str, history: Optional[str] = None, suggestion: Optional[str] = None) -> TaskPlanningOutput:
        """Generate a task plan for the given goal.
        
        Uses the task planner to break down a high-level goal into a structured
        set of subtasks with their inputs, outputs, and dependencies.
        
        Args:
            goal: High-level goal or task to plan for
            history: Optional string containing previous planning attempts/history
            suggestion: Optional string containing suggestions for planning
            
        Returns:
            A TaskPlanningOutput containing the structured subtasks and their details
        """

        history = "" if history is None else history
        suggestion = "" if suggestion is None else suggestion
        task_planner: TaskPlanner = self.task_planner
        task_planning_action_data = {"goal": goal, "history": history, "suggestion": suggestion}
        task_planning_action_name = task_planner.task_planning_action_name
        message: Message = task_planner.execute(
            action_name=task_planning_action_name,
            action_input_data=task_planning_action_data,
            return_msg_type=MessageType.REQUEST
        )
        return message.content
    
    def generate_agents(
        self, 
        goal: str, 
        workflow: WorkFlowGraph,
        existing_agents: Optional[List[Agent]] = None,
        # history: Optional[str] = None, 
        # suggestion: Optional[str] = None
    ) -> WorkFlowGraph:
        """Generate or assign agents for each task in the workflow.
        
        For each subtask in the workflow, uses the agent generator to either
        create new agents with appropriate capabilities or assign existing agents
        that match the task requirements.
        
        Args:
            goal: High-level goal of the workflow
            workflow: WorkFlowGraph containing tasks that need agents
            existing_agents: Optional list of pre-existing agents that could be assigned
            
        Returns:
            The updated WorkFlowGraph with agents assigned to each task
            
        Notes:
            Currently handles only generated agents, not assignment from existing agents
        """
        
        agent_generator: AgentGenerator = self.agent_generator
        workflow_desc = workflow.get_workflow_description()
        agent_generation_action_name = agent_generator.agent_generation_action_name
        for subtask in workflow.nodes:
            subtask_fields = ["name", "description", "reason", "inputs", "outputs"]
            subtask_data = {key: value for key, value in subtask.to_dict(ignore=["class_name"]).items() if key in subtask_fields}
            subtask_desc = json.dumps(subtask_data, indent=4)
            agent_generation_action_data = {"goal": goal, "workflow": workflow_desc, "task": subtask_desc}
            logger.info(f"Generating agents for subtask: {subtask_data['name']}")
            agents: AgentGenerationOutput = agent_generator.execute(
                action_name=agent_generation_action_name, 
                action_input_data=agent_generation_action_data,
                return_msg_type=MessageType.RESPONSE
            ).content
            # todo I only handle generated agents
            generated_agents = []
            for agent in agents.generated_agents:
                agent_dict = agent.to_dict(ignore=["class_name"])
                # agent_dict["llm_config"] = self.llm.config.to_dict()
                generated_agents.append(agent_dict)
            subtask.set_agents(agents=generated_agents)
        return workflow
    
    # def review_plan(self, goal: str, )

    def build_workflow_from_plan(self, goal: str, plan: TaskPlanningOutput) -> WorkFlowGraph:
        """Construct a workflow graph from a task planning output.
        
        Creates a WorkFlowGraph by extracting subtasks from the plan and
        inferring edges (dependencies) based on input-output relationships
        between tasks.
        
        Args:
            goal: High-level goal of the workflow
            plan: TaskPlanningOutput containing subtasks with their inputs and outputs
            
        Returns:
            A WorkFlowGraph with nodes (tasks) and edges (dependencies)
            
        Notes:
            Edges are inferred automatically based on parameter matching between
            outputs of one task and inputs of another, creating a data-flow graph
        """

        nodes: List[WorkFlowNode] = plan.sub_tasks
        # infer edges from sub-tasks' inputs and outputs
        edges: List[WorkFlowEdge] = []
        for node in nodes:
            for another_node in nodes:
                if node.name == another_node.name:
                    continue
                node_output_params = [param.name for param in node.outputs]
                another_node_input_params = [param.name for param in another_node.inputs]
                if any([param in another_node_input_params for param in node_output_params]):
                    edges.append(WorkFlowEdge(edge_tuple=(node.name, another_node.name)))
        workflow = WorkFlowGraph(goal=goal, nodes=nodes, edges=edges)
        return workflow
    