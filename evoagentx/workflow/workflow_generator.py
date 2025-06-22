import json
from typing import Optional, List
from pydantic import Field, PositiveInt 

import time
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
from ..tools.tool import ToolKit

class WorkFlowGenerator(BaseModule):
    """
    Automated workflow generation system based on high-level goals.
    
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
    tools: Optional[List[ToolKit]] = Field(default=None, description="A list of tools that can be used in the workflow.")
    
    def init_module(self):
        if self.tools is not None:
            self.get_tool_info()
        else:
            self.tool_info = {}
        if self.task_planner is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `task_planner` is None")
            self.task_planner = TaskPlanner(llm=self.llm)
        
        if self.agent_generator is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `agent_generator` is None")
            self.agent_generator = AgentGenerator(llm=self.llm, tool_info=self.tool_info)
        
        # TODO add WorkFlowReviewer
        # if self.workflow_reviewer is None:
        #     if self.llm is None:
        #         raise ValueError(f"Must provide `llm` when `workflow_reviewer` is None")
        #     self.workflow_reviewer = WorkFlowReviewer(llm=self.llm)

    def get_tool_info(self):
        self.tool_info =[
            {
                tool.name: [
                    s["function"]["description"] for s in tool.get_tool_schemas()
                ],
            }
            for tool in self.tools
        ]

    def _execute_with_retry(self, operation_name: str, operation, retries_left: int = 1, **kwargs):
        """Helper method to execute operations with retry logic.
        
        Args:
            operation_name: Name of the operation for logging
            operation: Callable that performs the operation
            retries_left: Number of retry attempts remaining
            **kwargs: Additional arguments to pass to the operation
            
        Returns:
            Tuple of (operation_result, number_of_retries_used)
            
        Raises:
            ValueError: If operation fails after all retries are exhausted
        """
        cur_retries = 0

        while cur_retries <= retries_left:  # Changed < to <= to include the initial try
            try:
                logger.info(f"{operation_name} (attempt {cur_retries + 1}/{retries_left + 1}) ...")
                result = operation(**kwargs)
                return result, cur_retries
            except Exception as e:
                if cur_retries == retries_left:
                    raise ValueError(f"Failed to {operation_name} after {cur_retries + 1} attempts.\nError: {e}")
                sleep_time = 2 ** cur_retries
                logger.error(f"Failed to {operation_name} in {cur_retries + 1} attempts. Retry after {sleep_time} seconds.\nError: {e}")
                time.sleep(sleep_time)
                cur_retries += 1

    def generate_workflow(self, goal: str, existing_agents: Optional[List[Agent]] = None, retry: int = 1, **kwargs) -> WorkFlowGraph:
        # Validate input
        if not goal or len(goal.strip()) < 10:
            raise ValueError("Goal must be at least 10 characters and descriptive")

        plan_history, plan_suggestion = "", ""

        # Generate the initial workflow plan
        cur_retries = 0
        plan, added_retries = self._execute_with_retry(
            operation_name="Generating a workflow plan",
            operation=self.generate_plan,
            retries_left=retry,
            goal=goal,
            history=plan_history,
            suggestion=plan_suggestion
        )
        cur_retries += added_retries

        # Build workflow from plan
        workflow, added_retries = self._execute_with_retry(
            operation_name="Building workflow from plan",
            operation=self.build_workflow_from_plan,
            retries_left=retry - cur_retries,
            goal=goal,
            plan=plan
        )
        cur_retries += added_retries

        # Validate initial workflow structure
        logger.info("Validating initial workflow structure...")
        workflow._validate_workflow_structure()
        logger.info(f"Successfully generate the following workflow:\n{workflow.get_workflow_description()}")

        # generate / assigns the initial agents
        logger.info("Generating agents for the workflow ...")
        workflow, added_retries = self._execute_with_retry(
            operation_name="Generating agents for the workflow",
            operation=self.generate_agents,
            retries_left=retry - cur_retries,
            goal=goal,
            workflow=workflow,
            existing_agents=existing_agents
        )

        # Validate workflow after agent generation
        logger.info("Validating workflow after agent generation...")
        workflow._validate_workflow_structure()
        # Validate that all nodes have agents
        for node in workflow.nodes:
            if not node.agents:
                raise ValueError(f"Node {node.name} has no agents assigned after agent generation")

        return workflow

    def generate_plan(self, goal: str, history: Optional[str] = None, suggestion: Optional[str] = None) -> TaskPlanningOutput:
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
    
