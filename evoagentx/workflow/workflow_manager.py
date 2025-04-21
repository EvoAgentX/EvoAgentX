from pydantic import Field
from itertools import chain
from collections import defaultdict
from typing import Union, Optional, Tuple, Dict, List 

from ..core.module import BaseModule
# from ..core.base_config import Parameter
from ..core.message import Message, MessageType
from ..models.base_model import BaseLLM, LLMOutputParser
# from ..agents.agent import Agent
from ..actions.action import Action
from ..agents.agent_manager import AgentManager
from .action_graph import ActionGraph
from .environment import Environment, TrajectoryState
from .workflow_graph import WorkFlowNode, WorkFlowGraph
from ..prompts.workflow.workflow_manager import (
    DEFAULT_TASK_SCHEDULER, 
    DEFAULT_ACTION_SCHEDULER, 
    OUTPUT_EXTRACTION_PROMPT
)


class Scheduler(Action):
    """Base interface for workflow schedulers.
    
    Provides a common interface for all scheduler types within the workflow
    system. Schedulers are responsible for making decisions about what to 
    execute next in a workflow, whether at the task or action level.
    
    Inherits from Action to leverage the common action interface and functionality.
    """
    pass


class TaskSchedulerOutput(LLMOutputParser):
    """Structured output format for task scheduling decisions.
    
    Parses and structures the output from the task scheduler LLM into
    a standardized format containing the scheduling decision, task name,
    and the reasoning behind the decision.
    
    Attributes:
        decision: The decision made by the scheduler (re-execute, iterate, forward)
        task_name: The name of the task that should be executed next
        reason: The rationale explaining why this task was selected
    """
    decision: str = Field(description="The decision made by the scheduler, whether to re-execute, iterate or forward a certain task.")
    task_name: str = Field(description="The name of the scheduled task.")
    reason: str = Field(description="The rationale behind the scheduling decision, explaining why the task was scheduled.")

    def to_str(self, **kwargs) -> str:
        """Convert the structured output to a human-readable string.
        
        Returns:
            A formatted string describing the scheduling decision and reason
        """
        return f"Based on the workflow execution results, the next subtask to be executed is '{self.task_name}' because {self.reason}"
    

class TaskScheduler(Action):
    """Determines the next task to execute in a workflow.
    
    Analyzes the workflow graph, current node statuses, and execution history
    to decide which task should be executed next. Uses an LLM to make intelligent
    decisions about task sequencing, particularly when multiple options are available.
    
    Attributes:
        max_num_turns: Maximum number of execution turns allowed in the workflow
    """
    def __init__(self, **kwargs):
        """Initialize the task scheduler with default or custom configuration.
        
        Args:
            **kwargs: Configuration parameters that may include:
                name: Custom name for the scheduler
                description: Custom description of the scheduler
                prompt: Custom prompt template for the scheduler
                max_num_turns: Maximum number of execution turns
        """
        name = kwargs.pop("name", None) if "name" in kwargs else DEFAULT_TASK_SCHEDULER["name"]
        description = kwargs.pop("description", None) if "description" in kwargs else DEFAULT_TASK_SCHEDULER["description"]
        prompt = kwargs.pop("prompt", None) if "prompt" in kwargs else DEFAULT_TASK_SCHEDULER["prompt"]
        super().__init__(name=name, description=description, prompt=prompt, outputs_format=TaskSchedulerOutput, **kwargs)
        self.max_num_turns = kwargs.get("max_num_turns", DEFAULT_TASK_SCHEDULER["max_num_turns"])

    def get_predecessor_tasks(self, graph: WorkFlowGraph, tasks: List[WorkFlowNode]) -> List[str]:
        """Find all predecessor tasks for a list of tasks.
        
        Identifies all unique predecessor nodes for the given tasks in the workflow graph.
        
        Args:
            graph: The workflow graph containing task dependencies
            tasks: List of task nodes to find predecessors for
            
        Returns:
            List of unique predecessor task names
        """
        predecessors = [] 
        for task in tasks:
            candidates = graph.get_node_predecessors(node=task)
            for candidate in candidates:
                if candidate not in predecessors:
                    predecessors.append(candidate)
        return predecessors
    
    def execute(self, llm: Optional[BaseLLM] = None, graph: WorkFlowGraph = None, env: Environment = None, sys_msg: Optional[str] = None, return_prompt: bool=False, **kwargs) -> Union[TaskSchedulerOutput, Tuple[TaskSchedulerOutput, str]]:
        """Determine the next executable tasks in the workflow.
        
        Analyzes the workflow graph and execution environment to identify
        the next task that should be executed. When multiple candidate tasks
        are available, uses an LLM to make an intelligent decision based on
        task descriptions, input/output relationships, and execution history.
        
        Args:
            llm: Language model to use for decision making
            graph: The workflow graph containing task definitions and dependencies
            env: The execution environment with history and output data
            sys_msg: Optional system message for the LLM
            return_prompt: Whether to return the prompt used for the LLM
            **kwargs: Additional parameters passed to the LLM
            
        Returns:
            If return_prompt is False:
                TaskSchedulerOutput with the scheduling decision
            If return_prompt is True:
                Tuple containing TaskSchedulerOutput and the prompt used
                
        Raises:
            AssertionError: If graph or env is not provided
        """
        assert graph is not None and env is not None, "must provide 'graph' and 'env' when executing TaskScheduler"

        candidate_tasks: List[WorkFlowNode] = graph.next()
        if not candidate_tasks:
            return None
        
        # directly return the task if there is only one single candidate task
        if len(candidate_tasks) == 1:
            task_name = candidate_tasks[0].name
            scheduled_task = TaskSchedulerOutput(
                decision="forward", 
                task_name=task_name,
                reason = f"Only one candidate task '{task_name}' is available."
            )
            return (scheduled_task, None) if return_prompt else scheduled_task
        
        workflow_graph_representation = graph.get_workflow_description()
        execution_history = " -> ".join(env.task_execution_history)
        # in execution_ouputs only consider the predecessors of candidate tasks
        predecessor_tasks = self.get_predecessor_tasks(graph=graph, tasks=candidate_tasks)
        execution_outputs = "\n\n".join([str(msg)  for msg in env.get_task_messages(tasks=predecessor_tasks)])
        candidate_tasks_info = "\n\n".join([task.get_task_info() for task in candidate_tasks])
        prompt_inputs = {
            "workflow_graph_representation": workflow_graph_representation, 
            "execution_history": execution_history,
            "execution_outputs": execution_outputs, 
            "candidate_tasks": candidate_tasks_info,
            "max_num_turns": self.max_num_turns
        }
        prompt = self.prompt.format(**prompt_inputs)
        scheduled_task = llm.generate(prompt=prompt, system_message=sys_msg, parser=self.outputs_format)
        
        if return_prompt:
            return scheduled_task, prompt
        return scheduled_task


class NextAction(LLMOutputParser):
    """Structured output format for action scheduling decisions.
    
    Parses and structures the output from the action scheduler LLM into
    a standardized format containing the agent, action, reasoning, and
    action graph information.
    
    Represents a decision about what action should be executed next
    for a particular workflow task, either as an agent-action pair or
    a predefined action graph.
    
    Attributes:
        agent: Name of the agent to execute the action (if using agent-action pair)
        action: Name of the action to execute (if using agent-action pair)
        reason: Explanation for why this agent and action were selected
        action_graph: Predefined action graph to execute (alternative to agent-action)
    """
    agent: Optional[str] = Field(default=None, description="The name of the selected agent responsible for executing the next action in the workflow.")
    action: Optional[str] = Field(default=None, description="The name of the action that the selected agent will execute to continue progressing the subtask.")
    reason: Optional[str] = Field(default=None, description= "The justification for selecting this agent and action, explaining how it contributes to subtask execution based on workflow requirements and execution history.")
    action_graph: Optional[ActionGraph] = Field(default=None, description="The predefined action graph to be executed.")

    def to_str(self, **kwargs) -> str:
        """Convert the structured output to a human-readable string.
        
        Returns:
            A formatted string describing the next action to be executed
            
        Raises:
            ValueError: If neither agent-action pair nor action_graph is provided
        """
        if self.agent is not None and self.action is not None:
            return f"Based on the tasks' execution results, the next action to be executed is the '{self.action}' action of '{self.agent}' agent."
        elif self.action_graph is not None:
            return f"The predefined action graph '{type(self.action_graph).__name__}' will be executed."
        else:
            raise ValueError("must provide either both agent (str) and action (str), or action_graph (ActionGraph).")


class ActionScheduler(Action):
    """Determines the next action to execute for a task.
    
    Analyzes the available agents and their actions for a given task,
    along with execution history and task requirements, to decide 
    which action should be executed next. Uses an LLM to make intelligent
    decisions when multiple options are available.
    """
    def __init__(self, **kwargs):
        """Initialize the action scheduler with default or custom configuration.
        
        Args:
            **kwargs: Configuration parameters that may include:
                name: Custom name for the scheduler
                description: Custom description of the scheduler
                prompt: Custom prompt template for the scheduler
        """
        name = kwargs.pop("name", None) if "name" in kwargs else DEFAULT_ACTION_SCHEDULER["name"]
        description = kwargs.pop("description", None) if "description" in kwargs else DEFAULT_ACTION_SCHEDULER["description"]
        prompt = kwargs.pop("prompt", None) if "prompt" in kwargs else DEFAULT_ACTION_SCHEDULER["prompt"]
        super().__init__(name=name, description=description, prompt=prompt, outputs_format=NextAction, **kwargs)

    def format_task_input_data(self, data: dict) -> str:
        """Format task input data into a readable string.
        
        Converts a dictionary of task input data into a formatted string
        with sections for each input parameter.
        
        Args:
            data: Dictionary of task input parameters and values
            
        Returns:
            Formatted string representation of the task inputs
        """
        info_list = [] 
        for key, value in data.items():
            info_list.append("## {}\n{}".format(key, value))
        return "\n\n".join(info_list)
    
    def check_candidate_action(self, task_name: str, actions: List[str], agent_actions_map: Dict[str, List[str]]):
        """Validate that all candidate actions exist in the agent actions map.
        
        Ensures that all actions in the candidate list are actually implemented
        by at least one agent for this task.
        
        Args:
            task_name: Name of the task being executed
            actions: List of candidate action names to validate
            agent_actions_map: Mapping of agent names to their available actions
            
        Raises:
            ValueError: If any action is not found in any agent's action list
        """
        unknown_actions = []
        merged_actions = set(chain.from_iterable(agent_actions_map.values()))
        for action in actions:
            if action not in merged_actions:
                unknown_actions.append(action)
        if unknown_actions:
            raise ValueError(f"Unknown actions: {unknown_actions} specified in the `next_actions`. All available actions defined for the task ({task_name}) are {merged_actions}.")
    
    def get_agent_action_pairs(self, action: str, agent_actions_map: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """Find all agent-action pairs for a given action.
        
        Identifies all agents that can execute the specified action.
        
        Args:
            action: The action name to find implementations for
            agent_actions_map: Mapping of agent names to their available actions
            
        Returns:
            List of (agent_name, action_name) tuples for all agents that can execute this action
        """
        pairs = [] 
        for agent, actions in agent_actions_map.items():
            if action in actions:
                pairs.append((agent, action))
        return pairs

    def execute(
        self, 
        llm: Optional[BaseLLM] = None, 
        task: WorkFlowNode = None, 
        agent_manager: AgentManager = None, 
        env: Environment = None, 
        sys_msg: Optional[str] = None, 
        return_prompt: bool=True, 
        **kwargs
    ) -> Union[NextAction, Tuple[NextAction, str]]:
        """Determine the next action to execute for a task.
        
        Analyzes the task, available agents and their actions, execution history,
        and task requirements to decide which action should be executed next.
        
        Decision making logic:
        1. If the task has a predefined action_graph, use that
        2. If there's only one agent with one action, use that
        3. If the previous execution step specified next_actions, use those
        4. Otherwise, use an LLM to decide based on task requirements and history
        
        Args:
            llm: Language model to use for decision making
            task: The workflow task node being executed
            agent_manager: Manager for accessing agent capabilities
            env: Execution environment with history and data
            sys_msg: Optional system message for the LLM
            return_prompt: Whether to return the prompt used for the LLM
            **kwargs: Additional parameters passed to the LLM
            
        Returns:
            If return_prompt is True:
                Tuple containing NextAction decision and the prompt used
            If return_prompt is False:
                NextAction decision only
                
        Raises:
            ValueError: If the task has no agents defined
        """
        # the task has a action_graph, directly return the action_graph for execution 
        if task.action_graph is not None:
            next_action = NextAction(action_graph=task.action_graph)
            return (next_action, None) if return_prompt else next_action
        
        # Otherwise, schedule an agent to execute the task.
        task_agent_names = task.get_agents()
        if not task_agent_names:
            raise ValueError(f"The task '{task.name}' does not provide any agents for execution!")
        
        task_agents = [agent_manager.get_agent(name) for name in task_agent_names]
        task_agent_actions_map = {agent.name: [action.name for action in agent.get_all_actions()] for agent in task_agents}
        
        next_action = None
        candidate_agent_actions = defaultdict(set)

        # if a previous message has specified next_actions, select from these actions
        task_execution_messages = env.get_task_messages(task.name)
        if task_execution_messages and task_execution_messages[-1].next_actions:
            predefined_next_actions = task_execution_messages[-1].next_actions
            # check whether all the predefined_next_actions are present in the actions of task_agents
            self.check_candidate_action(task.name, predefined_next_actions, task_agent_actions_map)
            if len(predefined_next_actions) == 1:
                predefined_next_action = predefined_next_actions[0]
                agent_action_pairs = self.get_agent_action_pairs(predefined_next_action, task_agent_actions_map)
                if len(agent_action_pairs) == 1:
                    next_action = NextAction(
                        agent=agent_action_pairs[0][0], 
                        action=agent_action_pairs[0][1],
                        reason=f"Selected because task history indicates a single predefined next action: {predefined_next_action}"
                    )
                else:
                    for agent, action in agent_action_pairs:
                        candidate_agent_actions[agent].add(action)
            else:
                for predefined_next_action in predefined_next_actions:
                    agent_action_pairs = self.get_agent_action_pairs(predefined_next_action, task_agent_actions_map)
                    for agent, action in agent_action_pairs:
                        candidate_agent_actions[agent].add(action)
        
        # if there are only one agent and one action, directly return the action
        if not next_action and len(task_agent_names) == 1 and len(task_agent_actions_map[task_agent_names[0]]) == 1:
            task_agent_name = task_agent_names[0]
            task_action_name = task_agent_actions_map[task_agent_name][0]
            next_action = NextAction(
                agent=task_agent_name, 
                action=task_action_name, 
                reason=f"Only one agent ('{task_agent_name}') is available, and it has only one action ('{task_action_name}'), making it the obvious choice."
            )
        
        if next_action is not None:
            return (next_action, None) if return_prompt else next_action

        # prepare candidate agent & action information 
        # agent_actions_info = "\n\n".join([agent.get_agent_profile() for agent in task_agents])
        candidate_agent_actions = candidate_agent_actions or task_agent_actions_map
        agent_actions_info = "\n\n".join(
            [
                agent.get_agent_profile(action_names=candidate_agent_actions[agent.name]) \
                    for agent in task_agents if agent.name in candidate_agent_actions
            ]
        )

        # prepare task and execution information
        task_info = task.get_task_info()
        task_input_names = [param.name for param in task.inputs]
        task_input_data: dict = env.get_execution_data(task_input_names)
        task_input_data_info = self.format_task_input_data(data=task_input_data)
        task_execution_history = "\n\n".join([str(msg) for msg in task_execution_messages])

        prompt_inputs = {
            "task_info": task_info, 
            "task_inputs": task_input_data_info, 
            "task_execution_history": task_execution_history, 
            "agent_action_list": agent_actions_info,
        }
        prompt = self.prompt.format(**prompt_inputs)
        next_action = llm.generate(prompt=prompt, system_message=sys_msg, parser=self.outputs_format)
        if return_prompt:
            return next_action, prompt
        return next_action


class WorkFlowManager(BaseModule):
    """Coordinates the scheduling and execution of workflow tasks and actions.
    
    Serves as the orchestration layer for workflow execution, making high-level
    decisions about task sequencing and action selection. Uses specialized
    schedulers for task and action level decisions.
    
    The WorkflowManager is responsible for:
    1. Determining which task to execute next in the workflow
    2. Determining which action to execute for the current task
    3. Extracting final output from workflow execution
    
    Attributes:
        llm: Language model used for decision making
        action_scheduler: Scheduler for determining next actions within a task
        task_scheduler: Scheduler for determining next tasks in the workflow
    """
    llm: BaseLLM
    action_scheduler: ActionScheduler = Field(default_factory=ActionScheduler)
    task_scheduler: TaskScheduler = Field(default_factory=TaskScheduler)

    def init_module(self):
        """Initialize the workflow manager module.
        
        Sets up the fields to ignore when saving the module, particularly
        the LLM which should not be serialized.
        """
        self._save_ignore_fields = ["llm"]

    def schedule_next_task(self, graph: WorkFlowGraph, env: Environment = None, **kwargs) -> WorkFlowNode:
        """Schedule the next task to execute in the workflow.
        
        Uses the task scheduler to determine which task should be executed next
        based on the workflow graph and execution environment.
        
        Args:
            graph: The workflow graph containing task definitions and dependencies
            env: The execution environment with history and data
            **kwargs: Additional parameters passed to the task scheduler
            
        Returns:
            The next task node to execute, or None if no more tasks can be executed
        """
        execution_results = self.task_scheduler.execute(llm=self.llm, graph=graph, env=env, return_prompt=True, **kwargs)
        if execution_results is None:
            return None
        scheduled_task, prompt, *other = execution_results
        message = Message(
            content=scheduled_task, agent=type(self).__name__, action=self.task_scheduler.name, \
                prompt=prompt, msg_type=MessageType.COMMAND, wf_goal=graph.goal
        )
        env.update(message=message, state=TrajectoryState.COMPLETED)
        task: WorkFlowNode = graph.get_node(scheduled_task.task_name)
        return task

    def schedule_next_action(self, goal: str, task: WorkFlowNode, agent_manager: AgentManager, env: Environment = None, **kwargs) -> NextAction:
        """Schedule the next action to execute for a task.
        
        Uses the action scheduler to determine which action should be executed next
        for the given task based on available agents and execution history.
        
        Args:
            goal: The high-level goal of the workflow
            task: The workflow task node being executed
            agent_manager: Manager for accessing agent capabilities
            env: Execution environment with history and data
            **kwargs: Additional parameters passed to the action scheduler
            
        Returns:
            The next action to execute, or None if the task is completed
        """
        execution_results = self.action_scheduler.execute(llm=self.llm, task=task, agent_manager=agent_manager, env=env, return_prompt=True, **kwargs)
        if execution_results is None:
            return None
        next_action, prompt, *_ = execution_results
        message = Message(
            content=next_action, agent=type(self).__name__, action=self.action_scheduler.name, \
                prompt=prompt, msg_type=MessageType.COMMAND, wf_goal=goal, wf_task=task.name, wf_task_desc=task.description 
        )
        env.update(message=message, state=TrajectoryState.COMPLETED)
        return next_action
    
    def extract_output(self, graph: WorkFlowGraph, env: Environment, **kwargs) -> str:
        """Extract the final output from workflow execution.
        
        Uses the LLM to synthesize a final output from the execution results of the
        workflow, focusing on end tasks and their immediate predecessors.
        
        Args:
            graph: The completed workflow graph
            env: The execution environment with all execution data
            **kwargs: Additional parameters passed to the LLM
            
        Returns:
            Synthesized output text representing the workflow result
        """
        # obtain the output for end tasks
        end_tasks = graph.find_end_nodes()
        end_task_predecesssors = sum([graph.get_node_predecessors(node=end_task) for end_task in end_tasks], [])
        candidate_taks_with_output = list(set(end_tasks)|set(end_task_predecesssors))
        candidate_msgs_with_output = [] 
        for task in candidate_taks_with_output:
            # only task the final output of the task
            candidate_msgs_with_output.extend(env.get_task_messages(tasks=task, n=1))
        candidate_msgs_with_output = Message.sort_by_timestamp(messages=candidate_msgs_with_output)

        prompt = OUTPUT_EXTRACTION_PROMPT.format(
            goal=graph.goal, 
            workflow_graph_representation=graph.get_workflow_description(), 
            workflow_execution_results="\n\n".join([str(msg) for msg in candidate_msgs_with_output]), 
        )
        llm_output: LLMOutputParser = self.llm.generate(prompt=prompt)
        return llm_output.content

    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        """Save the workflow manager to a file.
        
        Serializes the workflow manager configuration, ignoring the LLM and
        any other specified fields.
        
        Args:
            path: The file path to save to
            ignore: Additional fields to ignore when saving
            **kwargs: Additional parameters for saving
            
        Returns:
            The path where the module was saved
        """
        ignore_fields = self._save_ignore_fields + ignore
        super().save_module(path=path, ignore=ignore_fields, **kwargs)
