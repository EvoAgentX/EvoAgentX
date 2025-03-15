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
    """
    An interface for the workflow schedulers
    """
    pass


class TaskSchedulerOutput(LLMOutputParser):

    decision: str = Field(description="The decision made by the scheduler, whether to re-execute, iterate or forward a certain task.")
    task_name: str = Field(description="The name of the scheduled task.")
    reason: str = Field(description="The rationale behind the scheduling decision, explaining why the task was scheduled.")

    def to_str(self, **kwargs) -> str:
        return f"Based on the workflow execution results, the next subtask to be executed is '{self.task_name}' because {self.reason}"
    

class TaskScheduler(Action):

    """
    Determines the next task to execute based on the workflow graph and node statuses.
    """
    def __init__(self, **kwargs):
        name = kwargs.pop("name", None) if "name" in kwargs else DEFAULT_TASK_SCHEDULER["name"]
        description = kwargs.pop("description", None) if "description" in kwargs else DEFAULT_TASK_SCHEDULER["description"]
        prompt = kwargs.pop("prompt", None) if "prompt" in kwargs else DEFAULT_TASK_SCHEDULER["prompt"]
        super().__init__(name=name, description=description, prompt=prompt, outputs_format=TaskSchedulerOutput, **kwargs)

    def get_predecessor_tasks(self, graph: WorkFlowGraph, tasks: List[WorkFlowNode]) -> List[str]:
        predecessors = [] 
        for task in tasks:
            candidates = graph.get_node_predecessors(node=task)
            for candidate in candidates:
                if candidate not in predecessors:
                    predecessors.append(candidate)
        return predecessors
    
    def execute(self, llm: Optional[BaseLLM] = None, graph: WorkFlowGraph = None, env: Environment = None, sys_msg: Optional[str] = None, return_prompt: bool=False, **kwargs) -> Union[TaskSchedulerOutput, Tuple[TaskSchedulerOutput, str]]:

        """
        Determine the next executable tasks.

        Args:
            graph (WorkFlowGraph): The workflow graph.
            env (Environment): The execution environment. 
        
        Returns:
            str: the name of the task to execute. 
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
            "candidate_tasks": candidate_tasks_info
        }
        prompt = self.prompt.format(**prompt_inputs)
        scheduled_task = llm.generate(prompt=prompt, system_message=sys_msg, parser=self.outputs_format)
        
        if return_prompt:
            return scheduled_task, prompt
        return scheduled_task


class NextAction(LLMOutputParser):

    agent: Optional[str] = Field(default=None, description="The name of the selected agent responsible for executing the next action in the workflow.")
    action: Optional[str] = Field(default=None, description="The name of the action that the selected agent will execute to continue progressing the subtask.")
    reason: Optional[str] = Field(default=None, description= "The justification for selecting this agent and action, explaining how it contributes to subtask execution based on workflow requirements and execution history.")
    action_graph: Optional[ActionGraph] = Field(default=None, description="The predefined action graph to be executed.")

    def to_str(self, **kwargs) -> str:
        if self.agent is not None and self.action is not None:
            return f"Based on the tasks' execution results, the next action to be executed is the '{self.action}' action of '{self.agent}' agent."
        elif self.action_graph is not None:
            return f"The predefined action graph '{type(self.action_graph).__name__}' will be executed."
        else:
            raise ValueError("must provide either both agent (str) and action (str), or action_graph (ActionGraph).")


class ActionScheduler(Action):

    """
    Determines the next action(s) to execute for a given task using an LLM.
    """
    def __init__(self, **kwargs):
        name = kwargs.pop("name", None) if "name" in kwargs else DEFAULT_ACTION_SCHEDULER["name"]
        description = kwargs.pop("description", None) if "description" in kwargs else DEFAULT_ACTION_SCHEDULER["description"]
        prompt = kwargs.pop("prompt", None) if "prompt" in kwargs else DEFAULT_ACTION_SCHEDULER["prompt"]
        super().__init__(name=name, description=description, prompt=prompt, outputs_format=NextAction, **kwargs)

    def format_task_input_data(self, data: dict) -> str:
        info_list = [] 
        for key, value in data.items():
            info_list.append("## {}\n{}".format(key, value))
        return "\n\n".join(info_list)
    
    def check_candidate_action(self, task_name: str, actions: List[str], agent_actions_map: Dict[str, List[str]]):
        unknown_actions = []
        merged_actions = set(chain.from_iterable(agent_actions_map.values()))
        for action in actions:
            if action not in merged_actions:
                unknown_actions.append(action)
        if unknown_actions:
            raise ValueError(f"Unknown actions: {unknown_actions} specified in the `next_actions`. All available actions defined for the task ({task_name}) are {merged_actions}.")
    
    def get_agent_action_pairs(self, action: str, agent_actions_map: Dict[str, List[str]]) -> List[Tuple[str, str]]:
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
        """
        Determine the next actions to take for the given task. 
        If the last message stored in ``next_actions`` specifies the ``next_actions``, choose an action from these actions to execute. 

        Returns:
            NextAction: The next action to execute for the task.
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
    """
    Responsible for the scheduling and decision-making when executing a workflow. 

    Attributes:
        task_scheduler (TaskScheduler): Determines the next task(s) to execute based on the workflow graph and node states.
        action_scheduler (ActionScheduler): Determines the next action(s) to take for the selected task using an LLM.
    """
    llm: BaseLLM
    action_scheduler: ActionScheduler = Field(default_factory=ActionScheduler)
    task_scheduler: TaskScheduler = Field(default_factory=TaskScheduler)

    def init_module(self):
        self._save_ignore_fields = ["llm"]

    def schedule_next_task(self, graph: WorkFlowGraph, env: Environment = None, **kwargs) -> WorkFlowNode:
        """
        Return the next task to execute. 
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
        """
        return the next action to execute. If the task is completed, return None.
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
        ignore_fields = self._save_ignore_fields + ignore
        super().save_module(path=path, ignore=ignore_fields, **kwargs)
