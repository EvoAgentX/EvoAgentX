import inspect
import asyncio
from copy import deepcopy
from pydantic import Field, create_model
from typing import Optional, List
from ..core.logging import logger
from ..core.module import BaseModule
from ..core.message import Message, MessageType
from ..core.module_utils import generate_id
from ..models.base_model import BaseLLM
from ..agents.agent import Agent
from ..agents.agent_manager import AgentManager, AgentState
from ..storages.base import StorageHandler
from .environment import Environment, TrajectoryState
from .workflow_manager import WorkFlowManager, NextAction
from .workflow_graph import WorkFlowNode, WorkFlowGraph
from .action_graph import ActionGraph
from ..hitl import HITLManager, HITLBaseAgent
from ..utils.utils import generate_dynamic_class_name
from ..actions import ActionInput, ActionOutput

class WorkFlow(BaseModule):

    graph: WorkFlowGraph
    llm: Optional[BaseLLM] = None
    agent_manager: AgentManager = Field(default=None, description="Responsible for managing agents")
    workflow_manager: WorkFlowManager = Field(default=None, description="Responsible for task and action scheduling for workflow execution")
    environment: Environment = Field(default_factory=Environment)
    storage_handler: StorageHandler = None
    workflow_id: str = Field(default_factory=generate_id)
    version: int = 0 
    max_execution_steps: int = Field(default=5, description="The maximum number of steps to complete a subtask (node) in the workflow")
    hitl_manager: HITLManager = Field(default=None, description="Responsible for HITL work management")

    def init_module(self):
        if self.workflow_manager is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `workflow_manager` is None")
            self.workflow_manager = WorkFlowManager(llm=self.llm)
        if self.agent_manager is None:
            logger.warning("agent_manager is NoneType when initializing a WorkFlow instance")

    def execute(self, inputs: dict = {}, **kwargs) -> str:
        """
        Synchronous wrapper for async_execute. Creates a new event loop and runs the async method.
        
        Args:
            inputs: Dictionary of inputs for workflow execution
            **kwargs (Any): Additional keyword arguments
            
        Returns:
            str: The output of the workflow execution
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.async_execute(inputs, **kwargs))
        finally:
            loop.close()

    async def async_execute(self, inputs: dict = {}, **kwargs) -> str:
        """
        Asynchronously execute the workflow.
        
        Args:
            inputs: Dictionary of inputs for workflow execution
            **kwargs (Any): Additional keyword arguments
            
        Returns:
            str: The output of the workflow execution
        """
        goal = self.graph.goal
        # inputs.update({"goal": goal})
        inputs = self._prepare_inputs(inputs)

        # prepare for hitl functionalities
        if hasattr(self, "hitl_manager") and (self.hitl_manager is not None):
            self._prepare_hitl()

        # check the inputs and outputs of the task 
        self._validate_workflow_structure(inputs=inputs, **kwargs)
        inp_message = Message(content=inputs, msg_type=MessageType.INPUT, wf_goal=goal)
        self.environment.update(message=inp_message, state=TrajectoryState.COMPLETED)

        failed = False
        error_message = None
        while not self.graph.is_complete and not failed:
            try:
                task: WorkFlowNode = await self.get_next_task()
                if task is None:
                    break
                logger.info(f"Executing subtask: {task.name}")
                await self.execute_task(task=task)
            except Exception as e:
                failed = True
                error_message = Message(
                    content=f"An Error occurs when executing the workflow: {e}",
                    msg_type=MessageType.ERROR, 
                    wf_goal=goal
                )
                self.environment.update(message=error_message, state=TrajectoryState.FAILED, error=str(e))
        
        if failed:
            logger.error(error_message.content)
            return "Workflow Execution Failed"
        
        logger.info("Extracting WorkFlow Output ...")
        output: str = await self.workflow_manager.extract_output(graph=self.graph, env=self.environment)
        return output
    
    def _prepare_inputs(self, inputs: dict) -> dict:
        """
        Prepare the inputs for the workflow execution. Mainly determine whether the goal should be added to the inputs.
        """
        initial_node_names = self.graph.find_initial_nodes()
        initial_node_required_inputs = set()
        for initial_node_name in initial_node_names:
            initial_node = self.graph.get_node(initial_node_name)
            if initial_node.inputs:
                initial_node_required_inputs.update([inp.name for inp in initial_node.inputs if inp.required])
        if "goal" in initial_node_required_inputs and "goal" not in inputs:
            inputs.update({"goal": self.graph.goal})
            
        return inputs 
    
    async def get_next_task(self) -> WorkFlowNode:
        task_execution_history = " -> ".join(self.environment.task_execution_history)
        if not task_execution_history:
            task_execution_history = "None"
        logger.info(f"Task Execution Trajectory: {task_execution_history}. Scheduling next subtask ...")
        task: WorkFlowNode = await self.workflow_manager.schedule_next_task(graph=self.graph, env=self.environment)
        logger.info(f"The next subtask to be executed is: {task.name}")
        return task
        
    async def execute_task(self, task: WorkFlowNode):
        """
        Asynchronously execute a workflow task.
        
        Args:
            task: The workflow node to execute
        """
        last_executed_task = self.environment.get_last_executed_task()
        self.graph.step(source_node=last_executed_task, target_node=task)
        next_action: NextAction = await self.workflow_manager.schedule_next_action(
            goal=self.graph.goal,
            task=task, 
            agent_manager=self.agent_manager, 
            env=self.environment
        )
        if next_action.action_graph is not None:
            await self._async_execute_task_by_action_graph(task=task, next_action=next_action)
        else:
            await self._async_execute_task_by_agents(task=task, next_action=next_action)
        self.graph.completed(node=task)

    async def _async_execute_task_by_action_graph(self, task: WorkFlowNode, next_action: NextAction):
        """
        Asynchronously execute a task using an action graph.
        
        Args:
            task: The workflow node to execute
            next_action: The next action to perform with its action graph
        """
        action_graph: ActionGraph = next_action.action_graph
        async_execute_source = inspect.getsource(action_graph.async_execute)
        if "NotImplementedError" in async_execute_source:
            execute_function = action_graph.execute
            async_execute = False
        else:
            execute_function = action_graph.async_execute
            async_execute = True
        # execute_signature = inspect.signature(type(action_graph).async_execute)
        execute_signature = inspect.signature(execute_function)
        execute_params = {}
        action_input_data = self.environment.get_all_execution_data() 
        for param_name, param_obj in execute_signature.parameters.items():
            if param_name in ["self", "args", "kwargs"]:
                continue
            # execute_params.append(param)
            if param_name in action_input_data:
                execute_params[param_name] = action_input_data[param_name]
            elif param_obj.default is not param_obj.empty:
                execute_params[param_name] = param_obj.default 
            else:
                execute_params[param_name] = None
        # action_input_data = self.environment.get_all_execution_data()
        # execute_inputs = {param: action_input_data.get(param, "") for param in execute_params}
        # action_graph_output: dict = await action_graph.async_execute(**execute_inputs)
        if async_execute:
            action_graph_output: dict = await action_graph.async_execute(**execute_params)
        else:
            action_graph_output: dict = action_graph.execute(**execute_params)

        message = Message(
            content=action_graph_output, action=action_graph.name, msg_type=MessageType.RESPONSE,
            wf_goal=self.graph.goal, wf_task=task.name, wf_task_desc=task.description
        )
        self.environment.update(message=message, state=TrajectoryState.COMPLETED)
    
    async def _async_execute_task_by_agents(self, task: WorkFlowNode, next_action: NextAction):
        """
        Asynchronously execute a task using agents.
        
        Args:
            task: The workflow node to execute
            next_action: The next action to perform using agents
        """
        num_execution = 0 
        while next_action:
            if num_execution >= self.max_execution_steps:
                raise ValueError(
                    f"Maximum number of steps ({self.max_execution_steps}) reached when executing {task.name}. "
                    "Please check the workflow structure (e.g., inputs and outputs of the nodes and the agents) "
                    "or increase the `max_execution_steps` parameter."
                )
            agent: Agent = self.agent_manager.get_agent(agent_name=next_action.agent)
            if not self.agent_manager.wait_for_agent_available(agent_name=agent.name, timeout=300):
                raise TimeoutError(f"Timeout waiting for agent {agent.name} to become available")
            self.agent_manager.set_agent_state(agent_name=next_action.agent, new_state=AgentState.RUNNING)
            try:
                # message = await agent.async_execute(
                #     action_name=next_action.action,
                #     action_input_data=self.environment.get_all_execution_data(),
                #     return_msg_type=MessageType.RESPONSE, 
                #     wf_goal=self.graph.goal,
                #     wf_task=task.name, 
                #     wf_task_desc=task.description
                # )
                message = await self._async_execute_action(task=task, agent=agent, next_action=next_action)
                self.environment.update(message=message, state=TrajectoryState.COMPLETED)
            finally:
                self.agent_manager.set_agent_state(agent_name=next_action.agent, new_state=AgentState.AVAILABLE)
            if self.is_task_completed(task=task):
                break
            next_action: NextAction = await self.workflow_manager.schedule_next_action(
                goal=self.graph.goal,
                task=task,
                agent_manager=self.agent_manager, 
                env=self.environment
            )
            num_execution += 1 

    async def _async_execute_action(self, task: WorkFlowNode, agent: Agent, next_action: NextAction) -> Message:
        """
        Asynchronously execute an action using an agent.
        """
        action_name = next_action.action
        all_execution_data = self.environment.get_all_execution_data()

        # hitl part
        if hasattr(self, "hitl_manager") and (self.hitl_manager is not None):
            hitl_manager = self.hitl_manager
        else:
            hitl_manager = None

        action_inputs_format = agent.get_action(action_name).inputs_format
        action_input_data = {} 
        if action_inputs_format:
            for input_name in action_inputs_format.get_attrs():
                if input_name in all_execution_data:
                    action_input_data[input_name] = all_execution_data[input_name]
            action_required_input_names = action_inputs_format.get_required_input_names()
            if not all(inp in action_input_data for inp in action_required_input_names):
                # could not find all the required inputs in the execution data
                predecessors = self.graph.get_node_predecessors(node=task)
                predecessors_messages = self.environment.get_task_messages(
                    tasks=predecessors + [task.name], include_inputs=True
                )
                predecessors_messages = [
                    message for message in predecessors_messages 
                    if message.msg_type in [MessageType.INPUT, MessageType.RESPONSE]
                ]
                message, extracted_data = await agent.async_execute(
                    action_name=action_name, 
                    msgs=predecessors_messages,
                    return_msg_type=MessageType.RESPONSE,
                    return_action_input_data=True,
                    wf_goal=self.graph.goal,
                    wf_task=task.name,
                    wf_task_desc=task.description,
                    hitl_manager=hitl_manager
                )
                self.environment.update_execution_data_from_context_extraction(extracted_data)
                return message
        
        message = await agent.async_execute(
            action_name=action_name,
            action_input_data=action_input_data,
            return_msg_type=MessageType.RESPONSE,
            wf_goal=self.graph.goal,
            wf_task=task.name,
            wf_task_desc=task.description,
            hitl_manager=hitl_manager
        )
        return message
    
    def is_task_completed(self, task: WorkFlowNode) -> bool:
        task_outputs = [output.name for output in task.outputs]
        current_execution_data = self.environment.get_all_execution_data()
        return all(output in current_execution_data for output in task_outputs)
    
    def _validate_workflow_structure(self, inputs: dict, **kwargs):

        # check the inputs and outputs of the nodes 
        input_names = set(inputs.keys())
        for node in self.graph.nodes:
            node_input_names = deepcopy(input_names)
            is_initial_node = True
            for name in self.graph.get_node_predecessors(node):
                is_initial_node = False 
                predecessor = self.graph.get_node(name)
                node_input_names.update(predecessor.get_output_names())
            node_required_input_names = set(node.get_input_names(required=True))
            if not all(input_name in node_input_names for input_name in node_required_input_names):
                missing_required_inputs = node_required_input_names - node_input_names 
                if is_initial_node:
                    raise ValueError(
                        f"The initial node '{node.name}' is missing required inputs: {list(missing_required_inputs)}. "
                        "You should provide these inputs by specifying the `inputs={'input_name': 'input_value'}` parameter in the `execute` method, "
                        "or return the valid inputs in the `collate_func` when using `Evaluator`."
                    )
                else:
                    raise ValueError(
                        f"The node '{node.name}' is missing required inputs: {list(missing_required_inputs)}. "
                        f"You may need to check the `inputs` and `outputs` of the nodes to ensure that all the required inputs of node '{node.name}' are provided "
                        f"by either its predecessors or the `inputs` parameter in the `execute` method."
                    )
        
        for node in self.graph.nodes:
            for agent in node.agents:
                if hasattr(agent, "forbidden_in_workflow") and (agent.forbidden_in_workflow):
                    raise ValueError(f"The Agent of class {agent.__class__} is forbidden to be used in the workflow.")

    def _prepare_single_hitl_agent(self, agent: Agent, node: WorkFlowNode):
        """
        add complementary information and settings which need dynamically setting up to a single hitl agent
        For example, the `inputs_format` attribute, this needs a dynamical setting up.
        Up to Now, we only consider a HITL agent must be the only agent in its WorkFlowNode instance, this condition may be changed in the future
        Args:
            agent (Agent): a single HITL Agent instance 
            node (WorkFlowNode): a single WorkFlowNode instane which contains exactly the agent of previous param.
        """
        predecessors: List[str] = self.graph.get_node_predecessors(node)
        hitl_action = None
        for action in agent.actions:
            if (action.inputs_format) and (action.outputs_format):
                continue
            elif hasattr(action, "interaction_type"):
                hitl_action = action
                break
        if not hitl_action:
            raise ValueError(f"Can not find a HITL action in agent {agent}")

        hitl_inputs_data_fields = {}

        # set up inputs_format and outputs_format
        for predecessor in predecessors:
            predecessor_node = self.graph.get_node(predecessor)
            for param in predecessor_node.outputs:
                if param.required:
                    hitl_inputs_data_fields[param.name] = (str, Field(description=param.description))
                else:
                    hitl_inputs_data_fields[param.name] = (Optional[str], Field(description=param.description))
        inputs_format = create_model(
            agent._get_unique_class_name(
                generate_dynamic_class_name(hitl_action.class_name+" action_input")
            ),
            **(hitl_inputs_data_fields or {}),
            __base__= ActionInput
        )
        
        successors: List[str] = self.graph.get_node_children(node)
        hitl_outputs_data_fields = {}
        if successors == []:
            # hitl node as the ending node, not allowed for now
            raise ValueError("WorkFlowNode with a HITL Agent can not be set as the ending node.")
        for successor in successors:
            successor_node = self.graph.get_node(successor)
            for param in successor_node.inputs:
                if param.required:
                    hitl_outputs_data_fields[param.name] = (str, Field(description=param.description))
                else:
                    hitl_outputs_data_fields[param.name] = (Optional[str], Field(description=param.description))
        outputs_format = create_model(
            agent._get_unique_class_name(
                generate_dynamic_class_name(hitl_action.class_name+" action_output")
            ),
            **(hitl_outputs_data_fields or {}),
            __base__=ActionOutput
        )
        hitl_action.inputs_format = inputs_format
        hitl_action.outputs_format = outputs_format

        ## check hitl data field mapping
        if self.hitl_manager.hitl_input_output_mapping is None:
            raise ValueError("hitl_input_output_mapping attribute missing in HITLManager instance.")
        return

    def _prepare_hitl(self):
        """
        Prepare hitl settings before executing the WorkFlow
        """
        if self.hitl_manager is None:
            return
        hitl_agents: List[Agent]  = []
        node_with_hitl_agents = []
        for node in self.graph.nodes:
            agents = node.agents
            found_hitl_agent = False
            for agent in agents:
                # transfer to Agent instance
                if isinstance(agent, dict):
                    agent = self.agent_manager.get_agent(self.agent_manager.get_agent_name(agent))
                elif isinstance(agent, str):
                    agent = self.agent_manager.get_agent(agent)
                elif isinstance(agent, Agent):
                    pass
                # judgement
                if isinstance(agent, HITLBaseAgent):
                    found_hitl_agent = True
                    if agent not in hitl_agents:
                        hitl_agents.append(agent)
            if found_hitl_agent:
                node_with_hitl_agents.append(node)
                found_hitl_agent = False

        # Up to Now, we only consider a HITL agent must be the only agent in its WorkFlowNode instance, this condition may be changed in the future
        if len(hitl_agents) != len(node_with_hitl_agents):
            raise ValueError("Incorrect WorkFlowNode definition: A HITL Agent must be the only agent in its WorkFlowNode instance")

        # add complementary information and settings which need dynamically setting up to hitl agents
        for agent, node in zip(hitl_agents, node_with_hitl_agents):
            self._prepare_single_hitl_agent(agent, node)

        return 