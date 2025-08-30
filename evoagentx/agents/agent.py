import asyncio
import inspect 
from pydantic import Field
from typing import Type, Optional, Union, Tuple, List, Any, Coroutine

from ..core.module import BaseModule
from ..core.module_utils import generate_id
from ..core.message import Message, MessageType
from ..core.registry import MODEL_REGISTRY
from ..models.model_configs import LLMConfig
from ..models.base_model import BaseLLM
from ..memory.memory import ShortTermMemory
from ..memory.long_term_memory import LongTermMemory
from ..memory.memory_manager import MemoryManager
from ..storages.base import StorageHandler
from ..actions.action import Action
from ..actions.action import ContextExtraction


class Agent(BaseModule):
    """
    Base class for all agents. 
    
    Attributes:
        name (str): Unique identifier for the agent
        description (str): Human-readable description of the agent's purpose
        llm_config (Optional[LLMConfig]): Configuration for the language model. If provided, a new LLM instance will be created. 
            Otherwise, the existing LLM instance specified in the `llm` field will be used.   
        llm (Optional[BaseLLM]): Language model instance. If provided, the existing LLM instance will be used. 
        agent_id (Optional[str]): Unique ID for the agent, auto-generated if not provided
        system_prompt (Optional[str]): System prompt for the Agent.
        actions (List[Action]): List of available actions
        n (Optional[int]): Number of latest messages used to provide context for action execution. It uses all the messages in short term memory by default. 
        is_human (bool): Whether this agent represents a human user
        version (int): Version number of the agent, default is 0. 
    """

    name: str # should be unique
    description: str
    llm_config: Optional[LLMConfig] = None
    llm: Optional[BaseLLM] = None
    agent_id: Optional[str] = Field(default_factory=generate_id)
    system_prompt: Optional[str] = None
    short_term_memory: Optional[ShortTermMemory] = Field(default_factory=ShortTermMemory) # store short term memory for a single workflow.
    use_long_term_memory: Optional[bool] = False
    storage_handler: Optional[StorageHandler] = None
    long_term_memory: Optional[LongTermMemory] = None
    long_term_memory_manager: Optional[MemoryManager] = None
    actions: List[Action] = Field(default=None)
    n: int = Field(default=None, description="number of latest messages used to provide context for action execution. It uses all the messages in short term memory by default.")
    is_human: bool = Field(default=False)
    version: int = 0 

    def init_module(self):
        if not self.is_human:
            self.init_llm()
        if self.use_long_term_memory:
            self.init_long_term_memory()
        self.actions = [] if self.actions is None else self.actions
        self._action_map = {action.name: action for action in self.actions} if self.actions else dict()
        self._save_ignore_fields = ["llm", "llm_config"]
        self.init_context_extractor()

    # def __call__(self, *args, **kwargs) -> Message:
    #     """Make the agent callable and automatically choose between sync and async execution"""
    #     if asyncio.iscoroutinefunction(self.async_execute) and asyncio.get_event_loop().is_running():
    #         # If the operator is in an asynchronous environment and has an execute_async method, return a coroutine
    #         return self.async_execute(*args, **kwargs)
    #     # Otherwise, use the synchronous method
    #     return self.execute(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Union[dict, Coroutine[Any, Any, dict]]:
        """Make the operator callable and automatically choose between sync and async execution."""
        try:
            # Safe way to check if we're inside an async environment
            asyncio.get_running_loop()
            return self.async_execute(*args, **kwargs)
        except RuntimeError:
            # No running loop — likely in sync context or worker thread
            return self.execute(*args, **kwargs)
    
    def _prepare_execution(
        self,
        action_name: str,
        msgs: Optional[List[Message]] = None,
        action_input_data: Optional[dict] = None,
        **kwargs
    ) -> Tuple[Action, dict]:
        """Prepare for action execution by updating memory and getting inputs.
        
        Helper method used by both execute and aexecute methods.
        
        Args:
            action_name: The name of the action to execute
            msgs: Optional list of messages providing context for the action
            action_input_data: Optional pre-extracted input data for the action
            **kwargs: Additional workflow parameters
            
        Returns:
            Tuple containing the action object and input data
            
        Raises:
            AssertionError: If neither msgs nor action_input_data is provided
        """
        assert msgs is not None or action_input_data is not None, "must provide either `msgs` or `action_input_data`"
        action = self.get_action(action_name=action_name)

        # update short-term memory
        if msgs is not None:
            # directly add messages to short-term memory
            self.short_term_memory.add_messages(msgs)
        if action_input_data is not None:
            # create a message from action_input_data and add it to short-term memory
            input_message = Message(
                content = action_input_data,
                next_actions = [action_name],
                msg_type = MessageType.INPUT, 
                wf_goal = kwargs.get("wf_goal", None),
                wf_task = kwargs.get("wf_task", None),
                wf_task_desc = kwargs.get("wf_task_desc", None)
            )
            self.short_term_memory.add_message(input_message)
        
        # obtain action input data from short term memory if not provided
        action_input_data = action_input_data or self.get_action_inputs(action=action)
        
        return action, action_input_data
    
    def _create_output_message(
        self,
        action_output,
        prompt: str,
        action_name: str,
        return_msg_type: Optional[MessageType] = MessageType.UNKNOWN,
        **kwargs
    ) -> Message:
        """Create a message from execution results and update memory.
        
        Helper method used by both execute and aexecute methods.
        
        Args:
            action_output: The output from action execution
            prompt: The prompt used for execution
            action_name: The name of the executed action
            return_msg_type: Message type for the return message
            **kwargs: Additional workflow parameters
            
        Returns:
            Message object containing execution results
        """
        # formulate a message
        message = Message(
            content=action_output, 
            agent=self.name,
            action=action_name,
            prompt=prompt, 
            msg_type=return_msg_type,
            wf_goal = kwargs.get("wf_goal", None),
            wf_task = kwargs.get("wf_task", None),
            wf_task_desc = kwargs.get("wf_task_desc", None)
        )

        # update short-term memory
        self.short_term_memory.add_message(message)
        
        return message
    
    async def async_execute(
        self, 
        action_name: str, 
        msgs: Optional[List[Message]] = None, 
        action_input_data: Optional[dict] = None, 
        return_msg_type: Optional[MessageType] = MessageType.UNKNOWN,
        return_action_input_data: Optional[bool] = False, 
        **kwargs
    ) -> Union[Message, Tuple[Message, dict]]:
        """Execute an action asynchronously with the given context and return results.

        This is the async version of the execute method, allowing it to perform actions
        based on the current conversation context.

        Args:
            action_name: The name of the action to execute
            msgs: Optional list of messages providing context for the action
            action_input_data: Optional pre-extracted input data for the action
            return_msg_type: Message type for the return message
            **kwargs (Any): Additional parameters, may include workflow information
        
        Returns:
            Message: A message containing the execution results
        """
        action, action_input_data = self._prepare_execution(
            action_name=action_name,
            msgs=msgs,
            action_input_data=action_input_data,
            **kwargs
        )

        # execute action asynchronously
        async_execute_source = inspect.getsource(action.async_execute)
        if "NotImplementedError" in async_execute_source:
            # if the async_execute method is not implemented, use the execute method instead
            execution_results = action.execute(
                llm=self.llm, 
                inputs=action_input_data, 
                sys_msg=self.system_prompt,
                return_prompt=True,
                **kwargs
            )
        else:
            execution_results = await action.async_execute(
                llm=self.llm, 
                inputs=action_input_data, 
                sys_msg=self.system_prompt,
                return_prompt=True,
                **kwargs
        )
        action_output, prompt = execution_results

        message = self._create_output_message(
            action_output=action_output,
            prompt=prompt,
            action_name=action_name,
            return_msg_type=return_msg_type,
            **kwargs
        )
        if return_action_input_data:
            return message, action_input_data
        return message
    
    def execute(
        self, 
        action_name: str, 
        msgs: Optional[List[Message]] = None, 
        action_input_data: Optional[dict] = None, 
        return_msg_type: Optional[MessageType] = MessageType.UNKNOWN,
        return_action_input_data: Optional[bool] = False, 
        **kwargs
    ) -> Union[Message, Tuple[Message, dict]]:
        """Execute an action with the given context and return results.

        This is the core method for agent functionality, allowing it to perform actions
        based on the current conversation context.

        Args:
            action_name: The name of the action to execute
            msgs: Optional list of messages providing context for the action
            action_input_data: Optional pre-extracted input data for the action
            return_msg_type: Message type for the return message
            **kwargs (Any): Additional parameters, may include workflow information
        
        Returns:
            Message: A message containing the execution results
        """
        action, action_input_data = self._prepare_execution(
            action_name=action_name,
            msgs=msgs,
            action_input_data=action_input_data,
            **kwargs
        )

        # execute action
        execution_results = action.execute(
            llm=self.llm, 
            inputs=action_input_data, 
            sys_msg=self.system_prompt,
            return_prompt=True,
            **kwargs
        )
        action_output, prompt = execution_results

        message = self._create_output_message(
            action_output=action_output,
            prompt=prompt,
            action_name=action_name,
            return_msg_type=return_msg_type,
            **kwargs
        )
        if return_action_input_data:
            return message, action_input_data
        return message
    
    def init_llm(self):
        """
        Initialize the language model for the agent.
        """
        # Only initialize LLM if not human and LLM is provided
        if not self.is_human and (not self.llm_config and not self.llm):
            raise ValueError("must provide `llm_config` or `llm` when `is_human` is False")
        if not self.is_human and (self.llm_config or self.llm):
            if self.llm_config and not self.llm:
                llm_cls = MODEL_REGISTRY.get_model(self.llm_config.llm_type)
                self.llm = llm_cls(config=self.llm_config)
            if self.llm:
                self.llm_config = self.llm.config
        # If is_human=True or no LLM provided, self.llm remains None

    def init_long_term_memory(self):
        """
        Initialize long-term memory components.
        """
        assert self.storage_handler is not None, "must provide ``storage_handler`` when use_long_term_memory=True"
        # TODO revise the initialisation of long_term_memory and long_term_memory_manager
        if not self.long_term_memory:
            self.long_term_memory = LongTermMemory()
        if not self.long_term_memory_manager:
            self.long_term_memory_manager = MemoryManager(
                storage_handler=self.storage_handler,
                memory=self.long_term_memory
            )
    
    def init_context_extractor(self):
        """
        Initialize the context extraction action.
        """
        cext_action = ContextExtraction()
        self.cext_action_name = cext_action.name
        self.add_action(cext_action)

    def add_action(self, action: Type[Action]):
        """
        Add a new action to the agent's available actions.

        Args:
            action: The action instance to add
        """
        action_name  = action.name
        if action_name in self._action_map:
            return
        self.actions.append(action)
        self._action_map[action_name] = action

    def check_action_name(self, action_name: str):
        """
        Check if an action name is valid for this agent.
                
        Args:
            action_name: Name of the action to check
        """
        if action_name not in self._action_map:
            raise KeyError(f"'{action_name}' is an invalid action for {self.name}! Available action names: {list(self._action_map.keys())}")
    
    def get_action(self, action_name: str) -> Action:
        """
        Retrieves the Action instance associated with the given name.
        
        Args:
            action_name: Name of the action to retrieve
            
        Returns:
            The Action instance with the specified name
        """
        self.check_action_name(action_name=action_name)
        return self._action_map[action_name]
    
    def get_action_name(self, action_cls: Type[Action]) -> str:
        """
        Searches through the agent's actions to find one matching the specified type.
        
        Args:
            action_cls: The Action class type to search for
            
        Returns:
            The name of the matching action
        """
        for name, action in self._action_map.items():
            if isinstance(action, action_cls):
                return name
        raise ValueError(f"Couldn't find an action that matches Type '{action_cls.__name__}'")
    
    def get_action_inputs(self, action: Action) -> Union[dict, None]:
        """
        Uses the context extraction action to determine appropriate inputs
        for the specified action based on the conversation history.
        
        Args:
            action: The action for which to extract inputs
            
        Returns:
            Dictionary of extracted input data, or None if extraction fails
        """
        # return the input data of an action.
        context = self.short_term_memory.get(n=self.n)
        cext_action = self.get_action(self.cext_action_name)
        action_inputs = cext_action.execute(llm=self.llm, action=action, context=context)
        return action_inputs
    
    def get_all_actions(self) -> List[Action]:
        """Get all actions except the context extraction action.
        
        Returns:
            List of Action instances available for execution
        """
        actions = [action for action in self.actions if action.name != self.cext_action_name]
        return actions
    
    def get_agent_profile(self, action_names: List[str] = None) -> str:
        """Generate a human-readable profile of the agent and its capabilities.
        
        Args:
            action_names: Optional list of action names to include in the profile.
                          If None, all actions are included.
            
        Returns:
            A formatted string containing the agent profile
        """
        all_actions = self.get_all_actions()
        if action_names is None:
            # if `action_names` is None, return description of all actions 
            action_descriptions = "\n".join([f"  - {action.name}: {action.description}" for action in all_actions])
        else: 
            # otherwise, only return description of actions that matches `action_names`
            action_descriptions = "\n".join([f"  - {action.name}: {action.description}" for action in all_actions if action.name in action_names])
        profile = f"Agent Name: {self.name}\nDescription: {self.description}\nAvailable Actions:\n{action_descriptions}"
        return profile

    def clear_short_term_memory(self):
        """
        Remove all content from the agent's short-term memory.
        """
        pass 
        
    def __eq__(self, other: "Agent"):
        return self.agent_id == other.agent_id

    def __hash__(self):
        return self.agent_id
    
    def get_prompts(self) -> dict:
        """
        Get all the prompts of the agent.
        
        Returns:
            dict: A dictionary with keys in the format 'agent_name::action_name' and values
                containing the system_prompt and action prompt.
        """
        prompts = {}
        for action in self.get_all_actions():
            prompts[action.name] = {
                "system_prompt": self.system_prompt, 
                "prompt": action.prompt
            }
        return prompts
    
    def set_prompt(self, action_name: str, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """
        Set the prompt for a specific action of this agent.
        
        Args:
            action_name: Name of the action whose prompt should be updated
            prompt: New prompt text to set for the action
            system_prompt: Optional new system prompt to set for the agent
            
        Returns:
            bool: True if the prompt was successfully updated, False otherwise
            
        Raises:
            KeyError: If the action_name does not exist for this agent
        """
        try:
            action = self.get_action(action_name)
            action.prompt = prompt
            
            if system_prompt is not None:
                self.system_prompt = system_prompt
                
            return True
        except KeyError:
            raise KeyError(f"Action '{action_name}' not found in agent '{self.name}'")
        
    def set_prompts(self, prompts: dict) -> bool:
        """
        Set the prompts for all actions of this agent.
        
        Args:
            prompts: A dictionary with keys in the format 'action_name' and values
                containing the system_prompt and action prompt.
        
        Returns:
            bool: True if the prompts were successfully updated, False otherwise
        """
        for action_name, prompt_data in prompts.items():
            # self.set_prompt(action_name, prompt_data["prompt"], prompt_data["system_prompt"])
            if not isinstance(prompt_data, dict):
                raise ValueError(f"Invalid prompt data for action '{action_name}'. Expected a dictionary with 'prompt' and 'system_prompt' (optional) keys.")
            if "prompt" not in prompt_data:
                raise ValueError(f"Missing 'prompt' key in prompt data for action '{action_name}'.")
            self.set_prompt(action_name, prompt_data["prompt"], prompt_data.get("system_prompt", None))
        return True

    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        """Save the agent to persistent storage.
                
        Args:
            path: Path where the agent should be saved
            ignore: List of field names to exclude from serialization
            **kwargs (Any): Additional parameters for the save operation
            
        Returns:
            The path where the agent was saved
        """
        ignore_fields = self._save_ignore_fields + ignore
        super().save_module(path=path, ignore=ignore_fields, **kwargs)

    @classmethod
    def load_module(cls, path: str, llm_config: LLMConfig = None, **kwargs) -> "Agent":
        """
        load the agent from local storage. Must provide `llm_config` when loading the agent from local storage. 

        Args:
            path: The path of the file
            llm_config: The LLMConfig instance
        
        Returns:
            Agent: The loaded agent instance
        """
        agent = super().load_module(path=path, **kwargs)
        if llm_config is not None:
            agent["llm_config"] = llm_config.to_dict()
        return agent 
    
    def get_config(self) -> dict:
        """
        Get a dictionary containing all necessary configuration to recreate this agent.
        
        Returns:
            dict: A configuration dictionary that can be used to initialize a new Agent instance
            with the same properties as this one.
        """
        config = self.to_dict()
        return config