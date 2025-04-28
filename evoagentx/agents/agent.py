from pydantic import Field
from typing import Type, Optional, Union, Tuple, List

from ..core.module import BaseModule
from ..core.module_utils import generate_id
from ..core.message import Message, MessageType
from ..core.registry import MODEL_REGISTRY
from ..core.parser import Parser
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
        self._save_ignore_fields = ["llm"]
        self.init_context_extractor()

    def execute(
        self, 
        action_name: str, 
        msgs: Optional[List[Message]] = None, 
        action_input_data: Optional[dict] = None, 
        return_msg_type: Optional[MessageType] = MessageType.UNKNOWN,
        **kwargs
    ) -> Message:
        """Execute an action with the given context and return results.

        This is the core method for agent functionality, allowing it to perform actions
        based on the current conversation context. The method:
        1. Updates short-term memory with provided messages
        2. Extracts input data for the action if not provided
        3. Executes the action using the language model
        4. Creates a message with the results
        5. Updates short-term memory with the results

        Args:
            action_name: The name of the action to execute
            msgs: Optional list of messages providing context for the action
            action_input_data: Optional pre-extracted input data for the action
            return_msg_type: Message type for the return message
            **kwargs: Additional parameters, may include workflow information
        
        Returns:
            Message: A message containing the execution results
            
        Raises:
            AssertionError: If neither msgs nor action_input_data is provided
            KeyError: If the action_name is invalid
            
        Notes:
            - Either msgs or action_input_data must be provided
            - The action's results are formatted as a Message object
            - The message is added to short-term memory before being returned
        """
        assert msgs is not None or action_input_data is not None, "must provide either `msgs` or `action_input_data` in execute(...)"
        action = self.get_action(action_name=action_name)

        # update short-term memory
        if msgs is not None:
            self.short_term_memory.add_messages(msgs)
        
        # obtain action input data from short term memory
        action_input_data = action_input_data or self.get_action_inputs(action=action)

        # execute action
        execution_results: Tuple[Parser, str] = action.execute(
            llm=self.llm, 
            inputs=action_input_data, 
            sys_msg=self.system_prompt,
            return_prompt=True
        )
        action_output, prompt = execution_results

        # formulate a message
        message = Message(
            # content=action_output.to_str(),
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
    
    def init_llm(self):
        """
        Initialize the language model for the agent.
        """
        assert self.llm_config or self.llm, "must provide either 'llm_config' or 'llm' when is_human=False"
        if self.llm_config and not self.llm:
            llm_cls = MODEL_REGISTRY.get_model(self.llm_config.llm_type)
            self.llm = llm_cls(config=self.llm_config)
        if self.llm:
            self.llm_config = self.llm.config

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
        
    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        """Save the agent to persistent storage.
                
        Args:
            path: Path where the agent should be saved
            ignore: List of field names to exclude from serialization
            **kwargs: Additional parameters for the save operation
            
        Returns:
            The path where the agent was saved
        """
        ignore_fields = self._save_ignore_fields + ignore
        super().save_module(path=path, ignore=ignore_fields, **kwargs)

    @classmethod
    def load_module(cls, path: str, llm_config: LLMConfig, **kwargs):
        agent = super().load_module(path=path, **kwargs)
        agent["llm_config"] = llm_config.to_dict()
        return agent 