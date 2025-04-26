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
    """Represents an intelligent agent capable of executing actions and maintaining conversation memory.
    
    An Agent serves as the main executor of actions in the workflow, with capabilities to:
    - Execute actions using an underlying language model
    - Maintain short-term memory of conversation context
    - Optionally use long-term memory for persistent information
    - Extract context for action execution
    - Manage a collection of available actions
    
    Attributes:
        name (str): Unique identifier for the agent
        description (str): Human-readable description of the agent's purpose
        llm_config (LLMConfig): Configuration for the language model
        llm (BaseLLM): Language model instance 
        agent_id (str): Unique ID for the agent, auto-generated if not provided
        system_prompt (str): System prompt for the language model
        short_term_memory (ShortTermMemory): Memory for current conversation
        use_long_term_memory (bool): Whether to use persistent memory
        storage_handler (StorageHandler): Handler for storage operations
        long_term_memory (LongTermMemory): Persistent memory storage
        long_term_memory_manager (MemoryManager): Manager for long-term memory
        actions (List[Action]): List of available actions
        n (int): Number of latest messages to use for context
        is_human (bool): Whether this agent represents a human user
        version (int): Version number of the agent
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
        """Initialize the agent's components.
        
        This method sets up the various components of the agent:
        - Language model (if not a human agent)
        - Long-term memory (if enabled)
        - Actions list and map
        - Context extractor for action execution
        
        Notes:
            - Called automatically during instantiation
            - Sets up internal data structures for efficient operation
            - Creates a mapping of action names to action objects for quick lookup
        """
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
        """Initialize the language model for the agent.
        
        Sets up the language model based on the provided configuration or
        uses the existing model if already instantiated.
        
        Raises:
            AssertionError: If neither llm_config nor llm is provided for a non-human agent
            
        Notes:
            - Uses the MODEL_REGISTRY to instantiate the appropriate model type
            - Ensures the llm_config attribute is synchronized with the model
        """
        assert self.llm_config or self.llm, "must provide either 'llm_config' or 'llm' when is_human=False"
        if self.llm_config and not self.llm:
            llm_cls = MODEL_REGISTRY.get_model(self.llm_config.llm_type)
            self.llm = llm_cls(config=self.llm_config)
        if self.llm:
            self.llm_config = self.llm.config

    def init_long_term_memory(self):
        """Initialize long-term memory components.
        
        Sets up the long-term memory and memory manager if long-term memory
        usage is enabled.
        
        Raises:
            AssertionError: If storage_handler is not provided when use_long_term_memory is True
            
        Notes:
            - Creates default instances if not provided
            - Connects the memory to the storage handler
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
        """Initialize the context extraction action.
        
        Creates and adds a ContextExtraction action to the agent, which is
        used to extract relevant context from conversation history for other actions.
        
        Notes:
            - This action is a special utility action, not directly executed by users
            - Stored with a unique name for internal reference
        """
        cext_action = ContextExtraction()
        self.cext_action_name = cext_action.name
        self.add_action(cext_action)

    def add_action(self, action: Type[Action]):
        """Add a new action to the agent's available actions.
        
        Registers an action with the agent, making it available for execution.
        Does nothing if an action with the same name already exists.
        
        Args:
            action: The action instance to add
            
        Notes:
            - Actions are identified by their name attribute
            - Duplicate actions (by name) are silently ignored
            - Updates both the actions list and the action map
        """
        action_name  = action.name
        if action_name in self._action_map:
            return
        self.actions.append(action)
        self._action_map[action_name] = action

    def check_action_name(self, action_name: str):
        """Check if an action name is valid for this agent.
        
        Verifies that the given action name exists in the agent's action map.
        
        Args:
            action_name: Name of the action to check
            
        Raises:
            KeyError: If the action name is not found in the agent's actions
            
        Notes:
            - Used for validation before attempting to retrieve or execute an action
        """
        if action_name not in self._action_map:
            raise KeyError(f"'{action_name}' is an invalid action for {self.name}! Available action names: {list(self._action_map.keys())}")
    
    def get_action(self, action_name: str) -> Action:
        """Get an action by name.
        
        Retrieves the Action instance associated with the given name.
        
        Args:
            action_name: Name of the action to retrieve
            
        Returns:
            The Action instance with the specified name
            
        Raises:
            KeyError: If the action name is not found
            
        Notes:
            - Validates the action name before attempting retrieval
        """
        self.check_action_name(action_name=action_name)
        return self._action_map[action_name]
    
    def get_action_name(self, action_cls: Type[Action]) -> str:
        """Find the name of an action by its class type.
        
        Searches through the agent's actions to find one matching the specified type.
        
        Args:
            action_cls: The Action class type to search for
            
        Returns:
            The name of the matching action
            
        Raises:
            ValueError: If no action of the specified type is found
            
        Notes:
            - Useful for finding actions by their class rather than name
            - Returns the first matching action if multiple exist
        """
        for name, action in self._action_map.items():
            if isinstance(action, action_cls):
                return name
        raise ValueError(f"Couldn't find an action that matches Type '{action_cls.__name__}'")
    
    def get_action_inputs(self, action: Action) -> Union[dict, None]:
        """Extract input data for an action from conversation context.
        
        Uses the context extraction action to determine appropriate inputs
        for the specified action based on the conversation history.
        
        Args:
            action: The action for which to extract inputs
            
        Returns:
            Dictionary of extracted input data, or None if extraction fails
            
        Notes:
            - Retrieves context from short-term memory based on the agent's 'n' setting
            - Relies on the context extraction action to process the raw context
        """
        # return the input data of an action.
        context = self.short_term_memory.get(n=self.n)
        cext_action = self.get_action(self.cext_action_name)
        action_inputs = cext_action.execute(llm=self.llm, action=action, context=context)
        return action_inputs
    
    def get_all_actions(self) -> List[Action]:
        """Get all actions except the context extraction action.
        
        Retrieves a list of all actions available to the agent, excluding
        the special context extraction action used internally.
        
        Returns:
            List of Action instances available for execution
            
        Notes:
            - Filters out the context extraction action which is not meant for direct use
            - Used for generating agent documentation and interfaces
        """
        actions = [action for action in self.actions if action.name != self.cext_action_name]
        return actions
    
    def get_agent_profile(self, action_names: List[str] = None) -> str:
        """Generate a human-readable profile of the agent and its capabilities.
        
        Creates a string describing the agent, including its name, description,
        and available actions (optionally filtered by name).
        
        Args:
            action_names: Optional list of action names to include in the profile.
                          If None, all actions are included.
            
        Returns:
            A formatted string containing the agent profile
            
        Notes:
            - Useful for generating documentation or help text
            - Describes each action with its name and description
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
        """Remove all content from the agent's short-term memory.
        
        Resets the agent's conversation context by clearing its short-term memory.
        
        Notes:
            - Placeholder implementation (pass) - needs implementation
            - Useful for starting fresh conversations or resetting context
        """
        pass 
        
    def __eq__(self, other: "Agent"):
        """Compare two Agent instances for equality.
        
        Agents are considered equal if they have the same agent_id.
        
        Args:
            other: Another Agent instance to compare with
            
        Returns:
            True if the agents have the same ID, False otherwise
            
        Notes:
            - Used for comparing agents in collections
            - Equality is based solely on identity (agent_id), not capabilities
        """
        return self.agent_id == other.agent_id

    def __hash__(self):
        """Generate a hash value for the Agent instance.
        
        The hash is based on the agent's unique ID.
        
        Returns:
            Hash value for the agent
            
        Notes:
            - Allows Agent instances to be used in sets and as dictionary keys
            - Consistent with the equality implementation
        """
        return self.agent_id
        
    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        """Save the agent to persistent storage.
        
        Serializes and saves the agent's state to the specified path.
        
        Args:
            path: Path where the agent should be saved
            ignore: List of field names to exclude from serialization
            **kwargs: Additional parameters for the save operation
            
        Returns:
            The path where the agent was saved
            
        Notes:
            - Extends the parent class save_module method
            - Automatically ignores the LLM field which may not be serializable
            - Combines user-provided ignore list with internal _save_ignore_fields
        """
        ignore_fields = self._save_ignore_fields + ignore
        super().save_module(path=path, ignore=ignore_fields, **kwargs)

    @classmethod
    def load_module(cls, path: str, llm_config: LLMConfig, **kwargs):
        agent = super().load_module(path=path, **kwargs)
        agent["llm_config"] = llm_config.to_dict()
        return agent 