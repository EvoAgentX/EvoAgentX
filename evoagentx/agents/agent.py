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
        """
        Execute an action.

        Args:
            action_name (str): the name of the action to execute. 
            msgs (List[Message]): the context for the current task. 
        
        Returns:
            Message: a message that contains the execution results. 
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
        assert self.llm_config or self.llm, "must provide either 'llm_config' or 'llm' when is_human=False"
        if self.llm_config and not self.llm:
            llm_cls = MODEL_REGISTRY.get_model(self.llm_config.llm_type)
            self.llm = llm_cls(config=self.llm_config)
        if self.llm:
            self.llm_config = self.llm.config

    def init_long_term_memory(self):
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
        cext_action = ContextExtraction()
        self.cext_action_name = cext_action.name
        self.add_action(cext_action)

    def add_action(self, action: Type[Action]):
        action_name  = action.name
        if action_name in self._action_map:
            return
        self.actions.append(action)
        self._action_map[action_name] = action

    def check_action_name(self, action_name: str):
        if action_name not in self._action_map:
            raise KeyError(f"'{action_name}' is an invalid action for {self.name}! Available action names: {list(self._action_map.keys())}")
    
    def get_action(self, action_name: str) -> Action:
        self.check_action_name(action_name=action_name)
        return self._action_map[action_name]
    
    def get_action_name(self, action_cls: Type[Action]) -> str:
        for name, action in self._action_map.items():
            if isinstance(action, action_cls):
                return name
        raise ValueError(f"Couldn't find an action that matches Type '{action_cls.__name__}'")
    
    def get_action_inputs(self, action: Action) -> Union[dict, None]:
        # return the input data of an action.
        context = self.short_term_memory.get(n=self.n)
        cext_action = self.get_action(self.cext_action_name)
        action_inputs = cext_action.execute(llm=self.llm, action=action, context=context)
        return action_inputs
    
    def get_all_actions(self) -> List[Action]:
        actions = [action for action in self.actions if action.name != self.cext_action_name]
        return actions
    
    def get_agent_profile(self, action_names: List[str] = None) -> str:
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
        remove all the content in the short term memory.
        """
        pass 
        
    def __eq__(self, other: "Agent"):
        return self.agent_id == other.agent_id

    def __hash__(self):
        return self.agent_id
        
    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        ignore_fields = self._save_ignore_fields + ignore
        super().save_module(path=path, ignore=ignore_fields, **kwargs)
