import threading
from enum import Enum
from typing import Union, Optional, Dict, List
from pydantic import Field
from copy import deepcopy

from .agent import Agent
# from .agent_generator import AgentGenerator
from .customize_agent import CustomizeAgent
from ..core.module import BaseModule
from ..core.decorators import atomic_method
from ..storages.base import StorageHandler
from ..models.model_configs import LLMConfig
from ..tools.tool import Tool
class AgentState(str, Enum):
    AVAILABLE = "available"
    RUNNING = "running"


class AgentManager(BaseModule):
    """
    Responsible for creating and managing all Agent objects required for workflow operation.

    Attributes:
        storage_handler (StorageHandler): Used to load and save agents from/to storage.
        agents (List[Agent]): A list to keep track of all managed Agent instances.
        agent_states (Dict[str, AgentState]): A dictionary to track the state of each Agent by name.
    """
    agents: List[Agent] = Field(default_factory=list)
    agent_states: Dict[str, AgentState] = Field(default_factory=dict) # agent_name to AgentState mapping
    storage_handler: Optional[StorageHandler] = None # used to load and save agent from storage.
    # agent_generator: Optional[AgentGenerator] = None # used to generate agents for a specific subtask
    tools: Optional[List[Tool]] = None
    tools_mapping: Optional[Dict[str, Tool]] = None

    def init_module(self):
        self._lock = threading.Lock()
        self._state_conditions = {}
        self.init_tools()
        if self.agents:
            for agent in self.agents:
                self.agent_states[agent.name] = self.agent_states.get(agent.name, AgentState.AVAILABLE)
                if agent.name not in self._state_conditions:
                    self._state_conditions[agent.name] = threading.Condition()
            self.check_agents()
    
    def init_tools(self):
        if self.tools:
            self.tools_mapping = {tool.name: tool for tool in self.tools}
    
    def check_agents(self):
        """Validate agent list integrity and state consistency.
        
        Performs thorough validation of the agent manager's internal state:
        1. Checks for duplicate agent names
        2. Verifies that agent states exist for all agents
        3. Ensures agent list and state dictionary sizes match
        """
        # check that the names of self.agents should be unique
        duplicate_agent_names = self.find_duplicate_agents(self.agents)
        if duplicate_agent_names:
            raise ValueError(f"The agents should be unique. Found duplicate agent names: {duplicate_agent_names}!")
        # check agent states
        if len(self.agents) != len(self.agent_states):
            raise ValueError(f"The lengths of self.agents ({len(self.agents)}) and self.agent_states ({len(self.agent_states)}) are different!")
        missing_agents = self.find_missing_agent_states()
        if missing_agents:
            raise ValueError(f"The following agents' states were not found: {missing_agents}")

    def find_duplicate_agents(self, agents: List[Agent]) -> List[str]:
        # return the names of duplicate agents based on agent.name 
        unique_agent_names = set()
        duplicate_agent_names = set()
        for agent in agents:
            agent_name = agent.name
            if agent_name in unique_agent_names:
                duplicate_agent_names.add(agent_name)
            unique_agent_names.add(agent_name)
        return list(duplicate_agent_names)

    def find_missing_agent_states(self):
        missing_agents = [agent.name for agent in self.agents if agent.name not in self.agent_states]
        return missing_agents

    def list_agents(self) -> List[str]:
        return [agent.name for agent in self.agents]
    
    def has_agent(self, agent_name: str) -> bool:
        """Check if an agent with the given name exists in the manager.
        
        Args:
            agent_name: The name of the agent to check
            
        Returns:
            True if an agent with the given name exists, False otherwise
        """
        all_agent_names = self.list_agents()
        return agent_name in all_agent_names
    
    @property
    def size(self):
        """
        Get the total number of agents managed by this manager.
        """
        return len(self.agents)
    
    def load_agent(self, agent_name: str, **kwargs) -> Agent:
        """Load an agent from local storage through storage_handler.
        
        Retrieves agent data from storage and creates an Agent instance.
        
        Args:
            agent_name: The name of the agent to load
            **kwargs (Any): Additional parameters for agent creation
        
        Returns:
            Agent instance with data loaded from storage
        """
        if not self.storage_handler:
            raise ValueError("must provide ``self.storage_handler`` to use ``load_agent``")
        agent_data = self.storage_handler.load_agent(agent_name=agent_name)
        agent: Agent = self.create_customize_agent(agent_data=agent_data)
        return agent

    def load_all_agents(self, **kwargs):
        """Load all agents from storage and add them to the manager.
        
        Retrieves all available agents from storage and adds them to the
        managed agents collection.
        
        Args:
            **kwargs (Any): Additional parameters passed to storage handler
        """
        pass 
    
    def create_customize_agent(self, agent_data: dict, llm_config: Optional[Union[LLMConfig, dict]]=None, **kwargs) -> CustomizeAgent:
        """
        create a customized agent from the provided `agent_data`. 

        Args:
            agent_data: The data used to create an Agent instance, must contain 'name', 'description' and 'prompt' keys.
            llm_config (Optional[LLMConfig]): The LLM configuration to be used for the agent. 
                It will be used as the default LLM for agents without a `llm_config` key. 
                If not provided, the `agent_data` should contain a `llm_config` key. 
                If provided and `agent_data` contains a `llm_config` key, the `llm_config` in `agent_data` will be used.  
            **kwargs (Any): Additional parameters for agent creation
        
        Returns:
            Agent: the instantiated agent instance.
        """
        
        agent_data = deepcopy(agent_data)
        agent_llm_config = agent_data.get("llm_config", llm_config)
        if not agent_data.get("is_human", False) and not agent_llm_config:
            raise ValueError("`agent_data` should contain a `llm_config` key or `llm_config` should be provided.")

        if agent_llm_config:
            if isinstance(agent_llm_config, dict):
                agent_data["llm_config"] = agent_llm_config
            elif isinstance(agent_llm_config, LLMConfig):
                agent_data["llm_config"] = agent_llm_config.to_dict()
        
        if agent_data.get("tools", None):
            agent_data["tool_names"] = agent_data["tools"]
            agent_data["tool_dict"] = self.tools_mapping
        return CustomizeAgent.from_dict(data=agent_data)
    
    def get_agent_name(self, agent: Union[str, dict, Agent]) -> str:
        """Extract agent name from different agent representations.
        
        Handles different ways to specify an agent (string name, dictionary, or
        Agent instance) and extracts the agent name.
        
        Args:
            agent: Agent specified as a string name, dictionary with 'name' key,
                  or Agent instance
                  
        Returns:
            The extracted agent name as a string
        """
        if isinstance(agent, str):
            agent_name = agent
        elif isinstance(agent, dict):
            agent_name = agent["name"]
        elif isinstance(agent, Agent):
            agent_name = agent.name
        else:
            raise ValueError(f"{type(agent)} is not a supported type for ``get_agent_name``. Supported types: [str, dict, Agent].")
        return agent_name
    
    def create_agent(self, agent: Union[str, dict, Agent], llm_config: Optional[LLMConfig]=None, **kwargs) -> Agent:

        if isinstance(agent, str):
            if self.storage_handler is None:
                # if self.storage_handler is None, the agent (str) must exist in self.agents. Otherwise, a dictionary or an Agent instance should be provided.
                if not self.has_agent(agent_name=agent):
                    raise ValueError(f"Agent ``{agent}`` does not exist! You should provide a dictionary or an Agent instance when ``self.storage_handler`` is not provided.")
                return self.get_agent(agent_name=agent)
            else:
                # if self.storage_handler is not None, the agent (str) must exist in the storage and will be loaded from the storage.
                agent_instance = self.load_agent(agent_name=agent)
        elif isinstance(agent, dict):
            if not agent.get("is_human", False) and (llm_config is None and "llm_config" not in agent):
                raise ValueError("When providing an agent as a dictionary, you must either include 'llm_config' in the dictionary or provide it as a parameter.")
            agent_instance = self.create_customize_agent(agent_data=agent, llm_config=llm_config, **kwargs)
        elif isinstance(agent, Agent):
            agent_instance = agent
        else:
            raise ValueError(f"{type(agent)} is not a supported input type of ``create_agent``. Supported types: [str, dict, Agent].")
        return agent_instance
    
    @atomic_method
    def add_agent(self, agent: Union[str, dict, Agent], llm_config: Optional[LLMConfig]=None, **kwargs):
        """
        add a single agent, ignore if the agent already exists (judged by the name of an agent).

        Args:
            agent: The agent to be added, specified as:
                - String: Agent name to load from storage
                - Dictionary: Agent specification to create a CustomizeAgent
                - Agent: Existing Agent instance to add directly
            llm_config (Optional[LLMConfig]): The LLM configuration to be used for the agent. Only used when the `agent` is a dictionary, used to create a CustomizeAgent. 
            **kwargs (Any): Additional parameters for agent creation
        """
        # Check for 'tool' key and convert it to 'tools' if needed
        if isinstance(agent, dict) and "tool" in agent and "tools" not in agent:
            agent["tools"] = agent.pop("tool")
            
        agent_name = self.get_agent_name(agent=agent)
        if self.has_agent(agent_name=agent_name):
            return
        agent_instance = self.create_agent(agent=agent, llm_config=llm_config, **kwargs)
        self.agents.append(agent_instance)
        self.agent_states[agent_instance.name] = AgentState.AVAILABLE
        if agent_instance.name not in self._state_conditions:
            self._state_conditions[agent_instance.name] = threading.Condition()
        self.check_agents()

    def add_agents(self, agents: List[Union[str, dict, Agent]], llm_config: Optional[LLMConfig]=None, **kwargs):
        """
        add several agents by using self.add_agent().
        """
        for agent in agents:
            self.add_agent(agent=agent, llm_config=llm_config, **kwargs)
    
    def add_agents_from_workflow(self, workflow_graph, llm_config: Optional[LLMConfig]=None, **kwargs):
        """
        Initialize agents from the nodes of a given WorkFlowGraph and add these agents to self.agents. 

        Args:
            workflow_graph (WorkFlowGraph): The workflow graph containing nodes with agents information.
            llm_config (Optional[LLMConfig]): The LLM configuration to be used for the agents.
            **kwargs (Any): Additional parameters passed to add_agent
        """
        from ..workflow.workflow_graph import WorkFlowGraph
        if not isinstance(workflow_graph, WorkFlowGraph):
            raise TypeError("workflow_graph must be an instance of WorkFlowGraph")
        for node in workflow_graph.nodes:
            if node.agents:
                for agent in node.agents:
                    self.add_agent(agent=agent, llm_config=llm_config, **kwargs)
    
    def update_agents_from_workflow(self, workflow_graph, llm_config: Optional[LLMConfig]=None, **kwargs):
        """
        Update agents from a given WorkFlowGraph.

        Args:
            workflow_graph (WorkFlowGraph): The workflow graph containing nodes with agents information.
            llm_config (Optional[LLMConfig]): The LLM configuration to be used for the agents.
            **kwargs: Additional parameters passed to update_agent
        """
        from ..workflow.workflow_graph import WorkFlowGraph
        if not isinstance(workflow_graph, WorkFlowGraph):
            raise TypeError("workflow_graph must be an instance of WorkFlowGraph")
        for node in workflow_graph.nodes:
            if node.agents:
                for agent in node.agents:
                    agent_name = self.get_agent_name(agent=agent)
                    if self.has_agent(agent_name=agent_name):
                        # use the llm_config of the existing agent
                        agent_llm_config = self.get_agent(agent_name).llm_config
                        self.update_agent(agent=agent, llm_config=agent_llm_config, **kwargs)
                    else:
                        self.add_agent(agent=agent, llm_config=llm_config, **kwargs)

    def get_agent(self, agent_name: str, **kwargs) -> Agent:
        """Retrieve an agent by its name from managed agents.
        
        Searches the list of managed agents for an agent with the specified name.
        
        Args:
            agent_name: The name of the agent to retrieve
            **kwargs (Any): Additional parameters (unused)
            
        Returns:
            The Agent instance with the specified name
        """
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        raise ValueError(f"Agent ``{agent_name}`` does not exists!")
    
    def update_agent(self, agent: Union[dict, Agent], llm_config: Optional[LLMConfig]=None, **kwargs):
        """
        Update an agent in the manager.

        Args:
            agent: The agent to be updated, specified as:
                - Dictionary: Agent specification to update a CustomizeAgent
                - Agent: Existing Agent instance to update
            llm_config (Optional[LLMConfig]): The LLM configuration to be used for the agent.
        """
        agent_name = self.get_agent_name(agent=agent)
        self.remove_agent(agent_name=agent_name)
        self.add_agent(agent=agent, llm_config=llm_config, **kwargs)
    
    @atomic_method
    def remove_agent(self, agent_name: str, remove_from_storage: bool=False, **kwargs):
        """
        Remove an agent from the manager and optionally from storage.
        
        Args:
            agent_name: The name of the agent to remove
            remove_from_storage: If True, also remove the agent from storage
            **kwargs (Any): Additional parameters passed to storage_handler.remove_agent
        """
        self.agents = [agent for agent in self.agents if agent.name != agent_name]
        self.agent_states.pop(agent_name, None)
        self._state_conditions.pop(agent_name, None) 
        if remove_from_storage:
            self.storage_handler.remove_agent(agent_name=agent_name, **kwargs)
        self.check_agents()

    def get_agent_state(self, agent_name: str) -> AgentState:
        """
        Get the state of a specific agent by its name.

        Args:
            agent_name: The name of the agent.

        Returns:
            AgentState: The current state of the agent.
        """
        return self.agent_states[agent_name]
    
    @atomic_method
    def set_agent_state(self, agent_name: str, new_state: AgentState) -> bool:
        """
        Changes an agent's state and notifies any threads waiting on that agent's state.
        Thread-safe operation for coordinating multi-threaded agent execution.
        
        Args:
            agent_name: The name of the agent
            new_state: The new state to set
        
        Returns:
            True if the state was updated successfully, False otherwise
        """
        
        # if agent_name in self.agent_states and isinstance(new_state, AgentState):
        #     # self.agent_states[agent_name] = new_state
        #     with self._state_conditions[agent_name]:
        #         self.agent_states[agent_name] = new_state
        #         self._state_conditions[agent_name].notify_all()
        #     self.check_agents()
        #     return True
        # else:
        #     return False
        if agent_name in self.agent_states and isinstance(new_state, AgentState):
            if agent_name not in self._state_conditions:
                self._state_conditions[agent_name] = threading.Condition()
            with self._state_conditions[agent_name]:
                self.agent_states[agent_name] = new_state
                self._state_conditions[agent_name].notify_all()
            return True
        return False

    def get_all_agent_states(self) -> Dict[str, AgentState]:
        """Get the states of all managed agents.

        Returns:
            Dict[str, AgentState]: A dictionary mapping agent names to their states.
        """
        return self.agent_states
    
    @atomic_method
    def save_all_agents(self, **kwargs):
        """Save all managed agents to persistent storage.
                
        Args:
            **kwargs (Any): Additional parameters passed to the storage handler
        """
        pass 
    
    @atomic_method
    def clear_agents(self):
        """
        Remove all agents from the manager.
        """
        self.agents = [] 
        self.agent_states = {}
        self._state_conditions = {}
        self.check_agents()

    def wait_for_agent_available(self, agent_name: str, timeout: Optional[float] = None) -> bool:
        """Wait for an agent to be available.
        
        Args:
            agent_name: The name of the agent to wait for
            timeout: Maximum time to wait in seconds, or None to wait indefinitely
            
        Returns:
            True if the agent became available, False if timed out
        """
        if agent_name not in self._state_conditions:
            self._state_conditions[agent_name] = threading.Condition()
        condition = self._state_conditions[agent_name]

        with condition:
            return condition.wait_for(
                lambda: self.agent_states.get(agent_name) == AgentState.AVAILABLE,
                timeout=timeout
            )

