import threading
from enum import Enum
from typing import Union, Optional, Dict, List

from .agent import Agent
# from .agent_generator import AgentGenerator
from .customize_agent import CustomizeAgent
from ..core.module import BaseModule
from ..core.decorators import atomic_method
from ..storages.base import StorageHandler
from ..models.model_configs import LLMConfig

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
    agents: List[Agent] = []
    agent_states: Dict[str, AgentState] = {} # agent_name to AgentState mapping
    storage_handler: Optional[StorageHandler] = None # used to load and save agent from storage.
    # agent_generator: Optional[AgentGenerator] = None # used to generate agents for a specific subtask

    def init_module(self):
        """Initialize the agent manager module.
        
        Sets up internal state tracking for agents, including thread locks and conditions
        for thread-safe agent state management. Also performs validation of existing agents.
        
        Notes:
            - Creates a thread lock for atomic operations
            - Initializes condition variables for each agent for state change notifications
            - Sets initial states for existing agents
            - Validates agent uniqueness and state consistency
        """
        self._lock = threading.Lock()
        self._state_conditions = {}
        if self.agents:
            for agent in self.agents:
                self.agent_states[agent.name] = self.agent_states.get(agent.name, AgentState.AVAILABLE)
                if agent.name not in self._state_conditions:
                    self._state_conditions[agent.name] = threading.Condition()
            self.check_agents()
    
    def check_agents(self):
        """Validate agent list integrity and state consistency.
        
        Performs thorough validation of the agent manager's internal state:
        1. Checks for duplicate agent names
        2. Verifies that agent states exist for all agents
        3. Ensures agent list and state dictionary sizes match
        
        Raises:
            ValueError: If duplicate agent names are found
            ValueError: If agent states dictionary size doesn't match agents list
            ValueError: If any agent is missing its state entry
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
        """Find duplicate agent names in a list of agents.
        
        Scans a list of Agent objects and identifies any duplicate names.
        
        Args:
            agents: List of Agent objects to check for duplicates
            
        Returns:
            List of agent names that appear more than once
            
        Notes:
            - Uses sets for efficient duplicate detection
            - Important for maintaining agent name uniqueness
        """
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
        """Find agents that don't have corresponding state entries.
        
        Identifies any agents in the agents list that don't have corresponding
        entries in the agent_states dictionary.
        
        Returns:
            List of agent names missing from the state dictionary
            
        Notes:
            - Critical for ensuring agent state tracking consistency
        """
        missing_agents = [agent.name for agent in self.agents if agent.name not in self.agent_states]
        return missing_agents

    def list_agents(self) -> List[str]:
        """Return a list of all agent names managed by this manager.
        
        Provides a simple way to enumerate all available agents by name.
        
        Returns:
            List of string names for all agents in this manager
            
        Notes:
            - Used for quick access to agent inventory
            - Helpful for UI display and validation operations
        """
        return [agent.name for agent in self.agents]
    
    def has_agent(self, agent_name: str) -> bool:
        """Check if an agent with the given name exists in the manager.
        
        Args:
            agent_name: The name of the agent to check
            
        Returns:
            True if an agent with the given name exists, False otherwise
            
        Notes:
            - Used for checking agent existence before operations
            - Helps prevent duplicate agent creation
        """
        all_agent_names = self.list_agents()
        return agent_name in all_agent_names
    
    @property
    def size(self):
        """Get the total number of agents managed by this manager.
        
        Returns:
            Integer count of agents in the manager
            
        Notes:
            - Property accessor for easy size checking
            - Helpful for monitoring manager growth
        """
        return len(self.agents)
    
    def load_agent(self, agent_name: str, **kwargs) -> Agent:
        """Load an agent from local storage through storage_handler.
        
        Retrieves agent data from storage and creates an Agent instance.
        
        Args:
            agent_name: The name of the agent to load
            **kwargs: Additional parameters for agent creation
        
        Returns:
            Agent instance with data loaded from storage
            
        Raises:
            ValueError: If storage_handler is not provided
            
        Notes:
            - Requires storage_handler to be set
            - Creates agent using create_customize_agent
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
            **kwargs: Additional parameters passed to storage handler
            
        Notes:
            - Requires storage_handler to be configured
            - Placeholder implementation (pass) - needs implementation
        """
        pass 
    
    def create_customize_agent(self, agent_data: dict, llm_config: Optional[LLMConfig]=None, **kwargs) -> Agent:
        """
        create a customized agent from the provided `agent_data`. 

        Args:
            agent_data: The data used to create an Agent instance, must contain 'name' and 'description' keys
            llm_config (Optional[LLMConfig]): The LLM configuration to be used for the agent. If not provided, the `agent_data` should contain a `llm_config` key.  
            **kwargs: Additional parameters for agent creation
        
        Returns:
            Agent: the instantiated agent instance.
        
        Notes: 
            - Uses CustomizeAgent.from_dict() to create the agent instance
            - Enables dynamic agent creation from configuration data
        """
        if llm_config:
            if isinstance(llm_config, dict):
                agent_data["llm_config"] = llm_config
            elif isinstance(llm_config, LLMConfig):
                agent_data["llm_config"] = llm_config.to_dict()
            else:
                raise ValueError(f"llm_config must be a dictionary or an instance of LLMConfig. Got {type(llm_config)}.")
        return CustomizeAgent.from_dict(data=agent_data)
    
    def get_agent_name(self, agent: Union[str, dict, Agent]):
        """Extract agent name from different agent representations.
        
        Handles different ways to specify an agent (string name, dictionary, or
        Agent instance) and extracts the agent name.
        
        Args:
            agent: Agent specified as a string name, dictionary with 'name' key,
                  or Agent instance
                  
        Returns:
            The extracted agent name as a string
            
        Raises:
            ValueError: If agent is not of a supported type
            
        Notes:
            - Provides flexibility in specifying agents throughout the API
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
            **kwargs: Additional parameters for agent creation
            
        Notes:
            - Atomic operation protected by threading lock
            - Ignores agents with names that already exist in the manager
            - Automatically sets the agent's initial state to AVAILABLE
            - Creates a condition variable for the agent for thread synchronization
            - Validates agent manager's state after addition
        """
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
        
        Extracts agent information from workflow nodes and adds them to the agent manager.
        This allows automatic integration of workflow requirements with agent management.
        
        Args:
            workflow_graph: The workflow graph containing nodes with agent information
            **kwargs: Additional parameters passed to add_agent
            
        Raises:
            TypeError: If workflow_graph is not a WorkFlowGraph instance
            
        Notes:
            - Iterates through all nodes in the workflow graph
            - For each node, adds all associated agents
            - Useful for automatically initializing all agents needed for a workflow
        """
        from ..workflow.workflow_graph import WorkFlowGraph
        if not isinstance(workflow_graph, WorkFlowGraph):
            raise TypeError("workflow_graph must be an instance of WorkFlowGraph")
        for node in workflow_graph.nodes:
            if node.agents:
                for agent in node.agents:
                    self.add_agent(agent=agent, llm_config=llm_config, **kwargs)

    def get_agent(self, agent_name: str, **kwargs) -> Agent:
        """Retrieve an agent by its name from managed agents.
        
        Searches the list of managed agents for an agent with the specified name.
        
        Args:
            agent_name: The name of the agent to retrieve
            **kwargs: Additional parameters (unused)
            
        Returns:
            The Agent instance with the specified name
            
        Raises:
            ValueError: If no agent with the given name exists
            
        Notes:
            - O(n) search through the agents list
            - Important for accessing agent capabilities during workflow execution
        """
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        raise ValueError(f"Agent ``{agent_name}`` does not exists!")
    
    @atomic_method
    def remove_agent(self, agent_name: str, remove_from_storage: bool=False, **kwargs):
        """Remove an agent from the manager and optionally from storage.
        
        Removes an agent from the internal agents list, state tracking, and
        optionally from persistent storage.
        
        Args:
            agent_name: The name of the agent to remove
            remove_from_storage: If True, also remove the agent from storage
            **kwargs: Additional parameters passed to storage_handler.remove_agent
            
        Notes:
            - Atomic operation protected by threading lock
            - Removes agent from the agents list
            - Removes agent from the state dictionary
            - Removes agent's condition variable
            - Optionally removes from storage if remove_from_storage is True
            - Validates agent manager state after removal
        """
        self.agents = [agent for agent in self.agents if agent.name != agent_name]
        self.agent_states.pop(agent_name, None)
        self._state_conditions.pop(agent_name, None) 
        if remove_from_storage:
            self.storage_handler.remove_agent(agent_name=agent_name, **kwargs)
        self.check_agents()

    def get_agent_state(self, agent_name: str) -> AgentState:
        """Get the state of a specific agent by its name.

        Args:
            agent_name: The name of the agent.

        Returns:
            AgentState: The current state of the agent.
            
        Raises:
            KeyError: If agent_name does not exist in agent_states dictionary
            
        Notes:
            - Direct dictionary lookup, not protected by locks
            - For thread-safe operations, consider using with lock guards
        """
        return self.agent_states[agent_name]
    
    @atomic_method
    def set_agent_state(self, agent_name: str, new_state: AgentState) -> bool:
        """Update the state of a specific agent by its name.
        
        Changes an agent's state and notifies any threads waiting on that agent's state.
        Thread-safe operation for coordinating multi-threaded agent execution.
        
        Args:
            agent_name: The name of the agent
            new_state: The new state to set
        
        Returns:
            True if the state was updated successfully, False otherwise
            
        Notes:
            - Atomic operation protected by threading lock
            - Uses condition variables to notify waiting threads
            - Creates condition variable if it doesn't exist
            - Critical for synchronizing agent execution across threads
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
            
        Notes:
            - Returns a reference to the internal dictionary, not a copy
            - Changes to the returned dictionary will affect the manager's state
            - For thread-safe operations, consider proper synchronization
        """
        return self.agent_states
    
    @atomic_method
    def save_all_agents(self, **kwargs):
        """Save all managed agents to persistent storage.
        
        Persists all agents in the manager to storage using the storage_handler.
        
        Args:
            **kwargs: Additional parameters passed to the storage handler
            
        Notes:
            - Atomic operation protected by threading lock
            - Requires storage_handler to be configured
            - Placeholder implementation (pass) - needs implementation
        """
        pass 
    
    @atomic_method
    def clear_agents(self):
        """Remove all agents from the manager.
        
        Completely resets the agent manager's state by clearing:
        - The list of managed agents
        - The agent state dictionary
        - The state condition variables
        
        After clearing, validates the empty state with check_agents().
        
        Notes:
            - Atomic operation protected by threading lock
            - Use with caution as it completely resets manager state
            - Useful for reinitializing the agent manager
        """
        self.agents = [] 
        self.agent_states = {}
        self._state_conditions = {}
        self.check_agents()

    def wait_for_agent_available(self, agent_name: str, timeout: Optional[float] = None) -> bool:
        """Wait for an agent to be available.
        
        Blocks the calling thread until the specified agent becomes available
        or the timeout is reached. Uses condition variables for efficient waiting
        without busy-waiting.
        
        Args:
            agent_name: The name of the agent to wait for
            timeout: Maximum time to wait in seconds, or None to wait indefinitely
            
        Returns:
            True if the agent became available, False if timed out
            
        Notes:
            - Creates condition variable if it doesn't exist
            - Used for synchronizing workflows that depend on agent availability
            - Efficiently waits without consuming CPU cycles
            - Critical for coordinating agent usage across threads
        """
        if agent_name not in self._state_conditions:
            self._state_conditions[agent_name] = threading.Condition()
        condition = self._state_conditions[agent_name]

        with condition:
            return condition.wait_for(
                lambda: self.agent_states.get(agent_name) == AgentState.AVAILABLE,
                timeout=timeout
            )
        
