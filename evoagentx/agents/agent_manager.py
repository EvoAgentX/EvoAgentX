import threading
from enum import Enum
from typing import Union, Optional, Dict, List

from .agent import Agent
# from .agent_generator import AgentGenerator
from .customize_agent import CustomizeAgent
from ..core.module import BaseModule
from ..core.decorators import atomic_method
from ..storages.base import StorageHandler


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
            
        初始化代理管理器模块。
        
        设置代理的内部状态跟踪，包括线程锁和线程安全的代理状态管理条件变量。
        同时对现有代理进行验证。
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
            
        验证代理列表的完整性和状态一致性。
        
        对代理管理器的内部状态进行全面验证：
        1. 检查重复的代理名称
        2. 验证所有代理都存在状态记录
        3. 确保代理列表和状态字典大小匹配
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
            
        查找代理列表中的重复代理名称。
        
        扫描Agent对象列表并识别任何重复的名称。
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
            
        查找没有对应状态条目的代理。
        
        识别代理列表中没有在agent_states字典中对应条目的任何代理。
        """
        missing_agents = [agent.name for agent in self.agents if agent.name not in self.agent_states]
        return missing_agents

    def list_agents(self) -> List[str]:
        """
        return all the agent names in self.agents. 
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
            
        检查具有给定名称的代理是否存在于管理器中。
        """
        all_agent_names = self.list_agents()
        return agent_name in all_agent_names
    
    @property
    def size(self):
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
            
        通过storage_handler从本地存储加载代理。
        
        从存储中检索代理数据并创建一个Agent实例。
        """
        if not self.storage_handler:
            raise ValueError("must provide ``self.storage_handler`` to use ``load_agent``")
        agent_data = self.storage_handler.load_agent(agent_name=agent_name)
        agent: Agent = self.create_customize_agent(agent_data=agent_data)
        return agent

    def load_all_agents(self, **kwargs):
        """
        load all agents from storage and add them to self.agents. 
        """
        pass 
    
    def create_customize_agent(self, agent_data: dict, **kwargs) -> Agent:
        """Create a customized agent from the provided agent_data.
        
        Factory method for creating Agent instances from dictionary specifications.
        
        Args:
            agent_data: The data used to create an Agent instance, must contain
                       'name' and 'description' keys
            **kwargs: Additional parameters for agent creation
        
        Returns:
            Agent: the instantiated agent instance.
        
        Notes: 
            - Uses CustomizeAgent.from_dict() to create the agent instance
            - Enables dynamic agent creation from configuration data
            
        从提供的agent_data创建定制代理。
        
        从字典规范创建Agent实例的工厂方法。
        """
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
            
        从不同的代理表示中提取代理名称。
        
        处理指定代理的不同方式（字符串名称、字典或Agent实例）并提取代理名称。
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
    
    def create_agent(self, agent: Union[str, dict, Agent], **kwargs) -> Agent:
        """Create an Agent instance from different representations.
        
        Converts various agent specifications into concrete Agent instances:
        - String names are loaded from storage
        - Dictionaries are converted using create_customize_agent
        - Agent instances are returned as-is
        
        Args:
            agent: Agent specification as string name, dictionary, or Agent instance
            **kwargs: Additional parameters passed to underlying creation methods
            
        Returns:
            An instantiated Agent object
            
        Raises:
            ValueError: If agent is not of a supported type
            
        Notes:
            - Central factory method for creating agents in the system
            
        从不同表示创建Agent实例。
        
        将各种代理规范转换为具体的Agent实例：
        - 字符串名称从存储中加载
        - 字典使用create_customize_agent转换
        - Agent实例按原样返回
        """
        if isinstance(agent, str):
            agent_instance = self.load_agent(agent_name=agent)
        elif isinstance(agent, dict):
            agent_instance = self.create_customize_agent(agent_data=agent)
        elif isinstance(agent, Agent):
            agent_instance = agent
        else:
            raise ValueError(f"{type(agent)} is not a supported input type of ``create_agent``. Supported types: [str, dict, Agent].")
        return agent_instance
    
    @atomic_method
    def add_agent(self, agent: Union[str, dict, Agent], **kwargs):
        """Add a single agent to the manager.
        
        Adds an agent to the manager if it doesn't already exist. The agent can be
        specified in multiple formats, and will be created appropriately based on the format.
        
        Args:
            agent: The agent to be added, specified as:
                - String: Agent name to load from storage
                - Dictionary: Agent specification to create a CustomizeAgent
                - Agent: Existing Agent instance to add directly
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
        agent_instance = self.create_agent(agent=agent)
        self.agents.append(agent_instance)
        self.agent_states[agent_instance.name] = AgentState.AVAILABLE
        if agent_instance.name not in self._state_conditions:
            self._state_conditions[agent_instance.name] = threading.Condition()
        self.check_agents()

    def add_agents(self, agents: List[Union[str, dict, Agent]], **kwargs):
        """
        add several agents by using self.add_agent().
        """
        for agent in agents:
            self.add_agent(agent=agent, **kwargs)
    
    def add_agents_from_workflow(self, workflow_graph, **kwargs):
        """Initialize agents from the nodes of a given WorkFlowGraph and add these agents to self.agents. 
        
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
                    self.add_agent(agent=agent, **kwargs)

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
        """
        Get the state of a specific agent by its name.

        Args:
            agent_name (str): The name of the agent.

        Returns:
            AgentState: The current state of the agent, or None if not found.
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
        """
        Get the states of all managed agents.

        Returns:
            Dict[str, AgentState]: A dictionary mapping agent names to their states.
        """
        return self.agent_states
    
    @atomic_method
    def save_all_agents(self, **kwargs):
        """
        Save all agents to storage.
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
        