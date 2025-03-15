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
        self._lock = threading.Lock()
        if self.agents:
            for agent in self.agents:
                self.agent_states[agent.name] = self.agent_states.get(agent.name, AgentState.AVAILABLE)
            self.check_agents()
    
    def check_agents(self):
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
        """
        return all the agent names in self.agents. 
        """
        return [agent.name for agent in self.agents]
    
    def has_agent(self, agent_name: str) -> bool:
        all_agent_names = self.list_agents()
        return agent_name in all_agent_names
    
    @property
    def size(self):
        return len(self.agents)
    
    def load_agent(self, agent_name: str, **kwargs) -> Agent:

        """
        load an agent from local storage through self.storage_handler

        Args:
            agent (str): the name of the agent.
        
        Returns:
            Agent: the agent instance with the data loaded from local storage. 
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
        """
        create a customized agent from the provided `agent_data`. 

        Args:
            agent_data (dict): the data used to create an Agent instance, must contain the `name` and `description` keys.
        
        Returns:
            Agent: the instantiated agent instance.
        
        Notes: 
            - use CustomizeAgent.from_dict() to create the agent instance.
        """
        return CustomizeAgent.from_dict(data=agent_data)
    
    def get_agent_name(self, agent: Union[str, dict, Agent]):

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
        """
        add a single agent, ignore if the agent already exists (judged by the name of an agent).

        Args:
            agent (Union[str, dict, Agent]): The agent to be added.
                - Determine whether this agent should be added:
                    - If the agent's name is different from existing agents, add the agent. 
                    - If the agent's name already exists:
                        - check whether the data of this agent (if provided) is the same with existing one. If they are the same, ignore 
                - If a string is provided, it is treated as the agent's name. The agent will be loaded from storage using self.storage_handler.
                - If a dictionary is provided, CustomizeAgent.from_dict() will be used to create an Agent instance.
                - If an Agent instance is provided, it will be directly added to self.agents.
        """
        agent_name = self.get_agent_name(agent=agent)
        if self.has_agent(agent_name=agent_name):
            return
        agent_instance = self.create_agent(agent=agent)
        self.agents.append(agent_instance)
        self.agent_states[agent_instance.name] = AgentState.AVAILABLE
        self.check_agents()

    def add_agents(self, agents: List[Union[str, dict, Agent]], **kwargs):
        """
        add several agents by using self.add_agent().
        """
        for agent in agents:
            self.add_agent(agent=agent, **kwargs)
    
    def add_agents_from_workflow(self, workflow_graph, **kwargs):
        """
        Initialize agents from the nodes of a given WorkFlowGraph and add these agents to self.agents. 

        Args:
            workflow_graph (WorkFlowGraph): The workflow graph containing nodes with agents information.
        
        Notes:
            - The agent information is in workflow_graph.nodes: List[WorkFlowNode].
        """
        from ..workflow.workflow_graph import WorkFlowGraph
        if not isinstance(workflow_graph, WorkFlowGraph):
            raise TypeError("workflow_graph must be an instance of WorkFlowGraph")
        for node in workflow_graph.nodes:
            if node.agents:
                for agent in node.agents:
                    self.add_agent(agent=agent, **kwargs)

    def get_agent(self, agent_name: str, **kwargs) -> Agent:
        """
        Retrieve an agent by its name from self.agents. 
        """
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        raise ValueError(f"Agent ``{agent_name}`` does not exists!")
    
    @atomic_method
    def remove_agent(self, agent_name: str, remove_from_storage: bool=False, **kwargs):
        """
        remove an agent from self.agents (and storage). 

        Args:
            agent_name (str): the name of the agent to be removed. 
            remove_from_storage (boo): if True, remove the agent from storage if the agent is already in the storage.
        """
        self.agents = [agent for agent in self.agents if agent.name != agent_name]
        self.agent_states.pop(agent_name, None)
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
        """
        Update the state of a specific agent by its name.

        Args:
            agent_name (str): The name of the agent.
            new_state (AgentState): The new state to set.
        
        Returns:
            bool: True if the state was updated successfully, False otherwise.
        """
        if agent_name in self.agent_states and isinstance(new_state, AgentState):
            self.agent_states[agent_name] = new_state
            self.check_agents()
            return True
        else:
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
        """
        Remove all agents from the manager.
        """
        self.agents = [] 
        self.agent_states = {}
        self.check_agents()

