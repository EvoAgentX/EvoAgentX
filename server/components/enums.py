from enum import Enum


# used in user_query_router.py
class QueryType(Enum):
    """Enumeration of supported query types for front end or workflow modifications."""
    
    # Frontend modifications
    FRONT_END = "front_end"
    
    # Addition types (backend workflow)
    ADD_FUNCTIONALITY = "add_functionality"
    ADD_NODE = "add_node"
    ADD_AGENT = "add_agent"
    
    # Deletion types (backend workflow)
    DELETE_FUNCTIONALITY = "delete_functionality"
    DELETE_NODE = "delete_node"
    DELETE_AGENT = "delete_agent"
    
    # Modification types (backend workflow)
    MODIFY_FUNCTIONALITY = "modify_functionality"
    MODIFY_NODE = "modify_node"
    MODIFY_AGENT = "modify_agent"
    
    # Structural adjustment (backend workflow)
    RESTRUCTURE = "restructure"
    
    # Information and optimization
    QUERY = "query"
    OPTIMIZATION = "optimization"

query_type_description_dict = {
    QueryType.FRONT_END.value: "Changes to user interface, visual elements, interactions, styling, or frontend functionality.",
    QueryType.ADD_FUNCTIONALITY.value: "Add new features or capabilities to the workflow, in the way that concerns multiple nodes or agents, or modification of workflow in a general level.",
    QueryType.ADD_NODE.value: "Specifically add one new workflow node, with a clear purpose and key characteristics.",
    QueryType.ADD_AGENT.value: "Specifically add one new agent to existing nodes, with a clear purpose and key characteristics.",
    QueryType.DELETE_FUNCTIONALITY.value: "Remove features from the workflow, in the way that concerns multiple nodes or agents, or modification of workflow in a general level.",
    QueryType.DELETE_NODE.value: "Specifically remove one workflow nodes, with a clear purpose and key characteristics.",
    QueryType.DELETE_AGENT.value: "Specifically remove one agent from nodes, with a clear purpose and key characteristics.",
    QueryType.MODIFY_FUNCTIONALITY.value: "Modify existing workflow features, in the way that concerns multiple nodes or agents, or modification of workflow in a general level.",
    QueryType.MODIFY_NODE.value: "Specifically modify one existing workflow node, with a clear purpose and key characteristics.",
    QueryType.MODIFY_AGENT.value: "Specifically modify one existing agent, with a clear purpose and key characteristics.",
    QueryType.RESTRUCTURE.value: "Structural changes involving multiple elements",
    QueryType.QUERY.value: "Any other type of information requests about the workflow",
    QueryType.OPTIMIZATION.value: "Performance improvement suggestions"
}