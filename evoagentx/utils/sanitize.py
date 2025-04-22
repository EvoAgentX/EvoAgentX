# Acknowledgement: Modified from MetaGPT (https://github.com/geekan/MetaGPT/blob/main/metagpt/utils/sanitize.py) under MIT License 

import ast
import traceback
from enum import Enum
from typing import Dict, Generator, List, Optional, Set, Tuple

import tree_sitter_python
from tree_sitter import Language, Node, Parser


class NodeType(Enum):
    """Enumeration of syntax tree node types used for code analysis.
    
    Defines the different types of nodes that can be identified in the syntax tree
    when parsing Python code. Used for traversing and analyzing Python syntax trees.
    
    Attributes:
        CLASS: Class definition nodes
        FUNCTION: Function definition nodes
        IMPORT: Import statement nodes (both regular imports and from-imports)
        IDENTIFIER: Identifier nodes (variable names, function names, etc.)
        ATTRIBUTE: Attribute access nodes (e.g., object.attribute)
        RETURN: Return statement nodes
        EXPRESSION: Expression statement nodes
        ASSIGNMENT: Assignment operation nodes
    """
    CLASS = "class_definition"
    FUNCTION = "function_definition"
    IMPORT = ["import_statement", "import_from_statement"]
    IDENTIFIER = "identifier"
    ATTRIBUTE = "attribute"
    RETURN = "return_statement"
    EXPRESSION = "expression_statement"
    ASSIGNMENT = "assignment"


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    """Traverse the syntax tree structure starting from the given node.
    
    Performs a depth-first traversal of the tree-sitter syntax tree, yielding
    each node in the tree as it is visited. This allows for comprehensive
    analysis of code structure.
    
    Args:
        node: The root node to start the traversal from.
        
    Yields:
        Nodes in the tree in depth-first order.
        
    Notes:
        - Uses tree-sitter's cursor API for efficient traversal
        - Maintains proper depth tracking during traversal
        - Ensures all nodes in the tree are visited exactly once
    """
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def syntax_check(code, verbose=False):
    """Check if the given Python code has valid syntax.
    
    Attempts to parse the code using Python's AST parser to determine
    if it has valid syntax. Can optionally print stack traces for debugging.
    
    Args:
        code: The Python code as a string to check for syntax validity
        verbose: Whether to print the full stack trace on syntax errors
        
    Returns:
        Boolean indicating whether the code has valid syntax
        
    Notes:
        - Uses Python's built-in ast module for syntax validation
        - Returns False for both SyntaxError and MemoryError exceptions
        - Can help identify valid code snippets in larger text
    """
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def code_extract(text: str) -> str:
    """Extract the longest valid Python code snippet from a text string.
    
    Analyzes a string containing potential Python code mixed with other text,
    and extracts the longest continuous section that has valid Python syntax.
    
    Args:
        text: Input text that may contain valid Python code
        
    Returns:
        The longest continuous substring with valid Python syntax
        
    Notes:
        - Uses a sliding window approach to find valid code segments
        - Only counts non-empty lines when determining the longest segment
        - Helps extract useful code from documentation, comments, or mixed text
    """
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_definition_name(node: Node) -> str:
    """Extract the name from a class or function definition node.
    
    Searches through a node's children to find the identifier node
    that contains the name of the class or function being defined.
    
    Args:
        node: A syntax tree node representing a class or function definition
        
    Returns:
        The name of the class or function as a string, or None if not found
        
    Notes:
        - Works with class and function definition nodes
        - Returns the first identifier child node's text
        - Decodes the node text from UTF-8 bytes
    """
    for child in node.children:
        if child.type == NodeType.IDENTIFIER.value:
            return child.text.decode("utf8")


def has_return_statement(node: Node) -> bool:
    """Check if a function definition node contains a return statement.
    
    Traverses the syntax tree of a function to determine if it contains
    any return statements, which is useful for identifying functions
    that actually return values.
    
    Args:
        node: A syntax tree node (typically a function definition)
        
    Returns:
        Boolean indicating whether the node contains a return statement
        
    Notes:
        - Uses traverse_tree for efficient tree traversal
        - Returns True as soon as any return statement is found
        - Used to filter out function definitions that don't return values
    """
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == NodeType.RETURN.value:
            return True
    return False


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    """Extract dependencies between definitions in the code.
    
    Analyzes a list of named nodes to determine which identifiers each
    node references, building a dependency graph of the code.
    
    Args:
        nodes: A list of tuples containing (name, node) pairs
        
    Returns:
        A dictionary mapping definition names to sets of names they depend on
        
    Notes:
        - Uses depth-first search to find all identifiers in each node
        - Builds a complete dependency graph for the entire codebase
        - Used for determining which definitions are required by a given function
    """
    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == NodeType.IDENTIFIER.value:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    """Determine all functions reachable from an entrypoint in the call graph.
    
    Performs a breadth-first search through the call graph to find all
    functions that are directly or indirectly invoked by the entrypoint.
    
    Args:
        entrypoint: The name of the starting function
        call_graph: A dictionary mapping function names to sets of called function names
        
    Returns:
        A set of function names that are reachable from the entrypoint
        
    Notes:
        - Uses breadth-first search to traverse the call graph
        - Tracks visited functions to avoid cycles
        - Used to determine which functions to include in sanitized output
    """
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    """Sanitize and extract relevant parts of the given Python code.
    
    Parses the input code, extracts import statements, class and function definitions,
    and variable assignments. If an entrypoint is provided, it only includes definitions
    that are reachable from the entrypoint in the call graph.
    
    Args:
        code: The input Python code as a string
        entrypoint: Optional name of a function to use as the entrypoint for dependency analysis
        
    Returns:
        A sanitized version of the input code, containing only relevant parts
        
    Notes:
        - First extracts valid code using code_extract
        - Parses the code using tree-sitter
        - Identifies imports, classes, functions, and variable assignments
        - If entrypoint is specified, performs dependency analysis
        - Reconstructs the code with only the necessary components
        - Useful for extracting only the code needed to run a specific function
    """
    code = code_extract(code)
    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)
    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in NodeType.IMPORT.value:
            import_nodes.append(child)
        elif child.type == NodeType.CLASS.value:
            name = get_definition_name(child)
            if not (name in class_names or name in variable_names or name in function_names):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == NodeType.FUNCTION.value:
            name = get_definition_name(child)
            if not (name in function_names or name in variable_names or name in class_names) and has_return_statement(
                child
            ):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif child.type == NodeType.EXPRESSION.value and child.children[0].type == NodeType.ASSIGNMENT.value:
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (name in variable_names or name in function_names or name in class_names):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reacheable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b""

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and name not in reacheable:
            continue
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"
    return sanitized_output[:-1].decode("utf8")
