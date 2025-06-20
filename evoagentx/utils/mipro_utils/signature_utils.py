from dspy import Signature, InputField, OutputField
from ...prompts.template import PromptTemplate
from ...optimizers.engine.registry import ParamRegistry
from ...utils.mipro_utils.register_utils import MiproRegistry
# from dspy.signatures.signature import make_signature
import keyword
import re
import warnings
import ast
import typing
import importlib
from typing import Any, Dict, Optional, Tuple, Type, Union
from pydantic import Field, create_model
from pydantic.fields import FieldInfo
import types

def is_valid_identifier(key: str) -> bool:
    return key.isidentifier() and not keyword.iskeyword(key)

def check_input_placeholders(instruction: str, input_names: list[str], key: str):
    placeholders = set(re.findall(r"\{(\w+)\}", instruction))
    input_names_set = set(input_names or [])

    missing = placeholders - input_names_set
    if missing:
        warnings.warn(
            f"[{key}] Missing input_names for placeholders in instruction: {missing}"
        )

def _default_instructions(cls) -> str:
    inputs_ = ", ".join([f"`{field}`" for field in cls.input_fields])
    outputs_ = ", ".join([f"`{field}`" for field in cls.output_fields])
    return f"Given the fields {inputs_}, produce the fields {outputs_}."



def _parse_type_node(node, names=None) -> Any:
    """Recursively parse an AST node representing a type annotation.

    This function converts Python's Abstract Syntax Tree (AST) nodes into actual Python types.
    It's used to parse type annotations in signature strings like "x: List[int] -> y: str".

    Examples:
        - For "x: int", the AST node represents 'int' and returns the int type
        - For "x: List[str]", it processes a subscript node to return typing.List[str]
        - For "x: Optional[int]", it handles the Union type to return Optional[int]
        - For "x: MyModule.CustomType", it processes attribute access to return the actual type

    Args:
        node: An AST node from Python's ast module, representing a type annotation.
            Common node types include:
            - ast.Name: Simple types like 'int', 'str'
            - ast.Attribute: Nested types like 'typing.List'
            - ast.Subscript: Generic types like 'List[int]'
        names: Optional dictionary mapping type names to their actual type objects.
            Defaults to Python's typing module contents plus NoneType.

    Returns:
        The actual Python type represented by the AST node.

    Raises:
        ValueError: If the AST node represents an unknown or invalid type annotation.
    """

    if names is None:
        names = dict(typing.__dict__)
        names["NoneType"] = type(None)

    def resolve_name(type_name: str):
        # Check if it's a built-in known type or in the provided names
        if type_name in names:
            return names[type_name]
        # Common built-in types
        builtin_types = [int, str, float, bool, list, tuple, dict, set, frozenset, complex, bytes, bytearray]

        # Check if it matches any known built-in type by name
        for t in builtin_types:
            if t.__name__ == type_name:
                return t

        # Attempt to import a module with this name dynamically
        # This allows handling of module-based annotations like `dspy.Image`.
        try:
            mod = importlib.import_module(type_name)
            names[type_name] = mod
            return mod
        except ImportError:
            pass

        # If we don't know the type or module, raise an error
        raise ValueError(f"Unknown name: {type_name}")

    if isinstance(node, ast.Module):
        if len(node.body) != 1:
            raise ValueError(f"Code is not syntactically valid: {ast.dump(node)}")
        return _parse_type_node(node.body[0], names)

    if isinstance(node, ast.Expr):
        return _parse_type_node(node.value, names)

    if isinstance(node, ast.Name):
        return resolve_name(node.id)

    if isinstance(node, ast.Attribute):
        base = _parse_type_node(node.value, names)
        attr_name = node.attr
        if hasattr(base, attr_name):
            return getattr(base, attr_name)
        else:
            raise ValueError(f"Unknown attribute: {attr_name} on {base}")

    if isinstance(node, ast.Subscript):
        base_type = _parse_type_node(node.value, names)
        slice_node = node.slice
        if isinstance(slice_node, ast.Index):  # For older Python versions
            slice_node = slice_node.value

        if isinstance(slice_node, ast.Tuple):
            arg_types = tuple(_parse_type_node(elt, names) for elt in slice_node.elts)
        else:
            arg_types = (_parse_type_node(slice_node, names),)

        # Special handling for Union, Optional
        if base_type is typing.Union:
            return typing.Union[arg_types]
        if base_type is typing.Optional:
            if len(arg_types) != 1:
                raise ValueError("Optional must have exactly one type argument")
            return typing.Optional[arg_types[0]]

        return base_type[arg_types]

    if isinstance(node, ast.Tuple):
        return tuple(_parse_type_node(elt, names) for elt in node.elts)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Field":
        keys = [kw.arg for kw in node.keywords]
        values = []
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant):
                values.append(kw.value.value)
            else:
                values.append(_parse_type_node(kw.value, names))
        return Field(**dict(zip(keys, values)))

    raise ValueError(
        f"Failed to parse string-base Signature due to unhandled AST node type in annotation: {ast.dump(node)}. "
        "Please consider using class-based DSPy Signatures instead."
    )


def _parse_field_string(field_string: str) -> Dict[str, str]:
    """Extract the field name and type from field string in the string-based Signature.

    It takes a string like "x: int, y: str" and returns a dictionary mapping field names to their types.
    For example, "x: int, y: str" -> [("x", int), ("y", str)]. This function utitlizes the Python AST to parse the
    fields and types.
    """

    args = ast.parse(f"def f({field_string}): pass").body[0].args.args
    names = [arg.arg for arg in args]
    types = [str if arg.annotation is None else _parse_type_node(arg.annotation) for arg in args]
    return zip(names, types)


def _parse_signature(signature: str) -> Dict[str, Tuple[Type, Field]]:
    if signature.count("->") != 1:
        raise ValueError(f"Invalid signature format: '{signature}', must contain exactly one '->'.")

    inputs_str, outputs_str = signature.split("->")

    fields = {}
    for field_name, field_type in _parse_field_string(inputs_str):
        fields[field_name] = (field_type, InputField())
    for field_name, field_type in _parse_field_string(outputs_str):
        fields[field_name] = (field_type, OutputField())

    return fields

def make_signature(
    signature: Union[str, Dict[str, Tuple[type, FieldInfo]]],
    instructions: Optional[str] = None,
    signature_name: str = "StringSignature",
    extra_fields: Optional[Dict[str, Tuple[type, FieldInfo]]] = None,  # ✅ 新参数
) -> Type[Signature]:
    """Create a new Signature subclass with the specified fields and instructions."""

    fields = _parse_signature(signature) if isinstance(signature, str) else signature

    fixed_fields = {}
    for name, type_field in fields.items():
        if not isinstance(name, str):
            raise ValueError(f"Field names must be strings, but received: {name}.")
        if isinstance(type_field, FieldInfo):
            type_ = type_field.annotation
            field = type_field
        else:
            if not isinstance(type_field, tuple):
                raise ValueError(f"Field values must be tuples, but received: {type_field}.")
            type_, field = type_field
        if type_ is None:
            type_ = str
        if not isinstance(type_, (type, typing._GenericAlias, types.GenericAlias, typing._SpecialForm)):
            raise ValueError(f"Field types must be types, but received: {type_} of type {type(type_)}.")
        if not isinstance(field, FieldInfo):
            raise ValueError(f"Field values must be Field instances, but received: {field}.")
        fixed_fields[name] = (type_, field)

    # inject extra fields
    if extra_fields:
        fixed_fields.update(extra_fields)

    # Default prompt when no instructions are provided
    if instructions is None:
        sig = Signature(signature, "")
        instructions = _default_instructions(sig)

    return create_model(
        signature_name,
        __base__=Signature,
        __doc__=instructions,
        **fixed_fields,
    )

def signature_from_registry(
    registry: MiproRegistry,
) -> Dict[str, Type[Signature]]:
    
    signature_dict = {}

    signature_name2register_name = {}
    for key in registry.names():
        registered_element: Union[str, PromptTemplate] = registry.get(key)
        input_names = registry.get_input_names(key)
        output_names = registry.get_output_names(key)
        sig = {}

        # sig_dict[key] = (str, InputField(desc=f"The Input for prompt `{key}`."))

        if isinstance(registered_element, str):
            # For string prompts, create a simple signature with one input field
            instructions = registered_element
            
        elif isinstance(registered_element, PromptTemplate):
            instructions = registered_element.instruction
            # for field_name in registered_element.get_field_names():
        
        check_input_placeholders(instructions, input_names, key)

        for name in input_names:
            input_desc = registry.get_input_desc(key, name)
            if input_desc:
                sig[name] = (str, InputField(desc=input_desc))
            else:
                sig[name] = (str, InputField(desc=f"The Input for prompt `{key}`."))

        for name in output_names:
            output_desc = registry.get_output_desc(key, name)
            if output_desc:
                sig[name] = (str, OutputField(desc=output_desc))
            else:
                sig[name] = (str, OutputField(desc=f"The Output for prompt `{key}`."))

        if is_valid_identifier(key):
            signature_name = f"{key}Signature"
        else:
            # if the key is not a valid identifier, we need to add an underscore
            # 打印warning
            print(f"Warning: The key `{key}` is not a valid identifier, so we will add an underscore to it.")
            signature_name = f"DefaultSignature_{len(signature_dict)}"

        signature_class = make_signature(signature=sig, 
                                         instructions=instructions, 
                                         signature_name=signature_name,
                                         )
        # extra_fields={
        #                                     "register_name": (str, InputField(default=key))
        #                                 }
        
        signature_class.__pydantic_extra__ = {"register_name": key}

        # signature_class.__annotations__['register_name'] = str

        # setattr(signature_class, 'register_name', key)

        signature_dict[signature_name] = signature_class
        signature_name2register_name[signature_name] = key


    return signature_dict, signature_name2register_name


# Unused Function
def build_signature_class(
    registry: ParamRegistry,
    input_descs: Optional[Dict[str, str]] = None,
    output_name: str = "score",
    output_desc: str = "Final evaluation score of the agent output",
    output_type: type = float
):
    """
    unused function
    Dynamically builds a DSPy Signature class based on a parameter registry.
    
    This function creates a new DSPy Signature class that defines input and output fields
    based on the parameters in the registry. Each parameter becomes an input field in the
    signature, and an additional output field is added for the evaluation score.
    
    Parameters
    ----------
    registry : ParamRegistry
        Registry containing the tunable parameters that will become input fields
    input_descs : Optional[Dict[str, str]], default=None
        Optional descriptions for input parameters. Keys are parameter names,
        values are their descriptions. If not provided for a parameter,
        a default description will be generated.
    output_name : str, default="score"
        Name of the output field in the signature
    output_desc : str, default="Final evaluation score of the agent output"
        Description of the output field
    output_type : type, default=float
        Type annotation for the output field
        
    Returns
    -------
    type
        A new DSPy Signature subclass with dynamically defined input and output fields
        
    Examples
    --------
    >>> registry = ParamRegistry()
    >>> registry.register("temperature", 0.7)
    >>> signature = build_signature_class(
    ...     registry,
    ...     input_descs={"temperature": "Sampling temperature"}
    ... )
    """
    # Initialize empty descriptions dictionary if none provided
    input_descs = input_descs or {}
    
    # Get all parameter names from registry
    fields = registry.names()

    # Prepare class attributes and type annotations
    annotations = {}
    class_namespace = {"__doc__": "Auto-generated signature class."}

    # Create input fields for each parameter in registry
    for name in fields:
        annotations[name] = str  # All inputs are treated as strings
        class_namespace[name] = InputField(
            desc=input_descs.get(name, f"Tunable parameter: {name}")
        )

    # Add output field with specified configuration
    annotations[output_name] = output_type
    class_namespace[output_name] = OutputField(desc=output_desc)
    class_namespace['__annotations__'] = annotations

    # Create the signature class dynamically
    class PromptTuningSignature(Signature):
        __doc__ = class_namespace['__doc__']
        __annotations__ = annotations
        for k, v in class_namespace.items():
            if k not in ('__doc__', '__annotations__'):
                locals()[k] = v

    return PromptTuningSignature