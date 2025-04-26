import os 
import yaml
import json
import regex
from uuid import uuid4
from datetime import datetime, date 
from pydantic import BaseModel
from pydantic_core import PydanticUndefined, ValidationError
from typing import Union, Type, Any, List, Dict, get_origin, get_args

from .logging import logger 

def make_parent_folder(path: str):
    """Create parent directory for a file path if it doesn't exist.
    
    Args:
        path: File path whose parent directory needs to be created
        
    Notes:
        - Does nothing if the directory already exists
        - Creates all intermediate directories as needed
        - Does nothing if the path has no directory component
    """
    dir_folder = os.path.dirname(path)
    if len(dir_folder.strip()) == 0:
        return
    if not os.path.exists(dir_folder):
        os.makedirs(dir_folder, exist_ok=True)

def generate_id():
    """Generate a unique hexadecimal identifier.
    
    Returns:
        A unique string identifier based on UUID4
        
    Notes:
        - Uses uuid4 to generate a random UUID and converts it to hex
        - Useful for creating unique IDs for objects in the system
    """
    return uuid4().hex

def get_timestamp():
    """Get the current timestamp in a standard format.
    
    Returns:
        A string representation of the current time in YYYY-MM-DD HH:MM:SS format
        
    Notes:
        - Uses the local system time
        - Useful for logging and recording when events occur
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_json(path: str, type: str="json"):
    """Load data from a JSON or JSONL file.
    
    Args:
        path: Path to the JSON or JSONL file
        type: Format of the file, either "json" or "jsonl"
        
    Returns:
        For JSON files: The parsed JSON object (dict, list, etc.)
        For JSONL files: A list of parsed JSON objects, one per line
        
    Raises:
        AssertionError: If type is not "json" or "jsonl"
        Error logged: If the file doesn't exist or is not valid JSON
        
    Notes:
        - JSON files are loaded as a single object
        - JSONL files are loaded as a list of objects, one per line
    """
    assert type in ["json", "jsonl"] # only support json or jsonl format
    if not os.path.exists(path=path):
        logger.error(f"File \"{path}\" does not exists!")
    
    if type == "json":
        try:
            with open(path, "r", encoding="utf-8") as file:
                # outputs = yaml.safe_load(file.read()) # 用yaml.safe_load加载大文件的时候会非常慢
                outputs = json.loads(file.read())
        except Exception:
            logger.error(f"File \"{path}\" is not a valid json file!")
    
    elif type == "jsonl":
        outputs = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                # outputs.append(yaml.safe_load(line))
                outputs.append(json.loads(line))
    else:
        outputs = []
        
    return outputs

def save_json(data, path: str, type: str="json", use_indent: bool=True) -> str:

    """
    save data to a json file

    Args: 
        data: The json data to be saved. It can be a JSON str or a Serializable object when type=="json" or a list of JSON str or Serializable object when type=="jsonl".
        path(str): The path of the saved json file. 
        type(str): The type of the json file, chosen from ["json" or "jsonl"].
        use_indent: Whether to use indent when saving the json file. 
    
    Returns:
        path: the path where the json data is saved. 
    """

    assert type in ["json", "jsonl"] # only support json or jsonl format
    make_parent_folder(path)

    if type == "json":
        with open(path, "w", encoding="utf-8") as fout:
            if use_indent:
                fout.write(data if isinstance(data, str) else json.dumps(data, indent=4))
            else:
                fout.write(data if isinstance(data, str) else json.dumps(data))

    elif type == "jsonl":
        with open(path, "w", encoding="utf-8") as fout:
            for item in data:
                fout.write("{}\n".format(item if isinstance(item, str) else json.dumps(item)))

    return path

def escape_json_values(string: str) -> str:
    """Fix and escape problematic JSON strings to make them valid.
    
    Attempts to repair invalid JSON by properly escaping values that might cause
    parsing errors, such as newlines, quotes, and nested JSON objects.
    
    Args:
        string: The potentially invalid JSON string to fix
        
    Returns:
        A valid JSON string with properly escaped values
        
    Notes:
        - First tries to parse the string as JSON; returns it unchanged if valid
        - If initial parsing fails, attempts to fix common issues:
          1. Escapes double quotes not meant as JSON delimiters
          2. Fixes key formatting
          3. Escapes newlines in values
          4. Handles nested JSON objects
        - Returns the original string if fixing attempts fail
    """
    def escape_value(match):
        raw_value = match.group(1)
        raw_value = raw_value.replace('\n', '\\n')
        return f'"{raw_value}"'
    
    def fix_json(match):
        raw_key = match.group(1)
        raw_value = match.group(2)
        raw_value = raw_value.replace("\n", "\\n")
        raw_value = regex.sub(r'(?<!\\)"', '\\\"', raw_value)
        return f'"{raw_key}": "{raw_value}"'
    
    try:
        json.loads(string)
        return string
    except json.JSONDecodeError:
        pass

    try:
        string = regex.sub(r'(?<!\\)"', '\\\"', string) # replace " with \"
        pattern_key = r'\\"([^"]+)\\"(?=\s*:\s*)'
        string = regex.sub(pattern_key, r'"\1"', string) # replace \\"key\\" with "key"
        pattern_value = r'(?<=:\s*)\\"((?:\\.|[^"\\])*)\\"'
        string = regex.sub(pattern_value, escape_value, string, flags=regex.DOTALL) # replace \\"value\\" with "value"and change \n to \\n
        pattern_nested_json = r'"([^"]+)"\s*:\s*\\"([^"]*\{+[\S\s]*?\}+)[\r\n\\n]*"' # handle nested json in value
        string = regex.sub(pattern_nested_json, fix_json, string, flags=regex.DOTALL)
        json.loads(string)
        return string
    except json.JSONDecodeError:
        pass
    
    return string

def parse_json_from_text(text: str) -> List[str]:
    """
    Autoregressively extract JSON object from text 

    Args: 
        text (str): a text that includes JSON data 
    
    Returns:
        List[str]: a list of parsed JSON data
    """
    json_pattern = r"""(?:\{(?:[^{}]*|(?R))*\}|\[(?:[^\[\]]*|(?R))*\])"""
    pattern = regex.compile(json_pattern, regex.VERBOSE)
    matches = pattern.findall(text)
    matches = [escape_json_values(match) for match in matches]
    return matches

def parse_xml_from_text(text: str, label: str) -> List[str]:
    """Extract content from XML-like tags in text.
    
    Finds all content enclosed in the specified XML tags and returns them as a list.
    
    Args:
        text: The text containing XML-like tags
        label: The name of the tag to extract content from (without angle brackets)
        
    Returns:
        A list of strings containing the content between the opening and closing tags,
        with whitespace trimmed from the start and end
        
    Example:
        If text = "<answer>42</answer> and <answer>hello</answer>"
        and label = "answer", returns ["42", "hello"]
    """
    pattern = rf"<{label}>(.*?)</{label}>"
    matches: List[str] = regex.findall(pattern, text, regex.DOTALL)
    values = [] 
    if matches:
        values = [match.strip() for match in matches]
    return values

def parse_data_from_text(text: str, datatype: str):
    """Convert text to a specified data type.
    
    Parses a text string into the specified Python data type.
    
    Args:
        text: The text to parse
        datatype: The target data type, one of:
                 'str', 'int', 'float', 'bool', 'list', 'dict'
        
    Returns:
        The parsed data in the specified type
        
    Raises:
        ValueError: If datatype is not one of the supported types
        
    Notes:
        - For 'bool', values like "true", "yes", "1", "on", "True" return True
        - For 'list' and 'dict', uses Python's eval() function to parse the string
          (caution: potential security implications with untrusted input)
    """
    if datatype == "str":
        data = text
    elif datatype == "int":
        data = int(text)
    elif datatype == "float":
        data = float(text)
    elif datatype == "bool":
        data = text.lower() in ("true", "yes", "1", "on", "True")
    elif datatype == "list":
        data = eval(text)
    elif datatype == "dict":
        data = eval(text)
    else:
        raise ValueError(
            f"Invalid value '{datatype}' is detected for `datatype`. "
            "Available choices: ['str', 'int', 'float', 'bool', 'list', 'dict']"
        )
    return data

def parse_json_from_llm_output(text: str) -> dict:
    """
    Extract JSON str from LLM outputs and convert it to dict. 
    """
    json_list = parse_json_from_text(text=text)
    if json_list:
        json_text = json_list[0]
        try:
            data = yaml.safe_load(json_text)
        except Exception:
            raise ValueError(f"The following generated text is not a valid JSON string!\n{json_text}")
    else:
        raise ValueError(f"The follwoing generated text does not contain JSON string!\n{text}")
    return data

<<<<<<< HEAD
def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks enclosed in triple backticks from text.
    
    Finds and extracts all code blocks delimited by triple backticks (```),
    commonly used in Markdown to denote code sections.
    
    Args:
        text: The text containing code blocks
        
    Returns:
        A list of extracted code blocks with the triple backticks removed.
        If no code blocks are found, returns a list containing the original text.
        
    Notes:
        - Handles optional language specification after opening backticks
        - Removes leading and trailing whitespace from code blocks
        - If no code blocks are found, returns the original text as a single element list
=======
def extract_code_blocks(text: str, return_type: bool = False) -> Union[List[str], List[tuple]]:
    """
    Extract code blocks from text enclosed in triple backticks.
    
    Args:
        text (str): The text containing code blocks
        return_type (bool): If True, returns tuples of (language, code), otherwise just code
        
    Returns:
        Union[List[str], List[tuple]]: Either list of code blocks or list of (language, code) tuples
>>>>>>> origin/main
    """
    # Regular expression to match code blocks enclosed in triple backticks
    code_block_pattern = r"```((?:[a-zA-Z]*)?)\n*(.*?)\n*```"
    # Find all matches in the text
    matches = regex.findall(code_block_pattern, text, regex.DOTALL)

    # if no code blocks are found, return the text itself 
    if not matches:
        return [(None, text.strip())] if return_type else [text.strip()]
    
    if return_type:
        # Return tuples of (language, code)
        return [(lang.strip() or None, code.strip()) for lang, code in matches]
    else:
        # Return just the code blocks
        return [code.strip() for _, code in matches]

def remove_repr_quotes(json_string):
    """Remove quotes around Python representation strings in JSON.
    
    Identifies quoted Python object representations (e.g., "ClassName(params)")
    in a JSON string and removes the surrounding quotes to prevent them from 
    being interpreted as string literals.
    
    Args:
        json_string: JSON string potentially containing quoted object representations
        
    Returns:
        Modified JSON string with representation quotes removed
        
    Example:
        Input: '{"object": "Person(name='John', age=30)"}'
        Output: '{"object": Person(name='John', age=30)}'
    """
    pattern = r'"([A-Za-z_]\w*\(.*\))"'
    result = regex.sub(pattern, r'\1', json_string)
    return result

def custom_serializer(obj: Any): 
    """Convert non-serializable Python objects to JSON-serializable formats.
    
    Handles special cases of objects that are not natively JSON serializable
    by converting them to appropriate string representations.
    
    Args:
        obj: The object to serialize
        
    Returns:
        A JSON-serializable representation of the object
        
    Raises:
        TypeError: If the object cannot be converted to a JSON-serializable format
        
    Notes:
        Handles the following object types:
        - bytes/bytearray: Decoded to strings
        - datetime/date: Formatted as strings
        - sets: Converted to lists
        - file objects: Represented as descriptive strings
        - callables: Represented by their names
        - other objects: Represented by their string representation or class name
    """
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode()
    if isinstance(obj, (datetime, date)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "read") and hasattr(obj, "name"):
        return f"<FileObject name={getattr(obj, 'name', 'unknown')}>"
    if callable(obj):
        return obj.__name__
    if hasattr(obj, "__class__"):
        return obj.__repr__() if hasattr(obj, "__repr__") else obj.__class__.__name__
    
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# def get_type_name(type):
#     """
#     return the name of a type.
#     """
#     origin = get_origin(type)
#     args = get_args(type)
#     if origin:
#         type_name = f"{origin.__name__}[{', '.join(arg.__name__ for arg in args)}]"
#     else:
#         type_name = getattr(type, "__name__", str(type))

#     return type_name

def get_type_name(typ):
    """Get a string representation of a type, including generic types.
    
    Creates a readable string representation of Python types, handling special
    cases like Union types, generics (List[T], Dict[K, V]), and Type[T].
    
    Args:
        typ: The type to convert to a string representation
        
    Returns:
        A string representation of the type
        
    Examples:
        - int -> "int"
        - List[int] -> "list[int]"
        - Union[int, str] -> "int | str"
        - Dict[str, List[int]] -> "dict[str, list[int]]"
    """
    origin = get_origin(typ)
    if origin is None:
        return getattr(typ, "__name__", str(typ))
    
    if origin is Union:
        args = get_args(typ)
        return " | ".join(get_type_name(arg) for arg in args)
    
    if origin is type:
        return f"Type[{get_type_name(args[0])}]" if args else "Type[Any]"
    
    if origin in (list, tuple):
        args = get_args(typ)
        return f"{origin.__name__}[{', '.join(get_type_name(arg) for arg in args)}]"
    
    if origin is dict:
        key_type, value_type = get_args(typ)
        return f"dict[{get_type_name(key_type)}, {get_type_name(value_type)}]"
    
    return str(origin)

def get_pydantic_field_types(model: Type[BaseModel]) -> Dict[str, Union[str, dict]]:
    """Extract field types from a Pydantic model.
    
    Recursively analyzes a Pydantic model and returns a dictionary mapping
    field names to their type representations.
    
    Args:
        model: A Pydantic model class
        
    Returns:
        A dictionary where:
        - Keys are field names
        - Values are either string representations of types or nested dictionaries
          for fields that are themselves Pydantic models
          
    Notes:
        - Handles nested Pydantic models by recursively extracting their field types
        - For non-Pydantic fields, returns the string representation of the type
    """
    field_types = {}
    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        if hasattr(field_type, "model_fields"):
            field_types[field_name] = get_pydantic_field_types(field_type)
        else:
            type_name = get_type_name(field_type)           
            field_types[field_name] = type_name
    
    return field_types

def get_pydantic_required_field_types(model: Type[BaseModel]) -> Dict[str, str]:
    """Extract required field types from a Pydantic model.
    
    Analyzes a Pydantic model and returns a dictionary mapping required
    field names to their type representations.
    
    Args:
        model: A Pydantic model class
        
    Returns:
        A dictionary where:
        - Keys are field names of required fields
        - Values are string representations of their types
        
    Notes:
        - Only includes fields that are required (no default value)
        - Excludes fields with default values or default factories
    """
    required_field_types = {}
    for field_name, field_info in model.model_fields.items():
        if not field_info.is_required():
            continue
        if field_info.default is not PydanticUndefined or field_info.default_factory is not None:
            continue
        field_type = field_info.annotation
        type_name = get_type_name(field_type)
        required_field_types[field_name] = type_name
    
    return required_field_types

def format_pydantic_field_types(field_types: Dict[str, str]) -> str:
    """Format a dictionary of field types as a JSON-like string.
    
    Converts a dictionary mapping field names to type names into a string
    formatted like a JSON object.
    
    Args:
        field_types: Dictionary mapping field names to type representations
        
    Returns:
        A string representation of the field types in the format:
        {"field1": type1, "field2": type2, ...}
        
    Example:
        {"name": str, "age": int, "addresses": list[str]}
    """
    output = ", ".join(f"\"{field_name}\": {field_type}" for field_name, field_type in field_types.items())
    output = "{" + output + "}"
    return output

def get_error_message(errors: List[Union[ValidationError, Exception]]) -> str: 
    """Format a list of errors into a readable error message.
    
    Processes a list of errors, categorizing them as validation errors or exceptions,
    and formats them into a structured error message.
    
    Args:
        errors: A list of errors, can include ValidationError and other exceptions
        
    Returns:
        A formatted error message string
        
    Notes:
        - Groups errors by type (validation errors vs. exceptions)
        - Provides counts of each error type
        - Includes the full error messages with appropriate formatting
    """
    if not isinstance(errors, list):
        errors = [errors]
    
    validation_errors, exceptions = [], [] 
    for error in errors:
        if isinstance(error, ValidationError):
            validation_errors.append(error)
        else:
            exceptions.append(error)
    
    message = ""
    if len(validation_errors) > 0:
        message += f" >>>>>>>> {len(validation_errors)} Validation Errors: <<<<<<<<\n\n"
        message += "\n\n".join([str(error) for error in validation_errors])
    if len(exceptions) > 0:
        if len(message) > 0:
            message += "\n\n"
        message += f">>>>>>>> {len(exceptions)} Exception Errors: <<<<<<<<\n\n"
        message += "\n\n".join([str(type(error).__name__) + ": " +str(error) for error in exceptions])
    return message

def get_base_module_init_error_message(cls, data: Dict[str, Any], errors: List[Union[ValidationError, Exception]]) -> str:
    """Create a detailed error message for module initialization failures.
    
    Formats an error message for cases where a module class cannot be instantiated
    from the provided data, including the input data and all encountered errors.
    
    Args:
        cls: The class that failed to initialize
        data: The data that was used to try to initialize the class
        errors: The errors that occurred during initialization
        
    Returns:
        A formatted error message string including:
        - The class name that failed to initialize
        - The input data (formatted as JSON)
        - Detailed error messages
        
    Notes:
        - Uses custom_serializer for proper JSON formatting of the input data
        - Removes representation quotes from the formatted data
        - Includes error details from get_error_message
    """
    if not isinstance(errors, list):
        errors = [errors]
    
    message = f"Can not instantiate {cls.__name__} from: "
    formatted_data = json.dumps(data, indent=4, default=custom_serializer)
    formatted_data = remove_repr_quotes(formatted_data)
    message += formatted_data
    message += "\n\n" + get_error_message(errors)
    return message

