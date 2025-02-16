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

    dir_folder = os.path.dirname(path)
    if len(dir_folder.strip()) == 0:
        return
    if not os.path.exists(dir_folder):
        os.makedirs(dir_folder, exist_ok=True)

def generate_id():
    return uuid4().hex

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_json(path: str, type: str="json"):
    
    assert type in ["json", "jsonl"] # only support json or jsonl format
    if not os.path.exists(path=path):
        logger.error(f"File \"{path}\" does not exists!")
    
    if type == "json":
        try:
            with open(path, "r", encoding="utf-8") as file:
                outputs = yaml.safe_load(file.read())
        except Exception:
            logger.error(f"File \"{path}\" is not a valid json file!")
    
    elif type == "jsonl":
        outputs = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                outputs.append(yaml.safe_load(line))
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

def extract_code_blocks(text):

    # Regular expression to match code blocks enclosed in triple backticks
    code_block_pattern = r"```(?:[a-zA-Z]*)?\n*(.*?)\n*```"
    # Find all matches in the text
    code_blocks = regex.findall(code_block_pattern, text, regex.DOTALL)

    return code_blocks

def remove_repr_quotes(json_string):
    pattern = r'"([A-Za-z_]\w*\(.*\))"'
    result = regex.sub(pattern, r'\1', json_string)
    return result

def custom_serializer(obj: Any): 

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

    output = ", ".join(f"\"{field_name}\": {field_type}" for field_name, field_type in field_types.items())
    output = "{" + output + "}"
    return output

def get_error_message(errors: List[Union[ValidationError, Exception]]) -> str: 

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

    if not isinstance(errors, list):
        errors = [errors]
    
    message = f"Can not instantiate {cls.__name__} from: "
    formatted_data = json.dumps(data, indent=4, default=custom_serializer)
    formatted_data = remove_repr_quotes(formatted_data)
    message += formatted_data
    message += "\n\n" + get_error_message(errors)
    return message


