import os 
import yaml
import json 
from typing import Callable, Any, Dict, List
from pydantic import BaseModel, ValidationError
from pydantic._internal._model_construction import ModelMetaclass

from .logging import logger
from .callbacks import callback_manager, exception_buffer
from .module_utils import (
    save_json,
    custom_serializer,
    parse_json_from_text, 
    get_error_message,
    get_base_module_init_error_message
)
from .registry import register_module, MODULE_REGISTRY


class MetaModule(ModelMetaclass):
    """
    MetaModule is a metaclass that automatically registers all subclasses of BaseModule.

    
    Attributes:
        No public attributes
    """
    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Creates a new class and registers it in MODULE_REGISTRY.
        
        Args:
            mcs: The metaclass itself
            name: The name of the class being created
            bases: Tuple of base classes
            namespace: Dictionary containing the class attributes and methods
            **kwargs: Additional keyword arguments
        
        Returns:
            The created class object
        """
        cls = super().__new__(mcs, name, bases, namespace)
        register_module(name, cls)
        return cls 


class BaseModule(BaseModel, metaclass=MetaModule):
    """
    Base module class that serves as the foundation for all modules in the EvoAgentX framework.
    
    This class provides serialization/deserialization capabilities, supports creating instances from
    dictionaries, JSON, or files, and exporting instances to these formats.
    
    Attributes:
        class_name: The class name, defaults to None but is automatically set during subclass initialization
        model_config: Pydantic model configuration that controls type matching and behavior
    """

    class_name: str = None 
    model_config = {"arbitrary_types_allowed": True, "extra": "allow", "protected_namespaces": ()}

    def __init_subclass__(cls, **kwargs):
        """
        Subclass initialization method that automatically sets the class_name attribute.
        
        Args:
            cls: The subclass being initialized
            **kwargs: Additional keyword arguments
        """
        super().__init_subclass__(**kwargs)
        cls.class_name = cls.__name__
    
    def __init__(self, **kwargs):
        """
        Initializes a BaseModule instance.
        
        Args:
            **kwargs: Keyword arguments used to initialize the instance
        
        Raises:
            ValidationError: When parameter validation fails
            Exception: When other errors occur during initialization
        """

        try:
            for field_name, _ in type(self).model_fields.items():
                field_value = kwargs.get(field_name, None)
                if field_value:
                    kwargs[field_name] = self._process_data(field_value)
                # if field_value and isinstance(field_value, dict) and "class_name" in field_value:
                #     class_name = field_value.get("class_name")
                #     sub_cls = MODULE_REGISTRY.get_module(cls_name=class_name)
                #     kwargs[field_name] = sub_cls._create_instance(field_value)
            super().__init__(**kwargs) 
            self.init_module()
        except (ValidationError, Exception) as e:
            exception_handler = callback_manager.get_callback("exception_buffer")
            if exception_handler is None:
                error_message = get_base_module_init_error_message(
                    cls=self.__class__, 
                    data=kwargs, 
                    errors=e
                )
                logger.error(error_message)
                raise
            else:
                exception_handler.add(e)
    
    def init_module(self):
        """
        Module initialization method that subclasses can override to provide additional initialization logic.
        """
        pass

    def __str__(self):
        """
        Returns a string representation of the object.
        
        Returns:
            str: String representation of the object
        """
        return self.to_str()
    
    @property
    def kwargs(self):
        """
        Returns the extra fields of the model.
        
        Returns:
            dict: Dictionary containing all extra keyword arguments
        """
        return self.model_extra
    
    @classmethod
    def _create_instance(cls, data: Dict[str, Any]) -> "BaseModule":
        """
        Internal method for creating an instance from a dictionary.
        
        Args:
            data: Dictionary containing instance data
        
        Returns:
            BaseModule: The created instance
        """
        processed_data = {k: cls._process_data(v) for k, v in data.items()}
        # print(processed_data)
        return cls.model_validate(processed_data)

    @classmethod
    def _process_data(cls, data: Any) -> Any:
        """
        Recursive method for processing data, with special handling for dictionaries containing class_name.
        
        Args:
            data: Data to be processed
        
        Returns:
            Processed data
        """
        if isinstance(data, dict):
            if "class_name" in data:
                sub_class = MODULE_REGISTRY.get_module(data.get("class_name"))
                return sub_class._create_instance(data)
            else:
                return {k: cls._process_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [cls._process_data(x) for x in data]
        else:
            return data 

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs):
        """
        Instantiate the BaseModule from a dictionary.
        
        Args:
            data: Dictionary containing instance data
            **kwargs: Additional keyword arguments, can include log to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            Exception: When errors occur during initialization
        """
        use_logger = kwargs.get("log", True)
        with exception_buffer() as buffer:
            try:
                class_name = data.get("class_name", None)
                if class_name:
                    cls = MODULE_REGISTRY.get_module(class_name)
                module = cls._create_instance(data)
                # module = cls.model_validate(data)
                if len(buffer.exceptions) > 0:
                    error_message = get_base_module_init_error_message(cls, data, buffer.exceptions)
                    if use_logger:
                        logger.error(error_message)
                    raise Exception(get_error_message(buffer.exceptions))
            finally:
                pass
        return module
    
    @classmethod
    def from_json(cls, content: str, **kwargs):
        """
        Construct the BaseModule from a JSON string.
        
        This method uses yaml.safe_load to parse the JSON string into a Python object,
        which supports more flexible parsing than standard json.loads (including handling
        single quotes, trailing commas, etc). The parsed data is then passed to from_dict
        to create the instance.
        
        Args:
            content: JSON string
            **kwargs: Additional keyword arguments, can include `log` to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            ValueError: When the input is not a valid JSON string
        """
        use_logger = kwargs.get("log", True)
        try:
            data = yaml.safe_load(content)
        except Exception:
            error_message = f"Can not instantiate {cls.__name__}. The input to {cls.__name__}.from_json is not a valid JSON string."
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        
        if not isinstance(data, (list, dict)):
            error_message = f"Can not instantiate {cls.__name__}. The input to {cls.__name__}.from_json is not a valid JSON string."
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)

        return cls.from_dict(data, log=use_logger)
    
    @classmethod
    def from_str(cls, content: str, **kwargs):
        """
        Construct the BaseModule from a string that may contain JSON.
        
        This method is more forgiving than `from_json` as it can extract valid JSON
        objects embedded within larger text. It uses `parse_json_from_text` to extract 
        all potential JSON strings from the input text, then tries to create an instance 
        from each extracted JSON string until successful.
        
        Args:
            content: Text that may contain JSON strings
            **kwargs: Additional keyword arguments, can include `log` to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            ValueError: When the input does not contain valid JSON strings or the JSON is incompatible with the class
        """
        use_logger = kwargs.get("log", True)
        
        extracted_json_list = parse_json_from_text(content)
        if len(extracted_json_list) == 0:
            error_message = f"The input to {cls.__name__}.from_str does not contain any valid JSON str."
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        
        module = None
        for json_str in extracted_json_list:
            try:
                module = cls.from_json(json_str, log=False)
            except Exception:
                continue
            break
        
        if module is None:
            error_message = f"Can not instantiate {cls.__name__}. The input to {cls.__name__}.from_str either does not contain a valide JSON str, or the JSON str is incomplete or incompatable (incorrect variables or types) with {cls.__name__}."
            error_message += f"\nInput:\n{content}"
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        
        return module
    
    @classmethod 
    def load_module(cls, path: str, **kwargs) -> dict:
        """
        Load the values for a module from a file.
        
        By default, it opens the specified file and uses `yaml.safe_load` to parse its contents 
        into a Python object (typically a dictionary).
        
        Args:
            path: The path of the file
            **kwargs: Additional keyword arguments
        
        Returns:
            dict: The JSON object instantiated from the file
        """
        with open(path, mode="r", encoding="utf-8") as file:
            content = yaml.safe_load(file.read())
        return content

    @classmethod
    def from_file(cls, path: str, load_function: Callable=None, **kwargs):
        """
        Construct the BaseModule from a file.
        
        This method reads and parses a file into a data structure, then creates
        a module instance from that data. It first verifies that the file exists,
        then uses either the provided `load_function` or the default `load_module`
        method to read and parse the file content, and finally calls `from_dict`
        to create the instance.
        
        Args:
            path: The path of the file
            load_function: The function used to load the data, takes a file path as input and returns a JSON object
            **kwargs: Additional keyword arguments, can include `log` to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            ValueError: When the file does not exist
        """
        use_logger = kwargs.get("log", True)
        if not os.path.exists(path):
            error_message = f"File \"{path}\" does not exist!"
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        
        function = load_function or cls.load_module
        content = function(path, **kwargs)
        module = cls.from_dict(content, log=use_logger)

        return module
    
    # def to_dict(self, **kwargs) -> dict:
    #     """
    #     convert the BaseModule to a dict. 
    #     """
    #     return self.model_dump()

    def to_dict(self, exclude_none: bool = True, ignore: List[str] = [], **kwargs) -> dict:
        """
        Convert the BaseModule to a dictionary.
        
        Args:
            exclude_none: Whether to exclude fields with None values
            ignore: List of field names to ignore
            **kwargs: Additional keyword arguments
        
        Returns:
            dict: Dictionary containing the object data
        """
        data = {}
        for field_name, _ in type(self).model_fields.items():
            if field_name in ignore:
                continue
            field_value = getattr(self, field_name, None)
            if exclude_none and field_value is None:
                continue
            if isinstance(field_value, BaseModule):
                data[field_name] = field_value.to_dict(exclude_none=exclude_none, ignore=ignore)
            elif isinstance(field_value, list):
                data[field_name] = [
                    item.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(item, BaseModule) else item
                    for item in field_value
                ]
            elif isinstance(field_value, dict):
                data[field_name] = {
                    key: value.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(value, BaseModule) else value
                    for key, value in field_value.items()
                }
            else:
                data[field_name] = field_value
        
        return data
    
    def to_json(self, use_indent: bool=False, ignore: List[str] = [], **kwargs) -> str:
        """
        Convert the BaseModule to a JSON string.
        
        Args:
            use_indent: Whether to use indentation
            ignore: List of field names to ignore
            **kwargs: Additional keyword arguments
        
        Returns:
            str: The JSON string
        """
        if use_indent:
            kwargs["indent"] = kwargs.get("indent", 4)
        else:
            kwargs.pop("indent", None)
        if kwargs.get("default", None) is None:
            kwargs["default"] = custom_serializer
        data = self.to_dict(exclude_none=True)
        for ignore_field in ignore:
            data.pop(ignore_field, None)
        return json.dumps(data, **kwargs)
    
    def to_str(self, **kwargs) -> str:
        """
        Convert the BaseModule to a string. Use .to_json to output JSON string by default.
        
        Args:
            **kwargs: Additional keyword arguments
        
        Returns:
            str: The string
        """
        return self.to_json(use_indent=False)
    
    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        """
        Save the BaseModule to a file.
        
        This method will set non-serializable objects to None by default.
        If you want to save non-serializable objects, override this method.
        Remember to also override the `load_module` function to ensure the loaded
        object can be correctly parsed by `cls.from_dict`.
        
        Args:
            path: The path to save the file
            ignore: List of field names to ignore
            **kwargs: Additional keyword arguments
        
        Returns:
            str: The path where the file is saved, same as the input path
        """
        logger.info("Saving {} to {}", self.__class__.__name__, path)
        return save_json(self.to_json(use_indent=True, default=lambda x: None, ignore=ignore), path=path)
    

__all__ = ["BaseModule"]

