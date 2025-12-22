
from ...optimizers.engine.registry import ParamRegistry
from typing import Any, Optional, List, Dict
# from ...optimizers.engine.registry import OptimizableField

class MiproRegistry(ParamRegistry):
    """
    Extended ParamRegistry that supports storing input_names and output_names
    for each optimizable field. Compatible with all original track() usages.
    """

    def track(
        self,
        root_or_obj: Any,
        path_or_attr: str = None,
        *,
        name: Optional[str] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        input_descs: Optional[Dict[str, str]] = None,
        output_descs: Optional[Dict[str, str]] = None,
    ):
        # Support batch registration with list/tuple
        if isinstance(root_or_obj, (list, tuple)):
            for item in root_or_obj:
                if isinstance(item, dict):
                    self.track(**item)
                elif isinstance(item, (list, tuple)):
                    if len(item) == 7:
                        self.track(
                            item[0], item[1],
                            name=item[2],
                            input_names=item[3],
                            output_names=item[4],
                            input_descs=item[5],
                            output_descs=item[6]
                        )
                    else:
                        raise ValueError("Each tuple must be (obj, attr, name, input_names, output_names, input_descs, output_descs)")
            return self

        # Call parent to do normal tracking
        super().track(root_or_obj, path_or_attr, name=name)

        # Inject input/output names into the field
        key = name or path_or_attr
        field = self.fields[key]
        field.input_names = input_names or []
        field.output_names = output_names or []
        field.input_descs = input_descs or {}
        field.output_descs = output_descs or {}

        return self
    
    def get_input_names(self, name: str) -> List[str]:
        """Return the input_names for a registered field, or an empty list if not set."""
        return getattr(self.fields[name], "input_names", None) or []

    def get_output_names(self, name: str) -> List[str]:
        """Return the output_names for a registered field, or an empty list if not set."""
        return getattr(self.fields[name], "output_names", None) or []
    
    def get_input_desc_dict(self, name: str) -> Dict[str, str]:
        """Return the input_descs for a registered field, or an empty dict if not set."""
        return getattr(self.fields[name], "input_descs", {})

    def get_output_desc_dict(self, name: str) -> Dict[str, str]:
        """Return the output_descs for a registered field, or an empty dict if not set."""
        return getattr(self.fields[name], "output_descs", {})
    
    def get_input_desc(self, name: str, input_name: str) -> str:
        """Return the input_desc for a registered field, or an empty string if not set."""
        return self.get_input_desc_dict(name).get(input_name, "")

    def get_output_desc(self, name: str, output_name: str) -> str:
        """Return the output_desc for a registered field, or an empty string if not set."""
        return self.get_output_desc_dict(name).get(output_name, "")
    
    def describe(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dict of all fields and their metadata, including input/output names if present.
        """
        result = {}
        for name, field in self.fields.items():
            result[name] = {
                "value": field.get(),
                "input_names": getattr(field, "input_names", None),
                "output_names": getattr(field, "output_names", None),
                "input_descs": getattr(field, "input_descs", {}),
                "output_descs": getattr(field, "output_descs", {}),
            }
        return result