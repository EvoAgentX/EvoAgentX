from typing import Any

from .module import BaseModule


class Metadata(BaseModule):

    def __add__(self, other: "Metadata") -> "Metadata":
        if not isinstance(other, Metadata):
            raise TypeError(f"Cannot add {type(other)} to `Metadata`")

        def merge_values(v1: Any, v2: Any, field_name: str) -> Any:
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                return v1 + v2

            elif isinstance(v1, list) and isinstance(v2, list):
                return v1 + v2

            elif isinstance(v1, set) and isinstance(v2, set):
                return v1.union(v2)

            elif isinstance(v1, dict) and isinstance(v2, dict):
                merged = dict(v1)
                for key, val2 in v2.items():
                    if key in merged:
                        merged[key] = merge_values(merged[key], val2, key)
                    else:
                        merged[key] = val2
                return merged

            elif isinstance(v1, str) and isinstance(v2, str):
                if v1 == v2:
                    return v1
                else:
                    raise ValueError(f"Cannot add strings '{v1}' and '{v2}' for field '{field_name}'")

            elif isinstance(v1, bool) and isinstance(v2, bool):
                return v1 or v2

            elif isinstance(v1, Metadata) and isinstance(v2, Metadata):
                return v1 + v2

            else:
                raise TypeError(f"Cannot add values of types {type(v1)} and {type(v2)} for field '{field_name}'")

        merged_data = self.model_dump()
        merged_data.pop("class_name", None)
        other_data = other.model_dump()
        other_data.pop("class_name", None)

        for key, val2 in other_data.items():
            if key in merged_data:
                merged_data[key] = merge_values(merged_data[key], val2, key)
            else:
                merged_data[key] = val2

        if type(self) is type(other):
            return self.__class__(**merged_data)
        else:
            return Metadata(**merged_data)

