from typing import List, Dict, Any, Optional, Union
import string
from pydantic import Field

from evoagentx.core.module import BaseModule

class MetaPromptTemplate(BaseModule):
    """
    A meta prompt template that can be used to generate a prompt template for different situations.
    """
    prompt_template_sub_part: Dict[str, str] = Field(description="The sub parts of the prompt template.", default_factory=dict)

    def render_complete_prompt_template(self) -> str:
        """
        Render the complete prompt template.
        """
        template = ""
        for sub_part in self.prompt_template_sub_part.values():
            template += f"{sub_part}\n"
        return template
    
    def extract_full_field_expressions(fmt: str):
        formatter = string.Formatter()
        return [f for _, f, _, _ in formatter.parse(fmt) if f]

    
    def return_template_total_tokens(self) -> int:
        """
        estimate the total tokens of the prompt template(not equal to the final prompt)
        """
        raise NotImplementedError("This method is not implemented yet.")
    