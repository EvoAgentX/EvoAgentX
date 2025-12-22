from pydantic import Field
from typing import Optional

from ..core.logging import logger
from ..core.module_utils import extract_code_blocks
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput
from ..prompts.code_verification import CODE_VERIFICATION_ACTION


class CodeVerificationInput(ActionInput):

    code: str = Field(description="The code string to be verified for correctness and completeness.")
    requirements: Optional[str] = Field(default=None, description="Optional field containing requirements or specifications for the code.")


class CodeVerificationOutput(ActionOutput):

    analysis_summary: Optional[str] = Field(default=None, description="Brief summary of your findings, highlighting key issues or confirming overall quality.")
    issues_identified: Optional[str] = Field(default=None, description="Categorized list of issues found, with explanation of impact and severity.")
    thought_process: Optional[str] = Field(default=None, description="Detailed explanation of your verification reasoning and methodology applied.")
    modification_strategy: Optional[str] = Field(default=None, description="Describe the changes you made (or will make) to address the issues. Include any assumptions, design choices, or additional components you decided to add to make the code complete and robust.")
    verified_code: str = Field(description="The complete, corrected code if issues are found, or the original code if no issues are found.")


class CodeVerification(Action):

    def __init__(self, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else CODE_VERIFICATION_ACTION["name"]
        description = kwargs.pop("description") if "description" in kwargs else CODE_VERIFICATION_ACTION["description"]
        prompt = kwargs.pop("prompt") if "prompt" in kwargs else CODE_VERIFICATION_ACTION["prompt"]
        # inputs_format = kwargs.pop("inputs_format") if "inputs_format" in kwargs else CodeVerificationInput
        # outputs_format = kwargs.pop("outputs_format") if "outputs_format" in kwargs else CodeVerificationOutput
        inputs_format = kwargs.pop("inputs_format", None) or CodeVerificationInput
        outputs_format = kwargs.pop("outputs_format", None) or CodeVerificationOutput
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> CodeVerificationOutput:

        if not inputs:
            logger.error("CodeVerification action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to CodeVerification action is None or empty.')

        prompt_params_names = ["code", "requirements"]
        prompt_params_values = {param: inputs.get(param, "Not Provided") for param in prompt_params_names}
        prompt = self.prompt.format(**prompt_params_values)
        response = llm.generate(prompt = prompt, system_message=sys_msg)
        
        try:
            verification_result = self.outputs_format.parse(response.content, parse_mode="title")
        except Exception:
            try:
                code_blocks = extract_code_blocks(response.content, return_type=True)
                code = "\n\n".join([f"```{code_type}\n{code}\n```" for code_type, code in code_blocks])
                verification_result = self.outputs_format(verified_code=code)
            except Exception:
                raise ValueError(f"Failed to extract code blocks from the response: {response.content}")
        
        if return_prompt:
            return verification_result, prompt
        
        return verification_result