import json
from pydantic import Field
from typing import Type, Optional, List, Any

from ..core.module import BaseModule
from ..models.base_model import BaseLLM
from ..models.base_model import LLMOutputParser
from ..prompts.operators import (
    ANSWER_GENERATION_PROMPT,
    SC_ENSEMBLE_PROMPT
)

class OperatorOutput(LLMOutputParser):

    def to_str(self) -> str:
        return json.dumps(self.get_structured_data(), indent=4)


class Operator(BaseModule):

    name: str = Field(description="The name of the operator.")
    description: str = Field(description="The description of the operator.")

    llm: BaseLLM = Field(description="The LLM used to execute the operator.")
    outputs_format: Type[OperatorOutput] = Field(description="The structured content of the operator's output.")

    interface: Optional[str] = Field(description="The interface for calling the operator.")

    def init_module(self):
        self._save_ignore_fields = ["llm"]

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.execute(*args, **kwargs)
    
    def execute(self, *args, **kwargs) -> dict:
        raise NotImplementedError(f"The execute function for {type(self).__name__} is not implemented!")
    
    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        ignore_fields = self._save_ignore_fields + ignore
        super().save_module(path=path, ignore=ignore_fields, **kwargs)


## The following operators are inspired by AFlow's predefined operators: https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/operator.py 

class CustomOutput(OperatorOutput):
    response: str = Field(default="", description="Your solution for this problem")


class Custom(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):
        name = "Custom"
        description = "Generates anything based on customized input and instruction"
        interface = "custom(input: str, instruction: str) -> dict with key 'response' of type str"
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=CustomOutput, **kwargs)
    
    def execute(self, input: str, instruction: str) -> dict: 
        prompt = instruction + input
        response = self.llm.generate(prompt=prompt, parser=self.outputs_format, parse_mode="str")
        output =response.get_structured_data()
        return output 


class AnswerGenerateOutput(OperatorOutput):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")


class AnswerGenerate(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):
        name = "AnswerGenerate"
        description = "Generate step by step based on the input. The step by step thought process is in the field of 'thought', and the final answer is in the field of 'answer'."
        interface = "answer_generate(input: str) -> dict with key 'thought' of type str, 'answer' of type str"
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=AnswerGenerateOutput, **kwargs)
    
    def execute(self, input: str) -> dict:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = self.llm.generate(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        return response.get_structured_data()
    

class ScEnsembleOutput(OperatorOutput):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class ScEnsemble(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):
        name = "ScEnsemble"
        description = "Uses self-consistency to select the solution that appears most frequently in the solution list, improve the selection to enhance the choice of the best solution."
        interface = "sc_ensemble(solutions: List[str]) -> dict with key 'response' of type str"
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=ScEnsembleOutput, **kwargs)
    
    def execute(self, solutions: List[str]) -> dict:
        answer_mapping = {} 
        solution_text = "" 
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65+index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"
        prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        response = self.llm.generate(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        answer: str = response.get_structured_data().get("solution_letter", "")
        answer = answer.strip().upper()
        return {"response": solutions[answer_mapping[answer]]}
