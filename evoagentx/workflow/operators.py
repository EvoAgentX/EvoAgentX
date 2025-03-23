import json
from pydantic import Field
from typing import Coroutine, Type, Optional, List, Any

from ..core.module import BaseModule
from ..models.base_model import BaseLLM
from ..models.base_model import LLMOutputParser
from ..prompts.operators import (
    ANSWER_GENERATION_PROMPT,
    SC_ENSEMBLE_PROMPT,
    REFLECTION_ON_PUBLIC_TEST_PROMPT
)
from ..utils.utils import extract_test_cases_from_jsonl, test_case_2_test_function
import sys
import traceback
import logging
import re

logger = logging.getLogger(__name__)


class OperatorOutput(LLMOutputParser):

    def to_str(self) -> str:
        return json.dumps(self.get_structured_data(), indent=4)


class Operator(BaseModule):

    name: str = Field(description="The name of the operator.")
    description: str = Field(description="The description of the operator.")

    llm: BaseLLM = Field(description="The LLM used to execute the operator.")
    outputs_format: Type[OperatorOutput] = Field(description="The structured content of the operator's output.")

    interface: Optional[str] = Field(description="The interface for calling the operator.")
    prompt: Optional[str] = Field(default="", description="The prompt for calling the operator.")

    def init_module(self):
        self._save_ignore_fields = ["llm"]

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.execute(*args, **kwargs)
    
    def execute(self, *args, **kwargs) -> dict:
        raise NotImplementedError(f"The execute function for {type(self).__name__} is not implemented!")
    
    # async def __call__(self, *args: Any, **kwargs: Any) -> dict:
    #     return await self.execute_async(*args, **kwargs)
    
    async def execute_async(self, *args, **kwargs) -> dict:
        raise NotImplementedError(f"The execute function for {type(self).__name__} is not implemented!")
    
    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        ignore_fields = self._save_ignore_fields + ignore
        super().save_module(path=path, ignore=ignore_fields, **kwargs)

    def get_prompt(self, **kwargs) -> str:
        return self.prompt 
    
    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def set_operator(self, data: dict):
        self.name = data.get("name", self.name)
        self.description = data.get("description", self.description)
        self.interface = data.get("interface", self.interface)      
        self.prompt = data.get("prompt", self.prompt)
    

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
        output = response.get_structured_data()
        return output 
    
    async def execute_async(self, input: str, instruction: str) -> dict:
        prompt = instruction + input
        response = await self.llm.generate_async(prompt=prompt, parser=self.outputs_format, parse_mode="str")
        output = response.get_structured_data()
        return output 
    
    # def execute
    
    # def execute_async(self, *args, **kwargs) -> Coroutine[Any, Any, dict]:
    #     return super().execute_async(*args, **kwargs)   


class AnswerGenerateOutput(OperatorOutput):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")


class AnswerGenerate(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):
        name = "AnswerGenerate"
        description = "Generate step by step based on the input. The step by step thought process is in the field of 'thought', and the final answer is in the field of 'answer'."
        interface = "answer_generate(input: str) -> dict with key 'thought' of type str, 'answer' of type str"
        prompt = kwargs.pop("prompt", ANSWER_GENERATION_PROMPT)
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=AnswerGenerateOutput, prompt=prompt, **kwargs)
    
    def execute(self, input: str) -> dict:
        # prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        prompt = self.prompt.format(input=input)
        response = self.llm.generate(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        return response.get_structured_data()
    

class ScEnsembleOutput(OperatorOutput):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class ScEnsemble(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):
        name = "ScEnsemble"
        description = "Uses self-consistency to select the solution that appears most frequently in the solution list, improve the selection to enhance the choice of the best solution."
        interface = "sc_ensemble(solutions: List[str], problem: str) -> dict with key 'response' of type str"
        prompt = kwargs.pop("prompt", SC_ENSEMBLE_PROMPT)
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=ScEnsembleOutput, prompt=prompt, **kwargs)
    
    def execute(self, solutions: List[str], problem: str = None) -> dict:
        # breakpoint()
        answer_mapping = {} 
        solution_text = "" 
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65+index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"
        # prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        # breakpoint()
        prompt = self.prompt.format(solutions=solution_text)
        response = self.llm.generate(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        answer: str = response.get_structured_data().get("solution_letter", "")
        answer = answer.strip().upper()
        return {"response": solutions[answer_mapping[answer]]}
    
    async def execute_async(self, solutions: List[str], problem: str = None) -> dict:
        # breakpoint()
        answer_mapping = {} 
        solution_text = "" 
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65+index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"
        # prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)   
        prompt = self.prompt.format(solutions=solution_text)
        response = await self.llm.generate_async(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        answer: str = response.get_structured_data().get("solution_letter", "")
        answer = answer.strip().upper()
        return {"response": solutions[answer_mapping[answer]]}


class TestOutput(OperatorOutput):
    result: bool = Field(default=False, description="The result of the test")
    solution: str = Field(default="", description="The solution to the problem")
    
    @classmethod
    def validate_result(cls, value):
        """验证 result 字段，确保它是布尔值"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            # 尝试将字符串转换为布尔值
            if value.lower() in ('true', 'yes', '1'):
                return True
            elif value.lower() in ('false', 'no', '0'):
                return False
            # 如果无法转换，则默认为 False
            return False
        # 其他类型默认为 False
        return False
    
    @classmethod
    def model_validate(cls, obj, **kwargs):
        """重写 model_validate 方法，确保 result 字段是布尔值"""
        if isinstance(obj, dict) and "result" in obj:
            obj["result"] = cls.validate_result(obj["result"])
        return super().model_validate(obj, **kwargs)


class Test(Operator):
    
    def __init__(self, llm: BaseLLM, **kwargs):
        name = "Test"
        description = "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information"
        interface = "test(problem: str, solution: str, entry_point: str) -> str"
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=TestOutput, **kwargs)
        
    def exec_code(self, solution, entry_point):

        test_cases = extract_test_cases_from_jsonl(entry_point, dataset="HumanEval")
        # breakpoint()
                
        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            # breakpoint()
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"
        
    def _process_llm_response(self, response):
        try:
            content = response.content
            
            # 尝试提取JSON内容
            import json
            import re
            
            # 1. 尝试直接解析完整的JSON
            try:
                data = json.loads(content)
                if "reflection_and_solution" in data:
                    return data["reflection_and_solution"]
            except json.JSONDecodeError:
                pass
                
            # 2. 尝试用正则表达式提取JSON部分
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            match = re.search(json_pattern, content)
            if match:
                try:
                    json_str = match.group(1)
                    data = json.loads(json_str)
                    if "reflection_and_solution" in data:
                        return data["reflection_and_solution"]
                except json.JSONDecodeError:
                    pass
            
            # 3. 直接查找代码块
            code_pattern = r'```(?:python)?\s*([\s\S]*?)\s*```'
            code_matches = re.findall(code_pattern, content)
            if code_matches:
                # 返回最后一个代码块
                return code_matches[-1].strip()
                
            # 如果没有代码块，则返回原始内容
            return content
        except Exception as e:
            # 如果出现异常，记录错误并返回原始内容
            logger.error(f"Error processing LLM response: {e}")
            return response.content
        
    async def execute_async(
        self, problem, solution, entry_point, test_loop: int = 3
    ):
        try:
            for _ in range(test_loop):
                # breakpoint()
                result = self.exec_code(solution, entry_point)
                if result == "no error":
                    return {"result": True, "solution": solution}
                elif "exec_fail_case" in result:
                    result = result["exec_fail_case"]
                    prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                        problem=problem,
                        solution=solution,
                        exec_pass=f"executed unsuccessfully, error: \n {result}",
                        test_fail="executed unsucessfully",
                    )
                    response = await self.llm.generate_async(prompt=prompt, parser=None, parse_mode="str")
                    solution = self._process_llm_response(response)
                else:
                    prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                        problem=problem,
                        solution=solution,
                        exec_pass="executed successfully",
                        test_fail=result,
                    )
                    response = await self.llm.generate_async    (prompt=prompt, parser=None, parse_mode="str")
                    solution = self._process_llm_response(response)
            
            result = self.exec_code(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            else:
                # 确保返回的是有效的 TestOutput 格式
                return {
                    "result": False,  # 明确指定为布尔值 False
                    "solution": solution
                }
        except Exception as e:
            # 捕获所有异常，确保返回有效的 TestOutput 格式
            print(f"Error in Test.__call__: {e}")
            print(traceback.format_exc())
            return {
                "result": False,
                "solution": solution
            }
        

class CustomCodeGenerate(Operator):
    def __init__(self, llm: BaseLLM, **kwargs):
        name = "CustomCodeGenerate"
        description = "Generates code based on customized input and instruction"
        interface = "custom_code_generate(input: str, instruction: str) -> dict with key 'response' of type str"
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=CustomOutput, **kwargs)
    
    def execute(self, input: str, instruction: str) -> dict:
        prompt = instruction + input
        response = self.llm.generate(prompt=prompt, parser=self.outputs_format, parse_mode="str")
        output = response.get_structured_data()
        # logger.info(f"CustomCodeGenerate output is {output}")
        # print(f"CustomCodeGenerate output is {output}")
        return output 
    
    async def execute_async(self, input: str, instruction: str) -> dict:
        prompt = instruction + input
        response = await self.llm.generate_async(prompt=prompt, parser=self.outputs_format, parse_mode="str")
        output = response.get_structured_data()
        return output 