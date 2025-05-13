# Acknowledgement: Modified from AFlow (https://github.com/geekan/MetaGPT/tree/main/metagpt/ext/aflow) under MIT License

import sys
import json
import asyncio
import traceback
import concurrent 
from pydantic import Field
from typing import Type, Optional, List, Any, Tuple, Union, Coroutine

from ..core.logging import logger
from ..core.module import BaseModule
from ..models.base_model import BaseLLM
from ..models.base_model import LLMOutputParser
from ..prompts.operators import (
    ANSWER_GENERATION_PROMPT,
    QA_SC_ENSEMBLE_PROMPT,
    REFLECTION_ON_PUBLIC_TEST_PROMPT,
    SC_ENSEMBLE_PROMPT,
    PYTHON_CODE_VERIFIER_PROMPT
)
from ..utils.sanitize import sanitize
from ..benchmark.benchmark import Benchmark
from ..benchmark.humaneval import AFlowHumanEval
from ..benchmark.mbpp import AFlowMBPP
from ..utils.aflow_utils.data_utils import test_case_2_test_function

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

    # def __call__(self, *args: Any, **kwargs: Any) -> dict:
    #     """Make the operator callable and automatically choose between sync and async execution"""
    #     if asyncio.iscoroutinefunction(self.async_execute) and asyncio.get_event_loop().is_running():
    #         # If the operator is in an asynchronous environment and has an async_execute method, return a coroutine
    #         return self.async_execute(*args, **kwargs)
    #     # Otherwise, use the synchronous method
    #     return self.execute(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Union[dict, Coroutine[Any, Any, dict]]:
        """Make the operator callable and automatically choose between sync and async execution."""
        try:
            # Safe way to check if we're inside an async environment
            asyncio.get_running_loop()
            return self.async_execute(*args, **kwargs)
        except RuntimeError:
            # No running loop — likely in sync context or worker thread
            return self.execute(*args, **kwargs)
    
    def execute(self, *args, **kwargs) -> dict:
        raise NotImplementedError(f"The execute function for {type(self).__name__} is not implemented!")
    
    async def async_execute(self, *args, **kwargs) -> dict:
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
        output =response.get_structured_data()
        return output 
    
    async def async_execute(self, input: str, instruction: str) -> dict:
        prompt = instruction + input
        response = await self.llm.async_generate(prompt=prompt, parser=self.outputs_format, parse_mode="str")
        output = response.get_structured_data()
        return output 


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
    
    async def async_execute(self, input: str) -> dict:
        # prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        prompt = self.prompt.format(input=input)
        response = await self.llm.async_generate(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        return response.get_structured_data()
    

class ScEnsembleOutput(OperatorOutput):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class QAScEnsemble(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):
        name = "QAScEnsemble"
        description = "Uses self-consistency to select the solution that appears most frequently in the solution list, improve the selection to enhance the choice of the best solution."
        interface = "sc_ensemble(solutions: List[str]) -> dict with key 'response' of type str"
        prompt = kwargs.pop("prompt", QA_SC_ENSEMBLE_PROMPT)
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=ScEnsembleOutput, prompt=prompt, **kwargs)
    
    def _prepare_solutions(self, solutions: List[str]) -> Tuple[dict, str]:
        answer_mapping = {} 
        solution_text = "" 
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65+index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"
        return answer_mapping, solution_text
        
    def _process_response(self, response: LLMOutputParser, answer_mapping: dict, solutions: List[str]) -> dict:
        answer: str = response.get_structured_data().get("solution_letter", "")
        answer = answer.strip().upper()
        return {"response": solutions[answer_mapping[answer]]}
    
    def execute(self, solutions: List[str]) -> dict:
        answer_mapping, solution_text = self._prepare_solutions(solutions)
        prompt = self.prompt.format(solutions=solution_text)
        response = self.llm.generate(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        return self._process_response(response, answer_mapping, solutions)

    async def async_execute(self, solutions: List[str]) -> dict:
        answer_mapping, solution_text = self._prepare_solutions(solutions)
        prompt = self.prompt.format(solutions=solution_text)
        response = await self.llm.async_generate(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        return self._process_response(response, answer_mapping, solutions)


class ScEnsemble(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):
        name = "ScEnsemble" 
        description = "Uses self-consistency to select the solution that appears most frequently in the solution list, improve the selection to enhance the choice of the best solution."
        interface = "sc_ensemble(solutions: List[str], problem: str) -> dict with key 'response' of type str"
        prompt = kwargs.pop("prompt", SC_ENSEMBLE_PROMPT) 
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=ScEnsembleOutput, prompt=prompt, **kwargs)
    
    def _prepare_solutions(self, solutions: List[str]) -> Tuple[dict, str]:
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"
        return answer_mapping, solution_text
    
    def _process_response(self, response: LLMOutputParser, answer_mapping: dict, solutions: List[str]) -> dict:
        answer: str = response.get_structured_data().get("solution_letter", "")
        answer = answer.strip().upper()
        return {"response": solutions[answer_mapping[answer]]}
    
    def execute(self, solutions: List[str], problem: str) -> dict:
        answer_mapping, solution_text = self._prepare_solutions(solutions)
        prompt = self.prompt.format(problem=problem, solutions=solution_text)
        response = self.llm.generate(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        return self._process_response(response, answer_mapping, solutions)

    async def async_execute(self, solutions: List[str], problem: str) -> dict:
        answer_mapping, solution_text = self._prepare_solutions(solutions)
        prompt = self.prompt.format(problem=problem, solutions=solution_text)
        response = await self.llm.async_generate(prompt=prompt, parser=self.outputs_format, parse_mode="xml")
        return self._process_response(response, answer_mapping, solutions)



class CustomCodeGenerate(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):

        name = "CustomCodeGenerate"
        description = "Generates code based on customized input and instruction"
        interface = "custom_code_generate(problem: str, entry_point: str, instruction: str) -> dict with key 'response' of type str"
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=CustomOutput, **kwargs)
    
    def execute(self, problem: str, entry_point: str, instruction: str) -> dict:
        prompt = instruction + problem
        response = self.llm.generate(prompt=prompt, parser=self.outputs_format, parse_mode="str")
        code = sanitize(response.content, entrypoint=entry_point)
        return {"response": code}
    
    async def async_execute(self, problem: str, entry_point: str, instruction: str) -> dict:
        prompt = instruction + problem
        response = await self.llm.async_generate(prompt=prompt, parser=self.outputs_format, parse_mode="str")
        code = sanitize(response.content, entrypoint=entry_point)
        return {"response": code}


class TestOutput(OperatorOutput):
    
    result: bool = Field(default=False, description="The result of the test")
    solution: str = Field(default="", description="The solution to the problem")
    
    @classmethod
    def validate_result(cls, value):
        """Validate the result field, ensuring it is a boolean value"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            # Try to convert string to boolean
            if value.lower() in ('true', 'yes', '1'):
                return True
            elif value.lower() in ('false', 'no', '0'):
                return False
            # If conversion fails, default to False
            return False
        # Other types default to False
        return False
    
    @classmethod
    def model_validate(cls, obj, **kwargs):
        """Override model_validate method to ensure result field is boolean"""
        if isinstance(obj, dict) and "result" in obj:
            obj["result"] = cls.validate_result(obj["result"])
        return super().model_validate(obj, **kwargs)
    

class ReflectionTestOp(OperatorOutput):

    reflection_and_solution: str = Field(default="", description="Corrective solution for code execution errors or test case failures")


TEST_SUPPORTED_BENCHMARKS = (AFlowHumanEval, AFlowMBPP)

class Test(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):

        name = "Test"
        description = "Tests the solution using public test cases. If the solution fails, it reflects on the errors and attempts to modify the solution. Returns True and the solution if all tests pass after modifications. Returns False and the current solution if it still fails after modifications."
        interface = "test(problem: str, solution: str, entry_point: str, benchmark = self.benchmark) -> dict with key 'result' of type bool and key 'solution' of type str. Always include 'benchmark = self.benchmark' in the input."
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=TestOutput, **kwargs)
    
    def exec_code(self, solution: str, entry_point: str, benchmark: Benchmark):

        if any(isinstance(benchmark, benchmark_type) for benchmark_type in TEST_SUPPORTED_BENCHMARKS):
            test_cases = benchmark.extract_test_cases_with_entry_point(entry_point)
        else:
            supported_benchmarks = [typ.__name__ for typ in TEST_SUPPORTED_BENCHMARKS]
            raise ValueError(f"Benchmark {type(benchmark)} is not supported! Available benchmarks: {supported_benchmarks} and their subclasses")
        
        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                # with open("tester.txt", "a") as f:
                #     f.write("test_error of " + entry_point + "\n")
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
                # with open("tester.txt", "a") as f:
                #     f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    async def async_execute(self, problem, solution, entry_point, benchmark: Benchmark, test_loop: int = 3):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str, benchmark = self.benchmark) -> str"
        }
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point, benchmark)
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
                # response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                # solution = response["reflection_and_solution"]
                response = await self.llm.async_generate(prompt=prompt, parser=ReflectionTestOp, parse_mode="json")
                solution = sanitize(
                    response.get_structured_data().get("reflection_and_solution", response.content), 
                    entrypoint=entry_point
                )
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                # response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                # solution = response["reflection_and_solution"]
                response = await self.llm.async_generate(prompt=prompt, parser=ReflectionTestOp, parse_mode="json")
                solution = sanitize(
                    response.get_structured_data().get("reflection_and_solution", response.content), 
                    entrypoint=entry_point
                )
        
        result = self.exec_code(solution, entry_point, benchmark)

        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}
        

def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve']()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"
    

class CodeGenerateOutput(OperatorOutput):
    code: str = Field(default="", description="Your complete code solution for this problem") 


class Programmer(Operator):

    def __init__(self, llm: BaseLLM, **kwargs):
        name = "Programmer"
        description = "Automatically writes, executes Python code, and returns the solution based on the provided problem description and analysis. The `output` only contains the final answer. If you want to see the detailed solution process, it's recommended to retrieve the `code`."
        interface = "programmer(problem: str, analysis: str = 'None') -> dict with keys 'code' and 'output' of type str"
        prompt = kwargs.pop("prompt", PYTHON_CODE_VERIFIER_PROMPT)
        super().__init__(name=name, description=description, interface=interface, llm=llm, outputs_format=CodeGenerateOutput, prompt=prompt, **kwargs)

    async def exec_code(self, code, timeout=30):
        """
        Asynchronously execute code and return an error if timeout occurs.
        """
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            try:
                # Submit run_code task to the process pool
                future = loop.run_in_executor(executor, run_code, code)
                # Wait for the task to complete or timeout
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                # Timeout, attempt to shut down the process pool
                executor.shutdown(wait=False, cancel_futures=True)
                return "Error", "Code execution timed out"
            except Exception as e:
                return "Error", f"Unknown error: {str(e)}"
    
    async def code_generate(self, problem, analysis, feedback):
        """
        Asynchronous method to generate code.
        """
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self.llm.async_generate(prompt=prompt, parser=None, parse_mode="str")
        code = sanitize(response.content, entrypoint="solve")
        return {"code": code}
    
    async def async_execute(self, problem: str, analysis: str = "None"): 

        code = None 
        output = None 
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback)
            code = code_response.get("code")
            if not code:
                return {"code": code, "output": "No code generated"}
            status, output = await self.exec_code(code)
            if status == "Success":
                return {"code": code, "output": output}
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
        return {"code": code, "output": output}