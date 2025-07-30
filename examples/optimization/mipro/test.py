import os
from dotenv import load_dotenv
# import evoagentx
import json 
from typing import Any, Tuple

# 全局设置：关闭stream
os.environ["OPENAI_STREAM"] = "false"  # 如果支持的话

from evoagentx.benchmark import MATH
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers import MiproOptimizer
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.utils.mipro_utils.register_utils import MiproRegistry


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =====================
# prepare the benchmark data 
# =====================

class MathSplits(MATH):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # radnomly select 50 samples for training and 100 samples for test
        # self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._train_data = [full_test_data[idx] for idx in permutation[:5]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]

    # define the input keys. 
    # If defined, the corresponding input key and value will be passed to the __call__ method of the program, 
    # i.e., program.__call__(**{k: v for k, v in example.items() if k in self.get_input_keys()})
    # If not defined, the program will be executed with the entire input example, i.e., program.__call__(**example)
    def get_input_keys(self):
        return ["problem"]
    
    # the benchmark must have a `evaluate` method that receives the program's `prediction` (output from the program's __call__ method) 
    # and the `label` (obtained using the `self.get_label` method) and return a dictionary of metrics. 
    def evaluate(self, prediction: Any, label: Any) -> dict:
        return super().evaluate(prediction, label)


# =====================
# prepare the program
# =====================

# here we use a simple program to answer the math problem.
class CustomProgram: 

    def __init__(self, model: OpenAILLM):
        self.model = model 
        self.prompt = "Answer the following math problem: {problem}"
        self.system_prompt = "You are a helpful math assistant."
        self.follow_up_prompt = "Please explain your reasoning step by step."
    
    # the program must have a `save` and `load` method to save and load the program
    def save(self, path: str):
        params = {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "follow_up_prompt": self.follow_up_prompt
        }
        with open(path, "w") as f:
            json.dump(params, f)

    def load(self, path: str):
        with open(path, "r") as f:
            params = json.load(f)
            self.prompt = params["prompt"]
            self.system_prompt = params.get("system_prompt", "You are a helpful math assistant.")
            self.follow_up_prompt = params.get("follow_up_prompt", "Please explain your reasoning step by step.")
    
    # the program must have a `__call__` method to execute the program.
    # It receives the key-values (specified by `get_input_keys` in the benchmark) of an input example, 
    # and returns a tuple of (prediction, execution_data), 
    # where `prediction` is the program's output and `execution_data` is a dictionary that contains all the parameters' inputs and outputs. 
    def __call__(self, problem: str) -> Tuple[str, dict]:
        
        # 组合所有 prompt
        full_prompt = f"{self.system_prompt}\n\n{self.prompt.format(problem=problem)}\n\n{self.follow_up_prompt}"
        response = self.model.generate(prompt=full_prompt)
        solution = response.content
        return solution, {"problem": problem, "solution": solution}
    

def main():

    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=False, output_response=True)
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_config = OpenAILLMConfig(model="gpt-4o", openai_key=OPENAI_API_KEY, stream=False, output_response=True)
    optimizer_llm = OpenAILLM(config=optimizer_config)

    benchmark = MathSplits()
    program = CustomProgram(model=executor_llm)

    # register the parameters to optimize 
    registry = MiproRegistry()
    # MiproRegistry requires specify the input_names and output_names for the specific parameter. 
    # The input_names and output_names should appear in the execution_data returned by the program's __call__ method. 
    registry.track(program, "prompt", input_names=["problem"], output_names=["solution"])
    registry.track(program, "system_prompt", input_names=["problem"], output_names=["solution"])
    registry.track(program, "follow_up_prompt", input_names=["problem"], output_names=["solution"])

    # optimize the program 
    optimizer = MiproOptimizer(
        registry=registry, 
        program=program, 
        optimizer_llm=optimizer_llm,
        num_threads=2, 
        eval_rounds=2, 
        auto="light",
        
    )

    best_program = optimizer.optimize(dataset=benchmark)
    print("优化后的 prompts:")
    print(f"system_prompt: {best_program.system_prompt}")
    print(f"prompt: {best_program.prompt}")
    print(f"follow_up_prompt: {best_program.follow_up_prompt}")
    
    best_program.save("examples/output/mipro/math_plug_and_play.json")

if __name__ == "__main__":
    main()