import os
import json 
from dotenv import load_dotenv
from typing import Any, Tuple

from evoagentx.benchmark import MATH
from evoagentx.core.logging import logger
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
        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._test_data = [full_test_data[idx] for idx in permutation[100:200]]

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
        self.prompt = "Let's think step by step to answer the math question: {problem}"
    
    # the program must have a `save` and `load` method to save and load the program
    def save(self, path: str):
        params = {"prompt": self.prompt}
        with open(path, "w") as f:
            json.dump(params, f)

    def load(self, path: str):
        with open(path, "r") as f:
            params = json.load(f)
            self.prompt = params["prompt"]
    
    # the program must have a `__call__` method to execute the program.
    # It receives the key-values (specified by `get_input_keys` in the benchmark) of an input example, 
    # and returns a tuple of (prediction, execution_data), 
    # where `prediction` is the program's output and `execution_data` is a dictionary that contains all the parameters' inputs and outputs. 
    def __call__(self, problem: str) -> Tuple[str, dict]:
        
        prompt = self.prompt.format(problem=problem)
        response = self.model.generate(prompt=prompt)
        solution = response.content
        return solution, {"problem": problem, "solution": solution}
    

def main():

    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_config = OpenAILLMConfig(model="gpt-4o", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    optimizer_llm = OpenAILLM(config=optimizer_config)

    benchmark = MathSplits()
    program = CustomProgram(model=executor_llm)

    # register the parameters to optimize 
    registry = MiproRegistry()
    # MiproRegistry requires specify the input_names and output_names for the specific parameter. 
    # The input_names and output_names should appear in the execution_data returned by the program's __call__ method. 
    registry.track(program, "prompt", input_names=["problem"], output_names=["solution"])

    # optimize the program 
    # `evaluator` is optional. If not provided, the optimizer will construct an evaluator based on the `evaluate` method of the benchmark. 
    optimizer = MiproOptimizer(
        registry=registry, 
        program=program, 
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=4, 
        max_labeled_demos=4,
        num_threads=20,  
        eval_rounds=1, 
        auto="medium",
        save_path="examples/output/mipro/math_plug_and_play" 
    )

    logger.info("Optimizing program...")
    optimizer.optimize(dataset=benchmark)
    optimizer.restore_best_program()

    logger.info("Evaluating program on test set...")
    with suppress_logger_info():
        results = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    logger.info(f"Evaluation metrics (after optimization): {results}")


if __name__ == "__main__":
    main()
