import os 
from dotenv import load_dotenv
from typing import Any, Callable 

from evoagentx.benchmark import HotPotQA
from evoagentx.optimizers import AFlowOptimizer
from evoagentx.models import LiteLLMConfig, LiteLLM, OpenAILLMConfig, OpenAILLM 


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

EXPERIMENTAL_CONFIG = {
    "humaneval": {
        "question_type": "code", 
        "operators": ["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"] 
    }, 
    "mbpp": {
        "question_type": "code", 
        "operators": ["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"] 
    },
    "hotpotqa": {
        "question_type": "qa", 
        "operators": ["Custom", "AnswerGenerate", "QAScEnsemble"]
    },
    "gsm8k": {
        "question_type": "math", 
        "operators": ["Custom", "ScEnsemble", "Programmer"]
    },
    "math": {
        "question_type": "math", 
        "operators": ["Custom", "ScEnsemble", "Programmer"]
    }
    
}


class HotPotQASplits(HotPotQA):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._dev_data))
        full_test_data = self._dev_data 
        # radnomly select 50 samples for dev and 100 samples for test
        self._dev_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]
    
    async def async_evaluate(self, graph: Callable, example: Any) -> float:

        # generate solution 
        prompt = example["question"]
        paragraphs = [item[1] for item in example["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        inputs = f"Context: {context_str}\n\nQuestion: {prompt}\n\nAnswer:"
        solution = await graph(inputs)
        label = self._get_label(example)
        metrics = await super().async_evaluate(prediction=solution, label=label)
        return metrics["f1"]
    

def main():

    claude_config = LiteLLMConfig(model="anthropic/claude-3-5-sonnet-20240620", anthropic_key=ANTHROPIC_API_KEY)
    optimizer_llm = LiteLLM(config=claude_config)
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    executor_llm = OpenAILLM(config=openai_config)

    # load benchmark
    hotpotqa = HotPotQASplits()

    # create optimizer
    optimizer = AFlowOptimizer(
        graph_path = "examples/aflow/hotpotqa",
        optimized_path = "examples/aflow/hotpotqa/optimized",
        optimizer_llm=optimizer_llm,
        executor_llm=executor_llm,
        validation_rounds=3,
        eval_rounds=3,
        max_rounds=20,
        **EXPERIMENTAL_CONFIG["hotpotqa"]
    )

    # run optimization
    optimizer.optimize(hotpotqa)

    # run test 
    optimizer.test(hotpotqa) # use `test_rounds: List[int]` to specify the rounds to test 


if __name__ == "__main__":
    main() 