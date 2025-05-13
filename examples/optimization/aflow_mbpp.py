import os 
from dotenv import load_dotenv

from evoagentx.benchmark import MBPP, AFlowMBPP
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


class MBPPSplits(AFlowMBPP):

    def _load_data(self):

        # load the original MBPP data 
        mbpp_test_data = MBPP().get_test_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(mbpp_test_data))
        # radnomly select 50 samples for dev and 100 samples for test (be consistent with other models)
        dev_data_task_ids = [mbpp_test_data[idx]["task_id"] for idx in permutation[:50]]
        test_data_task_ids = [mbpp_test_data[idx]["task_id"] for idx in permutation[50:150]]

        super()._load_data() 
        full_data = self._dev_data + self._test_data
        self._dev_data = [example for example in full_data if example["task_id"] in dev_data_task_ids]
        self._test_data = [example for example in full_data if example["task_id"] in test_data_task_ids]

    

def main():

    claude_config = LiteLLMConfig(model="anthropic/claude-3-5-sonnet-20240620", anthropic_key=ANTHROPIC_API_KEY)
    optimizer_llm = LiteLLM(config=claude_config)
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    executor_llm = OpenAILLM(config=openai_config)

    # load benchmark
    mbpp = MBPPSplits()

    # create optimizer
    optimizer = AFlowOptimizer(
        graph_path = "examples/aflow/code_generation",
        optimized_path = "examples/aflow/mbpp/optimized",
        optimizer_llm=optimizer_llm,
        executor_llm=executor_llm,
        validation_rounds=3,
        eval_rounds=3,
        max_rounds=20,
        **EXPERIMENTAL_CONFIG["mbpp"]
    )

    # run optimization
    optimizer.optimize(mbpp)

    # run test 
    optimizer.test(mbpp) # use `test_rounds: List[int]` to specify the rounds to test 


if __name__ == "__main__":
    main() 