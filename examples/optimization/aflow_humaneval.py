import os 
from dotenv import load_dotenv
from evoagentx.optimizers import AFlowOptimizer
from evoagentx.models import LiteLLMConfig, LiteLLM, OpenAILLMConfig, OpenAILLM 
from evoagentx.benchmark import AFlowHumanEval

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

def main():

    claude_config = LiteLLMConfig(model="anthropic/claude-3-5-sonnet-20240620", anthropic_key=ANTHROPIC_API_KEY)
    optimizer_llm = LiteLLM(config=claude_config)
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    executor_llm = OpenAILLM(config=openai_config)

    # load benchmark
    humaneval = AFlowHumanEval()

    # create optimizer
    optimizer = AFlowOptimizer(
        graph_path = "examples/aflow/code_generation",
        optimized_path = "examples/aflow/humaneval/optimized",
        optimizer_llm=optimizer_llm,
        executor_llm=executor_llm,
        validation_rounds=5,
        eval_rounds=3,
        max_rounds=20,
        **EXPERIMENTAL_CONFIG["humaneval"]
    )

    # run optimization
    optimizer.optimize(humaneval)

    # run test 
    optimizer.test(humaneval) # use `test_rounds: List[int]` to specify the rounds to test 


if __name__ == "__main__":
    main() 