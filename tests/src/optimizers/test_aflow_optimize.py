from evoagentx.optimizers.aflow_optimizer import *
# from evoagentx.optimizers.aflow_optimizer import *
from evoagentx.models import OpenAILLMConfig, OpenAILLM

if __name__ == "__main__":
    

    dataset = "HumanEval"
    operators = EXPERIMENT_CONFIGS[dataset].operators
    question_type = EXPERIMENT_CONFIGS[dataset].question_type

    siliconflow_api_key = "sk-vhxzxtorlkllodmmcthkhjiidjvnkkgpaliydwrdkrqzilpw" # ruihong'key
    # model = "deepseek-ai/DeepSeek-V3"
    openai_api_key = "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"
    model = "gpt-4o-mini"

    executor_llm = OpenAILLM(config=OpenAILLMConfig(openai_key=openai_api_key,
                                                model="gpt-4o-mini"))
    optimizer_llm = OpenAILLM(config=OpenAILLMConfig(openai_key=openai_api_key,
                                                model="gpt-4o-mini"))
    action_graph_llm_config = OpenAILLMConfig(openai_key=openai_api_key,
                                                model="gpt-4o-mini")

    
    optimizer = AFlowOptimizer(
    dataset="HumanEval",
    operators=operators,
    question_type=question_type,
    check_convergence=True,
    initial_round=1,
    max_rounds=1,
    validation_rounds=5,
    optimized_path="evoagentx/ext/aflow/scripts/optimized",
    optimizer_llm=optimizer_llm,
    executor_llm=executor_llm,
    action_graph_llm_config=action_graph_llm_config
)
    
    optimizer.optimize('Test')