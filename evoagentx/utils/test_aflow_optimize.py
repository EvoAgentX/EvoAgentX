from evoagentx.optimizers.aflow_optimizer import AFlowOptimizer
from evoagentx.models.openai_model import OpenAILLM, OpenAILLMConfig
import os

def main():
    # Get API key from environment or use a default (replace with your valid key)
    api_key = os.environ.get("OPENAI_API_KEY", "your_valid_api_key_here")
    
    # 创建 OpenAI LLM 实例
    llm_config = OpenAILLMConfig(
        openai_key=api_key,
        model="gpt-4o-mini"
    )
    llm = OpenAILLM(config=llm_config)
    
    # 创建 AFlowOptimizer 实例
    optimizer = AFlowOptimizer(
        dataset="HumanEval",
        operators=["custom_code_generate", "sc_ensemble", "test"],
        question_type="code",
        check_convergence=True,
        initial_round=1,
        max_rounds=1,
        validation_rounds=5,
        optimized_path="evoagentx/ext/aflow/scripts/optimized",
        optimizer_llm=llm,
        executor_llm=llm,
        action_graph_llm_config=llm_config
    )
    
    # 运行优化
    optimizer.optimize("Test")

if __name__ == "__main__":
    main() 