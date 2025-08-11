"""
Quick Start Example: Node-wise Prompt Evolution

This example demonstrates how to use the new node-wise evolution architecture 
where each registered prompt evolves independently and combinations are evaluated together.

Key Benefits:
- Each prompt node evolves independently  
- Controllable evaluation cost via combination sampling
- No complex XML parsing needed
- Fine-grained performance tracking per node
"""

import asyncio
import os
from dotenv import load_dotenv

from evoagentx.optimizers.evo2_optimizer import GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.core.logging import logger

class SimpleClassifier:
    """
    A simple classifier that uses two different prompts for binary classification.
    """
    def __init__(self, model: OpenAILLM):
        self.model = model
        
        # Two prompts that will be evolved independently
        self.system_prompt = "You are an expert classifier. Analyze the input carefully."
        self.task_prompt = "Classify the following text as either (A) or (B). Respond with exactly (A) or (B)."

    def __call__(self, input: str) -> tuple[str, dict]:
        """Run classification using both prompts."""
        full_prompt = f"{self.system_prompt}\n\n{self.task_prompt}\n\nText: {input}"
        
        response = self.model.generate(prompt=full_prompt)
        prediction = response.content.strip()
        
        # Extract (A) or (B) from response
        if "(A)" in prediction.upper():
            return "(A)", {"raw_output": prediction}
        elif "(B)" in prediction.upper():
            return "(B)", {"raw_output": prediction}
        else:
            return "(A)", {"raw_output": prediction}  # Default fallback

async def main():
    """Quick start example for node-wise evolution."""
    
    # 🔧 Setup
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY in your environment")

    logger.info("🚀 Quick Start: Node-wise Prompt Evolution")
    logger.info("="*50)

    # Configure LLM
    llm_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=OPENAI_API_KEY,
        stream=False
    )
    llm = OpenAILLM(config=llm_config)

    # 📊 Setup benchmark (small sample for quick demo)
    benchmark = BIGBenchHard("snarks", sample_num=10)
    benchmark._load_data()

    # 🤖 Create program
    program = SimpleClassifier(model=llm)
    
    logger.info("📝 Initial prompts:")
    logger.info(f"  System: '{program.system_prompt}'")
    logger.info(f"  Task: '{program.task_prompt}'")

    # 📋 Register prompts as separate nodes
    registry = ParamRegistry()
    registry.track(program, "system_prompt", name="system_node")
    registry.track(program, "task_prompt", name="task_node")
    
    logger.info("✅ Registered 2 nodes for independent evolution")

    # 🧬 Setup optimizer
    optimizer = GAOptimizer(
        registry=registry,
        program=program,
        population_size=3,          # 3 variants per node
        iterations=2,               # 2 evolution rounds  
        llm_config=llm_config,
        concurrency_limit=20,
        combination_sample_size=5   # Evaluate 5 out of 3×3=9 possible combinations
    )
    
    logger.info("🔬 Configuration:")
    logger.info(f"  - Nodes: 2 (system_node, task_node)")
    logger.info(f"  - Population per node: 3")
    logger.info(f"  - Total combinations: 3×3 = 9")
    logger.info(f"  - Sampled combinations: 5")
    logger.info(f"  - Evolution rounds: 2")

    # 🚀 Run evolution
    logger.info("\n🧬 Starting evolution...")
    best_config, best_scores, avg_scores = await optimizer.optimize(benchmark=benchmark)

    # 📊 Show results
    logger.info("\n🎉 Evolution completed!")
    logger.info("="*50)
    
    logger.info("\n📝 Evolved prompts:")
    for node_name, evolved_prompt in best_config.items():
        logger.info(f"\n{node_name}:")
        logger.info(f"  Final: '{evolved_prompt}'")
    
    logger.info("\n📈 Performance by generation:")
    for gen_name in best_scores.keys():
        logger.info(f"\n{gen_name}:")
        for node_name in best_config.keys():
            best = best_scores[gen_name][node_name]
            avg = avg_scores[gen_name][node_name]
            logger.info(f"  {node_name}: Best={best:.3f}, Avg={avg:.3f}")

    # 🎯 Test final performance
    logger.info("\n🔬 Testing optimized prompts...")
    test_data = benchmark.get_test_data()[:5]  # Test on 5 examples
    
    correct = 0
    for example in test_data:
        prediction, _ = program(example["input"])
        label = benchmark.get_label(example)
        score_dict = benchmark.evaluate(prediction, label)
        if score_dict.get("em", 0.0) > 0:
            correct += 1
    
    accuracy = correct / len(test_data)
    logger.info(f"🎯 Final accuracy: {accuracy:.3f} ({correct}/{len(test_data)})")
    
    logger.info("\n✨ Summary:")
    logger.info("  ✓ Two prompts evolved independently")
    logger.info("  ✓ Controlled evaluation cost through sampling")
    logger.info("  ✓ No complex XML parsing required")
    logger.info("  ✓ Clear performance tracking per node")

if __name__ == "__main__":
    asyncio.run(main())
