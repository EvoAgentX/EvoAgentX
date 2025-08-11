import asyncio
import os
from typing import Dict, List
import re
from datetime import datetime
from tqdm.asyncio import tqdm as aio_tqdm
from dotenv import load_dotenv
from collections import Counter

from evoagentx.optimizers.evoprompt_optimizer import DEOptimizer, GAOptimizer
from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.core.logging import logger

class SarcasmClassifierProgram:
    """
    A program that uses a three-prompt majority vote ensemble to classify sarcasm.
    This version demonstrates the mega-prompt evolution strategy where individual 
    node prompts are concatenated and evolved as a single entity.
    """
    def __init__(self, model: OpenAILLM):
        self.model = model
        # Three different prompts acting as three independent voters.
        # These will be concatenated into a mega-prompt for evolution.
        self.prompt_direct = "From the two sentences provided, (A) and (B), determine which one is sarcastic. Respond with your final choice wrapped like this: FINAL_ANSWER((A))"
        self.prompt_expert = "You are an expert in linguistics and humor. Analyze the following two sentences, (A) and (B), and identify the sarcastic one. Your answer must be wrapped like this: FINAL_ANSWER((B))"
        self.prompt_cot = "Consider the context and potential double meanings in sentences (A) and (B). Which one uses sarcasm? Think step-by-step and conclude with your final choice, wrapped like this: FINAL_ANSWER((A))"

    def __call__(self, input: str) -> tuple[str, dict]:
        # --- Step 1: Get answers from all three prompts ---
        answers = []
        prompts = [self.prompt_direct, self.prompt_expert, self.prompt_cot]
        
        # This pattern now specifically looks for "FINAL_ANSWER((A))" or "FINAL_ANSWER((B))"
        # and captures only the inner parenthesized part.
        pattern = r'FINAL_ANSWER\((\([^)]*\))\)'

        for prompt in prompts:
            full_prompt = f"{prompt}\n\nText:\n{input}"
            # We assume the model call is synchronous for this example, 
            # but in a real high-performance scenario, these could be run concurrently.
            response = self.model.generate(prompt=full_prompt)
            prediction = response.content.strip()
            
            match = re.search(pattern, prediction)
            if match:
                # Group 1 captures the part inside the first parentheses of the pattern,
                # which is (\([^)]*\)), correctly extracting "(A)" or "(B)".
                answers.append(match.group(1))

        # --- Step 2: Majority Voting ---
        if not answers:
            return "N/A", {"votes": []}

        # Use Counter to find the most common answer
        vote_counts = Counter(answers)
        # most_common(1) returns a list of tuples, e.g., [('(A)', 2)]
        most_common_answer = vote_counts.most_common(1)[0][0]
        
        return most_common_answer, {"votes": answers}

    def save(self, path: str):
        logger.info(f"DUMMY: Saving program state to {path}")
        pass

    def load(self, path: str):
        logger.info(f"DUMMY: Loading program state from {path}")
        pass

async def run_mega_prompt_evolution_example(algorithm="DE"):
    """
    Run the mega-prompt evolution example.
    
    Args:
        algorithm: "DE" for Differential Evolution or "GA" for Genetic Algorithm
    """
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Configuration
    POPULATION_SIZE = 8  # Mega-prompt population size (same as node population size for fairness)
    ITERATIONS = 4
    CONCURRENCY_LIMIT = 50

    logger.info(f"🚀 Starting {algorithm} Mega-Prompt Evolution Example")
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Mega-prompt population size: {POPULATION_SIZE}")
    logger.info(f"   - Iterations: {ITERATIONS}")

    # Setup LLM
    llm_config = OpenAILLMConfig(
        model="gpt-4.1-nano",
        openai_key=OPENAI_API_KEY,
        stream=False
    )
    llm = OpenAILLM(config=llm_config)

    # Setup benchmark - using dev set for evolution
    benchmark = BIGBenchHard("snarks", dev_sample_num=25)  # 15 for dev, 5 for test
    benchmark._load_data()

    # Create program
    program = SarcasmClassifierProgram(model=llm)

    # Register prompts for mega-prompt evolution
    registry = ParamRegistry()
    registry.track(program, "prompt_direct", name="direct_prompt_node")
    registry.track(program, "prompt_expert", name="expert_prompt_node")  
    registry.track(program, "prompt_cot", name="cot_prompt_node")

    logger.info("📝 Registered 3 prompts for mega-prompt evolution:")
    logger.info(f"   1. direct_prompt_node: '{program.prompt_direct[:50]}...'")
    logger.info(f"   2. expert_prompt_node: '{program.prompt_expert[:50]}...'")
    logger.info(f"   3. cot_prompt_node: '{program.prompt_cot[:50]}...'")

    # Create initial mega-prompt
    from evoagentx.optimizers.evoprompt_optimizer import _build_mega_prompt_from_config
    # Build initial config manually from registry
    initial_config = {name: registry.get(name) for name in registry.names()}
    initial_mega_prompt = _build_mega_prompt_from_config(initial_config)
    
    logger.info("🔗 Initial mega-prompt structure:")
    logger.info(f"{initial_mega_prompt[:300]}...")

    # Choose optimizer
    if algorithm == "DE":
        optimizer = DEOptimizer(
            registry=registry,
            program=program,
            population_size=POPULATION_SIZE,
            iterations=ITERATIONS,
            llm_config=llm_config,
            concurrency_limit=CONCURRENCY_LIMIT,
            enable_logging=True,
            log_dir=f"mega_prompt_logs_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    elif algorithm == "GA":
        optimizer = GAOptimizer(
            registry=registry,
            program=program,
            population_size=POPULATION_SIZE,
            iterations=ITERATIONS,
            llm_config=llm_config,
            concurrency_limit=CONCURRENCY_LIMIT,
            enable_logging=True,
            log_dir=f"mega_prompt_logs_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    logger.info(f"🧬 Starting {algorithm} mega-prompt optimization...")

    # Run optimization
    best_mega_prompt, best_scores_per_gen, avg_scores_per_gen = await optimizer.optimize(benchmark=benchmark)

    logger.info("\n🎉 Mega-Prompt Evolution Complete!")

    # Display results
    logger.info("\n📊 Final Results:")
    logger.info("="*60)
    
    # Show initial vs final mega-prompt
    logger.info(f"\n--- MEGA-PROMPT EVOLUTION ---")
    logger.info(f"Initial: '{initial_mega_prompt[:200]}...'")
    logger.info(f"Final  : '{best_mega_prompt[:200]}...'")

    # Show final individual prompts
    from evoagentx.optimizers.evoprompt_optimizer import _split_mega_prompt_to_config
    final_config = _split_mega_prompt_to_config(best_mega_prompt)
    
    logger.info(f"\n--- FINAL INDIVIDUAL PROMPTS ---")
    for node_name, prompt in final_config.items():
        logger.info(f"{node_name}: '{prompt[:100]}...'")

    # Show performance evolution
    logger.info("\n📈 Performance Evolution:")
    for gen_name, score in best_scores_per_gen.items():
        avg_score = avg_scores_per_gen[gen_name]
        logger.info(f"{gen_name}: Best={score:.4f}, Avg={avg_score:.4f}")

    # Final test evaluation
    logger.info("\n🔬 Evaluating optimized program on test set...")
    test_data = benchmark.get_test_data()

    async def evaluate_example_concurrently(example: Dict) -> float:
        prediction, _ = await asyncio.to_thread(
            program,
            input=example["input"]
        )
        score_dict = benchmark.evaluate(prediction, benchmark.get_label(example))
        return score_dict.get("em", 0.0)

    if test_data:
        tasks = [evaluate_example_concurrently(ex) for ex in test_data]
        results = await aio_tqdm.gather(*tasks, desc="Final Test Evaluation")
        correct_count = sum(results)
        test_accuracy = correct_count / len(test_data)
    else:
        test_accuracy = 0.0

    logger.info(f"\n🎯 Final test set accuracy: {test_accuracy:.4f}")
    
    return best_mega_prompt, test_accuracy

async def main():
    """
    Main function demonstrating both GA and DE mega-prompt evolution.
    """
    logger.info("🌟 EvoAgentX Mega-Prompt Evolution Demonstration")
    logger.info("="*60)
    
    # Run DE example
    logger.info("\n1️⃣ Running Differential Evolution (DE) Example...")
    de_mega_prompt, de_accuracy = await run_mega_prompt_evolution_example(algorithm="DE")
    
    logger.info("\n" + "="*60)
    
    # Run GA example  
    logger.info("\n2️⃣ Running Genetic Algorithm (GA) Example...")
    ga_mega_prompt, ga_accuracy = await run_mega_prompt_evolution_example(algorithm="GA")
    
    # Final comparison
    logger.info("\n" + "="*60)
    logger.info("🏆 FINAL COMPARISON")
    logger.info("="*60)
    logger.info(f"DE Test Accuracy: {de_accuracy:.4f}")
    logger.info(f"GA Test Accuracy: {ga_accuracy:.4f}")
    logger.info(f"Winner: {'DE' if de_accuracy > ga_accuracy else 'GA' if ga_accuracy > de_accuracy else 'Tie'}")
    
    logger.info("\n✨ Key Features of Mega-Prompt Evolution:")
    logger.info("  1. 🔗 Concatenates individual node prompts into mega-prompt")
    logger.info("  2. 🧬 Evolves entire mega-prompt as single entity")
    logger.info("  3. 🔄 Splits evolved mega-prompt back to individual nodes")
    logger.info("  4. 📊 Simple population-based fitness evaluation")
    logger.info("  5. ⚡ Original EvoPrompt strategy with enhanced XML handling")

if __name__ == "__main__":
    asyncio.run(main())
