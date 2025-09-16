"""
Best Prompt Evaluation Script

This script evaluates baseline performance using simple prompts without optimization.
It serves as a baseline comparison for evolutionary prompt optimization results.
"""

import asyncio
import os
import re
import csv
from typing import Dict
from datetime import datetime

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as aio_tqdm

from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig


class SinglePromptSarcasmClassifier:
    """
    A simple classifier using a single fixed prompt for task processing.
    
    This serves as a baseline for comparison with evolved prompts.
    """
    
    def __init__(self, model: OpenAILLM):
        """
        Initialize the baseline classifier.
        
        Args:
            model: The language model to use for inference
        """
        self.model = model
        self.task_instruction = "After your reasoning, respond the answer only with option like this: the answer is (A)"
        self.chain_of_thought_prefix = "Let's think step by step."

    def __call__(self, input: str) -> tuple[str, dict]:
        """
        Process input with the fixed prompt.
        
        Args:
            input: The input text to process
            
        Returns:
            Tuple of (answer, metadata)
        """
        full_prompt = f"Question:{input}{self.task_instruction}"
        response = self.model.generate(prompt=full_prompt)
        prediction = response.content.strip()
        
        # Extract answer using regex pattern
        pattern = r"the answer is\s*(.*)"
        match = re.search(pattern, prediction, re.IGNORECASE)
        answer = match.group(1).strip().rstrip('.') if match else "N/A"
        
        return answer, {"full_prompt": full_prompt}


async def main():
    """Main execution function for baseline evaluation."""
    
    # Load environment configuration
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    # Models to evaluate
    model_list = [
        "gpt-4.1-nano-2025-04-14",
    ]

    # Evaluate each model
    for model_name in model_list:
        # Configure language model
        llm_config = OpenAILLMConfig(
            model=model_name,
            openai_key=OPENAI_API_KEY,
            stream=False,
        )
        llm = OpenAILLM(config=llm_config)
        
        # Set up benchmark and classifier
        benchmark = BIGBenchHard("geometric_shapes", dev_sample_num=50, seed=10)
        program = SinglePromptSarcasmClassifier(model=llm)
        
        print(f"\n--- Evaluating on Test Set with model {model_name} ---")
        test_data = benchmark.get_test_data()
        results_list = []
        task_name = benchmark.task

        async def evaluate_example_concurrently(example: Dict) -> float:
            """
            Evaluate a single example asynchronously.
            
            Args:
                example: The example to evaluate
                
            Returns:
                The evaluation score (0.0 or 1.0)
            """
            prediction, meta = await asyncio.to_thread(
                program,
                input=example["input"]
            )
            score_dict = benchmark.evaluate(prediction, benchmark.get_label(example))
            
            # Save detailed results for each sample
            results_list.append({
                "input": example["input"],
                "label": benchmark.get_label(example),
                "prediction": prediction,
                "em": score_dict.get("em", 0.0),
                "prompt": meta["full_prompt"],
                "model": model_name,
                "task": task_name
            })
            return score_dict.get("em", 0.0)

        # Run evaluation on test set
        if test_data:
            tasks = [evaluate_example_concurrently(ex) for ex in test_data]
            results = await aio_tqdm.gather(*tasks, desc="Evaluating on Test Set")
            correct_count = sum(results)
            test_accuracy = correct_count / len(test_data)
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            # Save results to CSV with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_name = f"results_{model_name}_{task_name}_{timestamp}.csv"
            csv_path = os.path.join(os.path.dirname(__file__), csv_name)
            
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "input", "label", "prediction", "em", "prompt", "model", "task"
                ])
                
                # Write average score at the top
                f.write(f"平均分数,{test_accuracy:.4f}\n")
                writer.writeheader()
                writer.writerows(results_list)
                
            print(f"详细结果已保存到: {csv_path}")
        else:
            test_accuracy = 0.0
            
    return test_accuracy


if __name__ == "__main__":
    asyncio.run(main())