import asyncio
import os
from typing import Dict
import re
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as aio_tqdm

from evoagentx.benchmark.bigbenchhard import BIGBenchHard
from evoagentx.models import OpenAILLM, OpenAILLMConfig

class SinglePromptSarcasmClassifier:
    """
    只用一个固定的提示词进行讽刺分类。
    """
    def __init__(self, model: OpenAILLM):
        self.model = model
        self.task_instruction = "From the two sentences provided, (A) and (B), determine which one is sarcastic. Respond with your final choice wrapped like this: FINAL_ANSWER((A))"
        self.chain_of_thought_prefix = "Break down and analyze each part of the problem in a step by step way to ensure the right answer is obtained."

    def __call__(self, input: str) -> tuple[str, dict]:
        full_prompt = f"{self.chain_of_thought_prefix}\n\n{self.task_instruction}\n\nText:\n{input}"
        response = self.model.generate(prompt=full_prompt)
        prediction = response.content.strip()
        pattern = r'FINAL_ANSWER\((\([^)]*\))\)'
        match = re.search(pattern, prediction)
        if match:
            answer = match.group(1)
        else:
            answer = "N/A"
        return answer, {"full_prompt": full_prompt}

async def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    llm_config = OpenAILLMConfig(
        model="gpt-3.5-turbo-0125",
        openai_key=OPENAI_API_KEY,
        stream=False,
        temperature=0,  # 设置温度以增加生成的多样性
    )
    llm = OpenAILLM(config=llm_config)
    benchmark = BIGBenchHard("snarks", dev_sample_num=50)
    benchmark._load_data()
    program = SinglePromptSarcasmClassifier(model=llm)
    print("\n--- Evaluating on Test Set ---")
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
        results = await aio_tqdm.gather(*tasks, desc="Evaluating on Test Set")
        correct_count = sum(results)
        test_accuracy = correct_count / len(test_data)
        print(f"Test Accuracy: {test_accuracy:.4f}")
    else:
        test_accuracy = 0.0
    return test_accuracy

if __name__ == "__main__":
    asyncio.run(main())
