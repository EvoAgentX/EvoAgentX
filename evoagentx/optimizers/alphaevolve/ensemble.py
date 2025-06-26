# Acknowledgement: Modified from OpenEvolve (https://github.com/codelion/openevolve/blob/main/openevolve/llm/ensemble.py) under Apache-2.0 license

import asyncio
import random
from typing import Dict, List, Optional

from evoagentx.core.logging import logger
from evoagentx.models.base_model import BaseLLM


class LLMEnsemble:
    """Ensemble of LLMs

    Args:
        llms: List of LLMs to be ensembled
        weights: List of weights for each LLM. LLMs will be sampled at random based on the weights.
    """

    def __init__(self, llms: List[BaseLLM], weights: Optional[List[float]] = None):

        if weights is not None:
            assert len(llms) == len(weights), "Number of LLMs and weights must match"
            assert sum(weights) == 1.0, "Sum of weights must be 1"
        else:
            weights = [1.0 / len(llms)] * len(llms)
        
        self.llms = llms
        self.weights = weights

        logger.info(
            "Initialized LLM ensemble with models: "
            + ", ".join(
                f"{llm.config.model} (weight: {weight:.2f})"
                for llm, weight in zip(self.llms, self.weights, strict=True)
            )
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        return await model.single_generate_async(messages=[{"role": "user", "content": prompt}], **kwargs)

    async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        return await model.single_generate_async(messages=messages, system_message=system_message, **kwargs)

    def _sample_model(self) -> BaseLLM:
        """Sample a model from the ensemble based on weights"""
        index = random.choices(range(len(self.llms)), weights=self.weights, k=1)[0]
        return self.llms[index]

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        """Generate text using a all available models"""
        responses = []
        for model in self.llms:
            response = await model.single_generate_async(system_message=system_message, messages=messages, **kwargs)
            responses.append(response)
        return responses
