import threading
from collections import defaultdict
from typing import Optional

import pandas as pd

from ..core.logging import logger
from ..core.decorators import atomic_method
from ..core.callbacks import suppress_cost_logs
from ..core.registry import MODEL_REGISTRY
from .model_configs import LLMConfig
from ..models.base_model import BaseLLM

def get_openai_model_cost() -> dict:
    import json 
    from importlib.resources import files
    # import importlib.resources
    # with importlib.resources.open_text('litellm', 'model_prices_and_context_window_backup.json') as f:
    #     model_cost = json.load(f)
    json_path = files('litellm') / 'model_prices_and_context_window_backup.json' 
    model_cost = json.loads(json_path.read_text(encoding="utf-8"))
    return model_cost

def infer_litellm_company_from_model(model: str) -> str:

    if "/" in model:
        company = model.split("/")[0]
    else:
        if "claude" in model or "anthropic" in model:
            company = "anthropic" 
        elif "gemini" in model:
            company = "gemini"
        elif "deepseek" in model:
            company = "deepseek"
        elif "openrouter" in model:
            company = "openrouter"
        elif "azure" in model.lower():
            company = "azure"
        else:
            company = "openai"
    return company


class Cost:

    def __init__(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        input_cost: Optional[float] = None,
        output_cost: Optional[float] = None,
        cost: Optional[float] = None
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        # Keep input_cost/output_cost only as temporary constructor compatibility.
        self.cost = cost if cost is not None else (input_cost or 0.0) + (output_cost or 0.0)

    @property
    def cost(self) -> float:
        return self._cost

    @cost.setter
    def cost(self, value: float):
        self._cost = value or 0.0


class CostManager:

    def __init__(self):

        self.input_tokens = defaultdict(int)
        self.output_tokens = defaultdict(int)
        self.total_tokens = defaultdict(int)

        self.cost_per_model = defaultdict(float)

        self._lock = threading.Lock()

    @property
    def total_llm_cost(self) -> float:
        return sum(self.cost_per_model.values())

    @property
    def total_llm_tokens(self) -> int:
        return sum(self.total_tokens.values())

    @property
    def total_cost(self) -> float:
        return self.total_llm_cost

    def add_llm_cost(self, cost: Cost, model: str):
        self.input_tokens[model] += (cost.input_tokens or 0)
        self.output_tokens[model] += (cost.output_tokens or 0)
        self.total_tokens[model] += (cost.input_tokens or 0) + (cost.output_tokens or 0)

        self.cost_per_model[model] += cost.cost

    @atomic_method
    def update_cost(self, cost: Cost, model: str):
        self.add_llm_cost(cost, model)

        total_tokens = self.total_llm_tokens
        total_llm_cost = self.total_llm_cost
        current_llm_cost = cost.cost
        current_total_tokens = (cost.input_tokens or 0) + (cost.output_tokens or 0)

        if not suppress_cost_logs.get():
            logger.info(f"Total LLM cost: ${total_llm_cost:.3f} | Total tokens: {total_tokens} | Current LLM cost: ${current_llm_cost:.3f} | Current tokens: {current_total_tokens}")

    def display_cost(self):

        data = {
            "Model": [],
            "Total Cost (USD)": [],
            "Total Tokens": [],
            "Total Input Tokens": [],
            "Total Output Tokens": [],
        }

        for model in self.total_tokens.keys():

            data["Model"].append(model)
            data["Total Cost (USD)"].append(round(self.cost_per_model[model], 4))

            data["Total Tokens"].append(self.total_tokens[model])
            data["Total Input Tokens"].append(self.input_tokens[model])
            data["Total Output Tokens"].append(self.output_tokens[model])

        df = pd.DataFrame(data)
        if len(df) > 1:
            summary = {
                "Model": "TOTAL",
                "Total Cost (USD)": df["Total Cost (USD)"].sum(),
                "Total Tokens": df["Total Tokens"].sum(),
                "Total Input Tokens": df["Total Input Tokens"].sum(),
                "Total Output Tokens": df["Total Output Tokens"].sum(),
            }
            df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

        print(df.to_string(index=False))

    def get_total_cost(self) -> float:
        return self.total_llm_cost


cost_manager = CostManager()


def create_llm_instance(llm_config: LLMConfig) -> BaseLLM:

    llm_cls = MODEL_REGISTRY.get_model(llm_config.llm_type)
    llm = llm_cls(config=llm_config)
    return llm 
