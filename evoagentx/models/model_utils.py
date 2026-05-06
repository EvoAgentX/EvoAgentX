import threading
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Optional

import pandas as pd

from ..core.callbacks import suppress_cost_logs
from ..core.decorators import atomic_method
from ..core.logging import logger
from ..core.registry import MODEL_REGISTRY
from ..models.base_model import BaseLLM
from ..utils.utils import (
    add_dict,
    get_cost_per_tool,
    get_provider_tool_cost,
    get_total_tool_cost,
)
from .model_configs import LLMConfig

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


cost_tracker = ContextVar("cost_tracker", default=None)


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
        self.input_cost = input_cost
        self.output_cost = output_cost

        if cost is None:
            self.cost = (self.input_cost or 0.) + (self.output_cost or 0.)
        else:
            self.cost = cost

    @property
    def cost(self) -> float:
        return self._cost

    @cost.setter
    def cost(self, value: float):
        self._validate_cost(value)
        self._cost = value

    def _validate_cost(self, value: float):
        if self.input_cost is not None or self.output_cost is not None:
            total_cost = (self.input_cost or 0.) + (self.output_cost or 0.)
            if total_cost != value:
                raise ValueError(f"Cost mismatch: provided cost {value} does not match the sum of input and output cost {total_cost}")


class CostManager:

    def __init__(self):

        self.input_tokens = defaultdict(int)
        self.output_tokens = defaultdict(int) 
        self.total_tokens = defaultdict(int) 

        self.input_cost = defaultdict(float)
        self.output_cost = defaultdict(float)
        self.cost_per_model = defaultdict(float)

        self.total_input_tokens = self.input_tokens
        self.total_output_tokens = self.output_tokens
        self.total_input_cost = self.input_cost
        self.total_output_cost = self.output_cost
        self.total_cost = self.cost_per_model

        self.tool_cost_breakdown = defaultdict(dict)
        self._lock = threading.Lock()

    @property
    def total_llm_cost(self) -> float:
        return sum(self.cost_per_model.values())

    @property
    def total_llm_tokens(self) -> int:
        return sum(self.total_tokens.values())

    @property
    def cost_per_tool(self) -> Dict[str, float]:
        return get_cost_per_tool(self.tool_cost_breakdown)

    @property
    def total_tool_cost(self) -> float:
        return get_total_tool_cost(self.tool_cost_breakdown)

    @property
    def openrouter_tool_cost(self) -> float:
        return get_provider_tool_cost(self.tool_cost_breakdown, "openrouter")

    @property
    def non_openrouter_tool_cost(self) -> float:
        non_openrouter_cost = self.total_tool_cost - self.openrouter_tool_cost
        return non_openrouter_cost

    def compute_total_cost(self):
        return self.total_llm_tokens, self.total_llm_cost

    @atomic_method
    def update_cost(self, cost: Cost, model: str):
        self.add_llm_cost(cost, model)

        tracker = cost_tracker.get()
        if tracker is not None and tracker is not self:
            tracker.add_llm_cost(cost, model)
        
        total_tokens = self.total_llm_tokens
        total_cost = self.total_llm_cost
        current_total_cost = cost.cost
        current_total_tokens = (cost.input_tokens or 0) + (cost.output_tokens or 0)

        if not suppress_cost_logs.get():
            logger.info(f"Total cost: ${total_cost:.3f} | Total tokens: {total_tokens} | Current cost: ${current_total_cost:.3f} | Current tokens: {current_total_tokens}")

    def add_llm_cost(self, cost: Cost, model: str):
        self.input_tokens[model] += (cost.input_tokens or 0)
        self.output_tokens[model] += (cost.output_tokens or 0)
        current_total_tokens = (cost.input_tokens or 0) + (cost.output_tokens or 0)
        self.total_tokens[model] += current_total_tokens

        self.input_cost[model] += (cost.input_cost or 0.0)
        self.output_cost[model] += (cost.output_cost or 0.0)
        self.cost_per_model[model] += cost.cost

    @atomic_method
    def update_tool_cost(self, tool_name: str, cost_breakdown: Dict[str, float]):
        self.add_tool_cost(tool_name, cost_breakdown)

        tracker = cost_tracker.get()
        if tracker is not None and tracker is not self:
            tracker.add_tool_cost(tool_name, cost_breakdown)
        
        cost = sum(cost_breakdown.values())
        if not suppress_cost_logs.get():
            logger.info(f"Total tool cost: ${self.total_tool_cost:.3f} | Current tool cost: ${cost:.3f} | Tool name: {tool_name}")

    def add_tool_cost(self, tool_name: str, cost_breakdown: Dict[str, float]):
        self.tool_cost_breakdown[tool_name] = add_dict(
            self.tool_cost_breakdown[tool_name],
            cost_breakdown
        )

    def display_cost(self):

        data = {
            "Model": [],
            "Total Cost (USD)": [], 
            "Total Input Cost (USD)": [], 
            "Total Output Cost (USD)": [], 
            "Total Tokens": [], 
            "Total Input Tokens": [], 
            "Total Output Tokens": [],
        }

        for model in self.total_tokens.keys():

            data["Model"].append(model)
            data["Total Cost (USD)"].append(round(self.cost_per_model[model], 4))
            data["Total Input Cost (USD)"].append(round(self.input_cost[model], 4))
            data["Total Output Cost (USD)"].append(round(self.output_cost[model], 4))

            data["Total Tokens"].append(self.total_tokens[model])
            data["Total Input Tokens"].append(self.input_tokens[model])
            data["Total Output Tokens"].append(self.output_tokens[model])
        
        # Convert to DataFrame for display
        df = pd.DataFrame(data)
        if len(df) > 1:
            summary = {
                "Model": "TOTAL",
                "Total Cost (USD)": df["Total Cost (USD)"].sum(),
                "Total Input Cost (USD)": df["Total Input Cost (USD)"].sum(),
                "Total Output Cost (USD)": df["Total Output Cost (USD)"].sum(),
                "Total Tokens": df["Total Tokens"].sum(),
                "Total Input Tokens": df["Total Input Tokens"].sum(),
                "Total Output Tokens": df["Total Output Tokens"].sum(),
            }
            df = df._append(summary, ignore_index=True)
        
        print(df.to_string(index=False))

    def get_total_cost(self):
        return self.total_llm_cost


cost_manager = CostManager()


def create_llm_instance(llm_config: LLMConfig) -> BaseLLM:

    llm_cls = MODEL_REGISTRY.get_model(llm_config.llm_type)
    llm = llm_cls(config=llm_config)
    return llm 


@contextmanager
def track_cost():
    tracker = CostManager()
    token = cost_tracker.set(tracker)

    try:
        yield tracker
    finally:
        cost_tracker.reset(token)
