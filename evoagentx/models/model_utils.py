import threading
import pandas as pd
from dataclasses import dataclass

from ..core.logging import logger
from ..core.decorators import atomic_method


def get_openai_model_cost() -> dict:
    import json 
    import importlib.resources
    with importlib.resources.open_text('litellm', 'model_prices_and_context_window_backup.json') as f:
        model_cost = json.load(f)
    return model_cost


@dataclass
class Cost:
    input_tokens: int 
    output_tokens: int
    input_cost: float 
    output_cost: float


class CostManager:

    def __init__(self):

        self.total_input_tokens = {}
        self.total_output_tokens = {} 
        self.total_tokens = {} 

        self.total_input_cost = {}
        self.total_output_cost = {}
        self.total_cost = {}

        self._lock = threading.Lock()

    def compute_total_cost(self):
        total_tokens, total_cost = 0, 0.0
        for _, value in self.total_tokens.items():
            total_tokens += value
        for _, value in self.total_cost.items():
            total_cost += value
        return total_tokens, total_cost

    @atomic_method
    def update_cost(self, cost: Cost, model: str):

        self.total_input_tokens[model] = self.total_input_tokens.get(model, 0) + cost.input_tokens
        self.total_output_tokens[model] = self.total_output_tokens.get(model, 0) + cost.output_tokens
        current_total_tokens = cost.input_tokens + cost.output_tokens
        self.total_tokens[model] = self.total_tokens.get(model, 0) + current_total_tokens

        self.total_input_cost[model] = self.total_input_cost.get(model, 0.0) + cost.input_cost
        self.total_output_cost[model] = self.total_output_cost.get(model, 0.0) + cost.output_cost
        current_total_cost = cost.input_cost + cost.output_cost
        self.total_cost[model] = self.total_cost.get(model, 0.0) + current_total_cost
        
        total_tokens, total_cost = self.compute_total_cost()
        logger.info(f"Total cost: ${total_cost:.3f} | Total tokens: {total_tokens} | Current cost: ${current_total_cost:.3f} | Current tokens: {current_total_tokens}")

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
            data["Total Cost (USD)"].append(round(self.total_cost[model], 4))
            data["Total Input Cost (USD)"].append(round(self.total_input_cost[model], 4))
            data["Total Output Cost (USD)"].append(round(self.total_output_cost[model], 4))

            data["Total Tokens"].append(self.total_tokens[model])
            data["Total Input Tokens"].append(self.total_input_tokens[model])
            data["Total Output Tokens"].append(self.total_output_tokens[model])
        
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


cost_manager = CostManager()