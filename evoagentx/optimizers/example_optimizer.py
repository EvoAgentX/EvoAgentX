import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from .engine.base import BaseOptimizer
from .engine.decorators import EntryPoint  # ensure this is implemented or mocked
from .engine.registry import ParamRegistry

class ExampleOptimizer(BaseOptimizer):
    def __init__(self,
                 registry: ParamRegistry,
                 evaluator: Callable[[Dict[str, Any]], float],
                 search_space: Dict[str, List[Any]],
                 n_trials: int = 10):
        """
        A simple random search optimizer example.

        Parameters:
        - registry (ParamRegistry): parameter registry
        - evaluator (Callable): evaluation function
        - search_space (Dict): dictionary mapping parameter names to possible values
        - n_trials (int): number of random trials to run
        """
        super().__init__(registry, evaluator)
        self.search_space = search_space
        self.n_trials = n_trials

    def optimize(self, program_entry: Optional[Callable] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if program_entry is None:
            program_entry = EntryPoint.get_entry()
            if program_entry is None:
                raise RuntimeError("No entry function provided or registered.")

        print(f"Starting optimization using {self.n_trials} random trials...")

        best_score = float("-inf")
        best_cfg = None
        history = []

        for i in range(self.n_trials):
            # Sample configuration
            cfg = {
                name: random.choice(choices)
                for name, choices in self.search_space.items()
            }

            # Apply and run
            self.apply_cfg(cfg)
            output = program_entry()
            score = self.evaluator(output)

            trial_result = {"cfg": cfg, "score": score}
            history.append(trial_result)

            print(f"Trial {i+1}/{self.n_trials}: Score = {score:.4f}, Config = {cfg}")

            if score > best_score:
                best_score = score
                best_cfg = cfg.copy()

        return best_cfg, history
    


def simple_accuracy_evaluator(output: Dict[str, Any]) -> float:
    """
    Example evaluator function that expects the output dict to contain:
    - 'correct' (int): number of correct predictions
    - 'total' (int): total predictions made
    """
    return output["correct"] / output["total"]
