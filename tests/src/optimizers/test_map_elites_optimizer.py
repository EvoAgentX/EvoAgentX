from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from evoagentx.optimizers.engine.decorators import EntryPoint
from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.optimizers.map_elites_optimizer import MapElitesOptimizer


@dataclass
class DummyProgram:
    x: int = 0
    y: int = 0

    def run(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y}


def test_map_elites_fills_multiple_cells_and_tracks_best():
    program = DummyProgram()
    registry = ParamRegistry()
    registry.track(program, "x", name="x")
    registry.track(program, "y", name="y")

    @EntryPoint()
    def entry():
        return program.run()

    def evaluator(output: Dict[str, Any]) -> Dict[str, Any]:
        x = output["x"]
        y = output["y"]
        score = float(x + y)
        return {"score": score, "complexity": float(x), "diversity": float(y)}

    opt = MapElitesOptimizer(
        registry=registry,
        evaluator=evaluator,
        search_space={"x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]},
        feature_dimensions=["complexity", "diversity"],
        feature_ranges={"complexity": (0.0, 4.0), "diversity": (0.0, 4.0)},
        feature_bins=5,
        n_iterations=80,
        exploration_ratio=0.5,
        random_seed=7,
    )

    best_cfg, result = opt.optimize()
    archive = result["archive"]
    best = result["best"]

    assert best_cfg is not None
    assert best is not None
    assert len(archive) >= 5
    assert best.fitness == 8.0
    assert best_cfg["x"] + best_cfg["y"] == 8
