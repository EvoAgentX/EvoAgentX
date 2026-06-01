import asyncio
import threading

import pytest

from evoagentx.optimizers.engine.base import BaseOptimizer
from evoagentx.optimizers.evoprompt_optimizer import EvopromptOptimizer


class FakeBenchmark:
    def get_input_keys(self):
        return ["case"]

    def get_label(self, example):
        return example["target"]

    def evaluate(self, prediction, label):
        return {"em": float(prediction == label)}


class RaisingBenchmark(FakeBenchmark):
    def evaluate(self, prediction, label):
        raise RuntimeError("evaluation failed")


class SimpleProgram:
    def __init__(self):
        self.prompt = "base"

    def __call__(self, case):
        return self.prompt, {"case": case}


class PromptRegistry:
    def __init__(self, program):
        self.program = program
        self.fields = {"prompt": object()}

    def get(self, name):
        assert name == "prompt"
        return self.program.prompt

    def set(self, name, value):
        assert name == "prompt"
        self.program.prompt = value

    def names(self):
        return ["prompt"]


class CoordinatedExampleProgram:
    def __init__(self):
        self._prompt = "base"
        self.slow_started = threading.Event()
        self.fast_returned = threading.Event()
        self.base_restored_after_fast = threading.Event()
        self.observed = {}

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        self._prompt = value
        if value == "base" and self.fast_returned.is_set():
            self.base_restored_after_fast.set()

    def __call__(self, case):
        if case == "fast":
            self.slow_started.wait(timeout=1.0)
            observed = self.prompt
            self.observed[case] = observed
            self.fast_returned.set()
            return observed, {"case": case}

        if case == "slow":
            self.slow_started.set()
            self.fast_returned.wait(timeout=1.0)
            self.base_restored_after_fast.wait(timeout=0.05)
            observed = self.prompt
            self.observed[case] = observed
            return observed, {"case": case}

        observed = self.prompt
        self.observed[case] = observed
        return observed, {"case": case}


class CrossCombinationProgram:
    def __init__(self):
        self.prompt = "base"
        self.combo_a_entered = threading.Event()
        self.combo_b_entered = threading.Event()
        self.observed = []

    def __call__(self, case):
        initial_prompt = self.prompt
        if initial_prompt == "combo-a":
            self.combo_a_entered.set()
            self.combo_b_entered.wait(timeout=0.05)
        elif initial_prompt == "combo-b":
            self.combo_b_entered.set()
            self.combo_a_entered.wait(timeout=0.05)

        observed = self.prompt
        self.observed.append((case, initial_prompt, observed))
        return observed, {"case": case}


def make_optimizer(program, concurrency_limit=2):
    registry = PromptRegistry(program)

    class TestOptimizer(EvopromptOptimizer):
        def __init__(self):
            BaseOptimizer.__init__(self, registry=registry, program=program)
            self.semaphore = asyncio.Semaphore(concurrency_limit)
            self._program_config_lock = asyncio.Lock()
            self._eval_cache = {}

        async def optimize(self):
            return None

    return TestOptimizer()


@pytest.mark.asyncio
async def test_combination_config_is_kept_until_all_examples_finish():
    program = CoordinatedExampleProgram()
    optimizer = make_optimizer(program, concurrency_limit=2)
    benchmark = FakeBenchmark()

    scores = await optimizer._evaluate_combination_on_examples(
        {"prompt": "optimized"},
        benchmark,
        [
            {"case": "fast", "target": "optimized"},
            {"case": "slow", "target": "optimized"},
        ],
    )

    assert scores == [1.0, 1.0]
    assert program.observed == {"fast": "optimized", "slow": "optimized"}
    assert program.prompt == "base"


@pytest.mark.asyncio
async def test_combination_config_is_restored_after_evaluation_error():
    program = SimpleProgram()
    optimizer = make_optimizer(program)

    scores = await optimizer._evaluate_combination_on_examples(
        {"prompt": "optimized"},
        RaisingBenchmark(),
        [{"case": "single", "target": "optimized"}],
    )

    assert scores == [0.0]
    assert program.prompt == "base"


@pytest.mark.asyncio
async def test_concurrent_combinations_do_not_share_temporary_config():
    program = CrossCombinationProgram()
    optimizer = make_optimizer(program, concurrency_limit=2)
    benchmark = FakeBenchmark()

    scores_a, scores_b = await asyncio.gather(
        optimizer._evaluate_combination_list(
            [{"prompt": "combo-a"}],
            benchmark,
            [{"case": "a", "target": "combo-a"}],
        ),
        optimizer._evaluate_combination_list(
            [{"prompt": "combo-b"}],
            benchmark,
            [{"case": "b", "target": "combo-b"}],
        ),
    )

    assert scores_a == [1.0]
    assert scores_b == [1.0]
    assert ("a", "combo-a", "combo-a") in program.observed
    assert ("b", "combo-b", "combo-b") in program.observed
    assert program.prompt == "base"
