"""
SEW prompt optimization on HumanEval (new optimizer engine).

This example optimizes the prompts of the built-in SEW coding workflow with
:class:`~evoagentx.optimizers.sew_optimizer.SEWOptimizer`. SEW is a PromptBreeder-style,
mutation-only ``(1 + λ)`` loop: each round it seeds from the current best snapshot and
re-mutates *every* target prompt at once, evaluates the variants, and keeps the best.

Pieces involved:
  * ``SEWWorkFlowGraph``   - a two-agent (task_parsing -> code_generation) coding workflow
                             whose agent prompts are the things we optimize.
  * ``SEWWorkFlowAdapter``  - wraps the graph so the engine can snapshot / mutate / run it.
                             It auto-derives one PROMPT unit per agent and runs the graph
                             end-to-end in ``execute`` / ``async_execute``.
  * ``SEWOptimizer``        - mutates the prompts each round (uses an OpenRouter model).
  * ``evaluate_fn``         - runs the (mutated) workflow over a dev split of HumanEval and
                             returns ``{"pass@1": <mean>}`` for the objective to rank.

Environment variables:
  * ``OPENAI_API_KEY``      - drives the workflow itself (the program being optimized).
  * ``OPENROUTER_API_KEY``  - drives the SEW mutation model that rewrites the prompts.
"""

import asyncio
import os

import numpy as np
from dotenv import load_dotenv

from evoagentx.benchmark import HumanEval
from evoagentx.core.logging import logger
from evoagentx.models import OpenRouterConfig, OpenRouterLLM
from evoagentx.optimizers.engine.objective import ScalarObjective
from evoagentx.optimizers.sew_optimizer import SEWOptimizer, SEWWorkFlowAdapter
from evoagentx.prompts import StringTemplate
from evoagentx.workflow import SequentialWorkFlowGraph

# ---------------------------------------------------------------------------
# Tunables (kept small so the example is cheap to run end-to-end).
# ---------------------------------------------------------------------------
DEV_SAMPLE_NUM = 30        # examples used to score each prompt variant during optimization
TEST_SAMPLE_NUM = 50       # examples used for the before/after report
MAX_TRIALS = 10            # total prompt variants the optimizer is allowed to evaluate
NUM_VARIANTS_PER_STEP = 2  # variants generated per round (the "λ" in (1 + λ))
EVAL_CONCURRENCY = 10      # how many dev examples to run concurrently per evaluation
EXECUTION_MODEL = "openai/gpt-4o-mini" # OpenRouter model used to run the workflow (the program being optimized)
OPTIMIZER_MODEL = "anthropic/claude-sonnet-4.6"  # OpenRouter model id used to mutate the prompts
SAVE_DIR = "debug/sew_humaneval_final" # where to save the intermediate variants and final best state (SEWOptimizer.async_optimize's save_dir)


# A single-node coding workflow defined as a SequentialWorkFlowGraph. Its one prompt is kept
# deliberately bare so the gain from SEW's prompt optimization is easy to see. That prompt
# template's ``instruction`` is the single optimizable PROMPT unit.
coding_graph_data = {
    "goal": "Generate functional and correct Python code that completes the given problem.",
    "tasks": [
        {
            "name": "code_generation",
            "description": "Generate the Python code that solves the given coding problem.",
            "inputs": [
                {"name": "question", "type": "str", "required": True, "description": "The coding problem (function signature and docstring)."}
            ],
            "outputs": [
                {"name": "code", "type": "str", "required": True, "description": "The generated Python code."}
            ],
            "prompt_template": StringTemplate(instruction="Write code for the problem."),
            "parse_mode": "str",
        },
    ],
}


class HumanEvalSplits(HumanEval):
    """HumanEval with a deterministic dev/test split (HumanEval ships only a test set)."""

    def _load_data(self):
        super()._load_data()
        np.random.seed(42)
        num_dev_samples = int(len(self._test_data) * 0.2)
        permutation = np.random.permutation(len(self._test_data))
        self._dev_data = [self._test_data[i] for i in permutation[:num_dev_samples]]
        self._test_data = [self._test_data[i] for i in permutation[num_dev_samples:]]


async def _score_one(adapter: SEWWorkFlowAdapter, benchmark: HumanEval, example: dict) -> float:
    """Run the workflow on one example and return its pass@1 (1.0 if it passes, else 0.0)."""
    try:
        code = await adapter.async_execute(inputs={"question": example["prompt"]})
        metrics = benchmark.evaluate(prediction=code, label=benchmark._get_label(example))
        return float(metrics.get("pass@1", 0.0))
    except Exception as exc:  # a single bad sample shouldn't abort the whole evaluation
        logger.warning(f"Evaluation failed for {example['task_id']}: {exc}")
        return 0.0


def make_evaluate_fn(benchmark: HumanEval, dataset: list, concurrency: int):
    """Build the async ``evaluate_fn`` the optimizer calls on every (baseline + trial) adapter."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(adapter, example):
        async with semaphore:
            return await _score_one(adapter, benchmark, example)

    async def evaluate_fn(adapter: SEWWorkFlowAdapter) -> dict:
        # No suppress_logger_info() needed: SEWWorkFlowAdapter.async_execute suppresses
        # the workflow's per-step logs itself (contextvar-scoped, so it's concurrency-safe).
        scores = await asyncio.gather(*[_bounded(adapter, ex) for ex in dataset])
        return {"pass@1": float(np.mean(scores)) if scores else 0.0}

    return evaluate_fn


async def main():
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY not found (drives the SEW prompt-mutation model).")

    # 1) The program to optimize: the built-in SEW coding workflow.
    llm_config = OpenRouterConfig(
        model=EXECUTION_MODEL,
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        top_p=0.85,
        temperature=0.2,
    )
    llm = OpenRouterLLM(config=llm_config)
    workflow_graph = SequentialWorkFlowGraph.from_dict(coding_graph_data)

    # 2) Wrap it: the (single) agent prompt becomes an optimizable PROMPT unit.
    adapter = SEWWorkFlowAdapter(graph=workflow_graph, llm=llm, llm_config=llm_config)
    logger.info(f"Optimizable prompts: {list(adapter.prompts.keys())}")

    # 3) Benchmark + evaluation/objective wiring.
    benchmark = HumanEvalSplits()
    dev_data = benchmark.get_dev_data(sample_k=DEV_SAMPLE_NUM, seed=42)
    test_data = benchmark.get_test_data(sample_k=TEST_SAMPLE_NUM, seed=42)
    objective = ScalarObjective(metric="pass@1", direction="maximize")
    dev_evaluate_fn = make_evaluate_fn(benchmark, dev_data, EVAL_CONCURRENCY)
    test_evaluate_fn = make_evaluate_fn(benchmark, test_data, EVAL_CONCURRENCY)

    # 4) The optimizer: re-mutate every prompt each round, steered by the workflow goal.
    optimizer = SEWOptimizer(
        adapter=adapter,
        optimizer_model=OPTIMIZER_MODEL,
        task_description=workflow_graph.goal,
        num_variants_per_step=NUM_VARIANTS_PER_STEP,
        order="first-order",  # "second-order" synthesizes a role-aware mutation directive first
        seed=42,
    )

    # Evaluate the un-optimized workflow on the held-out test split.
    before = await test_evaluate_fn(adapter)
    logger.info(f"Test pass@1 BEFORE optimization: {before['pass@1']:.4f}")

    # 5) Optimize. Returns the adapter rebuilt from the best-scoring snapshot.
    best_adapter = await optimizer.async_optimize(
        evaluate_fn=dev_evaluate_fn,
        objective=objective,
        max_trials=MAX_TRIALS,
        save_dir=SAVE_DIR,
    )

    # (Optional) load optimized adapter from a saved state 
    # best_adapter = optimizer.load_optimized(SAVE_DIR)

    # Evaluate the optimized workflow on the same test split.
    after = await test_evaluate_fn(best_adapter)
    logger.info(f"Test pass@1 AFTER optimization:  {after['pass@1']:.4f}")

    logger.info("Optimized prompts:")
    for name, prompt in best_adapter.prompts.items():
        logger.info(f"--- {name} ---\n{prompt}\n")


if __name__ == "__main__":
    asyncio.run(main())
