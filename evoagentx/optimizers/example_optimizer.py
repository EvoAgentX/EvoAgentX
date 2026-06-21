"""
Example optimizer: LLM-driven prompt-variant search.

This module is a worked end-to-end example of how a concrete evolutionary
algorithm plugs into the generic optimization engine defined in
``evoagentx/optimizers/engine``. It follows the intended authoring order:

    OptimizationUnit  ->  Optimizer  ->  ProgramAdapter  (->  WorkflowAdapter)

1. **What can be optimized?**  Each agent prompt is exposed as one
   ``OptimizationUnit`` of type ``PROMPT`` whose value is the prompt string.
2. **How do we search?**  ``PromptVariantOptimizer`` asks an LLM to rewrite the
   current best prompt into one or more *variants*, evaluates them, and keeps the
   best — a minimal mutation-only ``(1 + λ)`` evolutionary loop.
3. **What must the adapter provide?**  ``PromptVariantProgramAdapter`` is the
   adapter *paired with* this optimizer. It owns the generic snapshot / merge /
   reconstruct plumbing for a ``{unit_name: prompt_text}`` program, so a user only
   declares their prompts and implements ``execute`` — the single piece of
   program-specific glue. (A truly universal prompt adapter would also expose
   APPEND / partial-replace operations and per-unit ``json_schema`` constraints in
   ``register_units``; this demo keeps to whole-prompt REPLACE to match the
   optimizer it is bound to.)

A framework-native ``WorkflowAdapter`` (the fourth arrow) would subclass
``PromptVariantProgramAdapter`` and implement ``execute`` to run an EvoAgentX
workflow; that is intentionally left out of this minimal demo.

Run the closed-loop demo at the bottom with::

    OPENROUTER_API_KEY=sk-... PYTHONPATH=. python evoagentx/optimizers/example_optimizer.py
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Tuple

from ..models.model_configs import OpenRouterConfig
from ..models.openrouter_model import OpenRouterLLM
from .engine.adapter import ProgramAdapter, SnapShot
from .engine.base import (
    ChangeOperation,
    OptimizationProposal,
    OptimizationUnit,
    OptimizationUnitType,
    UnitChange,
)
from .engine.objective import Objective
from .engine.optimizer import Optimizer, OptimizationRunState
from .engine.utils import get_best_snapshot


# ---------------------------------------------------------------------------
# Optimizer: LLM-driven prompt variant generation (mutation-only search).
# ---------------------------------------------------------------------------
_DEFAULT_META_SYSTEM_PROMPT = (
    "You are an expert prompt engineer. You are given a prompt that is used by an "
    "AI agent to perform a task. Your job is to rewrite it into a single improved "
    "variant that is clearer, more specific, and more likely to elicit correct, "
    "high-quality responses, while preserving the original intent and any required "
    "input/output format. Respond with ONLY the rewritten prompt text — no "
    "explanations, no commentary, no surrounding quotes or markdown fences."
)


class PromptVariantOptimizer(Optimizer):
    """
    A minimal LLM-based prompt optimizer.

    Each round it takes the current best snapshot, picks one target prompt unit per
    proposal, and asks an LLM to produce an improved variant of that prompt. Every
    variant becomes one ``OptimizationProposal`` (a single REPLACE change). The
    engine evaluates them, and the best-scoring snapshot is carried forward as the
    seed for the next round — a simple ``(1 + λ)`` style evolutionary loop.

    Args:
        adapter: The program adapter to optimize (a ``PromptVariantProgramAdapter``).
        optimizer_model: OpenRouter model id used to generate variants (e.g.
            ``"openai/gpt-4o-mini"``). An ``OpenRouterLLM`` is built internally;
            the API key is read from ``openrouter_key`` or ``$OPENROUTER_API_KEY``.
        num_variants_per_step: How many variants (proposals) to generate per round.
        meta_system_prompt: System prompt steering the rewrite. Overridable.
        temperature: Sampling temperature for variant generation (higher = more diverse).
        openrouter_key: Optional explicit API key; falls back to ``$OPENROUTER_API_KEY``.
        seed: Optional RNG seed for reproducible target-unit selection.
    """

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset(
        {OptimizationUnitType.PROMPT}
    )

    def __init__(
        self,
        adapter: ProgramAdapter,
        optimizer_model: str,
        num_variants_per_step: int = 2,
        meta_system_prompt: str = _DEFAULT_META_SYSTEM_PROMPT,
        temperature: float = 1.0,
        openrouter_key: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(adapter, **kwargs)
        if not isinstance(optimizer_model, str) or not optimizer_model:
            raise ValueError("`optimizer_model` must be a non-empty OpenRouter model id string.")
        if num_variants_per_step < 1:
            raise ValueError("num_variants_per_step must be >= 1")
        self.num_variants_per_step = num_variants_per_step
        self.meta_system_prompt = meta_system_prompt
        self.temperature = temperature
        self._rng = random.Random(seed)
        self.model = self._build_model(optimizer_model, openrouter_key)

    @staticmethod
    def _build_model(optimizer_model: str, openrouter_key: Optional[str]) -> OpenRouterLLM:
        key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "No OpenRouter API key found. Pass `openrouter_key=...` or set $OPENROUTER_API_KEY."
            )
        config = OpenRouterConfig(model=optimizer_model, openrouter_key=key, output_response=True, stream=True)
        return OpenRouterLLM(config=config)

    # -- helpers ------------------------------------------------------------
    def _build_meta_prompt(self, unit: OptimizationUnit, current_prompt: str) -> str:
        return (
            f"Here is the current prompt named '{unit.name}'.\n"
            f"--- CURRENT PROMPT ---\n{current_prompt}\n--- END CURRENT PROMPT ---\n\n"
            "Rewrite it into one improved variant. Return only the new prompt text."
        )

    @staticmethod
    def _clean_variant(text: str) -> str:
        """Strip stray markdown fences / wrapping quotes the LLM may add."""
        cleaned = text.strip()
        fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _make_proposal(
        self,
        source_snapshot_id: str,
        unit: OptimizationUnit,
        current_prompt: str,
        variant: str,
        index: int,
    ) -> Optional[OptimizationProposal]:
        variant = self._clean_variant(variant)
        if not variant or variant == current_prompt:
            return None  # skip empty / no-op variants
        change = UnitChange.create(
            unit=unit,
            new_value=variant,
            old_value=current_prompt,
            operation=ChangeOperation.REPLACE,
            metadata={"strategy": "llm_variant", "variant_index": index},
        )
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=[change],
            metadata={"source": "PromptVariantOptimizer", "target_uid": unit.uid},
        )

    def _pick_target(self, snapshot: SnapShot) -> Tuple[OptimizationUnit, str]:
        unit = self._rng.choice(self.target_units)
        return unit, snapshot.unit_values[unit.uid]

    # -- proposal generation -----------------------------------------------
    def batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        budget_remaining: Optional[int] = None,
        **kwargs,
    ) -> List[OptimizationProposal]:
        snapshot = get_best_snapshot(state)
        source_id = snapshot.snapshot_id
        n = max(0, min(self.num_variants_per_step, budget_remaining) if budget_remaining is not None else self.num_variants_per_step)
        proposals: List[OptimizationProposal] = []
        for i in range(n):
            unit, current_prompt = self._pick_target(snapshot)
            variant = self.model.single_generate(
                messages=[
                    {"role": "system", "content": self.meta_system_prompt},
                    {"role": "user", "content": self._build_meta_prompt(unit, current_prompt)},
                ],
                temperature=self.temperature
            )
            proposal = self._make_proposal(source_id, unit, current_prompt, variant, i)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    async def async_batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        budget_remaining: Optional[int] = None,
        **kwargs,
    ) -> List[OptimizationProposal]:
        snapshot = get_best_snapshot(state)
        source_id = snapshot.snapshot_id
        n = max(0, min(self.num_variants_per_step, budget_remaining) if budget_remaining is not None else self.num_variants_per_step)
        targets = [self._pick_target(snapshot) for _ in range(n)]

        async def _gen(unit: OptimizationUnit, current_prompt: str) -> str:
            return await self.model.single_generate_async(
                messages=[
                    {"role": "system", "content": self.meta_system_prompt},
                    {"role": "user", "content": self._build_meta_prompt(unit, current_prompt)},
                ],
                temperature=self.temperature,
                output_response=False,
            )

        variants = await asyncio.gather(*[_gen(u, p) for u, p in targets])
        proposals: List[OptimizationProposal] = []
        for i, ((unit, current_prompt), variant) in enumerate(zip(targets, variants)):
            proposal = self._make_proposal(source_id, unit, current_prompt, variant, i)
            if proposal is not None:
                proposals.append(proposal)
        return proposals


# ---------------------------------------------------------------------------
# ProgramAdapter paired with PromptVariantOptimizer.
# ---------------------------------------------------------------------------
class PromptVariantProgramAdapter(ProgramAdapter):
    """
    Adapter for programs whose optimizable state is a set of named prompts, paired
    with :class:`PromptVariantOptimizer`.

    The snapshot / merge / reconstruct plumbing is fully generic for a
    ``{name: prompt_text}`` program, so a user only:

    * passes ``prompts`` -- a ``{name: prompt_text}`` mapping; each entry becomes one
      ``OptimizationUnit`` of type ``PROMPT`` (whole-prompt REPLACE), and
    * subclasses and implements ``execute`` -- how the program actually runs given
      its current prompts (the single piece of program-specific glue).
    """

    def __init__(self, prompts: Dict[str, str]) -> None:
        if not prompts:
            raise ValueError("PromptVariantProgramAdapter requires at least one prompt.")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in prompts.items()):
            raise TypeError("`prompts` must be a Dict[str, str] mapping prompt name -> prompt text.")
        self.prompts: Dict[str, str] = dict(prompts)

    # -- declare optimizable units -----------------------------------------
    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(
                name=name,
                uid=name,
                unit_type=OptimizationUnitType.PROMPT,
                json_schema={"type": "string", "description": "Full prompt text."},
                allowed_operations=[ChangeOperation.REPLACE],
            )
            for name in self.prompts
        ]

    # -- snapshot / merge / reconstruct (all generic) ----------------------
    def take_snapshot(self) -> SnapShot:
        return SnapShot(unit_values=dict(self.prompts))

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        new_values = dict(snapshot.unit_values)
        for change in changes:
            new_values[change.uid] = change.value  # REPLACE-only
        return SnapShot(unit_values=new_values, program_config=snapshot.program_config)

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> "PromptVariantProgramAdapter":
        # type(self) reconstructs the concrete subclass, provided it keeps this
        # single-arg constructor. A subclass with a different __init__ must override.
        return type(self)(prompts=dict(snapshot.unit_values))

    # -- run the program ----------------------------------------------------
    def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "Subclass PromptVariantProgramAdapter and implement execute() to run "
            "your program using self.prompts."
        )


__all__ = ["PromptVariantOptimizer", "PromptVariantProgramAdapter"]


# ---------------------------------------------------------------------------
# Toy closed-loop demo.
# ---------------------------------------------------------------------------
class ToyAdapter(PromptVariantProgramAdapter):
    """
    A toy program with two optimizable prompts (a marketing tagline generator).

    ``execute`` returns a fixed, deterministic list of strings — here just the two
    current prompt values, standing in for whatever a real program would emit. The
    paired ``toy_evaluate`` then maps those strings to a scalar score.
    """

    def __init__(self, prompts: Optional[Dict[str, str]] = None) -> None:
        super().__init__(prompts or {
            "headline": "Buy our product.",
            "call_to_action": "Click here.",
        })

    def execute(self, *args, **kwargs) -> List[str]:
        # Deterministic "program output": the current values of both prompt units.
        return [self.prompts["headline"], self.prompts["call_to_action"]]


# A hidden rubric the optimizer does NOT see: reward persuasive marketing words and
# a clear call to action, lightly penalise rambling. Deliberately NOT a length proxy
# — padding text with filler lowers the density score, so improvement must be real.
_REWARD_WORDS = {
    "free", "now", "today", "exclusive", "guaranteed", "save", "instantly",
    "limited", "proven", "results", "transform", "boost", "discover", "join",
}
_CTA_WORDS = {"start", "join", "get", "try", "claim", "shop", "subscribe", "discover", "unlock"}


def _score_text(text: str, reward_set: set) -> float:
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in reward_set)
    density = hits / len(tokens)          # reward relevant words per token (anti-padding)
    coverage = min(hits, 3) / 3.0          # reward hitting several distinct cues
    brevity = 1.0 if len(tokens) <= 14 else max(0.0, 1.0 - (len(tokens) - 14) / 20.0)
    return 0.6 * coverage + 0.3 * density + 0.1 * brevity


def toy_evaluate(adapter: ToyAdapter) -> Dict[str, float]:
    headline, cta = adapter.execute()
    score = 0.6 * _score_text(headline, _REWARD_WORDS) + 0.4 * _score_text(cta, _CTA_WORDS)
    return {"score": round(score, 4)}


# IMPORTANT: in this toy task the optimized units ARE the final marketing copy (not
# meta-instructions for another agent), so the rewrite instruction must match the
# evaluator's intent — punchy, persuasive, short. Using the optimizer's *default*
# meta_system_prompt (which treats each unit as an "agent prompt" to expand) pushes
# the LLM toward long generic instructions and the score never beats the baseline.
# This is the key lesson: the optimizer's rewrite objective must be aligned with how
# the adapter/evaluator actually consumes the unit value.
_TOY_META_SYSTEM_PROMPT = (
    "You are a world-class direct-response copywriter. You are given a short piece of "
    "marketing copy (a headline or a call-to-action button). Rewrite it into a single, "
    "punchy, high-converting variant. Lean into urgency, exclusivity, and concrete "
    "benefits, and end on a strong, action-oriented call to action. Keep it very short "
    "— ideally under 12 words, one line. Respond with ONLY the rewritten copy: no "
    "explanations, no quotes, no markdown."
)


def run_demo(
    optimizer_model: str = "deepseek/deepseek-v4-flash", # "openai/gpt-4o-mini",
    max_trials: int = 8,
    num_variants_per_step: int = 2,
) -> None:
    """Run the full evolutionary loop against the toy adapter and print progress."""
    from .engine.objective import ScalarObjective

    adapter = ToyAdapter()
    objective = ScalarObjective(metric="score", direction="maximize")
    optimizer = PromptVariantOptimizer(
        adapter=adapter,
        optimizer_model=optimizer_model,
        num_variants_per_step=num_variants_per_step,
        meta_system_prompt=_TOY_META_SYSTEM_PROMPT,
        temperature=1.1,
        seed=7,
    )

    print("=== baseline prompts ===")
    for name, value in adapter.prompts.items():
        print(f"  {name}: {value!r}")
    print(f"  baseline score: {toy_evaluate(adapter)['score']}\n")

    best_adapter: ToyAdapter = optimizer.optimize(
        evaluate_fn=toy_evaluate,
        objective=objective,
        max_trials=max_trials,
        save_dir="./debug/optimizers/toy_prompt_opt_run",
    )

    print("\n=== best evolved prompts ===")
    for name, value in best_adapter.prompts.items():
        print(f"  {name}: {value!r}")
    print(f"  best score: {toy_evaluate(best_adapter)['score']}")


def run_resume_demo(
    optimizer_model: str = "deepseek/deepseek-v4-flash",
    save_dir: str = "./debug/optimizers/toy_prompt_opt_run",
    max_trials: int = 14,
    num_variants_per_step: int = 2,
) -> None:
    """
    Resume a previously checkpointed run and continue optimizing.

    Loads the run state from ``save_dir`` (must already contain an
    ``optimization_state.json`` written by ``run_demo``), then continues into the
    SAME directory: the baseline is not re-evaluated, ``current_step`` picks up where
    it stopped, and only ``max_trials - current_step`` new trials run. Pass a larger
    ``max_trials`` than the original run, otherwise the loop has no budget left and
    exits immediately.
    """
    import os
    from .engine.objective import ScalarObjective
    from .engine.optimizer import OptimizationRunState

    state_path = os.path.join(save_dir, "optimization_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"No checkpoint at {state_path}. Run `run_demo()` first to produce one."
        )

    prior = OptimizationRunState.load_state(save_dir)
    print("=== resuming from checkpoint ===")
    print(f"  checkpoint: {state_path}")
    print(f"  trials so far (current_step): {prior.current_step}")
    print(f"  best score so far: {prior.best_metrics}")
    best_before = prior.best_snapshot_id

    adapter = ToyAdapter()
    objective = ScalarObjective(metric="score", direction="maximize")
    optimizer = PromptVariantOptimizer(
        adapter=adapter,
        optimizer_model=optimizer_model,
        num_variants_per_step=num_variants_per_step,
        meta_system_prompt=_TOY_META_SYSTEM_PROMPT,
        temperature=1.1,
        seed=123,  # different seed so resumed trials explore new variants
    )

    best_adapter: ToyAdapter = optimizer.optimize(
        evaluate_fn=toy_evaluate,
        objective=objective,
        max_trials=max_trials,
        resume_from=save_dir,   # load existing state ...
        save_dir=save_dir,      # ... and keep checkpointing into the same folder
    )

    after = OptimizationRunState.load_state(save_dir)
    print("\n=== after resume ===")
    print(f"  trials now (current_step): {after.current_step}")
    print(f"  best snapshot changed? {best_before} -> {after.best_snapshot_id} "
          f"({'improved' if after.best_snapshot_id != best_before else 'unchanged'})")
    for name, value in best_adapter.prompts.items():
        print(f"  {name}: {value!r}")
    print(f"  best score: {toy_evaluate(best_adapter)['score']}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_demo()
