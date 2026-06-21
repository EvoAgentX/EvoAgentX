from __future__ import annotations

import asyncio
import os
import random
import re
from typing import TYPE_CHECKING, Any, ClassVar, Dict, FrozenSet, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from ..workflow.workflow_graph import WorkFlowGraph, WorkFlowNode

from ..models.model_configs import OpenRouterConfig
from ..models.openrouter_model import OpenRouterLLM
from ..prompts.workflow.sew_optimizer import mutation_prompts, thinking_styles
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
# Optimizer: PromptBreeder-style mutation over all prompt units at once.
# ---------------------------------------------------------------------------
# ``[[TASK_DESCRIPTION]]`` is a substitution sentinel (str.replace, not str.format) so the
# literal ``{variable}`` placeholder text below is preserved untouched.
_META_SYSTEM_PROMPT = (
    "You are an expert prompt engineer improving the prompts that power an AI system.\n"
    "The system's overall goal is:\n"
    '"""\n[[TASK_DESCRIPTION]]\n"""\n\n'
    "You will be given ONE prompt from this system, a description of the specific role that "
    "prompt plays within it, and a mutation directive describing how to rewrite it. Apply the "
    "directive to produce a single improved version of that prompt that (a) better fulfils its "
    "specific role and (b) helps the system accomplish its overall goal, while preserving the "
    "original intent and any required input/output format — including placeholder tokens such "
    "as {variable}. Do not expand the prompt to take on responsibilities outside its stated "
    "role. Respond with ONLY the rewritten prompt text — no explanations, no commentary, no "
    "surrounding quotes or markdown fences."
)

# Hyper-mutation (second-order) produces a *mutation directive* tailored to the task and the
# unit's role, not a rewritten prompt, so it uses its own task-aware system prompt.
_HYPER_MUTATION_SYSTEM_PROMPT = (
    "You are an expert at designing prompt-improvement strategies for an AI system.\n"
    "The system's overall goal is:\n"
    '"""\n[[TASK_DESCRIPTION]]\n"""\n\n'
    "You will be given a thinking style and the specific role that one prompt plays in this "
    "system. Using the thinking style as inspiration, decide what FORM of change that prompt "
    "needs and propose a single concrete directive describing how it should be rewritten so it "
    "better fulfils its role and advances the overall goal. Keep the directive scoped to that "
    "one prompt's role. Respond with ONLY the directive — one or two imperative sentences "
    "addressed to a prompt engineer, no other text."
)


class SEWOptimizer(Optimizer):
    """
    LLM-driven prompt optimizer that re-mutates every target prompt each round.

    Each round it seeds from the current best snapshot and produces
    ``num_variants_per_step`` variants. A variant mutates **all** target prompt units
    simultaneously (one ``UnitChange`` per unit), so each variant is a single
    ``OptimizationProposal`` carrying several REPLACE changes. The engine evaluates the
    variants, keeps the best-scoring snapshot, and carries it forward as the seed for the
    next round — a mutation-only ``(1 + λ)`` loop.

    Rewrites are grounded in ``task_description`` (the system goal, in the system prompt)
    and each unit's role (in the user message, read from ``unit.metadata["role"]`` with a
    fallback to the unit name), so each prompt is improved for *its* job rather than the
    whole goal. See the module docstring for details.

    Args:
        adapter: The program adapter to optimize (a :class:`SEWProgramAdapter`).
        optimizer_model: OpenRouter model id used to mutate prompts (e.g.
            ``"openai/gpt-4o-mini"``). An ``OpenRouterLLM`` is built internally; the API
            key is read from ``openrouter_key`` or ``$OPENROUTER_API_KEY``.
        task_description: Required. Description of the system's overall goal the prompts
            serve (typically the workflow goal). Steers both the rewrite and (for
            ``second-order``) the hyper-mutation.
        num_variants_per_step: How many variants (full re-mutations) to generate per round.
        order: ``"first-order"`` (sample a predefined mutation directive) or
            ``"second-order"`` (synthesize a fresh, role-aware directive first).
        temperature: Sampling temperature for generation (higher = more diverse).
        openrouter_key: Optional explicit API key; falls back to ``$OPENROUTER_API_KEY``.
        seed: Optional RNG seed for reproducible mutation-directive / thinking-style sampling.
    """

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset(
        {OptimizationUnitType.PROMPT}
    )

    def __init__(
        self,
        adapter: ProgramAdapter,
        optimizer_model: str,
        task_description: str,
        num_variants_per_step: int = 1,
        order: Literal["first-order", "second-order"] = "first-order",
        temperature: float = 1.0,
        openrouter_key: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(adapter, **kwargs)
        if not isinstance(optimizer_model, str) or not optimizer_model:
            raise ValueError("`optimizer_model` must be a non-empty OpenRouter model id string.")
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError("`task_description` must be a non-empty string describing the system's goal.")
        if num_variants_per_step < 1:
            raise ValueError("num_variants_per_step must be >= 1")
        if order not in ("first-order", "second-order"):
            raise ValueError(f"order must be 'first-order' or 'second-order', got {order!r}")
        self.task_description = task_description.strip()
        self.num_variants_per_step = num_variants_per_step
        self.order = order
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

    # -- prompt construction ------------------------------------------------
    def _meta_system_prompt(self) -> str:
        return _META_SYSTEM_PROMPT.replace("[[TASK_DESCRIPTION]]", self.task_description)

    def _hyper_mutation_system_prompt(self) -> str:
        return _HYPER_MUTATION_SYSTEM_PROMPT.replace("[[TASK_DESCRIPTION]]", self.task_description)

    @staticmethod
    def _unit_role(unit: OptimizationUnit) -> Optional[str]:
        """Per-unit role description declared by the adapter, if any."""
        metadata = unit.metadata or {}
        role = metadata.get("role") or metadata.get("description")
        return role if isinstance(role, str) and role.strip() else None

    def _unit_descriptor(self, unit: OptimizationUnit) -> str:
        """Human-readable identification of the unit and its role for the user message."""
        role = self._unit_role(unit)
        if role:
            return f"the '{unit.name}' prompt, whose role in the system is:\n\"\"\"\n{role.strip()}\n\"\"\""
        return f"the '{unit.name}' prompt"

    def _select_mutation_prompt(self) -> str:
        """First-order: sample a predefined mutation directive (no LLM call)."""
        return self._rng.choice(mutation_prompts)

    def _build_hyper_user_prompt(self, unit: OptimizationUnit) -> str:
        """Second-order: ask the LLM for a role-aware directive, inspired by a thinking style."""
        thinking_style = self._rng.choice(thinking_styles)
        return (
            f"Thinking style:\n{thinking_style}\n\n"
            f"The prompt to be improved is {self._unit_descriptor(unit)}\n\n"
            "Propose the mutation directive for rewriting this prompt."
        )

    def _build_mutation_message(
        self, unit: OptimizationUnit, mutation_prompt: str, current_prompt: str
    ) -> str:
        return (
            f"The prompt below is {self._unit_descriptor(unit)}\n\n"
            f"Mutation directive:\n{mutation_prompt}\n\n"
            f"--- CURRENT PROMPT ---\n{current_prompt}\n--- END CURRENT PROMPT ---\n\n"
            "Apply the directive and return only the rewritten prompt text."
        )

    @staticmethod
    def _clean_variant(text: str) -> str:
        """Strip stray markdown fences / wrapping quotes the LLM may add."""
        cleaned = (text or "").strip()
        fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _mutate_prompt(self, unit: OptimizationUnit, current_prompt: str) -> str:
        """Produce one mutated prompt for ``unit`` (sync)."""
        if self.order == "second-order":
            mutation_prompt = self.model.single_generate(
                messages=[
                    {"role": "system", "content": self._hyper_mutation_system_prompt()},
                    {"role": "user", "content": self._build_hyper_user_prompt(unit)},
                ],
                temperature=self.temperature,
            )
        else:
            mutation_prompt = self._select_mutation_prompt()
        new_prompt = self.model.single_generate(
            messages=[
                {"role": "system", "content": self._meta_system_prompt()},
                {"role": "user", "content": self._build_mutation_message(unit, mutation_prompt, current_prompt)},
            ],
            temperature=self.temperature,
        )
        return self._clean_variant(new_prompt)

    async def _async_mutate_prompt(self, unit: OptimizationUnit, current_prompt: str) -> str:
        """Produce one mutated prompt for ``unit`` (async)."""
        if self.order == "second-order":
            mutation_prompt = await self.model.single_generate_async(
                messages=[
                    {"role": "system", "content": self._hyper_mutation_system_prompt()},
                    {"role": "user", "content": self._build_hyper_user_prompt(unit)},
                ],
                temperature=self.temperature,
                output_response=False,
            )
        else:
            mutation_prompt = self._select_mutation_prompt()
        new_prompt = await self.model.single_generate_async(
            messages=[
                {"role": "system", "content": self._meta_system_prompt()},
                {"role": "user", "content": self._build_mutation_message(unit, mutation_prompt, current_prompt)},
            ],
            temperature=self.temperature,
            output_response=False,
        )
        return self._clean_variant(new_prompt)

    def _make_proposal(
        self,
        source_snapshot_id: str,
        snapshot: SnapShot,
        new_prompts: Dict[str, str],
        variant_index: int,
    ) -> Optional[OptimizationProposal]:
        """Build one proposal mutating every target unit; drop empty / no-op changes."""
        changes: List[UnitChange] = []
        for unit in self.target_units:
            current_prompt = snapshot.unit_values[unit.uid]
            variant = new_prompts.get(unit.uid, "")
            if not variant or variant == current_prompt:
                continue  # skip empty / unchanged units
            changes.append(
                UnitChange.create(
                    unit=unit,
                    new_value=variant,
                    old_value=current_prompt,
                    operation=ChangeOperation.REPLACE,
                    metadata={
                        "strategy": "sew_mutation",
                        "order": self.order,
                        "variant_index": variant_index,
                    },
                )
            )
        if not changes:
            return None  # every unit was empty / unchanged
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=changes,
            metadata={
                "source": "SEWOptimizer",
                "order": self.order,
                "variant_index": variant_index,
                "target_uids": self.target_unit_uids,
            },
        )

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
            new_prompts = {
                unit.uid: self._mutate_prompt(unit, snapshot.unit_values[unit.uid])
                for unit in self.target_units
            }
            proposal = self._make_proposal(source_id, snapshot, new_prompts, i)
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

        # Mutate every (variant, unit) pair concurrently, then regroup by variant.
        index: List[Tuple[int, str]] = []
        tasks = []
        for i in range(n):
            for unit in self.target_units:
                index.append((i, unit.uid))
                tasks.append(self._async_mutate_prompt(unit, snapshot.unit_values[unit.uid]))
        results = await asyncio.gather(*tasks)

        per_variant: Dict[int, Dict[str, str]] = {i: {} for i in range(n)}
        for (i, uid), new_prompt in zip(index, results):
            per_variant[i][uid] = new_prompt

        proposals: List[OptimizationProposal] = []
        for i in range(n):
            proposal = self._make_proposal(source_id, snapshot, per_variant[i], i)
            if proposal is not None:
                proposals.append(proposal)
        return proposals


# ---------------------------------------------------------------------------
# ProgramAdapter paired with SEWOptimizer.
# ---------------------------------------------------------------------------
class SEWProgramAdapter(ProgramAdapter):
    """
    Adapter for programs whose optimizable state is a set of named prompts, paired with
    :class:`SEWOptimizer`.

    The snapshot / merge / reconstruct plumbing is fully generic for a
    ``{name: prompt_text}`` program, so a user only:

    * passes ``prompts`` -- a ``{name: prompt_text}`` mapping; each entry becomes one
      ``OptimizationUnit`` of type ``PROMPT`` (whole-prompt REPLACE);
    * optionally passes ``roles`` -- a ``{name: role_description}`` mapping giving each
      prompt's specific job within the system. ``SEWOptimizer`` reads these (via unit
      metadata) so it can rewrite each prompt for *its* role rather than the global goal.
      Names without a role fall back to the bare prompt name; and
    * subclasses and implements ``execute`` -- how the program actually runs given its
      current prompts (the single piece of program-specific glue).
    """

    def __init__(self, prompts: Dict[str, str], roles: Optional[Dict[str, str]] = None) -> None:
        if not prompts:
            raise ValueError("SEWProgramAdapter requires at least one prompt.")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in prompts.items()):
            raise TypeError("`prompts` must be a Dict[str, str] mapping prompt name -> prompt text.")
        self.prompts: Dict[str, str] = dict(prompts)

        roles = dict(roles) if roles else {}
        unknown = set(roles) - set(self.prompts)
        if unknown:
            raise ValueError(f"`roles` contains unknown prompt name(s): {sorted(unknown)}")
        if not all(isinstance(v, str) for v in roles.values()):
            raise TypeError("`roles` must be a Dict[str, str] mapping prompt name -> role description.")
        self.roles: Dict[str, str] = roles

    # -- declare optimizable units -----------------------------------------
    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(
                name=name,
                uid=name,
                unit_type=OptimizationUnitType.PROMPT,
                json_schema={"type": "string", "description": "Full prompt text."},
                allowed_operations=[ChangeOperation.REPLACE],
                metadata={"role": self.roles[name]} if name in self.roles else {},
            )
            for name in self.prompts
        ]

    # -- snapshot / merge / reconstruct (all generic) ----------------------
    def take_snapshot(self) -> SnapShot:
        # Roles are non-optimizable config; carry them in program_config so they survive
        # snapshot persistence and reconstruction.
        return SnapShot(
            unit_values=dict(self.prompts),
            program_config={"roles": self.roles} if self.roles else None,
        )

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        new_values = dict(snapshot.unit_values)
        for change in changes:
            new_values[change.uid] = change.value  # REPLACE-only
        return SnapShot(unit_values=new_values, program_config=snapshot.program_config)

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> "SEWProgramAdapter":
        # type(self) reconstructs the concrete subclass, provided it keeps this
        # constructor signature. A subclass with a different __init__ must override.
        roles = (snapshot.program_config or {}).get("roles")
        return type(self)(prompts=dict(snapshot.unit_values), roles=roles)

    # -- run the program ----------------------------------------------------
    def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "Subclass SEWProgramAdapter and implement execute() to run your program "
            "using self.prompts."
        )


# ---------------------------------------------------------------------------
# WorkFlowGraph-backed adapter.
# ---------------------------------------------------------------------------
class SEWWorkFlowAdapter(SEWProgramAdapter):
    """
    A :class:`SEWProgramAdapter` whose optimizable prompts are drawn from the agents of a
    :class:`~evoagentx.workflow.workflow_graph.WorkFlowGraph`.

    Given a workflow graph, this adapter automatically derives one ``PROMPT`` unit per agent
    that carries editable prompt text (keyed by agent name, since a node may hold several
    agents). For each agent the optimizable text is its ``prompt``, or — when a
    ``prompt_template`` is present (which takes precedence, mirroring ``CustomizeAgent``) —
    the template's ``instruction``. The agent's (or node's) description becomes that prompt's
    role, so :class:`SEWOptimizer` can rewrite each prompt for *its* job, and the graph's
    ``goal`` is the overall system goal. ``execute`` / ``async_execute`` run the graph
    end-to-end: they build a fresh ``AgentManager`` + ``WorkFlow`` and call
    ``workflow.execute`` / ``workflow.async_execute`` (cf. ``examples/workflow``).

    Args:
        graph: The ``WorkFlowGraph`` to optimize. Its nodes' agent prompts become the
            optimizable units.
        tools: Optional list of toolkits passed to the ``AgentManager`` at execution time.
        llm: The ``BaseLLM`` instance driving the ``WorkFlow`` (scheduling / output
            extraction). Required to actually run the workflow.
        llm_config: The ``LLMConfig`` used to instantiate each node's agent.
        max_execution_steps: Forwarded to ``WorkFlow.max_execution_steps``.
    """

    def __init__(
        self,
        graph: "WorkFlowGraph",
        tools: Optional[List[Any]] = None,
        llm: Optional[Any] = None,
        llm_config: Optional[Any] = None,
        max_execution_steps: int = 5,
        **kwargs,
    ) -> None:
        from ..workflow.workflow_graph import WorkFlowGraph

        if not isinstance(graph, WorkFlowGraph):
            raise TypeError(f"`graph` must be a WorkFlowGraph instance, got {type(graph).__name__}.")
        self.graph = graph
        self.tools = tools
        self.llm = llm
        self.llm_config = llm_config
        self.max_execution_steps = max_execution_steps

        prompts, roles = self._extract_prompts_and_roles(graph)
        if not prompts:
            raise ValueError(
                "No optimizable prompts found in the workflow graph: none of its agents "
                "carry an editable `prompt` or `prompt_template` instruction."
            )
        super().__init__(prompts=prompts, roles=roles)

    # -- prompt extraction --------------------------------------------------
    @staticmethod
    def _template_instruction(template: Any) -> Optional[str]:
        """Read the optimizable ``instruction`` from a prompt template (dict or PromptTemplate)."""
        if template is None:
            return None
        instruction = template.get("instruction") if isinstance(template, dict) else getattr(template, "instruction", None)
        return instruction if isinstance(instruction, str) and instruction.strip() else None

    @staticmethod
    def _set_template_instruction(template: Any, text: str) -> None:
        """Write ``text`` back as the prompt template's ``instruction`` (dict or PromptTemplate)."""
        if isinstance(template, dict):
            template["instruction"] = text
        elif hasattr(template, "set_instruction"):
            template.set_instruction(text)
        else:
            setattr(template, "instruction", text)

    @classmethod
    def _agent_prompt_text(cls, agent: Any) -> Optional[str]:
        """The active optimizable prompt text of an agent config dict.

        Mirrors ``CustomizeAgent``: when both are present ``prompt_template`` wins, so the
        template's ``instruction`` is the optimizable text; otherwise the raw ``prompt`` is.
        Returns ``None`` for agents (e.g. bare string references) with no editable prompt.
        """
        if not isinstance(agent, dict):
            return None
        if agent.get("prompt_template") is not None:
            return cls._template_instruction(agent["prompt_template"])
        prompt = agent.get("prompt")
        return prompt if isinstance(prompt, str) and prompt.strip() else None

    @classmethod
    def _set_agent_prompt_text(cls, agent: Dict[str, Any], text: str) -> None:
        """Inject ``text`` into the agent's active prompt slot (template instruction or prompt)."""
        if agent.get("prompt_template") is not None:
            cls._set_template_instruction(agent["prompt_template"], text)
        else:
            agent["prompt"] = text

    @classmethod
    def _iter_prompt_agents(cls, node: "WorkFlowNode") -> List[Tuple[str, Dict[str, Any]]]:
        """Yield ``(agent_name, agent_dict)`` for every agent on ``node`` with editable prompt text."""
        result: List[Tuple[str, Dict[str, Any]]] = []
        for agent in node.agents or []:
            if not isinstance(agent, dict) or cls._agent_prompt_text(agent) is None:
                continue
            name = agent.get("name")
            if name:
                result.append((name, agent))
        return result

    @classmethod
    def _extract_prompts_and_roles(
        cls, graph: "WorkFlowGraph"
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build ``{agent_name: prompt}`` and ``{agent_name: role}`` across all nodes' agents.

        A node may carry several agents, so units are keyed by (workflow-unique) agent name
        rather than node name. Each agent's role defaults to its own description, falling back
        to the node's description.
        """
        prompts: Dict[str, str] = {}
        roles: Dict[str, str] = {}
        for node in graph.nodes:
            for agent_name, agent in cls._iter_prompt_agents(node):
                if agent_name in prompts:
                    # Units are keyed by agent name; a duplicate would make reconstruction
                    # ambiguous (which agent does a mutated prompt belong to?).
                    raise ValueError(
                        f"Duplicate agent name {agent_name!r} found in the workflow graph. "
                        "Each agent must have a unique name so its prompt can be optimized "
                        "and reconstructed unambiguously."
                    )
                prompts[agent_name] = cls._agent_prompt_text(agent)
                role = agent.get("description") or node.description
                if role:
                    roles[agent_name] = role
        return prompts, roles

    def _copy_graph(self) -> "WorkFlowGraph":
        """Return an independent copy of the graph (its nodes/edges are deep-copied)."""
        from ..workflow.workflow_graph import WorkFlowGraph

        # Constructing from an existing graph deep-copies its nodes and edges while building
        # a fresh internal MultiDiGraph + lock, so the original is never mutated and we avoid
        # deep-copying the un-pickleable threading.Lock the graph holds. (cf. Evaluator.)
        return WorkFlowGraph(goal=self.graph.goal, graph=self.graph)

    def _rebuild_graph(self, prompts: Dict[str, str]) -> "WorkFlowGraph":
        """Deep-copy the base graph and inject the (possibly mutated) prompts into its agents."""
        new_graph = self._copy_graph()
        for node in new_graph.nodes:
            for agent_name, agent in self._iter_prompt_agents(node):
                if agent_name in prompts:
                    self._set_agent_prompt_text(agent, prompts[agent_name])
        new_graph.reset_graph()
        return new_graph

    # -- snapshot / reconstruct --------------------------------------------
    def take_snapshot(self) -> SnapShot:
        # Only the prompts are optimizable; everything needed to rebuild the graph (its
        # structure, the non-optimizable agent config, tools, llm) is carried on the live
        # adapter and re-applied in `from_snapshot`, so the snapshot stays prompt-only.
        # Roles are non-optimizable config; persist them so they survive reconstruction.
        return SnapShot(
            unit_values=dict(self.prompts),
            program_config={"roles": self.roles} if self.roles else None,
        )

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> "SEWWorkFlowAdapter":
        # Reconstruct the graph by injecting the snapshot's prompts into a copy of the
        # current graph, then build a fresh adapter around it. tools / llm / llm_config are
        # non-serializable runtime dependencies carried over from the live adapter.
        new_graph = self._rebuild_graph(dict(snapshot.unit_values))
        return type(self)(
            graph=new_graph,
            tools=self.tools,
            llm=self.llm,
            llm_config=self.llm_config,
            max_execution_steps=self.max_execution_steps,
        )

    # -- run the workflow ---------------------------------------------------
    def _build_workflow(self):
        """Reset the graph and assemble a fresh AgentManager + WorkFlow to run it."""
        from ..agents import AgentManager
        from ..workflow import WorkFlow

        # Clear any execution state on the graph so a prior run can't corrupt this one.
        self.graph.reset_graph()
        agent_manager = AgentManager(tools=self.tools)
        agent_manager.add_agents_from_workflow(self.graph, llm_config=self.llm_config)
        return WorkFlow(
            graph=self.graph,
            agent_manager=agent_manager,
            llm=self.llm,
            max_execution_steps=self.max_execution_steps,
        )

    def execute(self, inputs: Optional[dict] = None, **kwargs) -> Any:
        workflow = self._build_workflow()
        return workflow.execute(inputs=inputs or {}, **kwargs)

    async def async_execute(self, inputs: Optional[dict] = None, **kwargs) -> Any:
        workflow = self._build_workflow()
        return await workflow.async_execute(inputs=inputs or {}, **kwargs)


__all__ = ["SEWOptimizer", "SEWProgramAdapter", "SEWWorkFlowAdapter"]
