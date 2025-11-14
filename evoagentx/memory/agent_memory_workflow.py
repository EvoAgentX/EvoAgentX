"""
Agent Workflow Memory (AWM) support.

This module introduces three cooperative components that extend EvoAgentX memory system:

1. AWMExtractor – converts a workflow run trace into a normalized `AWMConcreteExample`.
2. AWMInducer – aggregates high quality examples and synthesizes reusable `AWMSummaryRule`.
3. AWMPromptInjector – retrieves summary rules + supporting examples to inject into system prompts.

Each component is designed to reuse the existing `LongTermMemory` + `MemoryManager` stack while
isolating AWM specific metadata in a dedicated corpus.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from textwrap import indent
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from ..core.logging import logger
from ..core.message import Message, MessageType
from ..models.base_model import BaseLLM
from ..models.model_configs import LLMConfig
from ..prompts.template import PromptTemplate
from ..agents.customize_memory_agent import (
    BaseMemoryBackend,
    BaseMemoryPolicy,
    CustomizeMemoryAgent,
    MemoryOperation,
    MemoryOrchestrator,
    MemoryRunContext,
)
from .long_term_memory import LongTermMemory


AWM_DEFAULT_CORPUS_ID = "awm_workflows"


class AWMMemoryType(str, Enum):
    """Memory entry type markers used in chunk metadata."""

    CONCRETE = "awm_concrete_example"
    SUMMARY = "awm_summary_rule"


class AWMTraceEvent(BaseModel):
    """One step in a workflow execution trace."""

    step_id: str = Field(default_factory=lambda: str(uuid4()))
    thought: str
    action: str
    action_input: Optional[str] = None
    observation: Optional[str] = None
    status: Optional[str] = Field(
        default=None, description="Optional status label (success/blocked/fail)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AWMRunRecord(BaseModel):
    """
    Input record describing the result of executing a workflow template once.
    """

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    template_id: str
    template_version: Optional[str] = None
    workflow_name: Optional[str] = None
    user_intent: str
    wf_goal: Optional[str] = None
    wf_task: Optional[str] = None
    domain: Optional[str] = None
    site: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    workflow_trace: List[AWMTraceEvent]
    generated_output: Optional[str] = None
    success: Optional[bool] = None
    evaluation_notes: Optional[str] = None
    failure_reason: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    raw_logs: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AWMConcreteExample(BaseModel):
    """
    Normalized unit stored for each workflow execution.
    """

    example_id: str = Field(default_factory=lambda: str(uuid4()))
    template_id: str
    abstract_signature: str
    signature_tokens: List[str] = Field(default_factory=list)
    template_version: Optional[str] = None
    workflow_name: Optional[str] = None
    run_id: str
    user_intent: str
    domain: Optional[str] = None
    site: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    success: bool = True
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    quality_label: str = "medium"
    quality_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    quality_rationale: str = ""
    trajectory_summary: str = ""
    generated_output: Optional[str] = None
    failure_reason: Optional[str] = None
    dedupe_key: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trace: List[AWMTraceEvent] = Field(default_factory=list)
    evaluation_notes: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def render_markdown(self, max_steps: int = 6) -> str:
        """
        Render the example as a compact markdown block for RAG retrieval.
        """
        head = [
            f"Template: {self.template_id}",
            f"Signature: {self.abstract_signature}",
            f"Success: {self.success} | Quality: {self.quality_label} ({self.quality_score:.2f})",
        ]
        if self.domain or self.site:
            head.append(
                "Context: "
                + ", ".join(filter(None, [self.domain, self.site]))
            )
        summary_lines = [
            "• " + self.trajectory_summary.strip()
            if self.trajectory_summary
            else "• Trajectory summary unavailable."
        ]
        if self.quality_rationale:
            summary_lines.append("• Evaluation: " + self.quality_rationale.strip())
        if self.generated_output:
            summary_lines.append("• Final Output: " + self.generated_output[:180].strip())

        step_lines: List[str] = []
        for idx, step in enumerate(self.trace[:max_steps], start=1):
            observation = step.observation or ""
            observation = observation.replace("\n", " ").strip()
            step_lines.append(
                f"{idx}. Thought: {step.thought.strip()} | Action: {step.action}"
                + (f" | Result: {observation}" if observation else "")
            )
        if len(self.trace) > max_steps:
            step_lines.append("...")

        return "\n".join(
            [
                "### AWM Concrete Example",
                "\n".join(head),
                "\n".join(summary_lines),
                "Steps:",
                indent("\n".join(step_lines), prefix="  "),
            ]
        )

    def prompt_excerpt(self, max_steps: int = 3) -> str:
        """Shortened excerpt for prompt injection."""
        excerpt_steps = []
        for idx, step in enumerate(self.trace[:max_steps], start=1):
            observation = (step.observation or "").replace("\n", " ").strip()
            excerpt_steps.append(
                f"{idx}. {step.action}: {observation or step.thought.strip()}"
            )
        if len(self.trace) > max_steps:
            excerpt_steps.append("   ...")
        joined_steps = "\n".join(excerpt_steps)
        lines = [
            f"- Intent: {self.user_intent[:180].strip()}",
            f"- Outcome: {'Success' if self.success else 'Failed'} | Quality {self.quality_score:.2f}",
            "- Key Steps:\n" + indent(joined_steps, "  "),
        ]
        if self.generated_output:
            lines.append("- Output Snippet: " + self.generated_output[:160].strip())
        return "\n".join(lines)


class AWMSummaryRule(BaseModel):
    """Generalized workflow rule derived from concrete examples."""

    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    template_id: str
    abstract_signature: str
    guidance: str
    rationale: str
    checklist: List[str] = Field(default_factory=list)
    risk_notes: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    support_example_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def render_prompt_block(self) -> str:
        checklist = "\n".join(f"  - {item}" for item in self.checklist) if self.checklist else "  - (Not specified)"
        lines = [
            f"#### Summary Rule ({self.abstract_signature})",
            f"- Guidance: {self.guidance.strip()}",
            f"- Rationale: {self.rationale.strip()}",
            "- Checklist:",
            checklist,
            f"- Confidence: {self.confidence:.2f} | Success Rate: {self.success_rate:.2f}",
        ]
        if self.risk_notes:
            lines.append(f"- Risks: {self.risk_notes.strip()}")
        return "\n".join(lines)


class AWMInductionConfig(BaseModel):
    """Parameters controlling summary induction behaviour."""

    template_id: Optional[str] = None
    abstract_signature: Optional[str] = None
    min_quality: float = Field(default=0.6, ge=0.0, le=1.0)
    require_success: bool = True
    max_examples_per_signature: int = Field(default=5, ge=1)
    summary_style: str = Field(
        default="concise",
        description="Style hint for the summarization LLM prompt.",
    )
    corpus_id: str = AWM_DEFAULT_CORPUS_ID


class AWMPromptContext(BaseModel):
    """Context information for constructing prompt injection memory blocks."""

    template_id: str
    goal: Optional[str] = None
    task: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    domain: Optional[str] = None
    site: Optional[str] = None
    step_hint: Optional[str] = None
    max_rules: int = Field(default=2, ge=1)
    max_examples: int = Field(default=2, ge=1)
    corpus_id: str = AWM_DEFAULT_CORPUS_ID


def _hash_signature(elements: Iterable[str]) -> str:
    canonical = "|".join(e.strip().lower() for e in elements if e)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def _call_llm_async(llm: BaseLLM, prompt: str) -> str:
    """
    Helper to obtain LLM response regardless of sync/async implementation on BaseLLM.
    """
    if hasattr(llm, "async_generate") and callable(getattr(llm, "async_generate")):
        response = await llm.async_generate(prompt=prompt)
        return response.content if hasattr(response, "content") else str(response)

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: llm.generate(prompt=prompt))
    return response.content if hasattr(response, "content") else str(response)


class AWMExtractor:
    """
    Transforms a workflow execution trace into a normalized concrete example and persists it.
    """

    def __init__(
        self,
        memory: LongTermMemory,
        corpus_id: str = AWM_DEFAULT_CORPUS_ID,
        summarizer_llm: Optional[BaseLLM] = None,
        judge_llm: Optional[BaseLLM] = None,
    ):
        self.memory = memory
        self.corpus_id = corpus_id
        self.summarizer_llm = summarizer_llm
        self.judge_llm = judge_llm

    async def extract(
        self,
        record: AWMRunRecord,
        auto_persist: bool = True,
    ) -> AWMConcreteExample:
        abstract_signature, signature_tokens = await self._build_signature(record)
        quality_score, quality_label, quality_confidence, rationale, success = await self._score_quality(record)
        dedupe_raw = [
            record.template_id,
            abstract_signature,
            record.domain or "",
            record.site or "",
            record.metadata.get("task_type", ""),
        ]
        dedupe_key = _hash_signature(dedupe_raw)

        trajectory_summary = await self._summarize_trace(record, abstract_signature)

        example = AWMConcreteExample(
            template_id=record.template_id,
            template_version=record.template_version,
            workflow_name=record.workflow_name,
            run_id=record.run_id,
            user_intent=record.user_intent,
            domain=record.domain,
            site=record.site,
            tags=list(dict.fromkeys(record.tags + record.metadata.get("tags", []))),
            success=success,
            quality_score=quality_score,
            quality_label=quality_label,
            quality_confidence=quality_confidence,
            quality_rationale=rationale,
            trajectory_summary=trajectory_summary,
            generated_output=record.generated_output,
            failure_reason=record.failure_reason,
            dedupe_key=dedupe_key,
            trace=record.workflow_trace,
            evaluation_notes=record.evaluation_notes,
            metrics=record.metrics,
            metadata=record.metadata,
            abstract_signature=abstract_signature,
            signature_tokens=signature_tokens,
        )

        if auto_persist:
            await self.persist(example)

        return example

    async def persist(self, example: AWMConcreteExample) -> str:
        """
        Store the example inside the AWM corpus and return memory id.
        """
        markdown = example.render_markdown()
        metadata_override = {
            "action": AWMMemoryType.CONCRETE.value,
            "wf_goal": example.template_id,
            "wf_task": example.abstract_signature,
            "wf_task_desc": "AWM Concrete Example",
            "custom_fields": {
                "awm_type": AWMMemoryType.CONCRETE.value,
                "template_id": example.template_id,
                "abstract_signature": example.abstract_signature,
                "quality_score": example.quality_score,
                "quality_label": example.quality_label,
                "success": example.success,
                "payload": example.model_dump(),
            },
        }
        message = Message(
            content=markdown,
            agent="AgentWorkflowMemory",
            action=AWMMemoryType.CONCRETE.value,
            msg_type=MessageType.RESPONSE,
            wf_goal=example.template_id,
            wf_task=example.abstract_signature,
            wf_task_desc="Concrete example captured for workflow memory",
        )
        memory_ids = self.memory.add(
            [message],
            corpus_id=self.corpus_id,
            metadata_overrides=[metadata_override],
            content_hashes=[example.dedupe_key],
        )
        return memory_ids[0] if memory_ids else ""

    async def _build_signature(
        self, record: AWMRunRecord
    ) -> Tuple[str, List[str]]:
        """Derive abstract signature and tokens from the trace."""
        action_tokens = [step.action for step in record.workflow_trace]
        condensed_actions = list(dict.fromkeys(action_tokens))
        heuristic_signature = " -> ".join(condensed_actions[:8])

        if not self.summarizer_llm:
            return heuristic_signature, condensed_actions

        prompt = (
            "You are summarizing an agent workflow trace for retrieval.\n"
            "Actions: "
            + ", ".join(action_tokens)
            + "\n"
            "Provide a short signature (max 12 words) capturing the workflow strategy."
        )
        try:
            response = await _call_llm_async(self.summarizer_llm, prompt)
            signature = response.strip().splitlines()[0]
            signature = signature or heuristic_signature
            return signature, condensed_actions
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("AWMExtractor signature generation failed: %s", exc)
            return heuristic_signature, condensed_actions

    async def _summarize_trace(self, record: AWMRunRecord, abstract_signature: str) -> str:
        """Generate high-level trajectory summary."""
        if not self.summarizer_llm:
            thoughts = [step.thought for step in record.workflow_trace[:3]]
            return (
                " | ".join(t.strip() for t in thoughts if t).strip()
                or f"Workflow followed signature {abstract_signature}."
            )

        steps_text = "\n".join(
            f"{idx+1}. Thought: {step.thought} | Action: {step.action} | Observation: {step.observation or ''}"
            for idx, step in enumerate(record.workflow_trace[:6])
        )
        prompt = (
            "Summarize the following workflow steps into a concise strategy overview (<=80 words):\n"
            f"Signature hint: {abstract_signature}\n"
            f"Steps:\n{steps_text}\n"
            "Summary:"
        )
        try:
            result = await _call_llm_async(self.summarizer_llm, prompt)
            return result.strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("AWMExtractor summary generation failed: %s", exc)
            return f"Workflow aligned with signature {abstract_signature}."

    async def _score_quality(
        self, record: AWMRunRecord
    ) -> Tuple[float, str, float, str, bool]:
        """Determine success/quality metrics, optionally via a judge LLM."""
        if record.success is not None:
            success = record.success
        elif record.metrics.get("auto_success") is not None:
            success = bool(record.metrics["auto_success"])
        else:
            success = True  # optimistic default

        if self.judge_llm:
            prompt = (
                "Evaluate the quality of this workflow run on a scale [0,1]. "
                "Return a JSON dictionary with keys score (float), label (string high/medium/low), "
                "confidence (float), rationale (string), success (bool).\n"
                f"User intent: {record.user_intent}\n"
                f"Generated output: {record.generated_output}\n"
                f"Evaluation notes: {record.evaluation_notes}\n"
                f"Failure reason: {record.failure_reason}\n"
                "Workflow steps:\n"
                + "\n".join(
                    f"{idx+1}. Thought={step.thought} | Action={step.action} | Observation={step.observation}"
                    for idx, step in enumerate(record.workflow_trace[:6])
                )
            )
            try:
                response = await _call_llm_async(self.judge_llm, prompt)
                parsed = self._safe_json_parse(response)
                if parsed:
                    return (
                        float(parsed.get("score", 0.7)),
                        parsed.get("label", "medium"),
                        float(parsed.get("confidence", 0.6)),
                        parsed.get("rationale", "").strip(),
                        bool(parsed.get("success", success)),
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("AWMExtractor quality judge failed: %s", exc)

        score = 0.85 if success else 0.35
        label = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
        rationale = record.evaluation_notes or (
            "Run succeeded with expected outputs." if success else f"Failure reason: {record.failure_reason}"
        )
        return score, label, 0.6, rationale, success

    @staticmethod
    def _safe_json_parse(raw: str) -> Optional[Dict[str, Any]]:
        try:
            import json

            cleaned = raw.strip().strip("`")
            return json.loads(cleaned)
        except Exception:
            return None


class AWMInducer:
    """
    Induces generalized workflow rules from a set of concrete examples.
    """

    def __init__(
        self,
        memory: LongTermMemory,
        llm: Optional[BaseLLM] = None,
        corpus_id: str = AWM_DEFAULT_CORPUS_ID,
    ):
        self.memory = memory
        self.llm = llm
        self.corpus_id = corpus_id

    async def induce(
        self,
        examples: Sequence[AWMConcreteExample],
        config: Optional[AWMInductionConfig] = None,
        auto_persist: bool = True,
    ) -> List[AWMSummaryRule]:
        if config is None:
            config = AWMInductionConfig()

        clusters: Dict[Tuple[str, str], List[AWMConcreteExample]] = defaultdict(list)
        for ex in examples:
            if config.template_id and ex.template_id != config.template_id:
                continue
            if config.abstract_signature and ex.abstract_signature != config.abstract_signature:
                continue
            if ex.quality_score < config.min_quality:
                continue
            if config.require_success and not ex.success:
                continue
            key = (ex.template_id, ex.abstract_signature)
            clusters[key].append(ex)

        summary_rules: List[AWMSummaryRule] = []
        for (template_id, signature), cluster_examples in clusters.items():
            cluster_examples = sorted(
                cluster_examples, key=lambda e: e.quality_score, reverse=True
            )[: config.max_examples_per_signature]
            success_rate = (
                sum(1 for e in cluster_examples if e.success) / len(cluster_examples)
            )
            confidence = sum(e.quality_score for e in cluster_examples) / len(cluster_examples)
            guidance, rationale, checklist, risks = await self._summarize_cluster(
                template_id, signature, cluster_examples, config.summary_style
            )
            rule = AWMSummaryRule(
                template_id=template_id,
                abstract_signature=signature,
                guidance=guidance,
                rationale=rationale,
                checklist=checklist,
                risk_notes=risks,
                confidence=confidence,
                success_rate=success_rate,
                support_example_ids=[ex.example_id for ex in cluster_examples],
                tags=list({tag for ex in cluster_examples for tag in ex.tags}),
                metadata={"cluster_size": len(cluster_examples)},
            )
            summary_rules.append(rule)
            if auto_persist:
                await self.persist(rule)

        return summary_rules

    async def persist(self, rule: AWMSummaryRule) -> str:
        markdown = rule.render_prompt_block()
        metadata_override = {
            "action": AWMMemoryType.SUMMARY.value,
            "wf_goal": rule.template_id,
            "wf_task": rule.abstract_signature,
            "wf_task_desc": "AWM Summary Rule",
            "custom_fields": {
                "awm_type": AWMMemoryType.SUMMARY.value,
                "template_id": rule.template_id,
                "abstract_signature": rule.abstract_signature,
                "confidence": rule.confidence,
                "success_rate": rule.success_rate,
                "payload": rule.model_dump(),
            },
        }
        message = Message(
            content=markdown,
            agent="AgentWorkflowMemory",
            action=AWMMemoryType.SUMMARY.value,
            msg_type=MessageType.RESPONSE,
            wf_goal=rule.template_id,
            wf_task=rule.abstract_signature,
            wf_task_desc="Induced workflow guidance",
        )
        memory_ids = self.memory.add(
            [message],
            corpus_id=self.corpus_id,
            metadata_overrides=[metadata_override],
            content_hashes=[_hash_signature([rule.template_id, rule.abstract_signature, rule.guidance])],
        )
        return memory_ids[0] if memory_ids else ""

    async def _summarize_cluster(
        self,
        template_id: str,
        signature: str,
        examples: Sequence[AWMConcreteExample],
        style: str,
    ) -> Tuple[str, str, List[str], Optional[str]]:
        if not self.llm:
            guidance = examples[0].trajectory_summary or signature
            rationale = (
                examples[0].quality_rationale
                or f"Derived from {len(examples)} successful executions."
            )
            checklist = [
                token for token in examples[0].signature_tokens[:5] if token and token.strip()
            ]
            risks = examples[0].failure_reason
            return guidance, rationale, checklist, risks

        example_blurbs = []
        for ex in examples:
            example_blurbs.append(
                f"- Quality {ex.quality_score:.2f} | Success={ex.success}\n"
                f"  Summary: {ex.trajectory_summary}\n"
                f"  Steps: {', '.join(ex.signature_tokens[:6])}\n"
                f"  Output: {(ex.generated_output or '')[:120]}"
            )
        prompt = (
            "You synthesize reusable workflow guidance for an agent.\n"
            f"Template: {template_id}\n"
            f"Abstract signature: {signature}\n"
            f"Style: {style}\n"
            "Examples:\n"
            + "\n".join(example_blurbs)
            + "\n"
            "Answer using the following four lines exactly:\n"
            "Guidance: <practical instructions>\n"
            "Rationale: <short why it works>\n"
            "Checklist: <comma separated key checkpoints>\n"
            "Risks: <potential pitfalls or 'None'>\n"
        )
        try:
            response = await _call_llm_async(self.llm, prompt)
            guidance, rationale, checklist, risks = self._parse_rule_response(response)
            return guidance, rationale, checklist, risks
        except Exception as exc:  # pragma: no cover
            logger.warning("AWMInducer cluster summarization failed: %s", exc)
            fallback = examples[0]
            checklist = [token for token in fallback.signature_tokens[:5] if token]
            return (
                fallback.trajectory_summary,
                fallback.quality_rationale or "Empirically observed from concrete examples.",
                checklist,
                fallback.failure_reason,
            )

    @staticmethod
    def _parse_rule_response(raw: str) -> Tuple[str, str, List[str], Optional[str]]:
        guidance = rationale = ""
        checklist: List[str] = []
        risks: Optional[str] = None
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("Guidance:"):
                guidance = stripped[len("Guidance:") :].strip()
            elif stripped.startswith("Rationale:"):
                rationale = stripped[len("Rationale:") :].strip()
            elif stripped.startswith("Checklist:"):
                checklist_text = stripped[len("Checklist:") :].strip()
                checklist = [item.strip() for item in checklist_text.split(",") if item.strip()]
            elif stripped.startswith("Risks:"):
                risks = stripped[len("Risks:") :].strip()
        return guidance, rationale, checklist, risks or None


class AWMPromptInjector:
    """
    Retrieves the most relevant summary rules and supporting examples to inject
    into the system prompt for a new workflow task.
    """

    def __init__(
        self,
        memory: LongTermMemory,
        corpus_id: str = AWM_DEFAULT_CORPUS_ID,
    ):
        self.memory = memory
        self.corpus_id = corpus_id

    async def build_memory_block(self, context: AWMPromptContext) -> str:
        query_terms = filter(None, [context.goal, context.task] + context.keywords)
        query = " | ".join(query_terms) or context.template_id

        # Retrieve more candidates for rules, then filter down; add a fallback if none remain after metadata filters.
        rule_chunks = await self.memory.search_async(
            query=query,
            n=max(context.max_rules * 10, 20),
            metadata_filters={
                "action": AWMMemoryType.SUMMARY.value,
                "wf_goal": context.template_id,
            },
            return_chunk=True,
            corpus_id=context.corpus_id,
        )
        if not rule_chunks:
            # Fallback: use template_id-only query and a larger candidate pool
            rule_chunks = await self.memory.search_async(
                query=context.template_id,
                n=50,
                metadata_filters={
                    "action": AWMMemoryType.SUMMARY.value,
                    "wf_goal": context.template_id,
                },
                return_chunk=True,
                corpus_id=context.corpus_id,
            )

        example_chunks = await self.memory.search_async(
            query=context.task or context.goal or context.template_id,
            n=max(context.max_examples * 5, 10),
            metadata_filters={
                "action": AWMMemoryType.CONCRETE.value,
                "wf_goal": context.template_id,
            },
            return_chunk=True,
            corpus_id=context.corpus_id,
        )

        summary_rules = self._decode_rules(rule_chunks)
        examples = self._decode_examples(example_chunks)

        blocks: List[str] = ["### Agent Workflow Memory"]
        if summary_rules:
            blocks.append("## Summary Rules")
            for rule in summary_rules[: context.max_rules]:
                blocks.append(rule.render_prompt_block())
        else:
            blocks.append("## Summary Rules\n(No matching rules found)")

        if examples:
            blocks.append("## Concrete Examples")
            for idx, example in enumerate(examples[: context.max_examples], start=1):
                blocks.append(f"### Example {idx}\n{example.prompt_excerpt()}")
        else:
            blocks.append("## Concrete Examples\n(No recent runs logged)")

        return "\n\n".join(blocks)

    def _decode_examples(self, chunks: Sequence[Tuple[Any, str]]) -> List[AWMConcreteExample]:
        decoded: List[AWMConcreteExample] = []
        for chunk, _ in chunks:
            payload = getattr(chunk.metadata, "custom_fields", {}).get("payload")
            if not payload:
                continue
            try:
                decoded.append(AWMConcreteExample.model_validate(payload))
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to parse AWM concrete example payload: %s", exc)
        return decoded

    def _decode_rules(self, chunks: Sequence[Tuple[Any, str]]) -> List[AWMSummaryRule]:
        decoded: List[AWMSummaryRule] = []
        for chunk, _ in chunks:
            payload = getattr(chunk.metadata, "custom_fields", {}).get("payload")
            if not payload:
                continue
            try:
                decoded.append(AWMSummaryRule.model_validate(payload))
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to parse AWM summary rule payload: %s", exc)
        return decoded


@dataclass
class AgentWorkflowMemoryToolkit:
    """
    Convenience wrapper that bundles extractor, inducer and injector around
    a shared LongTermMemory instance.
    """

    extractor: AWMExtractor
    inducer: AWMInducer
    injector: AWMPromptInjector

    @classmethod
    def from_memory(
        cls,
        memory: LongTermMemory,
        corpus_id: str = AWM_DEFAULT_CORPUS_ID,
        summarizer_llm: Optional[BaseLLM] = None,
        judge_llm: Optional[BaseLLM] = None,
        inducer_llm: Optional[BaseLLM] = None,
    ) -> "AgentWorkflowMemoryToolkit":
        extractor = AWMExtractor(
            memory=memory,
            corpus_id=corpus_id,
            summarizer_llm=summarizer_llm,
            judge_llm=judge_llm or summarizer_llm,
        )
        inducer = AWMInducer(
            memory=memory,
            llm=inducer_llm or summarizer_llm,
            corpus_id=corpus_id,
        )
        injector = AWMPromptInjector(memory=memory, corpus_id=corpus_id)
        return cls(extractor=extractor, inducer=inducer, injector=injector)


class AWMMemoryPolicy(BaseMemoryPolicy):
    """Custom policy that interprets prompt contexts for the AWM backend."""

    def _infer_corpus_id(self, agent: CustomizeMemoryAgent) -> str:
        backend = next(iter(agent.memory_backends.values()), None)
        if backend and hasattr(backend, "corpus_id"):
            return getattr(backend, "corpus_id")
        return AWM_DEFAULT_CORPUS_ID

    def _coerce_context(self, value: Any, agent: CustomizeMemoryAgent) -> Optional[AWMPromptContext]:
        if value is None:
            return None
        if isinstance(value, AWMPromptContext):
            return value
        if isinstance(value, dict):
            payload = {"corpus_id": self._infer_corpus_id(agent)}
            payload.update(value)
            try:
                return AWMPromptContext.model_validate(payload)
            except ValidationError:
                logger.warning("AWMMemoryPolicy: invalid prompt context payload: %s", value)
        return None

    def prepare_query(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        **kwargs: Any,
    ) -> Optional[Any]:
        context = self._coerce_context(run_context.metadata.get("prompt_context"), agent)
        if context:
            return context

        if run_context.inputs:
            context = self._coerce_context(run_context.inputs.get("awm_prompt_context"), agent)
            if context:
                return context

        return super().prepare_query(agent=agent, run_context=run_context, **kwargs)


class AWMMemoryBackend(BaseMemoryBackend):
    """Memory backend that wires AWM components into the CustomizeMemoryAgent lifecycle."""

    memory: LongTermMemory
    corpus_id: str = Field(default=AWM_DEFAULT_CORPUS_ID)
    summarizer_llm: Optional[BaseLLM] = None
    judge_llm: Optional[BaseLLM] = None
    inducer_llm: Optional[BaseLLM] = None
    induction_min_quality: float = Field(default=0.55, ge=0.0, le=1.0)
    name: str = Field(default="awm_memory")
    priority: int = Field(default=10)
    supported_operations: Tuple[MemoryOperation, ...] = (
        MemoryOperation.RETRIEVE,
        MemoryOperation.INJECT,
        MemoryOperation.REFLECT,
        MemoryOperation.PERSIST,
    )

    def init_module(self):
        super().init_module()
        self.extractor = AWMExtractor(
            memory=self.memory,
            corpus_id=self.corpus_id,
            summarizer_llm=self.summarizer_llm,
            judge_llm=self.judge_llm or self.summarizer_llm,
        )
        self.inducer = AWMInducer(
            memory=self.memory,
            llm=self.inducer_llm or self.summarizer_llm,
            corpus_id=self.corpus_id,
        )
        self.injector = AWMPromptInjector(memory=self.memory, corpus_id=self.corpus_id)

    async def on_retrieve(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        query: Any,
        top_k: int,
        **_: Any,
    ) -> Any:
        context = self._normalize_context(query, run_context)
        if context is None:
            logger.debug("AWMMemoryBackend: retrieval skipped due to missing context.")
            return {}

        summary_rules, examples = await self._collect_memories(context, top_k)
        payload = {
            "context": context,
            "summary_rules": summary_rules,
            "examples": examples,
        }
        run_context.metadata.setdefault("awm_memory", {})["retrieval"] = payload
        return payload

    async def on_inject(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        retrieved: Any,
        **_: Any,
    ) -> Any:
        if not retrieved:
            return None
        context = retrieved.get("context")
        summary_rules = retrieved.get("summary_rules", [])
        examples = retrieved.get("examples", [])
        if context is None:
            logger.debug("AWMMemoryBackend: injection skipped (no context).")
            return None

        block = self._render_memory_block(context, summary_rules, examples)
        run_context.inputs.setdefault("memory_context", block)
        run_context.metadata.setdefault("awm_memory", {})["prompt_block"] = block
        return block

    async def on_reflect(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        execution_result: Message,
        **_: Any,
    ) -> Any:
        record = self._resolve_run_record(run_context, execution_result)
        if record is None:
            logger.debug("AWMMemoryBackend: no run record available for reflection.")
            return None

        example = await self.extractor.extract(record, auto_persist=False)
        should_induce = bool(record.success) and example.quality_score >= self.induction_min_quality
        payload = {"record": record, "example": example, "should_induce": should_induce}
        run_context.metadata.setdefault("awm_memory", {})["reflection"] = payload
        return payload

    async def on_persist(
        self,
        agent: CustomizeMemoryAgent,
        run_context: MemoryRunContext,
        reflection: Any,
        **_: Any,
    ) -> Any:
        if not reflection:
            logger.debug("AWMMemoryBackend: persist skipped (empty reflection payload).")
            return {}

        example: Optional[AWMConcreteExample] = reflection.get("example")
        if example is None:
            logger.debug("AWMMemoryBackend: persist skipped (no example).")
            return {}

        result: Dict[str, Any] = {}
        example_id = await self.extractor.persist(example)
        result["example_id"] = example_id

        if reflection.get("should_induce"):
            try:
                summary_rules = await self.inducer.induce([example])
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("AWMMemoryBackend: induction failed: %s", exc)
                summary_rules = []

            rule_ids: List[str] = []
            for rule in summary_rules:
                rule_id = await self.inducer.persist(rule)
                if rule_id:
                    rule_ids.append(rule_id)

            if rule_ids:
                result["summary_rule_ids"] = rule_ids

        run_context.metadata.setdefault("awm_memory", {})["persist"] = result
        return result

    def _normalize_context(self, query: Any, run_context: MemoryRunContext) -> Optional[AWMPromptContext]:
        context = self._coerce_prompt_context(query)
        if context:
            return context

        context = self._coerce_prompt_context(run_context.metadata.get("prompt_context"))
        if context:
            return context

        context = self._coerce_prompt_context(run_context.inputs.get("awm_prompt_context"))
        if context:
            return context

        if isinstance(query, str) and query:
            return AWMPromptContext(template_id=query, corpus_id=self.corpus_id)
        return None

    def _coerce_prompt_context(self, value: Any) -> Optional[AWMPromptContext]:
        if value is None:
            return None
        if isinstance(value, AWMPromptContext):
            return value
        if isinstance(value, dict):
            payload = {"corpus_id": value.get("corpus_id", self.corpus_id)}
            payload.update(value)
            try:
                return AWMPromptContext.model_validate(payload)
            except ValidationError:
                logger.warning("AWMMemoryBackend: invalid prompt context: %s", value)
        return None

    async def _collect_memories(
        self,
        context: AWMPromptContext,
        top_k: int,
    ) -> Tuple[List[AWMSummaryRule], List[AWMConcreteExample]]:
        rule_limit = context.max_rules or 2
        example_limit = context.max_examples or 2
        if top_k:
            rule_limit = min(rule_limit, top_k)
            example_limit = min(example_limit, top_k)

        query_terms = [context.goal, context.task] + context.keywords
        rule_query = " | ".join(filter(None, query_terms)) or context.template_id

        rule_chunks = await self.memory.search_async(
            query=rule_query,
            n=max(rule_limit * 5, 20),
            metadata_filters={
                "action": AWMMemoryType.SUMMARY.value,
                "wf_goal": context.template_id,
            },
            return_chunk=True,
            corpus_id=context.corpus_id,
        )
        if not rule_chunks:
            rule_chunks = await self.memory.search_async(
                query=context.template_id,
                n=max(rule_limit * 5, 20),
                metadata_filters={
                    "action": AWMMemoryType.SUMMARY.value,
                    "wf_goal": context.template_id,
                },
                return_chunk=True,
                corpus_id=context.corpus_id,
            )

        example_chunks = await self.memory.search_async(
            query=context.task or context.goal or context.template_id,
            n=max(example_limit * 5, 10),
            metadata_filters={
                "action": AWMMemoryType.CONCRETE.value,
                "wf_goal": context.template_id,
            },
            return_chunk=True,
            corpus_id=context.corpus_id,
        )

        summary_rules = self.injector._decode_rules(rule_chunks)[:rule_limit]
        examples = self.injector._decode_examples(example_chunks)[:example_limit]
        return summary_rules, examples

    def _render_memory_block(
        self,
        context: AWMPromptContext,
        summary_rules: Sequence[AWMSummaryRule],
        examples: Sequence[AWMConcreteExample],
    ) -> str:
        blocks: List[str] = ["### Agent Workflow Memory"]
        if summary_rules:
            blocks.append("## Summary Rules")
            for rule in summary_rules:
                blocks.append(rule.render_prompt_block())
        else:
            blocks.append("## Summary Rules\n(No matching rules found)")

        if examples:
            blocks.append("## Concrete Examples")
            for idx, example in enumerate(examples, start=1):
                blocks.append(f"### Example {idx}\n{example.prompt_excerpt()}")
        else:
            blocks.append("## Concrete Examples\n(No recent runs logged)")
        return "\n\n".join(blocks)

    def _resolve_run_record(
        self,
        run_context: MemoryRunContext,
        execution_result: Message,
    ) -> Optional[AWMRunRecord]:
        for candidate in (
            run_context.metadata.get("run_record"),
            run_context.metadata.get("awm_run_record"),
            execution_result.content,
        ):
            record = self._coerce_run_record(candidate)
            if record:
                return record
        return None

    def _coerce_run_record(self, value: Any) -> Optional[AWMRunRecord]:
        if value is None:
            return None
        if isinstance(value, AWMRunRecord):
            return value
        if isinstance(value, dict):
            try:
                return AWMRunRecord.model_validate(value)
            except ValidationError:
                logger.warning("AWMMemoryBackend: invalid run record payload: %s", value)
        return None


def create_awm_memory_agent(
    memory: LongTermMemory,
    *,
    name: str = "awm_memory_agent",
    description: str = "CustomizeMemoryAgent with Agent Workflow Memory integration.",
    corpus_id: str = AWM_DEFAULT_CORPUS_ID,
    prompt: Optional[str] = None,
    llm_config: Optional[LLMConfig] = None,
    summarizer_llm: Optional[BaseLLM] = None,
    judge_llm: Optional[BaseLLM] = None,
    inducer_llm: Optional[BaseLLM] = None,
    induction_min_quality: float = 0.55,
    backend_name: str = "awm_memory",
    backend_priority: int = 10,
    default_top_k: int = 3,
    memory_policy: Optional[BaseMemoryPolicy] = None,
    memory_orchestrator: Optional[MemoryOrchestrator] = None,
    inputs: Optional[List[dict]] = None,
    outputs: Optional[List[dict]] = None,
    **agent_kwargs: Any,
) -> CustomizeMemoryAgent:
    """
    Helper to instantiate a CustomizeMemoryAgent pre-wired with the AWM backend.
    """

    backend = AWMMemoryBackend(
        name=backend_name,
        priority=backend_priority,
        memory=memory,
        corpus_id=corpus_id,
        summarizer_llm=summarizer_llm,
        judge_llm=judge_llm,
        inducer_llm=inducer_llm,
        induction_min_quality=induction_min_quality,
    )
    policy = memory_policy or AWMMemoryPolicy(default_top_k=default_top_k)
    orchestrator = memory_orchestrator or MemoryOrchestrator()

    base_prompt = prompt or (
        "You are an agent workflow memory assistant. Leverage retrieved examples and rules to respond succinctly.\n\n"
        "User request:\n{user_input}\n\n"
        "Memory context:\n{memory_context}"
    )
    prompt_template = PromptTemplate(instruction=base_prompt)
    inputs = inputs or [
        {"name": "user_input", "type": "str", "description": "Primary user instruction."},
        {"name": "memory_context", "type": "str", "description": "Injected memory block.", "required": False},
    ]
    outputs = outputs or [
        {"name": "response", "type": "str", "description": "Agent response."},
    ]

    agent_kwargs.setdefault("parse_mode", "str")
    agent_kwargs.setdefault("is_human", True)

    return CustomizeMemoryAgent(
        name=name,
        description=description,
        prompt=None,
        prompt_template=prompt_template,
        llm_config=llm_config,
        inputs=inputs,
        outputs=outputs,
        memory_backends={backend.name: backend},
        memory_policies={"default": policy},
        memory_orchestrator=orchestrator,
        storage_handler=memory.storage_handler,
        **agent_kwargs,
    )
