import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from evoagentx.agents import AgentManager
from evoagentx.agents.customize_memory_agent import CustomizeMemoryAgent
from evoagentx.core.message import Message, MessageType
from evoagentx.memory.agent_memory_workflow import (
    AWMPromptContext,
    AWMRunRecord,
    AWMTraceEvent,
    AWMMemoryType,
    create_awm_memory_agent,
)
from evoagentx.memory.long_term_memory import LongTermMemory
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.rag.rag_config import (
    ChunkerConfig,
    EmbeddingConfig,
    IndexConfig,
    RAGConfig,
    ReaderConfig,
    RetrievalConfig,
)
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import DBConfig, StoreConfig, VectorStoreConfig
from evoagentx.tools import ArxivToolkit
from evoagentx.tools.file_tool import FileToolkit
from evoagentx.workflow import WorkFlow, WorkFlowGenerator, WorkFlowGraph
from evoagentx.workflow.environment import TrajectoryState

load_dotenv()


TEMPLATE_ID = "arxiv_daily_digest"
AWM_CORPUS_ID = "awm_workflows"
OUTPUT_ROOT = Path("EvoAgentX/examples/output/awm_memory")
WORKFLOW_ROOT = OUTPUT_ROOT / "workflows"

TRAIN_SCENARIOS = [
    {
        "keywords": "medical, multiagent",
        "max_results": 8,
        "date_from": "2024-01-01",
        "categories": ["cs.AI", "cs.LG"],
        "label": "medical_multiagent_digest",
    },
    {
        "keywords": "reinforcement learning triage",
        "max_results": 6,
        "date_from": "2023-12-01",
        "categories": ["cs.AI", "cs.LG", "cs.CY"],
        "label": "rl_triage_digest",
    },
]

TEST_SCENARIO = {
    "keywords": "neurosymbolic planning healthcare",
    "max_results": 6,
    "date_from": "2024-01-01",
    "categories": ["cs.AI", "cs.RO"],
    "label": "neurosymbolic_planning_digest",
}


def _build_memory_components(openai_api_key: str) -> LongTermMemory:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    store_config = StoreConfig(
        dbConfig=DBConfig(
            db_name="sqlite",
            path=str(OUTPUT_ROOT / "awm_memory.sqlite"),
        ),
        vectorConfig=VectorStoreConfig(
            vector_name="faiss",
            dimensions=1536,
            index_type="flat_l2",
        ),
        graphConfig=None,
        path=str(OUTPUT_ROOT / "index"),
    )
    storage_handler = StorageHandler(storageConfig=store_config)

    rag_config = RAGConfig(
        reader=ReaderConfig(),
        chunker=ChunkerConfig(strategy="simple", chunk_size=768, chunk_overlap=64),
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-small",
            api_key=openai_api_key,
        ),
        index=IndexConfig(index_type="vector"),
        retrieval=RetrievalConfig(retrivel_type="vector", top_k=5, similarity_cutoff=0.3),
    )

    ltm = LongTermMemory(storage_handler=storage_handler, rag_config=rag_config)
    ltm.init_module()
    return ltm


def _create_agent(memory: LongTermMemory) -> CustomizeMemoryAgent:
    return create_awm_memory_agent(
        memory=memory,
        name="arxiv_memory_agent",
        description="Arxiv workflow agent with Agent Workflow Memory integration.",
        corpus_id=AWM_CORPUS_ID,
    )


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def _build_goal(keywords: str, max_results: int, date_from: str, categories: List[str]) -> str:
    search_constraints = f"""
    Search constraints:
    - Query keywords: {keywords}
    - Max results: {max_results}
    - Date from: {date_from}
    - Categories: {', '.join(categories)}
    """

    return f"""Create a daily research paper recommendation assistant that takes user keywords and pushes new relevant papers with summaries.

The assistant should:
1. Use the ArxivToolkit to search for the latest papers using the given keywords.
2. Apply the following search constraints:
{search_constraints}
3. Summarize the search results.
4. Compile the summaries into a well-formatted Markdown digest.

### Output
daily_paper_digest
"""


def _safe_str(value: Any, max_len: int = 1500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            s = str(value)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _create_workflow_instance(
    goal: str,
    openai_config: OpenAILLMConfig,
    module_path: Path,
    tools: List[Any],
) -> WorkFlow:
    module_path.parent.mkdir(parents=True, exist_ok=True)
    llm = OpenAILLM(config=openai_config)

    if module_path.exists():
        workflow_graph = WorkFlowGraph.from_file(str(module_path))
    else:
        wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
        workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
        workflow_graph.save_module(str(module_path))

    # Ensure prompts include {goal} when declared in inputs to satisfy CustomizeAgent validation
    _ensure_goal_placeholder(workflow_graph)

    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    return workflow


def _ensure_goal_placeholder(workflow_graph: WorkFlowGraph) -> None:
    """If a node declares an input named 'goal' but its agent prompt lacks '{goal}',
    append a small 'User Goal' section to the end of the prompt.

    This preserves existing prompt content while satisfying CustomizeAgent's
    placeholder validation for raw string prompts.
    """
    try:
        for node in getattr(workflow_graph, "nodes", []) or []:
            # Collect input names from node definition (Parameter or dict)
            input_names: List[str] = []
            for p in getattr(node, "inputs", []) or []:
                name = getattr(p, "name", None)
                if name is None and isinstance(p, dict):
                    name = p.get("name")
                if isinstance(name, str):
                    input_names.append(name)
            if "goal" not in input_names:
                continue

            agents = getattr(node, "agents", None)
            if not agents:
                continue
            for agent in agents:
                if not isinstance(agent, dict):
                    continue
                # If using a template, skip; the strict check applies to raw prompts
                if agent.get("prompt_template"):
                    continue
                prompt = agent.get("prompt")
                if isinstance(prompt, str) and "{goal}" not in prompt:
                    agent["prompt"] = prompt.rstrip() + "\n\nUser Goal:\n{goal}\n"
    except Exception:
        # Best-effort patch; ignore issues silently
        pass


def _workflow_to_run_record(
    workflow: WorkFlow,
    goal: str,
    output: str,
    success: bool,
    scenario: Dict[str, Any],
    run_stage: str,
) -> AWMRunRecord:
    trace_events: List[AWMTraceEvent] = []
    for step in workflow.environment.trajectory:
        msg = step.message
        if msg is None:
            continue
        thought = _safe_str(msg.prompt) if msg.prompt else ""
        observation = _safe_str(msg.content)
        action_name = msg.action or (msg.agent or "workflow")
        trace_events.append(
            AWMTraceEvent(
                thought=thought,
                action=action_name,
                action_input=thought or None,
                observation=observation,
                status=("ok" if step.status == TrajectoryState.COMPLETED else "fail"),
                metadata={
                    "agent": msg.agent,
                    "wf_task": msg.wf_task,
                    "msg_type": msg.msg_type.value if msg.msg_type else None,
                },
            )
        )

    tags = ["arxiv", run_stage] + scenario.get("categories", [])
    metadata = {
        "keywords": scenario["keywords"],
        "max_results": scenario["max_results"],
        "date_from": scenario["date_from"],
        "run_stage": run_stage,
    }

    return AWMRunRecord(
        template_id=TEMPLATE_ID,
        template_version="v1",
        workflow_name="Daily Arxiv Digest",
        user_intent=goal.strip(),
        wf_goal=goal.strip(),
        domain="research_digest",
        site="arxiv.org",
        tags=tags,
        workflow_trace=trace_events,
        generated_output=output,
        success=success,
        failure_reason=None if success else "Workflow execution failed",
        evaluation_notes=None,
        metrics={"auto_success": success},
        metadata=metadata,
    )


async def _run_workflow_scenario(
    scenario: Dict[str, Any],
    openai_config: OpenAILLMConfig,
    stage: str,
) -> Tuple[AWMRunRecord, str, bool]:
    goal = _build_goal(
        keywords=scenario["keywords"],
        max_results=scenario["max_results"],
        date_from=scenario["date_from"],
        categories=scenario["categories"],
    )
    module_path = WORKFLOW_ROOT / f"{_slugify(scenario['label'])}.json"
    tools = [ArxivToolkit(), FileToolkit()]

    workflow = _create_workflow_instance(goal, openai_config, module_path, tools)
    output = await workflow.async_execute()
    success = output is not None and output != "Workflow Execution Failed"
    run_record = _workflow_to_run_record(
        workflow=workflow,
        goal=goal,
        output=_safe_str(output),
        success=success,
        scenario=scenario,
        run_stage=stage,
    )
    return run_record, _safe_str(output), success


async def _ingest_run_records(agent: CustomizeMemoryAgent, records: List[AWMRunRecord]) -> None:
    backend_key = next(iter(agent.memory_backends.keys()))

    for record in records:
        print(f"  - Recording run {record.run_id[:8]} | success={record.success}")
        message = Message(
            content=record,
            agent=agent.name,
            action="arxiv_workflow.completed",
            msg_type=MessageType.RESPONSE,
            wf_goal=record.wf_goal,
            wf_task=record.metadata.get("run_stage", "arxiv_workflow"),
            wf_task_desc="Arxiv workflow execution",
        )

        reflection = await agent.reflect_and_learn(message, metadata={"run_record": record})
        backend_reflection = reflection.get(backend_key) if reflection else None
        if backend_reflection:
            example = backend_reflection.get("example")
            if example:
                print(
                    f"    Derived signature: {example.abstract_signature} "
                    f"(quality {example.quality_score:.2f})"
                )

        persist_results = await agent.persist(reflection_outputs=reflection or {}, metadata={"run_record": record})
        backend_persist = persist_results.get(backend_key, {})
        print(
            f"    Stored example_id={backend_persist.get('example_id', 'n/a')} "
            f"| summary_rules={len(backend_persist.get('summary_rule_ids', []))}"
        )


async def _render_prompt_block(agent: CustomizeMemoryAgent, scenario: Dict[str, Any]) -> str:
    context = AWMPromptContext(
        template_id=TEMPLATE_ID,
        goal="Create today's multi-agent medical research digest.",
        task="Summarize fresh Arxiv submissions on clinical decision making.",
        keywords=scenario["keywords"].split(","),
        domain="research_digest",
        site="arxiv.org",
        corpus_id=AWM_CORPUS_ID,
    )
    retrieved = await agent.retrieve(context, metadata={"prompt_context": context})
    enriched_inputs = await agent.inject_context(
        inputs={"goal": context.goal or "", "task": context.task or "", "keywords": scenario["keywords"]},
        metadata={"prompt_context": context},
        retrieved=retrieved,
    )
    return enriched_inputs.get("memory_context", "")


async def _display_memory(memory: LongTermMemory, corpus_id: str = AWM_CORPUS_ID) -> None:
    """Print a compact summary of what was persisted in memory for this template, grouped by stage.

    It lists all concrete examples (with stage train/test inferred from payload.metadata.run_stage)
    and all summary rules (with stage inferred from supporting examples when possible).

    Section header: "=== memory display ===".
    """
    try:
        # Fetch concrete examples and summary rules for this template
        example_chunks = await memory.search_async(
            query=TEMPLATE_ID,
            n=200,
            metadata_filters={
                "action": AWMMemoryType.CONCRETE.value,
                "wf_goal": TEMPLATE_ID,
            },
            return_chunk=True,
            corpus_id=corpus_id,
        )
        rule_chunks = await memory.search_async(
            query=TEMPLATE_ID,
            n=200,
            metadata_filters={
                "action": AWMMemoryType.SUMMARY.value,
                "wf_goal": TEMPLATE_ID,
            },
            return_chunk=True,
            corpus_id=corpus_id,
        )

        # Build example_id -> stage map and group examples by stage
        examples_by_stage: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": [], "unknown": []}
        example_stage_map: Dict[str, str] = {}

        for chunk, _mid in example_chunks:
            cf = getattr(chunk.metadata, "custom_fields", {}) or {}
            payload = cf.get("payload") or {}
            ex_id = payload.get("example_id") or ""
            stage = (payload.get("metadata", {}) or {}).get("run_stage") or "unknown"
            example_stage_map[ex_id] = stage
            examples_by_stage.setdefault(stage, []).append(payload)

        # Infer rule stage from support_example_ids
        rules_grouped: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": [], "mixed": [], "unknown": []}
        for chunk, _mid in rule_chunks:
            cf = getattr(chunk.metadata, "custom_fields", {}) or {}
            payload = cf.get("payload") or {}
            support_ids = payload.get("support_example_ids", []) or []
            stages = {example_stage_map.get(eid, "unknown") for eid in support_ids}
            if not stages:
                label = "unknown"
            elif len(stages) == 1:
                label = next(iter(stages))
            else:
                label = "mixed"
            rules_grouped.setdefault(label, []).append(payload)

        # Render
        print("\n=== Memory Display ===")

        # Concrete Examples by stage
        for stage in ("train", "test", "unknown"):
            items = examples_by_stage.get(stage, [])
            print(f"\n[examples:{stage}] count={len(items)}")
            for ex in items:
                sig = ex.get("abstract_signature", "")
                q = ex.get("quality_score", 0.0)
                succ = ex.get("success", False)
                ex_id = ex.get("example_id", "")
                summary = (ex.get("trajectory_summary") or "").strip()
                if summary:
                    summary = (summary[:120] + "...") if len(summary) > 120 else summary
                print(f"  - example_id={ex_id} | success={succ} | quality={q:.2f} | signature={sig}")
                if summary:
                    print(f"      summary: {summary}")

        # Summary Rules grouped by inferred stage
        print("\n[summary_rules]")
        for stage in ("train", "test", "mixed", "unknown"):
            rules = rules_grouped.get(stage, [])
            if not rules:
                continue
            print(f"  ({stage}) count={len(rules)}")
            for rule in rules:
                rid = rule.get("rule_id", "")
                sig = rule.get("abstract_signature", "")
                guidance = (rule.get("guidance") or "").strip()
                if guidance:
                    guidance = (guidance[:100] + "...") if len(guidance) > 100 else guidance
                supports = rule.get("support_example_ids", []) or []
                print(f"    - rule_id={rid} | signature={sig} | supports={len(supports)}")
                if guidance:
                    print(f"        guidance: {guidance}")
    except Exception as exc:  # best-effort reporting
        print(f"\n=== memory display ===\n(error while displaying memory: {exc})")


async def main():
    parser = argparse.ArgumentParser(description="AWM example (train/test/demo)")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "demo"],
        default="demo",
        help="demo=train+test in one run",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use for workflow generation and execution.",
    )
    args = parser.parse_args()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in your environment to run this example.")

    openai_config = OpenAILLMConfig(
        model=args.model,
        openai_key=openai_api_key,
        stream=False,
        output_response=True,
        max_tokens=8000,
    )

    long_term_memory = _build_memory_components(openai_api_key)
    agent = _create_agent(long_term_memory)

    db_path = OUTPUT_ROOT / "awm_memory.sqlite"

    if args.mode in ("train", "demo"):
        print("Generating workflow runs and seeding memory...")
        train_records: List[AWMRunRecord] = []
        for scenario in TRAIN_SCENARIOS:
            record, output, success = await _run_workflow_scenario(scenario, openai_config, stage="train")
            print(f"\n=== Training Digest ({scenario['label']}) ===")
            print(output)
            print(f"Success: {success}\n")
            train_records.append(record)
        await _ingest_run_records(agent, train_records)
        long_term_memory.save(corpus_id=AWM_CORPUS_ID)
        print(f"Saved memory index and DB for corpus '{AWM_CORPUS_ID}' to: {db_path}")
        if args.mode == "train":
            # Show memory contents after training-only runs
            await _display_memory(long_term_memory, corpus_id=AWM_CORPUS_ID)

    if args.mode in ("test", "demo"):
        restored = await long_term_memory.aload(corpus_id=AWM_CORPUS_ID)
        print(
            f"Loaded {len(restored) if isinstance(restored, list) else 'N/A'} memory ids for "
            f"corpus '{AWM_CORPUS_ID}' from: {db_path}"
        )

        test_record, test_output, test_success = await _run_workflow_scenario(
            TEST_SCENARIO, openai_config, stage="test"
        )
        print("\n=== Test Digest (fresh run) ===")
        print(test_output)
        print(f"Success: {test_success}\n")

        evaluation_agent = _create_agent(long_term_memory)
        prompt_block = await _render_prompt_block(evaluation_agent, TEST_SCENARIO)

        print("Prompt injection block derived from memory:\n")
        print(prompt_block)

        await _ingest_run_records(agent, [test_record])
    # Final memory overview (train + test if in demo)
    await _display_memory(long_term_memory, corpus_id=AWM_CORPUS_ID)


if __name__ == "__main__":
    asyncio.run(main())
