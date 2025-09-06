from typing import Dict, Any, List, Optional, Tuple
import random
import json
import os
from pydantic import Field

from ...workflow.action_graph import ActionGraph
from ...models.model_configs import LLMConfig, OpenAILLMConfig
from ...models.base_model import LLMOutputParser
from ...workflow.operators import QAScEnsemble
from .utils import (
    format_transcript,
    collect_last_round_candidates,
    collect_round_candidates,
)
from ...prompts.workflow.multi_agent_debate import (
    DEBATER_AGENT_PROMPT,
    JUDGE_AGENT_PROMPT,
    build_agent_prompt,
    build_judge_prompt,
    get_default_personas,
)
from .pruning import PruningPipeline
from ...agents.customize_agent import CustomizeAgent


class DebateAgentOutput(LLMOutputParser):
    """Output structure for individual debater in a round."""

    thought: str = Field(default="", description="Thinking process")
    argument: str = Field(default="", description="Argument or rebuttal for this round")
    answer: Optional[str] = Field(default=None, description="Current answer for this round (optional)")


class DebateJudgeOutput(LLMOutputParser):
    """Final judgment from judge after debate."""

    rationale: str = Field(default="", description="Judging rationale")
    winning_agent_id: int = Field(default=0, description="Winning debater ID (starting from 0)")
    final_answer: str = Field(default="", description="Final answer")


class MultiAgentDebateActionGraph(ActionGraph):
    """Multi-Agent Debate ActionGraph implementation (Google MAD style)."""

    name: str = "MultiAgentDebate"
    description: str = "Multi-agent debate workflow framework"
    # Unified LLM configuration: default configuration for all agents
    llm_config: LLMConfig = Field(default_factory=lambda: OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=os.getenv("OPENAI_API_KEY")
    ), description="Default LLM configuration for all agents")
    # Unified agent configuration: all debaters use the same agent configuration
    debater_agents: Optional[List[CustomizeAgent]] = Field(default=None, description="Optional: multiple debater CustomizeAgents, randomly selected during execution")
    judge_agent: Optional[CustomizeAgent] = Field(default=None, description="Optional: judge CustomizeAgent, used for judging phase if provided")
    # Optional model pool: for random selection of different models
    llm_config_pool: Optional[List[LLMConfig]] = Field(default=None, description="Optional: LLM configuration pool for random selection, provides choices for agents without specified models")
    # Group workflow mode (each position occupied by an independent workflow graph instead of agent)
    group_graphs_enabled: bool = Field(default=False, description="Enable group graph mode: replace individual debaters with workflow graphs")
    group_graphs: Optional[List[ActionGraph]] = Field(default=None, description="When group graph mode is enabled, provide workflow graph list (length >= 1)")

    # Runtime attributes
    _sc_ensemble: Optional[QAScEnsemble] = None

    def init_module(self):
        """Initialize module (create LLM, construct reusable operators)."""
        super().init_module()
        
        # Configuration conflict and integrity validation
        if self.group_graphs_enabled and self.debater_agents:
            raise ValueError(
                "Configuration conflict: cannot configure debater_agents when group_graphs_enabled is enabled."
            )
        if self.group_graphs_enabled and (not self.group_graphs or len(self.group_graphs) == 0):
            raise ValueError(
                "Configuration error: must provide non-empty group_graphs list when group graph mode is enabled."
            )
        if (not self.group_graphs_enabled) and self.group_graphs:
            raise ValueError(
                "Configuration error: provided group_graphs but did not enable group_graphs_enabled. Please enable both or remove group_graphs."
            )
        
        self._sc_ensemble = QAScEnsemble(self._llm)
           
    def _create_default_debater_agent(self) -> CustomizeAgent:
        """Create default debater CustomizeAgent (XML parsing thought/argument/answer)."""
        llm_config = random.choice(self.llm_config_pool) if self.llm_config_pool else self.llm_config
        
        return CustomizeAgent(
            name="DebaterAgent",
            description="Generate argument/rebuttal and optional answer per debate round.",
            prompt=DEBATER_AGENT_PROMPT,
            llm_config=llm_config,
            inputs=[
                {"name": "problem", "type": "str", "description": "Problem statement"},
                {"name": "transcript_text", "type": "str", "description": "Formatted debate transcript so far"},
                {"name": "role", "type": "str", "description": "Debater role/persona"},
                {"name": "agent_id", "type": "str", "description": "Debater id (string)"},
                {"name": "round_index", "type": "str", "description": "1-based round index"},
                {"name": "total_rounds", "type": "str", "description": "Total rounds"},
            ],
            outputs=[
                {"name": "thought", "type": "str", "description": "Brief reasoning", "required": True},
                {"name": "argument", "type": "str", "description": "Argument or rebuttal", "required": True},
                {"name": "answer", "type": "str", "description": "Optional current answer", "required": False},
            ],
            parse_mode="xml",
        )

    def _create_default_judge_agent(self) -> CustomizeAgent:
        """Create default judge CustomizeAgent (XML parsing rationale/winning_agent_id/final_answer)."""
        llm_config = random.choice(self.llm_config_pool) if self.llm_config_pool else self.llm_config
        
        return CustomizeAgent(
            name="JudgeAgent",
            description="Deliver final decision and answer based on debate transcript.",
            prompt=JUDGE_AGENT_PROMPT,
            llm_config=llm_config,
            inputs=[
                {"name": "problem", "type": "str", "description": "Problem statement"},
                {"name": "transcript_text", "type": "str", "description": "Formatted debate transcript"},
                {"name": "roles_text", "type": "str", "description": "Roles listing text"},
            ],
            outputs=[
                {"name": "rationale", "type": "str", "description": "Judging rationale", "required": True},
                {"name": "winning_agent_id", "type": "str", "description": "Winning agent id (integer as string)", "required": True},
                {"name": "final_answer", "type": "str", "description": "Final answer", "required": True},
            ],
            parse_mode="xml",
        )

    def execute(
        self,
        problem: str,
        num_agents: int = 3,
        num_rounds: int = 3,
        judge_mode: str = "llm_judge",  # options: ["llm_judge", "self_consistency"]
        personas: Optional[List[str]] = None,
        return_transcript: bool = True,
        agent_llm_configs: Optional[List[LLMConfig]] = None,
        enable_pruning: bool = False,
        pruning_qp_threshold: float = 0.15,
        pruning_dp_similarity_threshold: float = 0.92,
        pruning_enable_mr: bool = False,
        pruning_mr_llm_config: Optional[LLMConfig] = None,
        pruning_snapshot_mode: bool = False,
        transcript_mode: str = "prev",  # options: ["prev", "all"]
        **kwargs,
    ) -> dict:
        """Execute debate workflow (synchronous)."""
        state = self._setup_debate(problem, num_agents, num_rounds, personas, agent_llm_configs)
        transcript = self._run_debate_rounds(problem, state, transcript_mode)
        # Pruning (before consensus, filter and optionally correct last round candidates)
        pruning_info = None
        pruning_debug = None
        pruning_rounds_debug: Optional[List[Dict[str, Any]]] = None
        if enable_pruning:
            # Set minimum retention based on participants (default ratio 0.3, at least 1)
            min_keep = max(1, int(round(state["num_agents"] * 0.3)))
            pipeline = PruningPipeline(
                enable_qp=True,
                enable_dp=True,
                enable_mr=pruning_enable_mr,
                qp_threshold=pruning_qp_threshold,
                dp_similarity_threshold=pruning_dp_similarity_threshold,
                mr_llm_config=pruning_mr_llm_config,
                min_keep_count=min_keep,
            )
            if pruning_snapshot_mode:
                # Generate snapshot for each round (doesn't change main flow, only for display)
                pruning_rounds_debug = []
                for r in range(state["num_rounds"]):
                    rcands = collect_round_candidates(
                        transcript=transcript, num_agents=state["num_agents"], round_index=r
                    )
                    info_r = pipeline.apply(problem=problem, candidates=rcands)
                    pruning_rounds_debug.append(
                        {
                            "round": r,
                            "before_candidates": rcands,
                            "after_candidates": info_r.get("candidates", []),
                            "mr_suggested": info_r.get("mr_suggested"),
                        }
                    )
            candidates = collect_last_round_candidates(
                transcript=transcript, num_agents=state["num_agents"], last_round_index=state["num_rounds"] - 1
            )
            pruning_info = pipeline.apply(problem=problem, candidates=candidates)
            try:
                pruning_debug = {
                    "before_candidates": candidates,
                    "after_candidates": pruning_info.get("candidates", []),
                    "mr_suggested": pruning_info.get("mr_suggested"),
                }
            except Exception:
                pruning_debug = None
        consensus = self._generate_consensus(problem, state, transcript, judge_mode, pruning_info)
        result: Dict[str, Any] = {
            "final_answer": consensus["final_answer"],
            "winner": consensus.get("winner"),
            "rationale": consensus.get("rationale"),
        }
        if return_transcript:
            result["transcript"] = transcript
        if enable_pruning and pruning_debug is not None:
            result["pruning"] = pruning_debug
        if enable_pruning and pruning_snapshot_mode and pruning_rounds_debug is not None:
            result["pruning_rounds"] = pruning_rounds_debug
        return result

    async def async_execute(
        self,
        problem: str,
        num_agents: int = 3,
        num_rounds: int = 3,
        judge_mode: str = "llm_judge",
        personas: Optional[List[str]] = None,
        return_transcript: bool = True,
        agent_llm_configs: Optional[List[LLMConfig]] = None,
        enable_pruning: bool = False,
        pruning_qp_threshold: float = 0.15,
        pruning_dp_similarity_threshold: float = 0.92,
        pruning_enable_mr: bool = False,
        pruning_mr_llm_config: Optional[LLMConfig] = None,
        pruning_snapshot_mode: bool = False,
        transcript_mode: str = "prev",  # options: ["prev", "all"]
        **kwargs,
    ) -> dict:
        """Execute debate workflow (asynchronous)."""
        state = self._setup_debate(problem, num_agents, num_rounds, personas, agent_llm_configs)
        transcript = await self._run_debate_rounds_async(problem, state, transcript_mode)
        pruning_info = None
        pruning_debug = None
        pruning_rounds_debug: Optional[List[Dict[str, Any]]] = None
        if enable_pruning:
            min_keep = max(1, int(round(state["num_agents"] * 0.3)))
            pipeline = PruningPipeline(
                enable_qp=True,
                enable_dp=True,
                enable_mr=pruning_enable_mr,
                qp_threshold=pruning_qp_threshold,
                dp_similarity_threshold=pruning_dp_similarity_threshold,
                mr_llm_config=pruning_mr_llm_config,
                min_keep_count=min_keep,
            )
            if pruning_snapshot_mode:
                pruning_rounds_debug = []
                for r in range(state["num_rounds"]):
                    rcands = collect_round_candidates(
                        transcript=transcript, num_agents=state["num_agents"], round_index=r
                    )
                    info_r = pipeline.apply(problem=problem, candidates=rcands)
                    pruning_rounds_debug.append(
                        {
                            "round": r,
                            "before_candidates": rcands,
                            "after_candidates": info_r.get("candidates", []),
                            "mr_suggested": info_r.get("mr_suggested"),
                        }
                    )
            candidates = collect_last_round_candidates(
                transcript=transcript, num_agents=state["num_agents"], last_round_index=state["num_rounds"] - 1
            )
            pruning_info = pipeline.apply(problem=problem, candidates=candidates)
            try:
                pruning_debug = {
                    "before_candidates": candidates,
                    "after_candidates": pruning_info.get("candidates", []),
                    "mr_suggested": pruning_info.get("mr_suggested"),
                }
            except Exception:
                pruning_debug = None
        consensus = await self._generate_consensus_async(problem, state, transcript, judge_mode, pruning_info)
        result: Dict[str, Any] = {
            "final_answer": consensus["final_answer"],
            "winner": consensus.get("winner"),
        }
        if return_transcript:
            result["transcript"] = transcript
        if enable_pruning and pruning_debug is not None:
            result["pruning"] = pruning_debug
        if enable_pruning and pruning_snapshot_mode and pruning_rounds_debug is not None:
            result["pruning_rounds"] = pruning_rounds_debug
        return result

    def _setup_debate(
        self,
        problem: str,
        num_agents: int,
        num_rounds: int,
        personas: Optional[List[str]],
        agent_llm_configs: Optional[List[LLMConfig]] = None,
    ) -> dict:
        """Setup debate environment."""
        if num_agents <= 1:
            raise ValueError("num_agents must be greater than 1")
        if num_rounds <= 0:
            raise ValueError("num_rounds must be positive")

        roles: List[str] = personas or get_default_personas(num_agents)
        # Based on user input or model pool, prepare each agent object (fixed for entire battle cycle, not randomly changed per round)
        agents_for_ids: List[CustomizeAgent] = self._prepare_runtime_debaters(num_agents, agent_llm_configs)
        state: Dict[str, Any] = {
            "problem": problem,
            "num_agents": num_agents,
            "num_rounds": num_rounds,
            "roles": roles,
            "agents": agents_for_ids,
        }
        return state

    def _prepare_runtime_debaters(self, num_agents: int, agent_llm_configs: Optional[List[LLMConfig]]) -> List[CustomizeAgent]:
        """Select CustomizeAgent for each agent_id that remains unchanged throughout the debate.
        Priority:
        1) User explicitly passes debater_agents → cycle/truncate by length and assign to each position
        2) Pass agent_llm_configs → create default debater for each position
        3) Use llm_config_pool random selection → create default debater for each position (prioritized over default llm_config)
        4) Fallback to default llm_config
        """
        # If group graph mode is enabled, don't generate internal agents (driven by group_graphs instead)
        if self.group_graphs_enabled:
            return []
        
        # 1) Explicit multiple agents
        if self.debater_agents:
            agents: List[CustomizeAgent] = []
            for i in range(num_agents):
                agents.append(self.debater_agents[i % len(self.debater_agents)])
            return agents
        
        # 2) Create based on passed agent_llm_configs
        if agent_llm_configs and len(agent_llm_configs) > 0:
            return [
                self._create_debater_agent_with_llm(agent_llm_configs[i % len(agent_llm_configs)])
                for i in range(num_agents)
            ]
        
        # 3) Use llm_config_pool random selection
        if self.llm_config_pool and len(self.llm_config_pool) > 0:
            return [self._create_debater_agent_with_llm(random.choice(self.llm_config_pool)) for _ in range(num_agents)]
        
        # 4) Default: use llm_config
        default_agent = self._create_default_debater_agent()
        return [default_agent for _ in range(num_agents)]

    def _create_debater_agent_with_llm(self, llm_cfg: LLMConfig) -> CustomizeAgent:
        """Create a debater agent with given LLM configuration that is consistent with default structure."""
        return CustomizeAgent(
            name="DebaterAgent",
            description="Generate argument/rebuttal and optional answer per debate round.",
            prompt=DEBATER_AGENT_PROMPT,
            llm_config=llm_cfg,
            inputs=[
                {"name": "problem", "type": "str", "description": "Problem statement"},
                {"name": "transcript_text", "type": "str", "description": "Formatted debate transcript so far"},
                {"name": "role", "type": "str", "description": "Debater role/persona"},
                {"name": "agent_id", "type": "str", "description": "Debater id (string)"},
                {"name": "round_index", "type": "str", "description": "1-based round index"},
                {"name": "total_rounds", "type": "str", "description": "Total rounds"},
            ],
            outputs=[
                {"name": "thought", "type": "str", "description": "Brief reasoning", "required": True},
                {"name": "argument", "type": "str", "description": "Argument or rebuttal", "required": True},
                {"name": "answer", "type": "str", "description": "Optional current answer", "required": False},
            ],
            parse_mode="xml",
        )

    def _run_debate_rounds(self, problem: str, state: dict, transcript_mode: str = "prev") -> List[dict]:
        """Run debate rounds (synchronous). Return transcript.
        
        Args:
            transcript_mode: Control transcript range accessible to agents
                - "prev": Can only see n-1 round speeches (default)
                - "all": Can see all previous round speeches
        """
        transcript: List[dict] = []
        num_agents: int = state["num_agents"]
        num_rounds: int = state["num_rounds"]
        roles: List[str] = state["roles"]

        for round_index in range(num_rounds):
            for agent_id in range(num_agents):
                # Group graph mode: call corresponding graph.execute()
                if self.group_graphs_enabled and self.group_graphs:
                    graph = self.group_graphs[agent_id % len(self.group_graphs)]
                    # Get corresponding transcript based on access mode
                    transcript_text = self._get_transcript_for_agent(
                        transcript, round_index, agent_id, transcript_mode, num_agents
                    )
                    g_inputs = {
                        "problem": problem,
                        "transcript_text": transcript_text,
                        "role": roles[agent_id],
                        "agent_id": str(agent_id),
                        "round_index": str(round_index + 1),
                        "total_rounds": str(num_rounds),
                    }
                    g_out = graph.execute(**g_inputs)
                    structured = {
                        "argument": g_out.get("argument", g_out.get("output", "")),
                        "answer": g_out.get("answer"),
                        "thought": g_out.get("thought", ""),
                    }
                else:
                    # If _setup_debate has determined agent for each position, use directly
                    selected_agent: Optional[CustomizeAgent] = None
                    agents_for_ids: Optional[List[CustomizeAgent]] = state.get("agents")
                    if agents_for_ids:
                        selected_agent = agents_for_ids[agent_id]
                    elif self.debater_agents:
                        selected_agent = random.choice(self.debater_agents)

                    if selected_agent is not None:
                        try:
                            # Get corresponding transcript based on access mode
                            transcript_text = self._get_transcript_for_agent(
                                transcript, round_index, agent_id, transcript_mode, num_agents
                            )
                            inputs = {
                                "problem": problem,
                                "transcript_text": transcript_text,
                                "role": roles[agent_id],
                                "agent_id": str(agent_id),
                                "round_index": str(round_index + 1),
                                "total_rounds": str(num_rounds),
                            }
                            msg = selected_agent(inputs=inputs)
                            structured = msg.content.get_structured_data()
                        except Exception as e:
                            print(f"Agent execution error: {e}")
                            structured = {"argument": "", "answer": "", "thought": ""}
                    else:
                        # Get corresponding transcript based on access mode
                        transcript_text = self._get_transcript_for_agent(
                            transcript, round_index, agent_id, transcript_mode, num_agents
                        )
                        prompt = build_agent_prompt(
                            problem=problem,
                            transcript_text=transcript_text,
                            role=roles[agent_id],
                            agent_id=agent_id,
                            round_index=round_index,
                            total_rounds=num_rounds,
                        )
                        response = self._llm.generate(
                            prompt=prompt,
                            parser=DebateAgentOutput,
                            parse_mode="xml",
                        )
                        structured = response.get_structured_data()
                transcript.append(
                    {
                        "agent_id": agent_id,
                        "round": round_index,
                        "role": roles[agent_id],
                        "argument": structured.get("argument", ""),
                        "answer": structured.get("answer"),
                        "thought": structured.get("thought", ""),
                    }
                )
                # Print each agent's output immediately (synchronous, complete content)
                try:
                    arg_full = str(structured.get("argument", "")).strip()
                    ans_full = str(structured.get("answer") or "").strip()
                    print(
                        f"[Round {round_index + 1}] Agent#{agent_id} ({roles[agent_id]})\n"
                        f"Argument: {arg_full}\n"
                        f"Answer: {ans_full}\n"
                    )
                except Exception:
                    pass
        return transcript

    def _get_transcript_for_agent(self, transcript: List[dict], round_index: int, agent_id: int, 
                                 transcript_mode: str, num_agents: int) -> str:
        """根据访问模式获取agent可以访问的transcript。
        
        Args:
            transcript: 完整的transcript
            round_index: 当前轮次索引
            agent_id: 当前agent的ID
            transcript_mode: 访问模式
                - "prev": 只能看到n-1轮次的发言（默认）
                - "all": 可以看到之前所有轮次的发言
            num_agents: agent总数
            
        Returns:
            str: 格式化后的transcript文本
        """
        if transcript_mode == "prev":
            # 只包含当前轮次之前的transcript（n-1轮次）
            filtered_transcript = [t for t in transcript if t["round"] < round_index]
        elif transcript_mode == "all":
            # 包含当前轮次之前的所有transcript + 当前轮次中当前agent之前的发言
            filtered_transcript = []
            for t in transcript:
                if t["round"] < round_index:
                    # 之前轮次的所有发言
                    filtered_transcript.append(t)
                elif t["round"] == round_index and t["agent_id"] < agent_id:
                    # 当前轮次中，当前agent之前的发言
                    filtered_transcript.append(t)
        else:
            # 默认使用prev
            filtered_transcript = [t for t in transcript if t["round"] < round_index]
        
        return format_transcript(filtered_transcript)

    async def _run_debate_rounds_async(self, problem: str, state: dict, transcript_mode: str = "prev") -> List[dict]:
        """运行辩论轮次（异步）。返回 transcript。
        
        Args:
            transcript_mode: 控制agent可以访问的transcript范围
                - "prev": 只能看到n-1轮次的发言（默认）
                - "all": 可以看到之前所有轮次的发言
        """
        transcript: List[dict] = []
        num_agents: int = state["num_agents"]
        num_rounds: int = state["num_rounds"]
        roles: List[str] = state["roles"]

        for round_index in range(num_rounds):
            # 每个回合内可并行生成
            # 注意：在异步版本中，我们需要为每个agent单独计算transcript
            # 若分组图模式：逐个调用 graph.execute()（并发复杂性留待后续）
            if self.group_graphs_enabled and self.group_graphs:
                for agent_id in range(num_agents):
                    graph = self.group_graphs[agent_id % len(self.group_graphs)]
                    # 根据访问模式获取相应的transcript
                    transcript_text = self._get_transcript_for_agent(
                        transcript, round_index, agent_id, transcript_mode, num_agents
                    )
                    g_inputs = {
                        "problem": problem,
                        "transcript_text": transcript_text,
                        "role": roles[agent_id],
                        "agent_id": str(agent_id),
                        "round_index": str(round_index + 1),
                        "total_rounds": str(num_rounds),
                    }
                    g_out = graph.execute(**g_inputs)
                    structured = {
                        "argument": g_out.get("argument", g_out.get("output", "")),
                        "answer": g_out.get("answer"),
                        "thought": g_out.get("thought", ""),
                    }
                    transcript.append(
                        {
                            "agent_id": agent_id,
                            "round": round_index,
                            "role": roles[agent_id],
                            "argument": structured.get("argument", ""),
                            "answer": structured.get("answer"),
                            "thought": structured.get("thought", ""),
                        }
                    )
                    # 打印
                    try:
                        print(
                            f"[Round {round_index + 1}] Agent#{agent_id} ({roles[agent_id]})\n"
                            f"Argument: {str(structured.get('argument','')).strip()}\n"
                            f"Answer: {str(structured.get('answer') or '').strip()}\n"
                        )
                    except Exception:
                        pass
            # 若有外部注入的Agent，则并发调用Agent；否则走LLM批量生成
            elif state.get("agents") or self.debater_agents or self.debater_agent is not None:
                import asyncio
                tasks = []
                id_list: List[int] = []
                for agent_id in range(num_agents):
                    agents_for_ids: Optional[List[CustomizeAgent]] = state.get("agents")
                    if agents_for_ids:
                        selected_agent = agents_for_ids[agent_id]
                    elif self.debater_agents:
                        selected_agent = random.choice(self.debater_agents)
                    else:
                        selected_agent = None
                    # 根据访问模式获取相应的transcript
                    transcript_text = self._get_transcript_for_agent(
                        transcript, round_index, agent_id, transcript_mode, num_agents
                    )
                    inputs = {
                        "problem": problem,
                        "transcript_text": transcript_text,
                        "role": roles[agent_id],
                        "agent_id": str(agent_id),
                        "round_index": str(round_index + 1),
                        "total_rounds": str(num_rounds),
                    }
                    tasks.append(selected_agent(inputs=inputs))
                    id_list.append(agent_id)
                messages = await asyncio.gather(*tasks)
                for agent_id, msg in zip(id_list, messages):
                    structured = msg.content.get_structured_data()
                    transcript.append(
                        {
                            "agent_id": agent_id,
                            "round": round_index,
                            "role": roles[agent_id],
                            "argument": structured.get("argument", ""),
                            "answer": structured.get("answer"),
                            "thought": structured.get("thought", ""),
                        }
                    )
                # 即时打印本轮所有 agent 的产出（异步并发完成后逐个打印，完整内容）
                try:
                    for agent_id, msg in zip(id_list, messages):
                        st = msg.content.get_structured_data()
                        arg_full = str(st.get("argument", "")).strip()
                        ans_full = str(st.get("answer") or "").strip()
                        print(
                            f"[Round {round_index + 1}] Agent#{agent_id} ({roles[agent_id]})\n"
                            f"Argument: {arg_full}\n"
                            f"Answer: {ans_full}\n"
                        )
                except Exception:
                    pass
            else:
                prompts: List[Tuple[int, str]] = []
                for agent_id in range(num_agents):
                    # 根据访问模式获取相应的transcript
                    transcript_text = self._get_transcript_for_agent(
                        transcript, round_index, agent_id, transcript_mode, num_agents
                    )
                    prompt = build_agent_prompt(
                        problem=problem,
                        transcript_text=transcript_text,
                        role=roles[agent_id],
                        agent_id=agent_id,
                        round_index=round_index,
                        total_rounds=num_rounds,
                    )
                    prompts.append((agent_id, prompt))

                results = await self._llm.batch_generate_async(
                    batch_messages=[[{"role": "user", "content": p}] for _, p in prompts]
                )
                parsed_list = self._llm.parse_generated_texts(
                    texts=results, parser=DebateAgentOutput, parse_mode="xml"
                )
                for (agent_id, _), parsed in zip(prompts, parsed_list):
                    structured = parsed.get_structured_data()
                    transcript.append(
                        {
                            "agent_id": agent_id,
                            "round": round_index,
                            "role": roles[agent_id],
                            "argument": structured.get("argument", ""),
                            "answer": structured.get("answer"),
                            "thought": structured.get("thought", ""),
                        }
                    )
        return transcript

    def _generate_consensus(
        self, problem: str, state: dict, transcript: List[dict], judge_mode: str, pruning_info: Optional[Dict[str, Any]] = None
    ) -> dict:
        """根据 judge 模式生成最终共识（同步）。"""
        if judge_mode == "self_consistency":
            # 使用 QAScEnsemble 对各辩手最终答案做投票
            agent_final_answers = self._collect_agent_final_answers(state, transcript)
            if len(agent_final_answers) == 0:
                # 无人给出显式答案，则合并最后一轮的论据
                agent_final_answers = [t["argument"] for t in transcript if t.get("argument")]
            sc = self._sc_ensemble.execute(solutions=agent_final_answers)
            return {
                "final_answer": sc["response"],
                "winner": None,
            }

        # 默认 LLM 裁判；若外部注入 judge_agent，则优先使用
        if self.judge_agent is not None:
            roles_text = "\n".join([f"#{i}: {r}" for i, r in enumerate(state["roles"])])
            inputs = {
                "problem": problem,
                "transcript_text": format_transcript(transcript),
                "roles_text": roles_text,
            }
            # 将剪枝摘要附加给裁判（若存在 MR 建议，可以在裁判 prompt 中强调）
            if pruning_info and pruning_info.get("mr_suggested"):
                suggested = pruning_info["mr_suggested"].get("corrected", "")
                if suggested:
                    inputs["problem"] = problem + "\n\n(Consider corrected consolidation, if helpful.)"
            msg = self.judge_agent(inputs=inputs)
            jd = msg.content.get_structured_data()
        else:
            judge_prompt = build_judge_prompt(
                problem=problem,
                transcript_text=format_transcript(transcript),
                roles=state["roles"],
            )
            judge_resp = self._llm.generate(
                prompt=judge_prompt, parser=DebateJudgeOutput, parse_mode="xml"
            )
            jd = judge_resp.get_structured_data()
        
        winner_id = int(jd.get("winning_agent_id", 0))
        final_answer = jd.get("final_answer", "")
        
        # 获取获胜者的答案
        winner_answer = self._get_winner_answer(transcript, winner_id, state["num_rounds"])
        
        return {
            "final_answer": final_answer,
            "winner": winner_id,
            "winner_answer": winner_answer,
            "rationale": jd.get("rationale", ""),
        }

    async def _generate_consensus_async(
        self, problem: str, state: dict, transcript: List[dict], judge_mode: str, pruning_info: Optional[Dict[str, Any]] = None
    ) -> dict:
        """根据 judge 模式生成最终共识（异步）。"""
        if judge_mode == "self_consistency":
            agent_final_answers = self._collect_agent_final_answers(state, transcript)
            if len(agent_final_answers) == 0:
                agent_final_answers = [t["argument"] for t in transcript if t.get("argument")]
            sc = await self._sc_ensemble.async_execute(solutions=agent_final_answers)
            return {
                "final_answer": sc["response"],
                "winner": None,
            }

        if self.judge_agent is not None:
            roles_text = "\n".join([f"#{i}: {r}" for i, r in enumerate(state["roles"])])
            inputs = {
                "problem": problem,
                "transcript_text": format_transcript(transcript),
                "roles_text": roles_text,
            }
            if pruning_info and pruning_info.get("mr_suggested"):
                suggested = pruning_info["mr_suggested"].get("corrected", "")
                if suggested:
                    inputs["problem"] = problem + "\n\n(Consider corrected consolidation, if helpful.)"
            msg = await self.judge_agent(inputs=inputs)
            jd = msg.content.get_structured_data()
        else:
            judge_prompt = build_judge_prompt(
                problem=problem,
                transcript_text=format_transcript(transcript),
                roles=state["roles"],
            )
            judge_resp = await self._llm.async_generate(
                prompt=judge_prompt, parser=DebateJudgeOutput, parse_mode="xml"
            )
            jd = judge_resp.get_structured_data()
        
        winner_id = int(jd.get("winning_agent_id", 0))
        final_answer = jd.get("final_answer", "")
        
        # 获取获胜者的答案
        winner_answer = self._get_winner_answer(transcript, winner_id, state["num_rounds"])
        
        return {
            "final_answer": final_answer,
            "winner": winner_id,
            "winner_answer": winner_answer,
            "rationale": jd.get("rationale", ""),
        }

    def _collect_agent_final_answers(self, state: dict, transcript: List[dict]) -> List[str]:
        """收集每位辩手的最终答案（若有）。"""
        num_agents = state["num_agents"]
        num_rounds = state["num_rounds"]
        final_answers: List[str] = []
        for agent_id in range(num_agents):
            # 查找该辩手在最后一轮的记录
            records = [t for t in transcript if t["agent_id"] == agent_id and t["round"] == num_rounds - 1]
            if len(records) == 0:
                continue
            ans = records[-1].get("answer")
            if ans and isinstance(ans, str) and len(ans.strip()) > 0:
                final_answers.append(ans)
        return final_answers

    def _get_winner_answer(self, transcript: List[dict], winner_id: int, num_rounds: int) -> Optional[str]:
        """获取获胜者在最后一轮的答案。"""
        # 查找获胜者在最后一轮的记录
        records = [t for t in transcript if t["agent_id"] == winner_id and t["round"] == num_rounds - 1]
        if len(records) == 0:
            return None
        
        answer = records[-1].get("answer")
        if answer and isinstance(answer, str) and len(answer.strip()) > 0:
            return answer.strip()
        
        # 如果没有明确的答案，返回最后一轮的论据
        argument = records[-1].get("argument", "")
        return argument.strip() if argument else None

    def save_module(self, path: str, ignore: List[str] = [], **kwargs) -> str:
        """保存模块配置（直接保存agents，不保存llm_config_pool）"""
        # 确保目录存在
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # 保存debater agents
        agent_pool_path = path.replace('.json', '_agents.json')
        if self.debater_agents:
            agent_data = []
            for i, agent in enumerate(self.debater_agents):
                # 为每个agent创建单独的保存路径
                agent_path = agent_pool_path.replace('.json', f'_{i}.json')
                agent.save_module(agent_path)
                agent_data.append({
                    "name": agent.name,
                    "description": agent.description,
                    "file_path": agent_path
                })
            
            with open(agent_pool_path, 'w', encoding='utf-8') as f:
                json.dump(agent_data, f, ensure_ascii=False, indent=2)
        
        # 保存judge agent
        judge_agent_path = path.replace('.json', '_judge.json')
        if self.judge_agent:
            self.judge_agent.save_module(judge_agent_path)
        
        # 保存debate配置（只保存基本配置，不保存llm_config_pool）
        config = {
            "llm_config": self._serialize_llm_config(self.llm_config),
            "name": self.name,
            "description": self.description,
            "agent_pool_file": agent_pool_path if self.debater_agents else None,
            "judge_agent_file": judge_agent_path if self.judge_agent else None
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"模块配置已保存到: {path}")
        return path
    
    def get_config(self) -> dict:
        """获取当前模块的配置字典（不包含llm_config_pool）"""
        config = {
            "llm_config": self._serialize_llm_config(self.llm_config),
            "name": self.name,
            "description": self.description,
        }
        
        # 序列化agent pool
        if self.debater_agents:
            agent_data = []
            for agent in self.debater_agents:
                agent_info = {
                    "name": agent.name,
                    "description": agent.description,
                    "config": agent.get_config()
                }
                agent_data.append(agent_info)
            config["debater_agents"] = agent_data
        
        # 序列化judge agent
        if self.judge_agent:
            config["judge_agent"] = {
                "name": self.judge_agent.name,
                "description": self.judge_agent.description,
                "config": self.judge_agent.get_config()
            }
        
        return config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> 'MultiAgentDebateActionGraph':
        """从配置字典创建MultiAgentDebateActionGraph实例（不重建llm_config_pool）"""
        # 创建实例
        instance = cls()
        
        # 重建llm_config
        if data.get("llm_config"):
            instance.llm_config = instance._deserialize_llm_config(data["llm_config"])
        
        # 设置其他属性
        if data.get("name"):
            instance.name = data["name"]
        
        if data.get("description"):
            instance.description = data["description"]
        
        # 重建agent pool
        if data.get("debater_agents"):
            agents = []
            for agent_info in data["debater_agents"]:
                try:
                    agent_config = agent_info.get("config", {})
                    llm_config = instance._deserialize_llm_config(agent_config.get("llm_config"))
                    
                    # 从agent_config中移除可能重复的字段
                    agent_config_clean = {k: v for k, v in agent_config.items() 
                                        if k not in ['name', 'description', 'llm_config']}
                    
                    agent = CustomizeAgent(
                        name=agent_info["name"],
                        description=agent_info["description"],
                        llm_config=llm_config,
                        **agent_config_clean
                    )
                    agents.append(agent)
                except Exception as e:
                    print(f"警告: 重建agent {agent_info.get('name', 'unknown')}失败: {e}")
                    continue
            
            instance.debater_agents = agents
        
        # 重建judge agent
        if data.get("judge_agent"):
            try:
                judge_info = data["judge_agent"]
                judge_config = judge_info.get("config", {})
                llm_config = instance._deserialize_llm_config(judge_config.get("llm_config"))
                
                # 从judge_config中移除可能重复的字段
                judge_config_clean = {k: v for k, v in judge_config.items() 
                                    if k not in ['name', 'description', 'llm_config']}
                
                instance.judge_agent = CustomizeAgent(
                    name=judge_info["name"],
                    description=judge_info["description"],
                    llm_config=llm_config,
                    **judge_config_clean
                )
            except Exception as e:
                print(f"警告: 重建judge agent失败: {e}")
        
        return instance
    
    @classmethod
    def load_module(cls, path: str, llm_config: LLMConfig = None, **kwargs) -> 'MultiAgentDebateActionGraph':
        """从文件加载MultiAgentDebateActionGraph实例（类方法，不重建llm_config_pool）"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"模块配置文件不存在: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"读取配置文件失败: {e}")
        
        # 创建实例
        instance = cls()
        
        # 重建llm_config
        if config.get("llm_config"):
            try:
                instance.llm_config = instance._deserialize_llm_config(config["llm_config"])
            except Exception as e:
                print(f"警告: 重建llm_config失败: {e}")
        
        # 设置其他属性
        if config.get("name"):
            instance.name = config["name"]
        
        if config.get("description"):
            instance.description = config["description"]
        
        # 加载agent pool
        agent_pool_file = config.get("agent_pool_file")
        if agent_pool_file and os.path.exists(agent_pool_file):
            try:
                with open(agent_pool_file, 'r', encoding='utf-8') as f:
                    agent_data = json.load(f)
                
                agents = []
                for agent_info in agent_data:
                    try:
                        agent_path = agent_info.get("file_path")
                        if agent_path and os.path.exists(agent_path):
                            # 使用agent的load_module方法
                            agent = CustomizeAgent.from_file(
                                path=agent_path,
                                llm_config=instance.llm_config or llm_config
                            )
                            agents.append(agent)
                        else:
                            print(f"警告: agent文件不存在: {agent_path}")
                    except Exception as e:
                        print(f"警告: 加载agent {agent_info.get('name', 'unknown')}失败: {e}")
                        continue
                
                instance.debater_agents = agents
                print(f"从 {agent_pool_file} 加载了 {len(agents)} 个agents")
            except Exception as e:
                print(f"警告: 加载agent pool失败: {e}")
        
        # 加载judge agent
        judge_agent_file = config.get("judge_agent_file")
        if judge_agent_file and os.path.exists(judge_agent_file):
            try:
                # 使用agent的from_file方法
                instance.judge_agent = CustomizeAgent.from_file(
                    path=judge_agent_file,
                    llm_config=instance.llm_config or llm_config
                )
                print(f"从 {judge_agent_file} 加载了judge agent")
            except Exception as e:
                print(f"警告: 加载judge agent失败: {e}")
        
        print(f"从 {path} 加载了模块配置")
        return instance
    

    
    def _serialize_llm_config(self, llm_config) -> Optional[Dict[str, Any]]:
        """序列化LLM配置（只保存模型名称和基本参数）"""
        if not llm_config:
            return None
        
        config_info = {
            "model": llm_config.model if hasattr(llm_config, 'model') else None,
            "temperature": llm_config.temperature if hasattr(llm_config, 'temperature') else None,
            "config_type": type(llm_config).__name__
        }
        
        return config_info
    
    def _deserialize_llm_config(self, config_info: Optional[Dict[str, Any]]) -> Optional[LLMConfig]:
        """反序列化LLM配置（从环境变量重建）"""
        if not config_info:
            return None
        
        config_type = config_info.get("config_type", "OpenAILLMConfig")
        
        if config_type == "OpenAILLMConfig":
            from ...models.model_configs import OpenAILLMConfig
            return OpenAILLMConfig(
                model=config_info.get("model", "gpt-4o-mini"),
                openai_key=os.getenv("OPENAI_API_KEY")
            )
        elif config_type == "OpenRouterConfig":
            from ...models.model_configs import OpenRouterConfig
            return OpenRouterConfig(
                model=config_info.get("model", "meta-llama/llama-3.1-70b-instruct"),
                openrouter_key=os.getenv("OPENROUTER_API_KEY")
            )
        
        return None
