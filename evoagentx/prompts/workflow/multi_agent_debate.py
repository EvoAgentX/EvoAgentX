"""
Multi-Agent Debate Prompts

This module contains all prompts and utilities for the Multi-Agent Debate framework.
"""

# Default personas for debaters
DEFAULT_PERSONAS = [
    "Rigorous statistician, emphasizes falsifiability and counterexamples",
    "Pragmatic engineer, focuses on executable plans and edge cases", 
    "Devil's advocate, challenges assumptions and proposes alternatives",
    "Knowledge graph expert, stresses structured evidence and provenance",
    "Intuition-driven heuristic scorer, provides quick scoring and heuristics",
]

def get_default_personas(num_agents: int) -> list:
    """Return default personas in English."""
    roles = []
    for i in range(num_agents):
        roles.append(DEFAULT_PERSONAS[i % len(DEFAULT_PERSONAS)])
    return roles

# Debater Agent Prompt Template
DEBATER_AGENT_PROMPT = """
You are debater #{agent_id} (role: {role}). This is round {round_index} of {total_rounds}.

Problem:
{problem}

Conversation so far:
{transcript_text}

Instructions:
- Think briefly (<= 120 words), then present your argument or rebuttal.
- If confident, provide your current answer for this round.
- Your output MUST follow this XML template:

<response>
  <thought>Your brief reasoning</thought>
  <argument>Your argument or rebuttal</argument>
  <answer>Optional current answer; leave empty if uncertain</answer>
</response>
"""

# Judge Agent Prompt Template  
JUDGE_AGENT_PROMPT = """
You are the judge. Based on the multi-round debate, deliver a final decision and answer.

Problem:
{problem}

Debater roles:
{roles_text}

Debate transcript:
{transcript_text}

Return the following XML:
<response>
  <rationale>Your judging rationale</rationale>
  <winning_agent_id>Winning debater ID (integer)</winning_agent_id>
  <final_answer>The final answer</final_answer>
</response>
"""

# Utility functions for building prompts
def build_agent_prompt(
    problem: str,
    transcript_text: str,
    role: str,
    agent_id: int,
    round_index: int,
    total_rounds: int,
) -> str:
    """Construct agent prompt (XML-structured output)."""
    return DEBATER_AGENT_PROMPT.format(
        agent_id=agent_id,
        role=role,
        round_index=round_index + 1,
        total_rounds=total_rounds,
        problem=problem,
        transcript_text=transcript_text
    )

def build_judge_prompt(problem: str, transcript_text: str, roles: list) -> str:
    """Construct judge prompt (XML-structured output)."""
    roles_text = "\n".join([f"#{i}: {r}" for i, r in enumerate(roles)])
    return JUDGE_AGENT_PROMPT.format(
        problem=problem,
        roles_text=roles_text,
        transcript_text=transcript_text
    )

def format_transcript(transcript: list) -> str:
    """Format debate transcript for display."""
    if not transcript:
        return "(empty)"
    lines = []
    for turn in transcript:
        agent = turn.get("agent_id")
        rd = turn.get("round")
        role = turn.get("role")
        arg = (turn.get("argument") or "").strip()
        ans = (turn.get("answer") or "").strip()
        if ans:
            lines.append(f"[Round {rd}] Agent#{agent} ({role})\nArgument: {arg}\nAnswer: {ans}\n")
        else:
            lines.append(f"[Round {rd}] Agent#{agent} ({role})\nArgument: {arg}\n")
    return "\n".join(lines)

def collect_last_round_candidates(
    transcript: list, num_agents: int, last_round_index: int
) -> list:
    """Collect candidates from the last round of debate.
    
    Returns list of [{"agent_id": int, "text": str}].
    Prioritizes answer field; falls back to argument if no valid answer.
    """
    candidates = []
    for agent_id in range(num_agents):
        # Find the agent's record in the last round (if multiple, take the last one)
        records = [
            t for t in transcript if t.get("agent_id") == agent_id and t.get("round") == last_round_index
        ]
        if not records:
            continue
        rec = records[-1]
        ans = (rec.get("answer") or "").strip()
        arg = (rec.get("argument") or "").strip()
        text = ans if ans else arg
        if text:
            candidates.append({"agent_id": agent_id, "text": text})
    return candidates

def collect_round_candidates(
    transcript: list, num_agents: int, round_index: int
) -> list:
    """Collect candidates from specified round (prioritizes answer, falls back to argument)."""
    candidates = []
    for agent_id in range(num_agents):
        records = [
            t for t in transcript if t.get("agent_id") == agent_id and t.get("round") == round_index
        ]
        if not records:
            continue
        rec = records[-1]
        ans = (rec.get("answer") or "").strip()
        arg = (rec.get("argument") or "").strip()
        text = ans if ans else arg
        if text:
            candidates.append({"agent_id": agent_id, "text": text})
    return candidates
