import os
from dotenv import load_dotenv

from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.models import OpenAILLMConfig


def get_llm_config():
    load_dotenv()
    # Only use OpenAI GPT-4o (can be overridden via OPENAI_MODEL, default gpt-4o)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("Set OPENAI_API_KEY for OpenAI access.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    return OpenAILLMConfig(
        model=model,
        openai_key=openai_key,
        max_completion_tokens=800,
        temperature=0.5,
        output_response=True,
    )


def run_self_consistency_example():
    llm_config = get_llm_config()
    debate = MultiAgentDebateActionGraph(
        name="MAD Minimal",
        description="Minimal runnable example for multi-agent debate",
        llm_config=llm_config,
    )

    fixed_problem = "How many labeled trees on 10 vertices are there such that vertex 1 has degree exactly 4? Return only the final integer."
    result = debate.execute(
        problem=fixed_problem,
        num_agents=3,
        num_rounds=5,
        judge_mode="self_consistency",
        return_transcript=True,
    )

    print("=== Example: Self-Consistency (Fixed Answer) ===")
    print("Final Answer:", result.get("final_answer"))
    print("Winner:", result.get("winner"))  # Should be None
    print("\nTranscript:")
    for turn in result.get("transcript", []):
        print(
            f"[Round {turn['round']}] Agent#{turn['agent_id']} ({turn['role']})\n"
            f"Argument: {turn.get('argument','').strip()}\n"
            f"Answer: {str(turn.get('answer') or '').strip()}\n"
        )


def run_llm_judge_example():
    llm_config = get_llm_config()
    debate = MultiAgentDebateActionGraph(
        name="MAD Minimal",
        description="Minimal runnable example for multi-agent debate",
        llm_config=llm_config,
    )

    open_problem = (
        "Should AI agent service engineers be required to take an algorithms exam to validate their"
        " competencies? Return a final Yes/No and up to five concise reasons, assuming responsibilities"
        " include tool/function orchestration, workflow design, RAG integration, evaluation/telemetry,"
        " reliability/safety, and rapid delivery."
    )
    result = debate.execute(
        problem=open_problem,
        num_agents=5,
        num_rounds=5,
        judge_mode="llm_judge",
        return_transcript=True,
    )

    print("=== Example: LLM Judge (Open Question) ===")
    print("Final Answer:", result.get("final_answer"))
    print("Winner:", result.get("winner"))
    print("\nTranscript:")
    for turn in result.get("transcript", []):
        print(
            f"[Round {turn['round']}] Agent#{turn['agent_id']} ({turn['role']})\n"
            f"Argument: {turn.get('argument','').strip()}\n"
            f"Answer: {str(turn.get('answer') or '').strip()}\n"
        )


def main():
    # Choose one to run: comment out the other one
    # run_self_consistency_example()
    run_llm_judge_example()


if __name__ == "__main__":
    main()



