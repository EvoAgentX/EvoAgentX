import asyncio  
import os
import pytest
from evoagentx.core.message import Message, MessageType
from evoagentx.agents.agent import Agent
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.actions.action import Action

# Define a DummyAction for testing
class DummyAction(Action):
    def __init__(self, **kwargs):
        super().__init__(
            name="DummyAction",
            description="A test action that echoes input",
            **kwargs
        )

    def execute(self, llm, inputs, sys_msg=None, return_prompt=False, **kwargs):
        text = inputs.get("text", "")
        output = f"[Echo] {text}"
        prompt = f"Prompt used: {text}"
        if return_prompt:
            return output, prompt
        return output

class ShortTermMemoryTestAgent(Agent):
    """Inherit Agent for short-term memory testing"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.short_term_memory.max_size = 5
        self.add_action(DummyAction())

async def test_short_term_memory_via_agent():
    config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1,
    )
    llm = OpenAILLM(config=config)

    # Initialize agent
    agent = ShortTermMemoryTestAgent(
        llm=llm,
        rag_config=None,
        storage_handler=None,
        name="STMTestAgent",
        description="Test short-term memory",
    )

    print("=== Add a single message ===")
    await agent.async_execute("DummyAction", action_input_data={"text": "Hello 1"})
    for m in agent.short_term_memory.get():
        print(f"- {m.content}")

    print("\n=== Add multiple messages, exceeding max_size to test circular queue ===")
    for i in range(2, 8):
        await agent.async_execute("DummyAction", action_input_data={"text": f"Msg {i}"})

    buffer_msgs = agent.short_term_memory.get()
    print("Short-term memory content (should be at most 5 messages):")
    for i, m in enumerate(buffer_msgs, 1):
        print(f"{i}: {m.content}")

    # Check FIFO behavior (only look at response messages)
    responses = [m for m in buffer_msgs if isinstance(m.content, str) and m.content.startswith("[Echo]")]
    assert len(responses) <= agent.short_term_memory.max_size, "Short-term memory exceeded max_size"
    assert responses[0].content == "[Echo] Msg 5", "Oldest response message should be overwritten"
    assert responses[-1].content == "[Echo] Msg 7", "Latest response message should be at the end"

    print("\n✅ Short-term memory test passed!")

if __name__ == "__main__":
    asyncio.run(test_short_term_memory_via_agent())
