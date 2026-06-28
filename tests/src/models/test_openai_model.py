import pytest

from evoagentx.models import OpenAILLMConfig, OpenAILLM, LLMOutputParser
from evoagentx.models.model_utils import cost_manager

from tests.src.models.mock_response import (
    mock_openai_completions_create,
    mock_openai_stream_completions_create,
    mock_openai_tool_call_create,
    mock_async_openai_create,
    mock_async_openai_tool_call_create,
)

OPENAI_MODEL = "gpt-4o-mini"
SYNC_PATCH = "openai.resources.chat.completions.Completions.create"
ASYNC_PATCH = "openai.resources.chat.completions.AsyncCompletions.create"

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cost_manager():
    cost_manager.input_tokens.clear()
    cost_manager.output_tokens.clear()
    cost_manager.total_tokens.clear()
    cost_manager.cost_per_model.clear()
    yield


def _make_llm(**kwargs) -> OpenAILLM:
    config = OpenAILLMConfig(
        model=OPENAI_MODEL,
        openai_key="mock_openai_key",
        output_response=False,
        **kwargs,
    )
    return OpenAILLM(config=config)


def _assert_cost_updated(model: str = OPENAI_MODEL):
    assert cost_manager.total_tokens[model] > 0, "No tokens recorded for cost tracking"


# ---------------------------------------------------------------------------
# 1. Sync — non-streaming (input format variants)
# ---------------------------------------------------------------------------

def test_sync_non_stream_prompt(mocker):
    mocker.patch(SYNC_PATCH, mock_openai_completions_create)
    llm = _make_llm(stream=False)
    prompt = "What is the capital of China?"
    system = "You are a geography expert."

    out = llm.generate(prompt=prompt, system_message=system)
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Beijing"

    out = llm.generate(prompt=[prompt], system_message=[system])
    assert isinstance(out, list)
    assert out[0].content == "Beijing"

    out = llm.generate(messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Beijing"

    out = llm.generate(messages=[[{"role": "system", "content": system}, {"role": "user", "content": prompt}]])
    assert isinstance(out, list)
    assert out[0].content == "Beijing"

    _assert_cost_updated()
    assert cost_manager.total_tokens[OPENAI_MODEL] == 23 * 4


# ---------------------------------------------------------------------------
# 2. Sync — streaming
# ---------------------------------------------------------------------------

def test_sync_stream(mocker):
    mocker.patch(SYNC_PATCH, mock_openai_stream_completions_create)
    llm = _make_llm(stream=True)
    out = llm.generate(prompt="What is the capital of France?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Paris"
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 3. Sync — tool call (non-streaming)
# ---------------------------------------------------------------------------

def test_sync_tool_call_non_stream(mocker):
    mocker.patch(SYNC_PATCH, mock_openai_tool_call_create)
    llm = _make_llm(stream=False, tools=[GET_WEATHER_TOOL], tool_choice="auto")
    out = llm.generate(prompt="What is the weather in Tokyo?")
    assert isinstance(out, LLMOutputParser)
    assert "<tool_call>" in out.content
    assert "get_weather" in out.content
    assert "Tokyo" in out.content
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 4. Sync — tool call (streaming)
# ---------------------------------------------------------------------------

def test_sync_tool_call_stream(mocker):
    mocker.patch(SYNC_PATCH, mock_openai_tool_call_create)
    llm = _make_llm(stream=True, tools=[GET_WEATHER_TOOL], tool_choice="auto")
    out = llm.generate(prompt="What is the weather in Tokyo?")
    assert isinstance(out, LLMOutputParser)
    assert "<tool_call>" in out.content
    assert "get_weather" in out.content
    assert "Tokyo" in out.content
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 5. Async — non-streaming
# ---------------------------------------------------------------------------

async def test_async_non_stream(mocker):
    mocker.patch(ASYNC_PATCH, mock_async_openai_create)
    llm = _make_llm(stream=False)
    out = await llm.async_generate(prompt="What is the capital of China?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Beijing"
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 6. Async — streaming
# ---------------------------------------------------------------------------

async def test_async_stream(mocker):
    mocker.patch(ASYNC_PATCH, mock_async_openai_create)
    llm = _make_llm(stream=True)
    out = await llm.async_generate(prompt="What is the capital of France?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Paris"
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 7. Async — tool call (non-streaming)
# ---------------------------------------------------------------------------

async def test_async_tool_call_non_stream(mocker):
    mocker.patch(ASYNC_PATCH, mock_async_openai_tool_call_create)
    llm = _make_llm(stream=False, tools=[GET_WEATHER_TOOL], tool_choice="auto")
    out = await llm.async_generate(prompt="What is the weather in Tokyo?")
    assert isinstance(out, LLMOutputParser)
    assert "<tool_call>" in out.content
    assert "get_weather" in out.content
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 8. Async — tool call (streaming)
# ---------------------------------------------------------------------------

async def test_async_tool_call_stream(mocker):
    mocker.patch(ASYNC_PATCH, mock_async_openai_tool_call_create)
    llm = _make_llm(stream=True, tools=[GET_WEATHER_TOOL], tool_choice="auto")
    out = await llm.async_generate(prompt="What is the weather in Tokyo?")
    assert isinstance(out, LLMOutputParser)
    assert "<tool_call>" in out.content
    assert "get_weather" in out.content
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 9. stream_options include_usage explicitly set
# ---------------------------------------------------------------------------

def test_stream_with_include_usage(mocker):
    mocker.patch(SYNC_PATCH, mock_openai_stream_completions_create)
    llm = _make_llm(stream=True, stream_options={"include_usage": True})
    out = llm.generate(prompt="What is the capital of France?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Paris"
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 10. Cost tracking — verify token counts are accumulated correctly
# ---------------------------------------------------------------------------

def test_cost_accumulation(mocker):
    mocker.patch(SYNC_PATCH, mock_openai_completions_create)
    llm = _make_llm(stream=False)
    prompt = "Test prompt"

    llm.generate(prompt=prompt)
    tokens_after_1 = cost_manager.total_tokens[OPENAI_MODEL]

    llm.generate(prompt=prompt)
    tokens_after_2 = cost_manager.total_tokens[OPENAI_MODEL]

    assert tokens_after_2 == tokens_after_1 * 2
    assert cost_manager.input_tokens[OPENAI_MODEL] > 0
    assert cost_manager.output_tokens[OPENAI_MODEL] > 0
