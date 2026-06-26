import pytest

from evoagentx.models.model_configs import OpenRouterConfig
from evoagentx.models.openrouter_model import OpenRouterLLM
from evoagentx.models import LLMOutputParser
from evoagentx.models.model_utils import cost_manager

from tests.src.models.mock_response import (
    mock_openrouter_completions_create,
    mock_openrouter_tool_call_create,
    mock_async_openrouter_create,
    mock_async_openrouter_tool_call_create,
)

OPENROUTER_MODEL = "openai/gpt-4o-mini"
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


def _make_llm(**kwargs) -> OpenRouterLLM:
    config = OpenRouterConfig(
        model=OPENROUTER_MODEL,
        openrouter_key="mock_or_key",
        output_response=False,
        **kwargs,
    )
    return OpenRouterLLM(config=config)


def _assert_cost_updated(model: str = OPENROUTER_MODEL):
    assert cost_manager.total_tokens[model] > 0, "No tokens recorded"
    assert cost_manager.cost_per_model[model] > 0, "No cost recorded"


# ---------------------------------------------------------------------------
# 1. Sync — non-streaming
# ---------------------------------------------------------------------------

def test_sync_non_stream(mocker):
    mocker.patch(SYNC_PATCH, mock_openrouter_completions_create)
    llm = _make_llm(stream=False)
    out = llm.generate(prompt="What is the capital of France?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Paris"
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 2. Sync — streaming
# ---------------------------------------------------------------------------

def test_sync_stream(mocker):
    mocker.patch(SYNC_PATCH, mock_openrouter_completions_create)
    llm = _make_llm(stream=True)
    out = llm.generate(prompt="What is the capital of France?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Paris"
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 3. Sync — tool call (non-streaming)
# ---------------------------------------------------------------------------

def test_sync_tool_call_non_stream(mocker):
    mocker.patch(SYNC_PATCH, mock_openrouter_tool_call_create)
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
    mocker.patch(SYNC_PATCH, mock_openrouter_tool_call_create)
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
    mocker.patch(ASYNC_PATCH, mock_async_openrouter_create)
    llm = _make_llm(stream=False)
    out = await llm.async_generate(prompt="What is the capital of France?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Paris"
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 6. Async — streaming
# ---------------------------------------------------------------------------

async def test_async_stream(mocker):
    mocker.patch(ASYNC_PATCH, mock_async_openrouter_create)
    llm = _make_llm(stream=True)
    out = await llm.async_generate(prompt="What is the capital of France?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Paris"
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 7. Async — tool call (non-streaming)
# ---------------------------------------------------------------------------

async def test_async_tool_call_non_stream(mocker):
    mocker.patch(ASYNC_PATCH, mock_async_openrouter_tool_call_create)
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
    mocker.patch(ASYNC_PATCH, mock_async_openrouter_tool_call_create)
    llm = _make_llm(stream=True, tools=[GET_WEATHER_TOOL], tool_choice="auto")
    out = await llm.async_generate(prompt="What is the weather in Tokyo?")
    assert isinstance(out, LLMOutputParser)
    assert "<tool_call>" in out.content
    assert "get_weather" in out.content
    _assert_cost_updated()


# ---------------------------------------------------------------------------
# 9. Cost — missing usage.cost logs warning and records 0
# ---------------------------------------------------------------------------

def test_missing_cost_warns(mocker):
    """When usage.cost is absent, cost is recorded as 0 without raising."""
    from unittest.mock import MagicMock

    def mock_no_cost_create(self, stream=False, **kwargs):
        resp = MagicMock()
        resp.id = "or_no_cost"
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15
        del usage.cost  # force getattr to fall back to MagicMock default
        usage.cost = None  # explicitly None → triggers the warning path
        resp.usage = usage
        resp.choices[0].message.content = "Hello"
        resp.choices[0].message.tool_calls = None
        return resp

    mocker.patch(SYNC_PATCH, mock_no_cost_create)
    llm = _make_llm(stream=False)
    out = llm.generate(prompt="Hello")
    assert isinstance(out, LLMOutputParser)
    assert cost_manager.cost_per_model[OPENROUTER_MODEL] == 0.0
    assert cost_manager.total_tokens[OPENROUTER_MODEL] == 15


# ---------------------------------------------------------------------------
# 10. Cost accumulation across multiple calls
# ---------------------------------------------------------------------------

def test_cost_accumulation(mocker):
    mocker.patch(SYNC_PATCH, mock_openrouter_completions_create)
    llm = _make_llm(stream=False)

    llm.generate(prompt="Call 1")
    tokens_1 = cost_manager.total_tokens[OPENROUTER_MODEL]
    cost_1 = cost_manager.cost_per_model[OPENROUTER_MODEL]

    llm.generate(prompt="Call 2")
    tokens_2 = cost_manager.total_tokens[OPENROUTER_MODEL]
    cost_2 = cost_manager.cost_per_model[OPENROUTER_MODEL]

    assert tokens_2 == tokens_1 * 2
    assert cost_2 == pytest.approx(cost_1 * 2)
