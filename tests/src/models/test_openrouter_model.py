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


# ---------------------------------------------------------------------------
# 11. prepare_request — provider-specific prompt caching
# ---------------------------------------------------------------------------

def _caching_llm(model: str, **kwargs) -> OpenRouterLLM:
    return OpenRouterLLM(
        config=OpenRouterConfig(
            model=model, openrouter_key="mock_or_key", output_response=False, **kwargs
        )
    )


@pytest.fixture
def reset_caching_warnings():
    from evoagentx.models import openrouter_model

    openrouter_model._PROMPT_CACHING_COST_WARNED.clear()
    yield
    openrouter_model._PROMPT_CACHING_COST_WARNED.clear()


def test_prepare_request_disabled_by_default_is_noop():
    llm = _caching_llm("anthropic/claude-haiku-4.5")
    messages = [{"role": "user", "content": "stable prompt"}]
    out_messages, params = llm.prepare_request(messages, {"temperature": 0})

    # No opt-in -> request untouched; the flag never reaches the wire.
    assert out_messages is messages
    assert params == {"temperature": 0}


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-haiku-4.5", "google/gemini-2.5-flash", "qwen/qwen-plus"],
)
def test_prepare_request_enables_block_cache_without_mutating_input(model, reset_caching_warnings):
    llm = _caching_llm(model)
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "stable user prompt"},
    ]
    out_messages, params = llm.prepare_request(
        messages, {"enable_prompt_caching": True}
    )

    # Flag popped (never sent to the OpenAI-compatible API).
    assert "enable_prompt_caching" not in params
    # Original messages untouched.
    assert messages[1]["content"] == "stable user prompt"
    # Latest user text block carries the breakpoint in the returned copy.
    assert out_messages[1]["content"] == [
        {
            "type": "text",
            "text": "stable user prompt",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_prepare_request_marks_existing_text_block(reset_caching_warnings):
    llm = _caching_llm("qwen/qwen-plus")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "stable block"},
                {"type": "text", "text": "question"},
            ],
        }
    ]
    out_messages, _ = llm.prepare_request(messages, {"enable_prompt_caching": True})

    assert "cache_control" not in messages[0]["content"][0]
    assert "cache_control" not in out_messages[0]["content"][0]
    assert out_messages[0]["content"][1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-haiku-4.5", "google/gemini-2.5-flash", "qwen/qwen-plus"],
)
def test_prepare_request_moves_cache_to_latest_text_block(model, reset_caching_warnings):
    llm = _caching_llm(model)
    messages = [
        {"role": "system", "content": "stable system prompt"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "initial task",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "assistant", "content": "thinking before using a tool"},
        {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "continue"},
                {"type": "text", "text": "latest question"},
            ],
        },
    ]

    out_messages, _ = llm.prepare_request(messages, {"enable_prompt_caching": True})

    # Original caller-owned messages are untouched, including any pre-existing marker.
    assert messages[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # The returned request has one breakpoint on the latest cacheable text block.
    assert "cache_control" not in out_messages[1]["content"][0]
    assert "cache_control" not in out_messages[4]["content"][0]
    assert out_messages[4]["content"][1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-haiku-4.5", "google/gemini-2.5-flash", "qwen/qwen-plus"],
)
def test_prepare_request_marks_last_tool_result_text(model, reset_caching_warnings):
    llm = _caching_llm(model)
    messages = [
        {"role": "system", "content": "stable system prompt"},
        {"role": "user", "content": "call a tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
    ]

    out_messages, _ = llm.prepare_request(messages, {"enable_prompt_caching": True})

    assert messages[-1]["content"] == "tool result"
    assert out_messages[-1]["content"] == [
        {
            "type": "text",
            "text": "tool result",
            "cache_control": {"type": "ephemeral"},
        }
    ]


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-5.4-mini", "z-ai/glm-4.5-air", "deepseek/deepseek-v3.2"],
)
def test_prepare_request_automatic_cache_providers_untouched(model, reset_caching_warnings):
    llm = _caching_llm(model)
    messages = [{"role": "user", "content": "stable prompt"}]
    out_messages, params = llm.prepare_request(messages, {"enable_prompt_caching": True})

    # Opted in, but these providers auto-cache: leave the request alone.
    assert out_messages is messages
    assert "enable_prompt_caching" not in params


def test_prepare_request_config_default_enables_caching(reset_caching_warnings):
    llm = _caching_llm("anthropic/claude-haiku-4.5", enable_prompt_caching=True)
    messages = [{"role": "user", "content": "stable prompt"}]
    out_messages, _ = llm.prepare_request(messages, {})

    assert out_messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_prepare_request_warns_once_per_model(reset_caching_warnings, mocker):
    warn = mocker.patch("evoagentx.models.openrouter_model.logger.warning")
    llm = _caching_llm("anthropic/claude-haiku-4.5")
    messages = [{"role": "user", "content": "stable prompt"}]

    llm.prepare_request(messages, {"enable_prompt_caching": True})
    llm.prepare_request(messages, {"enable_prompt_caching": True})

    assert warn.call_count == 1
    assert "cost" in warn.call_args[0][0].lower() or "premium" in warn.call_args[0][0].lower()
