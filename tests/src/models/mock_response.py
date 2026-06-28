import json
from unittest.mock import MagicMock

from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.chat.chat_completion_chunk import Choice as AChoice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function


# ---------------------------------------------------------------------------
# OpenAI — non-streaming helpers
# ---------------------------------------------------------------------------

def get_openai_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="xxxx",
        model="model_name",
        object="chat.completion",
        created=11111,
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Beijing"),
                logprobs=None,
            )
        ],
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=22, total_tokens=23),
    )


def get_openai_tool_call_completion() -> ChatCompletion:
    return ChatCompletion(
        id="tool_call_xxxx",
        model="model_name",
        object="chat.completion",
        created=11111,
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_abc123",
                            type="function",
                            function=Function(name="get_weather", arguments='{"city": "Tokyo"}'),
                        )
                    ],
                ),
                logprobs=None,
            )
        ],
        usage=CompletionUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30),
    )


# ---------------------------------------------------------------------------
# OpenAI — streaming helpers
# ---------------------------------------------------------------------------

def get_openai_chat_completion_chunk(usage_as_dict: bool = False) -> ChatCompletionChunk:
    usage = CompletionUsage(completion_tokens=1, prompt_tokens=22, total_tokens=23)
    usage = usage if not usage_as_dict else usage.model_dump()
    return ChatCompletionChunk(
        id="xxxx",
        model="model_name",
        object="chat.completion.chunk",
        created=11111,
        choices=[
            AChoice(
                delta=ChoiceDelta(role="assistant", content="Beijing"),
                finish_reason="stop",
                index=0,
                logprobs=None,
            )
        ],
        usage=usage,
    )


def get_openai_stream_chunks() -> list:
    """Content chunks followed by a usage-only final chunk."""
    content_chunk = ChatCompletionChunk(
        id="stream_xxxx",
        model="model_name",
        object="chat.completion.chunk",
        created=11111,
        choices=[
            AChoice(
                delta=ChoiceDelta(role="assistant", content="Paris"),
                finish_reason=None,
                index=0,
                logprobs=None,
            )
        ],
    )
    usage_chunk = ChatCompletionChunk(
        id="stream_xxxx",
        model="model_name",
        object="chat.completion.chunk",
        created=11111,
        choices=[],
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=22, total_tokens=23),
    )
    return [content_chunk, usage_chunk]


def get_openai_tool_call_chunks() -> list:
    """Streaming tool-call: name chunk → arguments chunk → usage chunk."""
    name_chunk = ChatCompletionChunk(
        id="tool_stream_xxxx",
        model="model_name",
        object="chat.completion.chunk",
        created=11111,
        choices=[
            AChoice(
                delta=ChoiceDelta(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_abc123",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=""),
                        )
                    ],
                ),
                finish_reason=None,
                index=0,
                logprobs=None,
            )
        ],
    )
    args_chunk = ChatCompletionChunk(
        id="tool_stream_xxxx",
        model="model_name",
        object="chat.completion.chunk",
        created=11111,
        choices=[
            AChoice(
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            function=ChoiceDeltaToolCallFunction(arguments='{"city": "Tokyo"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
                index=0,
                logprobs=None,
            )
        ],
    )
    usage_chunk = ChatCompletionChunk(
        id="tool_stream_xxxx",
        model="model_name",
        object="chat.completion.chunk",
        created=11111,
        choices=[],
        usage=CompletionUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30),
    )
    return [name_chunk, args_chunk, usage_chunk]


# ---------------------------------------------------------------------------
# Async iterator wrapper (for async streaming mocks)
# ---------------------------------------------------------------------------

class AsyncChunkIterator:
    def __init__(self, chunks: list):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


# ---------------------------------------------------------------------------
# OpenAI sync/async mock factories
# ---------------------------------------------------------------------------

default_resp = get_openai_chat_completion()
default_resp_chunk = get_openai_chat_completion_chunk()


def mock_openai_completions_create(self, stream: bool = False, **kwargs):
    if stream:
        class SyncIterator:
            def __iter__(self_inner):
                yield default_resp_chunk
        return SyncIterator()
    return default_resp


def mock_openai_stream_completions_create(self, stream: bool = False, **kwargs):
    """Returns proper multi-chunk stream with final usage chunk."""
    if stream:
        chunks = get_openai_stream_chunks()
        class SyncIterator:
            def __iter__(self_inner):
                for c in chunks:
                    yield c
        return SyncIterator()
    return get_openai_chat_completion()


def mock_openai_tool_call_create(self, stream: bool = False, **kwargs):
    if stream:
        chunks = get_openai_tool_call_chunks()
        class SyncIterator:
            def __iter__(self_inner):
                for c in chunks:
                    yield c
        return SyncIterator()
    return get_openai_tool_call_completion()


async def mock_async_openai_create(self, stream: bool = False, **kwargs):
    if stream:
        return AsyncChunkIterator(get_openai_stream_chunks())
    return get_openai_chat_completion()


async def mock_async_openai_tool_call_create(self, stream: bool = False, **kwargs):
    if stream:
        return AsyncChunkIterator(get_openai_tool_call_chunks())
    return get_openai_tool_call_completion()


# ---------------------------------------------------------------------------
# OpenRouter mock helpers (MagicMock-based to attach usage.cost)
# ---------------------------------------------------------------------------

def _make_or_usage(prompt_tokens=22, completion_tokens=1, cost=0.000015):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    usage.cost = cost
    return usage


def get_openrouter_chat_completion(content="Paris", cost=0.000015) -> MagicMock:
    resp = MagicMock()
    resp.id = "or_xxxx"
    resp.usage = _make_or_usage(cost=cost)
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    return resp


def get_openrouter_tool_call_completion(cost=0.000015) -> MagicMock:
    resp = MagicMock()
    resp.id = "or_tool_xxxx"
    resp.usage = _make_or_usage(prompt_tokens=20, completion_tokens=10, cost=cost)
    resp.choices[0].message.content = None
    # Build a minimal tool-call object compatible with format_tool_calls
    tc = MagicMock()
    tc.id = "call_or123"
    tc.function.name = "get_weather"
    tc.function.arguments = '{"city": "Tokyo"}'
    resp.choices[0].message.tool_calls = [tc]
    return resp


def get_openrouter_stream_chunks(content="Paris", cost=0.000015) -> list:
    content_chunk = MagicMock()
    content_chunk.usage = None
    content_chunk.choices = [MagicMock()]
    content_chunk.choices[0].delta.content = content
    content_chunk.choices[0].delta.tool_calls = None

    usage_chunk = MagicMock()
    usage_chunk.usage = _make_or_usage(cost=cost)
    usage_chunk.choices = []
    return [content_chunk, usage_chunk]


def get_openrouter_tool_call_stream_chunks(cost=0.000015) -> list:
    name_chunk = MagicMock()
    name_chunk.usage = None
    name_chunk.choices = [MagicMock()]
    name_chunk.choices[0].delta.content = None
    tc_name = MagicMock()
    tc_name.index = 0
    tc_name.id = "call_or123"
    tc_name.function.name = "get_weather"
    tc_name.function.arguments = ""
    name_chunk.choices[0].delta.tool_calls = [tc_name]

    args_chunk = MagicMock()
    args_chunk.usage = None
    args_chunk.choices = [MagicMock()]
    args_chunk.choices[0].delta.content = None
    tc_args = MagicMock()
    tc_args.index = 0
    tc_args.id = None
    tc_args.function.name = None
    tc_args.function.arguments = '{"city": "Tokyo"}'
    args_chunk.choices[0].delta.tool_calls = [tc_args]

    usage_chunk = MagicMock()
    usage_chunk.usage = _make_or_usage(prompt_tokens=20, completion_tokens=10, cost=cost)
    usage_chunk.choices = []
    return [name_chunk, args_chunk, usage_chunk]


def mock_openrouter_completions_create(self, stream: bool = False, **kwargs):
    if stream:
        chunks = get_openrouter_stream_chunks()
        class SyncIterator:
            def __iter__(self_inner):
                for c in chunks:
                    yield c
        return SyncIterator()
    return get_openrouter_chat_completion()


def mock_openrouter_tool_call_create(self, stream: bool = False, **kwargs):
    if stream:
        chunks = get_openrouter_tool_call_stream_chunks()
        class SyncIterator:
            def __iter__(self_inner):
                for c in chunks:
                    yield c
        return SyncIterator()
    return get_openrouter_tool_call_completion()


async def mock_async_openrouter_create(self, stream: bool = False, **kwargs):
    if stream:
        return AsyncChunkIterator(get_openrouter_stream_chunks())
    return get_openrouter_chat_completion()


async def mock_async_openrouter_tool_call_create(self, stream: bool = False, **kwargs):
    if stream:
        return AsyncChunkIterator(get_openrouter_tool_call_stream_chunks())
    return get_openrouter_tool_call_completion()
