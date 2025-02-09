from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as AChoice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage

def get_openai_chat_completion() -> ChatCompletion:
    
    openai_chat_completion = ChatCompletion(
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
    return openai_chat_completion


def get_openai_chat_completion_chunk(usage_as_dict: bool = False) -> ChatCompletionChunk:

    usage = CompletionUsage(completion_tokens=1, prompt_tokens=22, total_tokens=23)
    usage = usage if not usage_as_dict else usage.model_dump()
    openai_chat_completion_chunk = ChatCompletionChunk(
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
    return openai_chat_completion_chunk


default_resp = get_openai_chat_completion()
default_resp_chunk = get_openai_chat_completion_chunk()

def mock_openai_completions_create(self, stream: bool=False, **kwargs):
    if stream:
        class Iterator(object):
            def __iter__(self):
                yield default_resp_chunk
        return Iterator()
    else:
        return default_resp