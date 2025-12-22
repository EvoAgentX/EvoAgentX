from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.models import LLMOutputParser 
from evoagentx.models import cost_manager

from tests.src.models.mock_response import mock_openai_completions_create

    
def test_openai_generation(mocker):

    mocker.patch("openai.resources.chat.completions.Completions.create", mock_openai_completions_create)

    model_name = "gpt-4o-mini"
    config = OpenAILLMConfig(model=model_name, openai_key="mock_openai_key", output_response=False)
    model = OpenAILLM(config)

    prompt = "what is the capital city of China. Only output the answer."
    system_prompt = "You are an expert in geography"

    # test different input formats
    output = model.generate(prompt=prompt, system_message=system_prompt)
    assert isinstance(output, LLMOutputParser)
    assert output.content == "Beijing"
    assert str(output) == "Beijing"
    assert cost_manager.total_tokens[model_name] == 23 

    output = model.generate(prompt=[prompt], system_message=[system_prompt])
    assert isinstance(output, list) and isinstance(output[0], LLMOutputParser)
    assert output[0].content == "Beijing"
    assert str(output[0]) == "Beijing"
    assert cost_manager.total_tokens[model_name] == 23*2 

    output = model.generate(messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}])
    assert isinstance(output, LLMOutputParser)
    assert output.content == "Beijing"
    assert str(output) == "Beijing"
    assert cost_manager.total_tokens[model_name] == 23*3

    output = model.generate(messages=[[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]])
    assert isinstance(output, list) and isinstance(output[0], LLMOutputParser)
    assert output[0].content == "Beijing"
    assert str(output[0]) == "Beijing"
    assert cost_manager.total_tokens[model_name] == 23*4

    # test stream output 
    output = model.generate(prompt=prompt, system_message=system_prompt, stream=True)
    assert isinstance(output, LLMOutputParser)
    assert output.content == "Beijing"
    assert str(output) == "Beijing"
    assert cost_manager.total_tokens[model_name] > 23*4 

