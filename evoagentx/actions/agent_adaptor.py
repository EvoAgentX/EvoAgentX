import json
from typing import Optional, Tuple, Type, Union

from ..actions.action import Action, ActionInput, ActionOutput
from ..agents import Agent
from ..core.parser import Parser
from ..models import BaseLLM, OpenAILLM, OpenAILLMConfig
from ..prompts.agent_adaptor import DATA_TRANSFORM_PROMPT
from ..utils.utils import get_unique_class_name


class AgentAdaptorAction(Action):
    """
    Transforms the input to the format expected by the agent, and transforms the agent's output to the format specified by the user.
    """

    def __init__(
        self, 
        agent: Agent, 
        inputs_format: Type[ActionInput], 
        outputs_format: Type[ActionOutput],
        llm: Optional[BaseLLM] = None
    ):

        if llm is None:
            llm = OpenAILLM(config=OpenAILLMConfig(model="gpt-4o-mini"))

        # All agents have a `ContextExtraction` action, filter it out
        non_ContextExtraction_actions = [
            action for action in agent.actions if action.class_name != "ContextExtraction"
        ]

        if len(non_ContextExtraction_actions) > 1:
            raise ValueError(f"AgentAdaptorAction currently only supports agents with a single action. {agent.name} has {len(non_ContextExtraction_actions)} actions.")
        
        name = get_unique_class_name(agent.actions[0].class_name)
        description = agent.actions[0].description
        super().__init__(name=name, description=description, inputs_format=inputs_format, outputs_format=outputs_format)
        self.agent = agent
        self.llm = llm

    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> Optional[Union[Parser, Tuple[Parser, str]]]:
        prompt = []
        inputs_json = json.dumps(inputs, indent=4)
        
        input_transform_prompt = DATA_TRANSFORM_PROMPT.format(
            data=inputs_json,
            format=self.agent.actions[0].inputs_format.get_specification()
        )

        prompt.append("Input transform prompt:\n" + input_transform_prompt)

        transformed_input = self.llm.generate(
            prompt=input_transform_prompt,
            parser=self.agent.actions[0].inputs_format,
            parse_mode="json"
        )

        transformed_input_dict = transformed_input.get_structured_data()
        prompt.append("Input transform output:\n" + json.dumps(transformed_input_dict, indent=4, ensure_ascii=False))

        agent_output = self.agent.execute(
            action_name=self.agent.actions[0].name,
            action_input_data=transformed_input_dict
        )

        prompt.append(f"{self.agent.name} prompt:\n" + str(agent_output.prompt))

        agent_output_dict = agent_output.content.get_structured_data()
        agent_output_json = json.dumps(agent_output_dict, indent=4, ensure_ascii=False)
        prompt.append(f"{self.agent.name} output:\n" + agent_output_json)

        output_transform_prompt = DATA_TRANSFORM_PROMPT.format(
            data=agent_output_json, 
            format=self.outputs_format.get_specification()
        )

        prompt.append("Output transform prompt:\n" + output_transform_prompt)

        final_output = self.llm.generate(
            prompt=output_transform_prompt,
            parser=self.outputs_format,
            parse_mode="json"
        )

        if return_prompt:
            return final_output, "\n\n".join(prompt)

        return final_output


    async def async_execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> Optional[Union[Parser, Tuple[Parser, str]]]:
        prompt = []
        inputs_json = json.dumps(inputs, indent=4)
        
        input_transform_prompt = DATA_TRANSFORM_PROMPT.format(
            data=inputs_json,
            format=self.agent.actions[0].inputs_format.get_specification()
        )

        prompt.append("Input transform prompt:\n" + input_transform_prompt)

        transformed_input = await self.llm.async_generate(
            prompt=input_transform_prompt,
            parser=self.agent.actions[0].inputs_format,
            parse_mode="json"
        )

        transformed_input_dict = transformed_input.get_structured_data()
        prompt.append("Input transform output:\n" + json.dumps(transformed_input_dict, indent=4, ensure_ascii=False))

        agent_output = await self.agent.async_execute(
            action_name=self.agent.actions[0].name,
            action_input_data=transformed_input_dict
        )

        prompt.append(f"{self.agent.name} prompt:\n" + str(agent_output.prompt))

        agent_output_dict = agent_output.content.get_structured_data()
        agent_output_json = json.dumps(agent_output_dict, indent=4, ensure_ascii=False)
        prompt.append(f"{self.agent.name} output:\n" + agent_output_json)

        output_transform_prompt = DATA_TRANSFORM_PROMPT.format(
            data=agent_output_json, 
            format=self.outputs_format.get_specification()
        )

        prompt.append("Output transform prompt:\n" + output_transform_prompt)

        final_output = await self.llm.async_generate(
            prompt=output_transform_prompt,
            parser=self.outputs_format,
            parse_mode="json"
        )

        if return_prompt:
            return final_output, "\n\n".join(prompt)

        return final_output