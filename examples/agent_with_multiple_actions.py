import os 
from pydantic import Field
from dotenv import load_dotenv

from typing import Optional
from evoagentx.models import BaseLLM, OpenAILLMConfig
from evoagentx.agents import Agent
from evoagentx.actions import Action, ActionInput, ActionOutput

load_dotenv()

# Define the CodeGeneration action inputs
class TestCodeGenerationInput(ActionInput):
    requirement: str = Field(description="The requirement for the code generation")

# Define the CodeGeneration action outputs
class TestCodeGenerationOutput(ActionOutput):
    code: str = Field(description="The generated code")

# Define the CodeGeneration action
class TestCodeGeneration(Action): 

    def __init__(
        self, 
        name: str="TestCodeGeneration", 
        description: str="Generate code based on requirements", 
        prompt: str="Generate code based on requirements: {requirement}",
        inputs_format: ActionInput=None, 
        outputs_format: ActionOutput=None, 
        **kwargs
    ):
        inputs_format = inputs_format or TestCodeGenerationInput
        outputs_format = outputs_format or TestCodeGenerationOutput
        super().__init__(
            name=name, 
            description=description, 
            prompt=prompt, 
            inputs_format=inputs_format, 
            outputs_format=outputs_format, 
            **kwargs
        )
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> TestCodeGenerationOutput:
        
        action_input_attrs = self.inputs_format.get_attrs() # obtain the attributes of the action input 
        action_input_data = {attr: inputs.get(attr, "undefined") for attr in action_input_attrs}
        prompt = self.prompt.format(**action_input_data) # format the prompt with the action input data 
        output = llm.generate(
            prompt=prompt, 
            system_message=sys_msg, 
            parser=self.outputs_format, 
            parse_mode="str" # specify how to parse the output 
        )
        if return_prompt:
            return output, prompt
        return output


# Define the CodeReview action inputs
class TestCodeReviewInput(ActionInput):
    code: str = Field(description="The code to be reviewed")
    requirements: str = Field(description="The requirements for the code review")

# Define the CodeReview action outputs
class TestCodeReviewOutput(ActionOutput):
    review: str = Field(description="The review of the code")

# Define the CodeReview action
class TestCodeReview(Action):
    def __init__(
        self, 
        name: str="TestCodeReview", 
        description: str="Review the code based on requirements", 
        prompt: str="Review the following code based on the requirements:\n\nRequirements: {requirements}\n\nCode:\n{code}.\n\nYou should output a JSON object with the following format:\n```json\n{{\n'review': '...'\n}}\n```",
        inputs_format: ActionInput=None, 
        outputs_format: ActionOutput=None, 
        **kwargs
    ):
        inputs_format = inputs_format or TestCodeReviewInput
        outputs_format = outputs_format or TestCodeReviewOutput
        super().__init__(
            name=name, 
            description=description, 
            prompt=prompt, 
            inputs_format=inputs_format, 
            outputs_format=outputs_format, 
            **kwargs
        )
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> TestCodeReviewOutput:
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, "undefined") for attr in action_input_attrs}
        prompt = self.prompt.format(**action_input_data)
        output = llm.generate(
            prompt=prompt, 
            system_message=sys_msg,
            parser=self.outputs_format, 
            parse_mode="json" # specify how to parse the output 
        ) 
        if return_prompt:
            return output, prompt
        return output


def main():

    # Initialize the LLM
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=os.getenv("OPENAI_API_KEY")) 

    # Define the agent 
    developer = Agent(
        name="Developer", 
        description="A developer who can write code and review code",
        actions=[TestCodeGeneration(), TestCodeReview()], 
        llm_config=openai_config
    )

    # [optional] save the agent to a file
    # developer.save_module("examples/output/developer.json")
    # [optional] load the agent from a file
    # developer = Agent.from_file("examples/output/developer.json", llm_config=openai_config)

    # List all available actions on the agent
    actions = developer.get_all_actions()
    print(f"Available actions of agent {developer.name}:")
    for action in actions:
        print(f"- {action.name}: {action.description}")

    # Generate some code using the CodeGeneration action
    generation_result = developer.execute(
        action_name="TestCodeGeneration", # specify the action name
        action_input_data={ 
            "requirement": "Write a function that returns the sum of two numbers"
        }
    )

    # Access the generated code
    generated_code = generation_result.content.code
    print("Generated code:")
    print(generated_code)

    # Review the generated code using the CodeReview action
    review_result = developer.execute(
        action_name="TestCodeReview",
        action_input_data={
            "requirements": "Write a function that returns the sum of two numbers",
            "code": generated_code
        }
    )

    # Access the review results
    review = review_result.content.review
    print("\nReview:")
    print(review)


if __name__ == "__main__":
    main()