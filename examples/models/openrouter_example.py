import os 
from dotenv import load_dotenv
from evoagentx.core import Message 
from evoagentx.models import OpenRouterConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate 
# from evoagentx.models import OpenRouterLLM
from evoagentx.core.module_utils import extract_code_blocks

load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_config = OpenRouterConfig(
    model="deepseek/deepseek-r1-0528-qwen3-8b:free", 
    openrouter_key=OPEN_ROUTER_API_KEY, 
    output_response=True,
    max_tokens=1,
    temperature=0.5
)

def build_customize_agent():

    # Create an agent from a dictionary
    agent_data = {
        "name": "FirstAgent",
        "description": "A simple agent that prints hello world",
        "prompt": "Print 'hello world'", 
        "llm_config": openrouter_config
    }
    agent = CustomizeAgent.from_dict(agent_data)

    # Execute the agent
    message: Message = agent() # the output of an agent is a Message object 

    print(f"Response from {agent.name}:")
    print(message.content.content) # the content of a Message object is a LLMOutputParser object


def build_customize_agent_with_inputs():

    simple_agent = CustomizeAgent(
        name="SimpleAgent",
        description="A basic agent that responds to queries",
        prompt="Answer the following question: {question}",
        llm_config=openrouter_config,
        inputs=[
            {"name": "question", "type": "string", "description": "The question to answer"}
        ]
    )
    # Execute the agent
    response = simple_agent(inputs={"question": "What is a language model?"})
    print(f"Response from {simple_agent.name}:")
    print(response.content.content)  # Access the raw response content 


def build_customize_agent_with_inputs_and_outputs():

    code_writer = CustomizeAgent(
        name="CodeWriter",
        description="Writes Python code based on requirements",
        prompt="Write Python code that implements the following requirement: {requirement}",
        llm_config=openrouter_config,
        inputs=[
            {"name": "requirement", "type": "string", "description": "The coding requirement"}
        ],
        outputs=[
            {"name": "code", "type": "string", "description": "The generated Python code"}
        ],
        parse_mode="str" # use the raw LLM output as the value for each output field
    )

    message = code_writer(
        inputs={"requirement": "Write a function that returns the sum of two numbers"}
    )
    print(f"Response from {code_writer.name}:")
    print(message.content.code)


def build_customize_agent_with_custom_parse_func():

    code_writer = CustomizeAgent(
        name="CodeWriter",
        description="Writes Python code based on requirements",
        prompt="Write Python code that implements the following requirement: {requirement}",
        llm_config=openrouter_config,
        inputs=[
            {"name": "requirement", "type": "string", "description": "The coding requirement"}
        ],
        outputs=[
            {"name": "code", "type": "string", "description": "The generated Python code"}
        ],
        parse_mode="custom", 
        parse_func=lambda content: {"code": extract_code_blocks(content)[0]}  # Extract first code block
    )

    message = code_writer(
        inputs={"requirement": "Write a function that returns the sum of two numbers"}
    )
    print(f"Response from {code_writer.name}:")
    print(message.content.code)


def build_customize_agent_with_prompt_template():

    agent = CustomizeAgent(
        name="FirstAgent",
        description="A simple agent that prints hello world",
        prompt_template=StringTemplate(
            instruction="Print 'hello world'",
        ),
        llm_config=openrouter_config
    )

    message = agent()
    print(f"Response from {agent.name}:")
    print(message.content.content)


def build_customize_agent_with_inputs_and_outputs_and_prompt_template(): 

    code_writer = CustomizeAgent(
        name="CodeWriter",
        description="Writes Python code based on requirements",
        prompt_template= StringTemplate(
            instruction="Write Python code that implements the provided `requirement`",
            # demonstrations=[
            #     {
            #         "requirement": "print 'hello world'",
            #         "code": "print('hello world')"
            #     }, 
            #     {
            #         "requirement": "print 'Test Demonstration'",
            #         "code": "print('Test Demonstration')"
            #     }
            # ]
        ), # no need to specify input placeholders in the instruction of the prompt template
        llm_config=openrouter_config,
        inputs=[
            {"name": "requirement", "type": "string", "description": "The coding requirement"}
        ],
        outputs=[
            {"name": "code", "type": "string", "description": "The generated Python code"},
        ],
        parse_mode="custom", 
        parse_func=lambda content: {"code": extract_code_blocks(content)[0]}
    )

    message = code_writer(
        inputs={"requirement": "Write a function that returns the sum of two numbers"}
    )
    print(f"Response from {code_writer.name}:")
    print(message.content.code)


if __name__ == "__main__":
    build_customize_agent()
    build_customize_agent_with_inputs()
    build_customize_agent_with_inputs_and_outputs()
    build_customize_agent_with_custom_parse_func()
    build_customize_agent_with_prompt_template()
    build_customize_agent_with_inputs_and_outputs_and_prompt_template()