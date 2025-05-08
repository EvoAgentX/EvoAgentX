import os 
from dotenv import load_dotenv
from evoagentx.core import Message 
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def build_customize_agent():

    # Configure LLM
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True)

    describe_module = CustomizeAgent(
        name="DescribeModule",
        description="An agent that describes a module's functionality in a program",
        prompt="""
        Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of one of the specified module in this pipeline.
        {program_code}
        {program_example}
        {program_description}
        {module}
        """,
        inputs = [
            {"name": "program_code", "type": "str", "description": "Pseudocode for a language model program designed to solve a particular task"},
            {"name": "program_example", "type": "str", "description": "An example of the program in use"},
            {"name": "program_description", "type": "str", "description": "Summary of the task the program is designed to solve, and how it goes about solving it"},
            {"name": "module", "type": "str", "description": "The module in the program that we want to describe"}
        ],
        outputs = [
            {"name": "module_description", "type": "str", "description": "Description of the module's role in the broader program"}
        ],
        llm_config=openai_config,
        parse_mode="str",
    )
    
    # 获取 action 的输入和输出字段
    
    message = describe_module(inputs = {"program_code": "hello", "program_example": "world", "program_description": "world", "module": "hello"})
    print(message.content.module_description)
if __name__ == "__main__":
    build_customize_agent()