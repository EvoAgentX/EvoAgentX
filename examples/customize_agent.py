import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.core.module_utils import extract_code_blocks

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def build_customize_agent():

    # Configure LLM
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True)

    # Create an agent from a dictionary
    agent_data = {
        "name": "CodeWriter",
        "description": "Writes Python code based on requirements",
        "inputs": [
            {"name": "requirement", "type": "string", "description": "The coding requirement"}
        ],
        "outputs": [
            {"name": "code", "type": "string", "description": "The generated Python code"}
        ],
        "prompt": "Write Python code that implements the following requirement: {requirement}",
        "system_prompt": "You are an expert Python developer specialized in writing clean, efficient code. Respond only with code and brief explanations when necessary.",
        "llm_config": openai_config, 
        # "parse_mode": "str", # directly assign the LLM output to the attributes specified in the `outputs` field
        "parse_mode": "custom", 
        "parse_func": lambda content: {"code": extract_code_blocks(content)[0]} # use a custom function to parse the LLM output, `parse_mode` must be set to "custom" and the function must accept `content` as the input 
    }

    agent = CustomizeAgent.from_dict(agent_data) # create an agent from a dictionary

    # Execute the agent with input
    message = agent(
        inputs={"requirement": "Write a function that returns the sum of two numbers"}
    ) # the output of an agent is a Message object 
    content = message.content # the content of a Message object is a LLMOutputParser object

    print(f"Execution result of {agent.name}:")
    print(content.code) # use `code` attribute to get the code from the LLMOutputParser object


if __name__ == "__main__":
    build_customize_agent()