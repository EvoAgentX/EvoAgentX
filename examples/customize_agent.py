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

    # Create an agent from a dictionary
    agent_data = {
        "name": "FirstAgent",
        "description": "A simple agent that prints hello world",
        "prompt": "Print 'hello world'",
        "llm_config": openai_config
    }
    agent = CustomizeAgent.from_dict(agent_data)

    # Execute the agent
    message: Message = agent() # the output of an agent is a Message object 

    print(f"Response from {agent.name}:")
    print(message.content.content) # the content of a Message object is a LLMOutputParser object


if __name__ == "__main__":
    build_customize_agent()