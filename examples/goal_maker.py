import os 
from dotenv import load_dotenv 
from evoagentx.models import OpenAILLMConfig, OpenAILLM, LiteLLMConfig, LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification 
from evoagentx.core.module_utils import extract_code_blocks
from evoagentx.core.message import Message
load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") 


PROMPT_MAKER_PROMPT = """
You are a prompt engineer. You will be given a task description and you should output a prompt used for multi_agent workflow generation and execution.

Here are some key points you should consider when generating the prompt: 
- You should instruct to use tool calling action more often, especially there is a tool that can retrieve needed data.
- You should analyze the task and think about what information should be in the output. For example, a job analysis task should output the job title, company name, location, job description, job requirements, salary, job posting link, etc, and maybe a short summary on the resume at the beginning of the output.
- Your output should containing only the "goal", which is a detailed string input for the workflow generator and execution.

Here are some key points you should put into the generated prompts: 
- Usually, the input of a workflow is a long string called "goal" and no other inputs. This "goal" is what you will generate. In this situation, yous should state that this "goal" is the input of the workflow in your generated prompt.
- The output for a workflow is usally a markdown text, unless otherwise specified. 
- You should instruct to include tools whenever possible. Try the best to find the most relevant tools for the task.
- Instruct the agents to only retireve real data, never generate any data.
- You should remind the generator that the only initial input is the "goal" and no other inputs.

Here is the task description: 
{task_description}
"""

def making_goal(openai_config, task_description):
    """
    Generate a prompt for workflow generation and execution based on a task description.
    
    Args:
        task_description (str): The description of the task to generate a prompt for.
        
    Returns:
        str: The generated goal/prompt.
    """
    
    # Create a message in the format expected by OpenAI API
    input_message = [{"role": "user", "content": PROMPT_MAKER_PROMPT.format(task_description=task_description)}]
    
    # Generate the prompt
    prompt_maker = OpenAILLM(config=openai_config)
    goal = prompt_maker.single_generate(input_message)
    
    return goal

if __name__ == "__main__":
    task_description = "Read and analyze the pdf resume at examples/test_pdf.pdf, then find 5 real job opportunities to this client by search the website."
    
    print("Initial input:")
    print(task_description)
    print("--------------------------------")
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    goal = making_goal(openai_config, task_description)
    
    print("Generated goal:")
    print(goal)




