## ATTENTION:
# This project is powered by the "a-share-mcp-is-just-i-need" project. You would need to set it up first.
# Here are the steps:
# 1. Visit https://github.com/24mlight/a-share-mcp-is-just-i-need/ and install the project according to its instructions.
# 2. Set up the project path in the ./examples/output/a_shares/mcp_shares.config file.


import os 
from dotenv import load_dotenv 
import sys
import time

from evoagentx.models import OpenAILLMConfig, OpenAILLM, OpenRouterConfig, OpenRouterLLM
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.tools import MCPToolkit, FileToolkit, CMDToolkit, StorageToolkit
from evoagentx.workflow import WorkFlowGenerator
load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


REQUIREMENT_MAKER_PROMPT = """You are a experienced product manager and an insider of consulting company. Your job is to write requirements for a workflow.
You will be given short "description" and you are expected to write a detailed requirements document based on it, which will be further passed to a workflow generator.
You may not get further information from the user so it is counting on you to make a reasonable and good guess.

You must indicate the input of the workflow, which is a user-provided "goal" parameter. Do not guess what is in it, the user might tell you the content in the description.
You must indicate what is the output and what format it should be in (for example, saved as html file or .md file). In most cases, a .md file is enough. 
You should also indicate the contents in it. For example, there might be graph or tables in it.
If the output is a file, there will be a tool to save file and the formal output will be a boolean that indicates whether the file is saved successfully.

Now lets start. Here is the "description":
{description}
"""


output_file = f"examples/output/invest/data_cache/output_invest_{time.strftime('%Y%m%d%H%M%S')}.md"
working_directory = "examples/output/invest/data_cache/"
mcp_config_path = "examples/output/invest/mcp_invest.config"
module_save_path = "examples/output/invest/invest_demo_4o_mini.json"
data_folder = "examples/output/invest/data_cache/"
available_funds = 100000
current_positions = 500
type_of_position = "call"
date = "2025-07-22"

OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_config = OpenRouterConfig(model="moonshotai/kimi-k2", openrouter_key=OPEN_ROUTER_API_KEY, stream=True, output_response=True, max_tokens=16000)
llm = OpenRouterLLM(config=openrouter_config)

def main():

   # LLM configuration
   openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
   # Initialize the language model
   llm = OpenAILLM(config=openai_config)
   
   
   ## Get tools
   # mcp_toolkit = MCPToolkit(config_path=mcp_config_path)
   # tools = mcp_toolkit.get_toolkits()
   tools = [StorageToolkit(), CMDToolkit()]
   
   
   ## _______________ Workflow Generation _______________
   goal = """Create a daily trading decision workflow for A-share stocks.

## Workflow Overview:
A multi-step workflow for daily trading decisions with fixed capital, making trading decisions based on market data and current positions.

## Task Description:
**Name:** daily_trading_decision
**Description:** A comprehensive trading decision system that analyzes market data and generates daily trading operations with detailed analysis.

## Input:
- **goal** (string): Contains stock code, available funds, current positions, data folder path, output file path, and optional past report path

## Output:
- **trading_report** (string): A comprehensive daily trading report with complete analysis

## Analysis Requirements:
The workflow should analyze three key aspects of the stock:

1. **Background Analysis**: Market environment, industry trends, news sentiment, expert opinions, economic factors, and regulatory environment that affect stock prices
2. **Price Analysis**: Historical price patterns, technical indicators, support/resistance levels, and trading volume analysis
3. **Performance Review**: Past trading decisions, performance evaluation, and lessons learned from previous reports

## Workflow Structure:
- Start with file discovery to identify and categorize available data sources
- Perform the three analyses in parallel where possible for efficiency
- Compile all findings into a comprehensive trading report

## Agent Guidelines:
- Agents should use appropriate tools to discover and read files from the data folder
- Each analysis should focus on its specific domain without overlap
- Agents should filter out irrelevant files and focus on data relevant to their analysis
- All analysis must be based on actual data from files - no fake or estimated data
- Present complete data without omissions or truncations

## Report Structure:
The final report should include:
1. **Background Analysis**: Market environment and external factors
2. **Price Analysis**: Technical patterns and indicators
3. **Performance Review**: Historical performance and lessons learned
4. **Trading Recommendations**: Specific buy/sell/hold decisions with quantities and prices

## Critical Requirements:
- Base all analysis on actual data read from files
- If no relevant files are found, report this clearly and do not make up data
- Provide specific trading recommendations with quantities and price targets
- Consider current positions and available capital in decision making
- Structure the report with clear sections and data tables
- Return complete analysis without summarization
"""
   # goal = llm.generate(prompt=REQUIREMENT_MAKER_PROMPT.format(description=goal)).content
   # from pdb import set_trace; set_trace()
   
   # ## _______________ Workflow Creation _______________
   # wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
   # # workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
   # workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal, retry=5)
   # # [optional] save workflow 
   # workflow_graph.save_module(module_save_path)
   
   
   ## _______________ Workflow Execution _______________
   #[optional] load saved workflow 
   workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(module_save_path)
   workflow_graph.display()

   agent_manager = AgentManager(tools=tools)
   agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

   # Create working directory if it doesn't exist
   os.makedirs(working_directory, exist_ok=True)
   
   workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
   workflow.init_module()
   output = workflow.execute({"goal": f"""I need a daily trading decision for stock 300750. 
Available funds: {available_funds} RMB
Current positions: {current_positions} shares of 300750 at average price 280 RMB
Date: {date}
Type of position: {type_of_position}
Data folder: {data_folder}
Past report (if exists): {output_file.replace('.md', '_previous.md')}

Please read ALL files in the data folder and generate a comprehensive trading decision report in Chinese based on real data. Return the complete content.
"""})
   try:
      # Save the complete report to file
      with open(output_file, "w", encoding="utf-8") as f:
         f.write(output)
      print(f"Trading decision report saved to: {output_file}")
      
      # Also save a backup
      with open("examples/output/invest/output_invest_back.md", "w", encoding="utf-8") as f:
         f.write(output)
   except Exception as e:
      print(f"Error saving report: {e}")
   
if __name__ == "__main__": 
   
   main()