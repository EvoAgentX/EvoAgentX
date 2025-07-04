## ATTENTION:
# This project is powered by the "a-share-mcp-is-just-i-need" project. You would need to set it up first.
# Here are the steps:
# 1. Visit https://github.com/24mlight/a-share-mcp-is-just-i-need/ and install the project according to its instructions.
# 2. Set up the project path in the ./examples/output/a_shares/mcp_shares.config file.


import os 
from dotenv import load_dotenv 
import sys

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.workflow import WorkFlowGenerator
load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

output_html_path = "workplace/market_analysis.html"

SHARES_GOAL = """
**CRITICAL CONSTRAINTS: DATA VIA TOOL CALLS ONLY. OUTPUT IS A SINGLE, SELF-CONTAINED HTML FILE WITH JAVASCRIPT FOR ALL ANALYSIS & VISUALIZATION. NO PYTHON IN HTML.**

**Goal:** Generate a concise, **technically detailed HTML report** analyzing the **overall A-share market environment**. Focus on **key industry index performances** (past 3 months ending 2025-05-07).

**Core Requirements:**
1.  **Input:** This goal description is the sole input.
2.  **Data Collection:**
    *   All quantitative data (historical k-data: OHLCV for A-SHARE INDUSTRY INDEXES and benchmark HS300) MUST come from tool calls.
    *   Ensure the **entirety of the historical data for the specified 3-month period** is collected and utilized, not just a subset or the beginning of the period.
3.  **HTML/JavaScript Report Implementation:**
    *   Embed all tool-collected data as JavaScript variables.
    *   **Charts (JavaScript-generated, e.g., Chart.js/Plotly.js):**
        *   Line charts of indexed performance.
        *   Candlestick charts (if OHLC available).
        *   Volume charts (if volume data available).
        *   Ensure all charts are comprehensive, with clearly labeled axes, descriptive titles, appropriate legends, and distinctly visible data points.
    *   **Technical Indicators (JavaScript-calculated & plotted):**
        *   Calculate and chart 1-2 common moving averages (e.g., 20-day, 50-day).
        *   *Optional:* RSI or MACD if feasible with JS libraries and embedded data.
    *   **Written Analysis (HTML text, informed by JS analysis):**
        *   Objectively interpret current market conditions, trends (including the dynamics of **rises and falls within industry indexes**), support/resistance, and signals from JS-generated technicals.
        *   Describe the overall market environment based on industry index performance. Avoid buy/sell recommendations.
4.  **Process:**
    *   Fetch all data first.
    *   JavaScript performs all data processing, calculations, and chart generation.
    *   Logical flow: Benchmark technical view -> Comparative industry index technical analysis -> Summary of overall market environment.
    *   No individual stock data. No placeholder content.
"""






AGENT_SUGGESTIONS = """
CRITICAL WORKFLOW CONSTRAINTS:

1. DATA INTEGRITY:
   - ALL market data MUST come from tool calls -- data collection agents use should have tools in the config
   - NEVER generate fictional market data - this is a hard requirement
   - Each agent must verify data comes from tools before using it
   - Each agent must have inputs and outputs

2. AGENT ROLES:
   - Each agent performs ONE type of action: EITHER tool calls or analysis/coding

3. WORKFLOW STRUCTURE:
   - Collect all required data via tools BEFORE analysis
   - Data analysis should be based on the collected data
   - Verify data completeness before visualization
   - Document data sources throughout the process

Suggestions:
- Use the tools to collect data
- Use different agents to collect different types of data



The final output must be a complete HTML file where EVERY data point originated from tool calls.
"""

output_file = "examples/output/a_shares/output_shares.md"
mcp_config_path = "examples/output/a_shares/mcp_shares.config"
module_save_path = "examples/output/a_shares/shares_demo_4o_mini.json"

def main(goal=None):

   # LLM configuration
   openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
   # Initialize the language model
   llm = OpenAILLM(config=openai_config)
   
   
   ## Get tools
   mcp_Toolkit = MCPToolkit(config_path=mcp_config_path)
   tools = mcp_Toolkit.get_tools()
   
   ## _______________ Workflow Creation _______________
   wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
   # workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
   workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal, agent_suggestion=AGENT_SUGGESTIONS)
   # [optional] display workflow
   workflow_graph.display()
   # [optional] save workflow 
   workflow_graph.save_module(module_save_path)
   
   
   ## _______________ Workflow Execution _______________
   #[optional] load saved workflow 
   workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(module_save_path)
   workflow_graph.display()

   agent_manager = AgentManager(tools=tools)
   agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

   workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
   workflow.init_module()
   output = workflow.execute()
   
   
   ## _______________ Save Output _______________
#    try:
#        # Save the output HTML directly to file
#        with open(output_file, "w", encoding="utf-8") as f:
#            f.write(output)
#        print(f"Market analysis dashboard has been saved to {output_file}")
#    except Exception as e:
#        print(f"Error saving market analysis dashboard: {e}")
   
   
   try:
       # Save the output HTML directly to file
       with open(output_html_path, "w", encoding="utf-8") as f:
           f.write(output[8:-4])
       print(f"Market analysis dashboard has been saved to {output_html_path}")
   except Exception as e:
       print(f"Error saving market analysis dashboard: {e}")
   
   from pdb import set_trace
   set_trace()
    
    
if __name__ == "__main__": 
    # Get custom goal from positional argument if provided
    custom_goal = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not custom_goal:
        custom_goal = SHARES_GOAL
    
    # Run the main function with the provided goal
    main(custom_goal)
