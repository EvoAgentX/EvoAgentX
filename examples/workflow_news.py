## This example shows how to use the workflow to recommend a PHD direction for a candidate based on their resume.
## It uses the arxiv-mcp-server to search the papers. You may find the project here: https://github.com/blazickjp/arxiv-mcp-server/tree/main

import os 
from dotenv import load_dotenv 
import sys

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.agents import AgentManager
from evoagentx.tools import MCPToolkit, FileToolkit, RSSToolkit, GoogleFreeSearchToolkit, BrowserUseToolkit, RequestToolkit
load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

output_file = "debug/output/newsletter/output.md"
mcp_config_path = "examples/output/news/mcp_news.config"
target_directory = "examples/output/news/"
module_save_path = "examples/output/news/news_demo_4o_mini.json"

def main(goal=None):
    # LLM configuration
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    # Initialize the language model
    llm = OpenAILLM(config=openai_config)
    
    goal = f"""  User chooses what they are interested in and what websites they want to subscribe to. Fetch articles and select ones that user might be interested in. Summarise the content into nicely formatted newsletter. 
  - Input: user interests (string) and optional list of websites they want to source articles from (list of strings); optional output path for the newsletter (string), dont save if we dont have it.
  - Output: newsletter (markdown)
1） Use RSS feeds to fetch articles if available. Otherwise use web search. Only get articles published today.
2) Summarize the articles into a newsletter. Keep article titles (and url if available) and provide a short summary for each article.

Key things to note:
- If RSS urls are available, use tools to fetch the articles.
- You may also search for new RSS urls and relevant articles through tools.
- You MUST ensure all information is real, retirable, and correct. They can be collected using the websearch toolkit or the rss toolkit
- You must collect information through tools only instead of making up or guessing.
- You should use the file toolkit at the end to save the newsletter to a given path.
"""
    
    ## Get tools
    tools = [RSSToolkit(), FileToolkit(), BrowserUseToolkit(), GoogleFreeSearchToolkit(), RequestToolkit()]
    
    
    # ## _______________ Workflow Creation _______________
    # wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
    # workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
    # # [optional] save workflow 
    # workflow_graph.save_module(module_save_path)
    
    
    ## _______________ Workflow Execution _______________
    #[optional] load saved workflow 
    workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(module_save_path, llm_config=openai_config, tools=tools)

    # [optional] display workflow
    workflow_graph.display()
    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    # from pdb import set_trace; set_trace()

    
    topic = "AI"
    rss_urls = [
        # Technology & AI
        "https://www.theverge.com/rss/index.xml",
        "https://techcrunch.com/feed/",
        "https://www.wired.com/feed/rss",
        "https://www.engadget.com/rss.xml",
        "https://www.zdnet.com/news/rss.xml",
        
        # Business & Finance
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.ft.com/rss/home",
        
        # Science & Research
        "https://www.nature.com/nature.rss",
        "https://www.science.org/rss/news_current.xml",
        "https://www.technologyreview.com/feed/",
        "https://www.quantamagazine.org/feed/",
        
        # General News
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.reuters.com/Reuters/worldNews",
        "https://www.theguardian.com/world/rss"
    ]
    output_file = "examples/output/news/output.md"
    
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute(inputs={"goal": f"""Please write a newsletter on the topic of {topic}, here are some rss urls to collect articles: {rss_urls}; You should save the output into the file {output_file}"""})
    
    
    ## _______________ Save Output _______________
    try:
        # Write to file
        back_up_path = "examples/output/news/output_backup.md"
        with open(back_up_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Direction recommendations have been saved to {back_up_path}")
    except Exception as e:
        print(f"Error saving direction recommendations: {e}")
    
    # from pdb import set_trace; set_trace()
    print(output)

if __name__ == "__main__": 
    # Get custom goal from positional argument if provided
    custom_goal = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run the main function with the provided goal
    main(custom_goal)
