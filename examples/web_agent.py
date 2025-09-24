import os
from evoagentx.models import OpenAILLMConfig
from evoagentx.tools import SearchCollectionToolkit, Crawl4AICrawlToolkit, BrowserUseToolkit, BrowserToolkit, BrowserUseAutoToolkit
from dotenv import load_dotenv
from evoagentx.agents.websearch_agent import WebSearchAgent
from evoagentx.prompts.template import StringTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-5-mini", openai_key=OPENAI_API_KEY, output_response=True)

def test_websearch_action():
    tools = []
    # tools += [BrowserToolkit(headless = False), Crawl4AICrawlToolkit(), SearchCollectionToolkit()]
    tools += [BrowserUseToolkit(headless = False), Crawl4AICrawlToolkit(), SearchCollectionToolkit()]
    # tools += [BrowserUseAutoToolkit(headless = False)]
    
    websearch_agent = WebSearchAgent(
        name="WebSearchAgent",
        description="A web search agent that can use the tools provided by the MCP server",
        prompt_template= StringTemplate(
            instruction="You are a helpful assistant that can help with web search tasks."
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "requirement", "type": "string", "description": "The goal you need to achieve"}
        ],
        outputs=[
            {"name": "output", "type": "string", "description": "The result of the web search, containing all information required in the inputs."}
        ],
        tools=tools
    )
    
    message = websearch_agent(
        # inputs={"requirement": "Find me a suitable flight ticket from Tokyo to San Francisco. Give me some details aobut it like the price, the airline, the flight number, the departure time, the arrival time, the duration, the departure airport, the arrival airport."}
        # inputs={"requirement": "Help me find the cheapest single trip flight ticket from Tokyo to San Francisco."}
        # inputs={"requirement": "What is on the page https://en.wikipedia.org/wiki/Test?"}
        inputs={"requirement": "Find me a good laptop that is capable of running Windows 11 and has a good battery life on Japanese Amazon and put it into my shopping cart."}
        # inputs={"requirement": "I would like to buy a new laptop that is capable of running Windows 11 and has a good battery life. Please recommend me some options and give me some candidate shopping links."}
    )
    
    print("\n" + "=" * 60)
    print("WEBSEARCH AGENT FINAL RESULT:")
    print("=" * 60)
    print(message.content)
    print("=" * 60)

if __name__ == "__main__":
    test_websearch_action()