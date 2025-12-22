import os
from dotenv import load_dotenv

from evoagentx.models import OpenRouterConfig
from evoagentx.tools.alita_agent import create_alita_agent, AlitaDynamicToolkit


def main():
    """
    Simple example demonstrating how to build and use the Alita agent.

    The example assumes the following environment variables are set:
    - OPENROUTER_API_KEY: API key for the OpenRouter-compatible model
    - SERPAPI_KEY: (optional) API key for SerpAPI search
    """

    load_dotenv()

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required to run this example.")

    serpapi_key = os.getenv("SERPAPI_KEY")

    llm_config = OpenRouterConfig(
        model="openai/gpt-5-mini",
        openrouter_key=OPENROUTER_API_KEY,
        stream=True,
        output_response=True,
    )

    agent = create_alita_agent(
        llm_config=llm_config,
        persist_dynamic_tools=True,
        load_existing_dynamic_tools=True,
        dynamic_tools_path="./workplace/alita/dynamic_tools.json",
        use_docker=True,
        serpapi_api_key=serpapi_key,
    )

    # Example 1: ask Alita to search the web and store a summary to a file.
    print("===== Example 1: Web search + file write =====")
    message = agent(
        inputs={
            "instruction": (
                "Use web search to find the EvoAgentX GitHub repository and "
                "summarize what the project is. Save your summary to "
                "'project_summary.txt' under the Alita storage directory."
            )
        }
    )
    print("\nAlita result:\n", message.content.result)

    # Example 2: let the agent create and use a generated tool via prompt.
    print("\n===== Example 2: LLM-created generated tool =====")

    instruction2 = (
        "First, use the 'create_generated_tool' tool to create a new generated "
        "code tool named 'payload_text_analyzer'. The tool should read a "
        "'text' field from the payload and return a JSON object with keys "
        "'original', 'upper', and 'length'. After you have successfully "
        "created the tool (in one ToolCalling step), then use the stable "
        "'call_generated_tool' to run it with the payload "
        "{'text': 'Hello from Alita generated tool'} and see its actual "
        "result. Finally, based on the generated tool output you obtain, explain what "
        "the tool did and show the exact JSON result in your final answer."
    )

    message2 = agent(inputs={"instruction": instruction2})
    print("\nAlita example 2 result:\n", message2.content.result)

if __name__ == "__main__":
    main()
