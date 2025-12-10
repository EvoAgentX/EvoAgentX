import os
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.models.model_configs import OpenRouterConfig
from evoagentx.tools.finance_tool import FinanceToolkit

OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
financial_data_api_key = os.getenv("FINANCIAL_DATA_API_KEY")

def tool_test():
    financial_toolkit = FinanceToolkit(financial_data_api_key=financial_data_api_key, openrouter_api_key=OPENROUTER_API_KEY)
    financial_tool = financial_toolkit.tools[0]
    # print("="*100)
    # print("Test Normal Mode (default):")
    # print(financial_tool("APPL","daily","2024-01-01","2024-01-07"))
    # print("="*100)
    # print("Test CSV Mode:")
    # print(financial_tool("AAPL","daily","2024-01-01","2024-01-07","csv"))
    # print("="*100)
    # print("Test Minute Mode:")
    # print(financial_tool("MSFT","minute","2020-01-15","2020-01-15"))
    # print("="*100)
    # print("Test Fuzzy Matching:")
    # print(financial_tool("APPL","daily","2024-01-01","2024-01-07"))
    # print("="*100)
    # print("Test Corporate Name:")
    # print(financial_tool("Apple Inc.","daily","2024-01-01","2024-1-7"))
    # print("="*100)


def agent_with_tool_test():
    finance_toolkit = FinanceToolkit(financial_data_api_key=financial_data_api_key, openrouter_api_key=OPENROUTER_API_KEY)

    model_config=OpenRouterConfig(
        model="openai/gpt-4o-mini",
        openrouter_key=OPENROUTER_API_KEY,
        stream=True,
    )
    simple_agent = CustomizeAgent(
        name="SimpleAgent",
        description="A basic agent that responds to queries",
        prompt="Answer the following question: {question}",
        llm_config=model_config,
        tools=[finance_toolkit],
        inputs=[
            {
                "name": "question", 
                "type": "string", 
                "description": "The question to answer"
            }
        ],
        outputs=[
            {
                "name": "content",
                "type": "string",
                "description": "The answer to the question"
            }
        ]
    )

    result = simple_agent(inputs={"question": "苹果最近咋样?请给出详细分析"})

    print("--------------------------------")
    print(result.content.content)
    print("--------------------------------")

if __name__ == "__main__":
    # tool_test()
    agent_with_tool_test()