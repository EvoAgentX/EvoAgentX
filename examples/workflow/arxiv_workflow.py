import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.tools.file_tool import FileToolkit
from evoagentx.tools import ArxivToolkit   # 引入 Arxiv 工具

load_dotenv()  # 加载 .env 文件中的环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    # 初始化大模型配置
    openai_config = OpenAILLMConfig(
        model="gpt-4o",
        openai_key=OPENAI_API_KEY,
        stream=True,
        output_response=True,
        max_tokens=16000
    )
    llm = OpenAILLM(config=openai_config)

    # 设置文献关键词，推送文献数量，文献日期，文献分类
    keywords = "medical, multiagent"
    max_results = 10
    date_from = "2024-01-01"
    categories = ["cs.AI", "cs.LG"]

    # 构建搜索条件描述
    search_constraints = f"""
    Search constraints:
    - Query keywords: {keywords}
    - Max results: {max_results}
    - Date from: {date_from}
    - Categories: {', '.join(categories)}
    """

    # 助手的目标任务
    goal = f"""Create a daily research paper recommendation assistant that takes user keywords and pushes new relevant papers with summaries.

    The assistant should:
    1. Use the ArxivToolkit to search for the latest papers using the given keywords.
    2. Apply the following search constraints:
    {search_constraints}
    3. Summarize the search results.
    4. Compile the summaries into a well-formatted Markdown digest.

    ### Output
    daily_paper_digest
    """

    target_directory = "EvoAgentX/examples/output/paper_push"
    module_save_path = os.path.join(target_directory, "paper_push_workflow.json")
    result_path = os.path.join(target_directory, "daily_paper_digest.md")
    os.makedirs(target_directory, exist_ok=True)

    # ✅ 初始化 Arxiv 工具
    arxiv_toolkit = ArxivToolkit()
    tools = [arxiv_toolkit, FileToolkit()]

    # 生成工作流图
    wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    # 保存生成的工作流模块
    workflow_graph.save_module(module_save_path)

    # 展示可视化结构
    workflow_graph.display()

    # Agent 管理器初始化
    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

    # 构建与执行完整工作流
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()

    # 保存摘要结果为 Markdown
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"✅ 推送结果已保存到：{result_path}")
    print("📬 你可以设置定时任务每天自动运行此脚本来获取推荐")


if __name__ == "__main__":
    main()
