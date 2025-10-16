import os
import argparse
import json
import importlib
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.tools.mcp import MCPToolkit


load_dotenv()

def dynamic_import(module: str, cls: str):
    mod = importlib.import_module(module)
    return getattr(mod, cls)

def load_tools_from_json(workdir: str):
    """Load tools from tools.json in the workflow directory"""
    tools = []
    tools_file = os.path.join(workdir, "tools.json")
    if os.path.exists(tools_file):
        with open(tools_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for t in data.get("tools", []):
            ToolClass = dynamic_import(t["module"], t["class"])
            if "params" in t:
                tools.append(ToolClass(**t["params"]))
            else:
                tools.append(ToolClass())
    return tools

def main():
    parser = argparse.ArgumentParser(description="Universal workflow executor (goal only)")
    parser.add_argument("--workflow", required=True, help="Path to workflow.json")
    parser.add_argument("--goal", required=True, help="The goal input")
    parser.add_argument("--output", help="Where to save the result")
    args = parser.parse_args()

    # LLM config
    llm_config = OpenAILLMConfig(
        model="gpt-4o",
        openai_key=os.getenv("OPENAI_API_KEY"),
        stream=True,
        output_response=True,
        max_tokens=16000,
    )
    llm = OpenAILLM(config=llm_config)

    # Workflow 路径与目录
    workdir = os.path.dirname(args.workflow)

    
    tools = load_tools_from_json(workdir)

    # tool_names = [t.__class__.__name__ for t in tools]
    # if "MCPToolkit" in tool_names:
    
    #     mcp_config_path = "Wonderful_workflow_corpus/PhD_direction/mcp_direction.config"
    #     tools = [MCPToolkit(config_path=mcp_config_path) if isinstance(t, MCPToolkit) else t for t in tools]

    wf_graph = WorkFlowGraph.from_file(
        args.workflow,
        llm_config=llm_config,
        tools=tools,
    )

    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(wf_graph, llm_config=llm_config)
    workflow = WorkFlow(graph=wf_graph, agent_manager=agent_manager, llm=llm)

    output = workflow.execute(inputs={"goal": args.goal})

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"✅ Output saved to {args.output}")
    else:
        print("====== Workflow Output ======")
        print(output)

if __name__ == "__main__":
    main()
