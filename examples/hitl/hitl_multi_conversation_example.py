import asyncio
import os
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.hitl import HITLOutsideConversationAgent, HITLManager
# from evoagentx.workflow import WorkFlow, WorkFlowGraph
# from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
# from evoagentx.agents import AgentManager
# from evoagentx.core.base_config import Parameter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def main():
    """
    show how to use HITLOutsideConversationAgent to modify the JSON structure of WorkFlow
    """
    print("🚀 EvoAgentX HITL Outside Conversation Agent Example")
    print("=" * 80)
    
    # configure the LLM
    llm_config = OpenAILLMConfig(
        model="gpt-4o", 
        openai_key=OPENAI_API_KEY, 
        stream=True, 
        output_response=True
    )
    llm = OpenAILLM(llm_config)
    
    # create the HITLOutsideConversationAgent
    conversation_agent = HITLOutsideConversationAgent(
        name="WorkFlowModificationAgent",
        description="Support multi-turn conversation to modify the JSON structure of WorkFlow",
        llm_config=llm_config
    )
    
    # activate the HITL function
    hitl_manager = HITLManager()
    hitl_manager.activate()
    
    # example: load the JSON file and modify the WorkFlow
    print("\n📋 Example 1: Load the JSON file and modify the WorkFlow")
    print("-" * 50)
    
    # specify the JSON file path
    json_file_path = "examples/output/tetris_game/workflow_demo_4o_mini.json"
    
    try:
        # execute the multi-turn conversation modification
        
        result = await conversation_agent.conversation_action.async_execute(
            llm=llm,
            inputs={"workflow_json_path": json_file_path},
            hitl_manager=hitl_manager
        )
        
        print("\n✅ Successfully modified the WorkFlow from the JSON file!")
        print(f"Modified WorkFlow: {result[0]['final_workflow']}")
        
    except Exception as e:
        print(f"❌ Failed to modify the WorkFlow from the JSON file: {e}")
    
    hitl_manager.deactivate()
    
    print("\n🎉 Example run completed!")


if __name__ == "__main__":
    asyncio.run(main())
