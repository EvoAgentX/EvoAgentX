# This example shows how to use the workflow to have a role play chat, with specific roles defined by user or a role already exists on the Internet.
# By running this file, the final workflow will be saved in the examples/output/workflow_examples_output/role_play_demo_{model_name}.json, where {model_name} is the model name used in the workflow.

import os 
from dotenv import load_dotenv 
import sys
import asyncio

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents import CustomizeAgent, AgentManager
from evoagentx.prompts import StringTemplate, ChatTemplate
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.workflow import WorkFlow, WorkFlowGraph
from evoagentx.tools import GoogleFreeSearchToolkit, WikipediaSearchToolkit


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = "gpt-4o"
llm_config = OpenAILLMConfig(model=model_name, openai_key=OPENAI_API_KEY, stream=True, output_response=True)
llm = OpenAILLM(llm_config)
workflow_save_path = f"examples/workflow_examples/role_play_demo_{model_name}.json"


def chat_history_inputs_example() -> str:
    return [
        {"role": "user", "content": "Hello Ms.Ayanami, I am a fan of your work."},
    ]

def character_profile_inputs_example() -> str:
    return {
        "character_name": "Ayanami Rei",
        "character_description": "Ayanami Rei is a character from the anime Neon Genesis Evangelion.",
        "original_character": False,
        "chat_history": chat_history_inputs_example()
    }

async def main():

    # define the search_character agent
    search_character_agent = CustomizeAgent(
        name="search_character_agent",
        description="An agent to search for a character on the Internet.",
        prompt_template=StringTemplate(
            instruction="You will be given a character name and a character description, and you need to search for the character on the Internet and return the profile information of the character.",
            context="The character name is {character_name} and the character description is {character_description}.",
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "character_name", "type": "string", "description": "The name of the character to search for."},
            {"name": "character_description", "type": "string", "description": "The description of the character to search for."},
        ],
        outputs=[
            {"name": "character_profile", "type": "string", "description": "The profile information of the character."},
        ],
        tools=[GoogleFreeSearchToolkit(), WikipediaSearchToolkit()],
        system_prompt="You are a helpful assistant that can search for a character on the Internet.",
    )

    generate_character_profile_agent = CustomizeAgent(
        name="generate_character_profile_agent",
        description="An agent to generate a character profile.",
        prompt_template=StringTemplate(
            instruction="You will be given a character name and a character description, and you need to generate a profile information of the character.",
            context="The character name is {character_name} and the character description is {character_description}.",
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "character_name", "type": "string", "description": "The name of the character to generate a profile for."},
            {"name": "character_description", "type": "string", "description": "The description of the character to generate a profile for."},
        ],
        outputs=[
            {"name": "character_profile", "type": "string", "description": "The profile information of the character."},
        ],
        system_prompt="You are a helpful assistant that can generate a character profile.",
    )

    generate_character_response_agent = CustomizeAgent(
        name="generate_character_response_agent",
        description="An agent to generate a response for the user, based on the character profile and the chat history.",
        prompt_template=ChatTemplate(
            instruction="You will be given a character profile and a chat history, and you need to generate a response for the user.",
            context="The character profile is {character_profile} and the chat history is {chat_history}.",
        ),
        llm_config=llm_config,
        inputs=[
            {"name": "character_profile", "type": "string", "description": "The profile information of the character."},
            {"name": "chat_history", "type": "list", "description": "The chat history between the user and the character."},
        ],
        outputs=[
            {"name": "character_response", "type": "string", "description": "The response for the user."},
        ],
        system_prompt="You are a helpful assistant that can generate a response for the user.",
    )

    # define the workflow nodes
    nodes = [
        WorkFlowNode(
            name="collect_character_info_node",
            description="collect character info from the Internet, or create a new character profile based on user's inputs",
            agents=[search_character_agent, generate_character_response_agent],
            inputs=[
                {"name": "character_name", "type": "string", "description": "The name of the character to search for or to generate."},
                {"name": "character_description", "type": "string", "description": "The description of the character to search for or to generate."},
                {"name": "chat_history", "type": "list", "description": "The chat history between the user and the character."},
                {"name": "original_character", "type": "bool", "description": "Whether the character is original or not."},
            ],
            outputs=[
                {"name": "character_profile", "type": "string", "description": "The profile information of the character."},
            ]
        ),

        WorkFlowNode(
            name="generate_character_response_node",
            description="generate a response for the user, based on the character profile and the chat history",
            agents=[generate_character_response_agent],
            inputs=[
                {"name": "character_profile", "type": "string", "description": "The profile information of the character."},
                {"name": "chat_history", "type": "list", "description": "The chat history between the user and the character."},
            ],
            outputs=[
                {"name": "character_response", "type": "string", "description": "The response for the user."}
            ]
        )
    ]

    edges = [
        WorkFlowEdge(source="collect_character_info_node", target="generate_character_response_node"),
    ]

    graph = WorkFlowGraph(
        goal="generate a response for the user, based on the character profile and the chat history",
        nodes=nodes,
        edges=edges
    )

    agents = [search_character_agent, generate_character_profile_agent, generate_character_response_agent]

    workflow = WorkFlow(graph=graph, llm=llm, agent_manager=AgentManager(agents=agents))

    try:
        # save workflow
        graph.save_module(path=workflow_save_path)
        print(f"\n✅ workflow graph saved to {workflow_save_path}")
        print("\n📋 start to execute the workflow")
        result = await workflow.async_execute(inputs=character_profile_inputs_example())
        print(f"\n✅ workflow executed successfully!")
        print(f"result: {result}")
        
    except Exception as e:
        print(f"\n❌ workflow execution failed: {e}")

    # save json file
    workflow.save_module(workflow_save_path)


if __name__ == "__main__":
    asyncio.run(main()) 