import os
from dotenv import load_dotenv
from evoagentx.models import OpenRouterConfig, OpenRouterLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

load_dotenv() # Loads environment variables from .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def main():

    # LLM configuration
    openrouter_config = OpenRouterConfig(model="openai/gpt-5.4-mini", openrouter_key=OPENROUTER_API_KEY, stream=True, output_response=True, max_tokens=16000)
    # Initialize the language model
    llm = OpenRouterLLM(config=openrouter_config)

    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_directory = "examples/output/tetris_game"
    
    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    # [optional] display workflow
    workflow_graph.display()
    # [optional] save workflow 
    # workflow_graph.save_module(f"{target_directory}/workflow_demo_4o_mini.json")
    #[optional] load saved workflow 
    # workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(f"{target_directory}/workflow_demo_4o_mini.json")

    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openrouter_config)

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    result = workflow.execute(extract_output=True)
    if result.status != "success":
        raise RuntimeError(f"Workflow failed: {result.displayable_error or result.error_msg}")
    output = result.result
    if not isinstance(output, str):
        output = str(output)

    # verify the code
    code_verifier = CodeVerification()
    output = code_verifier.execute(
        llm=llm,
        inputs={
            "requirements": goal, 
            "code": output
        }
    ).verified_code

    # extract the code 
    os.makedirs(target_directory, exist_ok=True)
    code_blocks = extract_code_blocks(output)
    if len(code_blocks) == 1:
        file_path = os.path.join(target_directory, "index.html")
        with open(file_path, "w") as f:
            f.write(code_blocks[0])
        print(f"You can open this HTML file in a browser to play the Tetris game: {file_path}")
        return
    
    code_extractor = CodeExtraction()
    results = code_extractor.execute(
        llm=llm, 
        inputs={
            "code_string": output, 
            "target_directory": target_directory,
        }
    )

    print(f"Extracted {len(results.extracted_files)} files:")
    for filename, path in results.extracted_files.items():
        print(f"  - {filename}: {path}")
    
    if results.main_file:
        print(f"\nMain file: {results.main_file}")
        file_type = os.path.splitext(results.main_file)[1].lower()
        if file_type == '.html':
            print(f"You can open this HTML file in a browser to play the Tetris game")
        else:
            print(f"This is the main entry point for your application")
    

if __name__ == "__main__":
    main()
