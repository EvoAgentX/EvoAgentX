import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import MBPP  
from evoagentx.evaluators import Evaluator
from evoagentx.agents import AgentManager 
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info 


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class MBPPSplits(MBPP):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # radnomly select 50 samples for dev and 100 samples for test
        self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]


def collate_func(example: dict) -> dict:
    return {"problem": example["prompt"]}


mbpp_graph_data = {
    "goal": "Generate a functional and correct Python code for the given problem.",
    "tasks": [
        {
            "name": "code_generate",
            "description": "Code generation for MBPP.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "code", "type": "str", "required": True, "description": "The generated code."}
            ],
            "prompt": "Generate a functional and correct Python code for the given problem.\n\nProblem: {problem}",
            "parse_mode": "str"
        }
    ] 
}


def main(): 

    llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    model = OpenAILLM(config=llm_config)

    # load benchmark 
    benchmark = MBPPSplits()

    # load workflow 
    workflow_graph = SequentialWorkFlowGraph.from_file(r"C:\Users\31646\Desktop\EvoAgentX\examples\mipro\output\best_program_mbpp.json")
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)

    # create evaluator 
    evaluator = Evaluator(llm=model, agent_manager=agent_manager, collate_func=collate_func, num_workers=20, verbose=True)

    # initial evaluation
    with suppress_logger_info():
        results = evaluator.evaluate(graph=workflow_graph, benchmark=benchmark, eval_mode="test")
    
    print("Evaluation metrics: ", results)


if __name__ == "__main__":
    main() 