import os 
from dotenv import load_dotenv
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import MBPP  
from evoagentx.workflow import SequentialWorkFlowGraph, WorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info 
from evoagentx.optimizers import MiproOptimizer
from evoagentx.evaluators import Evaluator
import evoagentx
from evoagentx.utils.mipro_utils.settings import settings

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
        # randomly select 50 samples for dev and 50 samples for test
        self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:100]]

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
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    evoagentx.configure(lm=openai_config)
    
    benchmark = MBPPSplits()
    workflow_graph = SequentialWorkFlowGraph.from_dict(mbpp_graph_data)
    
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    
    evaluate = Evaluator(
        llm = OpenAILLM(config=openai_config),
        agent_manager = agent_manager,
        collate_func = collate_func,
        num_workers = 16,
        verbose = True
    )
    
    optimizer = MiproOptimizer(
        graph = workflow_graph,
        executor_llm = openai_config,
        max_bootstrapped_demos = 4,
        max_labeled_demos = 4,
        num_candidates = 5,
        auto = "light",
        num_threads = 16,
        save_path = "examples/mipro/output/logs",
        evaluator = evaluate,
        metric_instance = "pass@1",
        metric_threshold = 0.5
    )
    
    best_program = optimizer.optimize(
        benchmark = benchmark,
        collate_func = collate_func,
    )
    
    output_path = r"C:\Users\31646\Desktop\EvoAgentX\examples\mipro\output\best_program_mbpp.json"
    best_program.save_module(output_path)
    result = optimizer.evaluate(graph = best_program, benchmark = benchmark, eval_mode = "test")
    print(result)

if __name__ == "__main__":
    main()
