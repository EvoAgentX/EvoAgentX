import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import MATH  
from evoagentx.evaluators import Evaluator
from evoagentx.agents import AgentManager 
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info 


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class MathSplits(MATH):

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
    return {"problem": example["problem"]}


math_graph_data = {
    "goal": r"Answer the math question. The answer should be in box format, e.g., \boxed{123}",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer generation for Math.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "answer", "type": "str", "required": True, "description": "The generated answer."}
            ],
            "prompt": "Answer the math question. The answer should be in box format, e.g., \\boxed{{123}}\n\nProblem: {problem}",
            "parse_mode": "str"
        }
    ] 
}


def main(): 

    llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)
    model = OpenAILLM(config=llm_config)

    # load benchmark 
    benchmark = MathSplits()

    # load workflow 
    workflow_graph = SequentialWorkFlowGraph.from_file(r"C:\Users\31646\Desktop\EvoAgentX\examples\mipro\output\best_program_math.json")
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
