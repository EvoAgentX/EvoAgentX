import os
from dotenv import load_dotenv
from evoagentx.agents import AgentManager
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import MATH
from evoagentx.workflow import SequentialWorkFlowGraph # , WorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.optimizers import MiproOptimizer
from evoagentx.evaluators import Evaluator
import evoagentx
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
        # radnomly select 50 samples for training and 100 samples for test
        self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]

def collate_func(example: dict) -> dict:
    return {"problem": example["problem"]}


math_graph_data = {
    "goal": r"Answer the math question. The answer should be in box format, e.g., \boxed{{123}}",
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
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    evoagentx.configure(llm_config=openai_config)
    
    benchmark = MathSplits()
    trainset = [{"problem": collate_func(example)["problem"], "solution": example["solution"]} for example in benchmark._train_data]
    
    workflow_graph = SequentialWorkFlowGraph.from_dict(math_graph_data)
    
    def evaluate_metric(example, prediction, trace=None):
        
        example_ans = benchmark.extract_answer(example["solution"])
        prediction_ans = benchmark.extract_answer(prediction)
        
        return benchmark.math_equal(prediction_ans, example_ans)
    
    
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    
    evaluate = Evaluator(
        llm = OpenAILLM(config=openai_config),
        agent_manager = agent_manager,
        collate_func = collate_func,
        num_workers = 32,
        verbose = True
    )
    
    optimizer = MiproOptimizer(
        graph = workflow_graph,
        metric = evaluate_metric,
        executor_llm = openai_config,
        max_bootstrapped_demos = 4,
        max_labeled_demos = 4,
        num_candidates = 15,
        auto = "medium",
        num_threads = 32,
        save_path = "examples/mipro/output/logs",
        evaluator = evaluate
    )
    
    with suppress_logger_info():
        best_program = optimizer.optimize(
            trainset = trainset,
            collate_func = collate_func,
        )
    
    output_path = r"examples/mipro/output/best_program_math.json"
    best_program.save_module(output_path)
    result = optimizer.evaluate(graph = best_program, benchmark = benchmark, eval_mode = "test")
    print(result)

if __name__ == "__main__":
    main()
