import os
from dotenv import load_dotenv
from evoagentx.agents import AgentManager
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import MATH
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.evaluators import Evaluator
from evoagentx.core.logging import logger
from evoagentx.prompts import MiproPromptTemplate 
from evoagentx.optimizers.mipro_optimizer import WorkFlowMiproOptimizer 

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
        # self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._test_data = [full_test_data[idx] for idx in permutation[100:200]]

    def get_input_keys(self):
        return ["problem"]


def collate_func(example: dict) -> dict:
    return {"problem": example["problem"]}


math_graph_data = {
    "goal": r"Answer the math question. The answer should be in box format, e.g., \boxed{{123}}.",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer generation for Math.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "solution", "type": "str", "required": True, "description": "The generated answer."}
            ],
            "prompt_template": MiproPromptTemplate(
                instruction=r"Let's think step by step to answer the math question.", 
            ),
            "parse_mode": "title" 
        }
    ] 
}

def main():

    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_config = OpenAILLMConfig(model="gpt-4o", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    optimizer_llm = OpenAILLM(config=optimizer_config)
    
    benchmark = MathSplits()
    workflow_graph: SequentialWorkFlowGraph = SequentialWorkFlowGraph.from_dict(math_graph_data)
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

    # define the evaluator 
    evaluator = Evaluator(
        llm = executor_llm,
        agent_manager = agent_manager,
        collate_func = collate_func,
        num_workers = 20,
        verbose = True
    )
    
    # define the optimizer 
    optimizer = WorkFlowMiproOptimizer(
        graph = workflow_graph,
        evaluator = evaluator, 
        optimizer_llm = optimizer_llm, 
        max_bootstrapped_demos = 4, 
        max_labeled_demos = 4,
        eval_rounds = 1, 
        auto = "medium",
        save_path = "examples/output/mipro/math_mipro", 
    )

    logger.info("Optimizing workflow...")
    optimizer.optimize(dataset=benchmark)
    from pdb import set_trace; set_trace()
    optimizer.restore_best_program() # restore the best graph from the saved path 

    logger.info("Evaluating program on test set...")
    with suppress_logger_info():
        results = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    logger.info(f"Evaluation metrics (after optimization): {results}")
    

if __name__ == "__main__":
    main()
