from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import SEWWorkFlowGraph 
from evoagentx.agents import AgentManager
from evoagentx.benchmark import HumanEval 
from evoagentx.evaluators import Evaluator 
from evoagentx.optimizers import SEWOptimizer 
from evoagentx.core.callbacks import suppress_logger_info


# OPENAI_API_KEY = "OPENAI_API_KEY" 
import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class HumanEvalSplits(HumanEval):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        num_dev_samples = int(len(self._test_data) * 0.2)
        random_indices = np.random.permutation(len(self._test_data))
        self._dev_data = [self._test_data[i] for i in random_indices[:num_dev_samples]]
        self._test_data = [self._test_data[i] for i in random_indices[num_dev_samples:]]


def main():
    
    llm_config = OpenAILLMConfig(model="gpt-4o-mini-2024-07-18", openai_key=OPENAI_API_KEY, top_p=0.85, temperature=0.2, frequency_penalty=0.0, presence_penalty=0.0)
    llm = OpenAILLM(config=llm_config)

    # obtain SEW workflow 
    sew_graph = SEWWorkFlowGraph(llm_config=llm_config)
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(sew_graph, llm_config=llm_config)

    # obtain HumanEval benchmark
    humaneval = HumanEvalSplits()
    def collate_func(example: dict) -> dict:
        # convert raw example to the expected input for the SEW workflow
        return {"question": example["prompt"]}
    
    # obtain Evaluator
    evaluator = Evaluator(llm=llm, agent_manager=agent_manager, collate_func=collate_func, num_workers=20, verbose=True)

    # obtain SEWOptimizer
    optimizer = SEWOptimizer(
        graph=sew_graph, 
        evaluator=evaluator, 
        llm=llm, 
        max_steps=10,
        eval_rounds=1, 
        repr_scheme="python", 
        optimize_mode="prompt", 
        order="zero-order"
    )

    with suppress_logger_info():
        metrics = optimizer.evaluate(dataset=humaneval, eval_mode="test")
    print("Evaluation metrics: ", metrics)

    # optimize the SEW workflow
    optimizer.optimize(dataset=humaneval)

    # evaluate the optimized SEW workflow
    with suppress_logger_info():
        metrics = optimizer.evaluate(dataset=humaneval, eval_mode="test")
    print("Evaluation metrics: ", metrics)
    
    # save the optimized SEW workflow
    optimizer.save("debug/optimized_sew_workflow.json")

if __name__ == "__main__":
    main()
