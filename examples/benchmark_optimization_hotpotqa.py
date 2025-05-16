import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import HotPotQA 
from evoagentx.evaluators import Evaluator
from evoagentx.agents import AgentManager 
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info 


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class HotPotQASplits(HotPotQA):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._dev_data))
        full_test_data = self._dev_data 
        # radnomly select 50 samples for dev and 100 samples for test
        self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]


def collate_func(example: dict) -> dict:
    context_list = []
    for item in example["context"]:
        context = "Title: {}\nText: {}".format(item[0], " ".join([t.strip() for t in item[1]]))
        context_list.append(context)
    context = "\n\n".join(context_list)
    problem = "Context: {}\n\nQuestion: {}\n\nAnswer:".format(context, example["question"])
    return {"problem": problem}


hotpotqa_graph_data = {
    "goal": "Answer the question based on the context. The answer should be a direct response to the question, without including explanations or reasoning.",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer the question based on the context.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "answer", "type": "str", "required": True, "description": "The answer to the problem."}
            ],
            "prompt": "Think step by step to answer the question. You should explain your thinking process in the 'thought' field, and provide the final answer in the 'answer' field.\nFormat your output in xml format, such as <thought>xxx</thought> and <answer>xxx</answer>.\n\nProblem: {problem}",
            "parse_mode": "xml"
        }
    ] 
}


def main(): 

    llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    model = OpenAILLM(config=llm_config)

    # load benchmark 
    benchmark = HotPotQASplits()

    # load workflow 
    workflow_graph = SequentialWorkFlowGraph.from_file(r"C:\Users\31646\Desktop\EvoAgentX\examples\mipro\output\best_program_hotpotqa.json")
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