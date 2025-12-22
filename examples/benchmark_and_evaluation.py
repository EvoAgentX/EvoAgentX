from evoagentx.models import OpenAILLMConfig, OpenAILLM 
from evoagentx.benchmark import HotPotQA
from evoagentx.workflow import QAActionGraph 
from evoagentx.evaluators import Evaluator 
from evoagentx.core.callbacks import suppress_logger_info

import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main(): 

    llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    llm = OpenAILLM(config=llm_config)

    benchmark = HotPotQA(mode="dev")

    workflow = QAActionGraph(
        llm_config=llm_config,
        description="This workflow aims to address multi-hop QA tasks."
    )

    def collate_func(example: dict) -> dict:
        """
        Args:
            example (dict): A dictionary containing the raw example data.

        Returns: 
            The expected input for the (custom) workflow.
        """
        problem = "Question: {}\n\n".format(example["question"])
        context_list = []
        for item in example["context"]:
            context = "Title: {}\nText: {}".format(item[0], " ".join([t.strip() for t in item[1]]))
            context_list.append(context)
        context = "\n\n".join(context_list)
        problem += "Context: {}\n\n".format(context)
        problem += "Answer:" 
        return {"problem": problem}


    def output_postprocess_func(output: dict) -> dict:
        """
        Args:
            output (dict): The output from the workflow.

        Returns: 
            The processed output that can be used to compute the metrics. The output will be directly passed to the benchmark's `evaluate` method. 
        """
        return output["answer"]

    evaluator = Evaluator(
        llm=llm, 
        collate_func=collate_func,
        output_postprocess_func=output_postprocess_func,
        verbose=True, 
        num_workers=3 
    )

    with suppress_logger_info():
        results = evaluator.evaluate(
            graph=workflow, 
            benchmark=benchmark, 
            eval_mode="dev", # Evaluation split: train / dev / test 
            sample_k=6 # If set, randomly sample k examples from the benchmark for evaluation  
        )
    
    print("Evaluation metrics: ", results)

if __name__ == "__main__":
    main()