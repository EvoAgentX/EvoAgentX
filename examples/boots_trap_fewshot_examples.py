from evoagentx.optimizers.mipro_optimizer import MiproOptimizer
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.benchmark.math import MATH
from evoagentx.utils.mipro_utils.settings import settings
# OPENAI_API_KEY = "OPENAI_API_KEY" 
from evoagentx.evaluators.mipro_evaluator import Evaluate as mipro_evaluator
import os 
import logging
from dotenv import load_dotenv
from evoagentx.workflow.workflow import WorkFlowGraph
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

def main():
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    executor_llm = OpenAILLM(config=openai_config)
    settings.lm = openai_config

    def evaluate_metric(example, prediction, trace=None):
        example_ans = math.extract_answer(example["solution"])
        prediction_ans = math.extract_answer(prediction)
        
        return math.math_equal(prediction_ans, example_ans)

    math = MATH()
    math._load_data()
    trainset = math._test_data[:100]
    devset = math._test_data[100:200]
    
    graph = WorkFlowGraph.from_file("examples/output/mipro_demo.json")
    
    # zero-shot optimization
    optimizer = MiproOptimizer(
        graph = graph,
        metric = evaluate_metric,
        prompt_model = executor_llm,
        max_bootstrapped_demos = 0,
        max_labeled_demos = 0,
        auto = "medium",
        num_threads = 32,
        log_dir = "examples/output/logs",
        
    )
    

    # zero-shot evaluation
    evaluate = mipro_evaluator(
        devset=devset,
        metric=evaluate_metric,
        num_threads=32,
        display_progress=True,
        display_table=False,
    )
    
    
    evaluate(program = graph, with_inputs={"problem":"problem"})
    
    output_path = r"C:\Users\31646\Desktop\EvoAgentX\examples\mipro\output\best_progrma_math.json"

    best_program = optimizer.optimize(trainset=trainset, with_inputs={"problem":"problem"}, provide_traceback=True)
    
    evaluate(program = best_program, with_inputs={"problem":"problem"})
    
    best_program.save_module(output_path)
    

if __name__ == "__main__":
    main()