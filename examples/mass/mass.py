import os 
import json
from dotenv import load_dotenv
from typing import Any

from evoagentx.benchmark import MATH
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.optimizers import MiproOptimizer
from evoagentx.utils.mipro_utils.register_utils import MiproRegistry
from evoagentx.workflow.operators import Predictor, Summarizer
from evoagentx.workflow.blocks.summarize import summarize
from evoagentx.workflow.blocks.aggregate import aggregate
from evoagentx.workflow.blocks.reflect import reflect
from evoagentx.workflow.blocks.debate import debate
from evoagentx.workflow.blocks.execute import execute
from evoagentx.optimizers.mass_optimizer import MassOptimiser
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_BOOTSTRAPPED_DEMOS = 1
MAX_LABELED_DEMOS = 0
AUTO = "light"
NUM_THREADS = 16
EVALUATION_ROUNDS = 1
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

    # define the input keys. 
    # If defined, the corresponding input key and value will be passed to the __call__ method of the program, 
    # i.e., program.__call__(**{k: v for k, v in example.items() if k in self.get_input_keys()})
    # If not defined, the program will be executed with the entire input example, i.e., program.__call__(**example)
    def get_input_keys(self):
        return ["problem"]
    
    # the benchmark must have a `evaluate` method that receives the program's `prediction` (output from the program's __call__ method) 
    # and the `label` (obtained using the `self.get_label` method) and return a dictionary of metrics. 
    def evaluate(self, prediction: Any, label: Any) -> dict:
        return super().evaluate(prediction, label)

class WorkFlow():
    def __init__(self,
                 summarizer,
                 aggregater,
                 reflector,
                 debater,
                 executer) -> None:
        
        self.summarizer = summarizer
        self.aggregater = aggregater
        self.reflector = reflector
        self.debater = debater
        self.executer = executer
        self.blocks = [self.summarizer, self.aggregater, self.reflector, self.debater, self.executer]
    
    def __call__(self, problem: str, **kwargs):
        context = kwargs.get("context", None)
        testcases = kwargs.get("testcases", None)
        # Step 1: 获取总结的上下文
        if self.summarizer.n > 0:
            context = self.summarizer.execute(problem, context = context)

        # Step 2: 生成候选解决方案
        if self.debater.n > 0:
            candidate_solutions = self.aggregater.execute(problem, context = context)
        else:
            candidate_solutions = [self.aggregater(problem)]

        # Step 3: 对每个候选方案进行反思优化
        if self.reflector.n > 0:
            for i in range(len(candidate_solutions)):
                if self.executer.n > 0:
                    self.executer.n = self.reflector.n
                    refined_answer = self.executer.execute(problem, candidate_solutions[i], testcases = testcases)
                else:
                    refined_answer = self.reflector.execute(problem, candidate_solutions[i], context = context)
                candidate_solutions[i] = refined_answer

        # Step 4: 通过辩论选择最佳答案
        if self.debater.n > 0:
            final_answer = self.debater.execute(problem, candidate_solutions, context = context)
        else:
            final_answer = candidate_solutions[0]

        return final_answer, {"problem":problem, 
                              "context":context, 
                              "testcases": testcases,
                              "answer":final_answer}

    def save(self, path):
        params = {
            "summarizer": {
                "summarizer": self.summarizer.summarizer.prompt,
                "predictor": self.summarizer.predictor.prompt,
            },
            "aggregater":{
                "predictor": self.aggregater.predictor.prompt,
            },
            "reflector": {
                "reflector": self.reflector.reflector.prompt,
                "refiner": self.reflector.refiner.prompt,
            },
            "debater": {
                "debater": self.debater.debater.prompt,
                "predictor": self.debater.predictor.prompt,
            },
            "executer": {
                "predictor": self.executer.predictor.prompt,
                "codereflector": self.executer.codereflector.prompt,
            }
        }
        with open(path, "w") as f:
            json.dump(params, f)

    def load(self, path):
        with open(path, "r") as f:
            params = json.load(f)
            self.summarizer.summarizer.prompt = params["summarizer"]["summarizer"]
            self.summarizer.predictor.prompt = params["summarizer"]["predictor"]
            self.aggregater.predictor.prompt = params["aggregater"]["predictor"]
            self.reflector.reflector.prompt = params["reflector"]["reflector"]
            self.reflector.refiner.prompt = params["reflector"]["refiner"]
            self.debater.debater.prompt = params["debater"]["debater"]
            self.debater.predictor.prompt = params["debater"]["predictor"]
            self.executer.predictor.prompt = params["executer"]["predictor"]
            self.executer.codereflector.prompt = params["executer"]["codereflector"]

    def __repr__(self):
        lines = [
            "WorkFlow representation:",
            f"summarizer.prompt: {getattr(self.summarizer, 'prompt', getattr(self.summarizer, 'summarizer', None))}",
            f"aggregater.prompt: {getattr(self.aggregater, 'prompt', getattr(self.aggregater, 'predictor', None))}",
            f"reflector.prompt: {getattr(self.reflector, 'prompt', getattr(self.reflector, 'reflector', None))}",
            f"debater.prompt: {getattr(self.debater, 'prompt', getattr(self.debater, 'debater', None))}",
            f"executer.prompt: {getattr(self.executer, 'prompt', getattr(self.executer, 'predictor', None))}",
        ]
        return '\n'.join(lines)

    def __str__(self):
        return self.__repr__()


def get_save_path(program):
    return f"examples/mass/{program}"


def mipro_optimize(registry, block, optimizer_llm, save_path, benchmark):
    optimizer = MiproOptimizer(
        registry=registry,
        program=block,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MAX_LABELED_DEMOS,
        num_threads=NUM_THREADS,
        eval_rounds=EVALUATION_ROUNDS,
        auto=AUTO,
        save_path=save_path
    )
    
    optimizer.optimize(dataset=benchmark)
    return optimizer

def optimize_predictor(predictor, optimizer_llm, benchmark):
    registry = MiproRegistry()
    registry.track(predictor, "prompt", input_names=['problem'], output_names=['reasoning', 'answer'])
    
    optimizer = mipro_optimize(registry, predictor, optimizer_llm, get_save_path("predictor"), benchmark)
    return optimizer.evaluate(dataset = benchmark, eval_mode="test"), optimizer.restore_best_program()

def optimize_summarizer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score):
    block = summarize(predictor=optimized_predictor, llm=executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "summarizer.prompt", input_names=['problem'], output_names=['summary'])
    registry.track(block, "predictor.prompt", input_names=['problem'], output_names=['reasoning', 'answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/summarizer"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def optimize_aggregator(optimized_predictor, optimizer_llm, benchmark, predictor_score):
    block = aggregate(predictor=optimized_predictor)
    
    registry = MiproRegistry()
    registry.track(block, "predictor.prompt", input_names=['problem'], output_names=['reasoning', 'answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/aggregator"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def optimize_reflector(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score):
    block = reflect(predictor=optimized_predictor, llm=executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "predictor.prompt", input_names=['problem'], output_names=['predictor_reasoning', 'predictor_answer'])
    registry.track(block, "reflector.prompt", input_names=['problem'], output_names=['reflector_reasoning', 'reflector_feedback', 'reflector_correctness'])
    registry.track(block, "refiner.prompt", input_names=['problem'], output_names=['refiner_reasoning', 'refiner_answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/reflector"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def optimize_debater(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score):
    block = debate(predictor=optimized_predictor, llm=executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "debater.prompt", input_names=['problem'], output_names=['reasoning', 'answer'])
    registry.track(block, "predictor.prompt", input_names=['problem'], output_names=['predictor_reasoning', 'predictor_answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/debater"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def optimize_executer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score):
    block = execute(predictor=optimized_predictor, benchmark=benchmark, llm=executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "predictor.prompt", input_names=['problem'], output_names=['predictor_reasoning', 'predictor_answer'])
    registry.track(block, "codereflector.prompt", input_names=['problem'], output_names=['reasoning', 'correctness', 'answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/executer"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def main():
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    optimizer_llm = OpenAILLM(config=optimizer_config)

    benchmark = MathSplits()
    
    # Step 0: 优化 Predictor
    # predictor = Predictor(llm=executor_llm)
    
    # predictor_score, optimized_predictor = optimize_predictor(predictor, optimizer_llm, benchmark)
    predictor_score, optimized_predictor = 50.0, Predictor(llm = executor_llm)


    # Step 1: 逐个优化每个block
    print("优化 summarizer...")
    optimized_summarizer = optimize_summarizer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 aggregator...")
    optimized_aggregator = optimize_aggregator(optimized_predictor, optimizer_llm, benchmark, predictor_score)
    
    print("优化 reflector...")
    optimized_reflector = optimize_reflector(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 debater...")
    optimized_debater = optimize_debater(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 executer...")
    optimized_executer = optimize_executer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)

    # 构建最终工作流
    block_workflow = WorkFlow(
        summarizer=optimized_summarizer,
        aggregater=optimized_aggregator,
        reflector=optimized_reflector,
        debater=optimized_debater,
        executer=optimized_executer
    )

    mass = MassOptimiser(workflow = block_workflow,
                         optimizer_llm = optimizer_llm,
                         max_labeled_demso = 0,
                         auto = "light",
                         save_path = "examples/mass/mass_optimization",
                         num_threads = 16)

    best_program = mass.optimize(benchmark = benchmark)
    block_workflow.save_config("examples/mass/best_workflow_config.json")
    print("Best workflow config saved to examples/mass/best_workflow_config.json")

if __name__ == "__main__":
    main()