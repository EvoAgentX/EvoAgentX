from .optimizer import Optimizer
from typing import Literal, List, Dict
from ..core.logging import logger
import time
import asyncio
from ..models.base_model import BaseLLM
from ..utils.data_utils import DataUtils
from ..utils.experience_utils import ExperienceUtils
from ..utils.aflow.aflow_evaluation_utils import EvaluationUtils
from ..utils.aflow.aflow_utils import GraphUtils
from pydantic import BaseModel, Field
from ..workflow.action_graph import HumanEvalActionGraph
from ..models.model_configs import LLMConfig
from ..utils.aflow.aflow_convergence_utils import ConvergenceUtils
from tqdm import tqdm

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]
QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators
        
        
EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "DROP": ExperimentConfig(
        dataset="DROP",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "HotpotQA": ExperimentConfig(
        dataset="HotpotQA",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "MBPP": ExperimentConfig(
        dataset="MBPP",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
}

class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class AFlowOptimizer(Optimizer):

    def __init__(self, 
                 dataset: DatasetType,
                 operators: List,
                 question_type: QuestionType,
                 sample: int = 1,
                 check_convergence: bool = False,
                 initial_round: int = 1,
                 max_rounds: int = 20,
                 validation_rounds: int = 5,
                 optimized_path: str = None,
                 optimizer_llm: BaseLLM = None,
                 executor_llm: BaseLLM = None,
                 action_graph_llm_config: LLMConfig = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.dataset = dataset
        self.operators = operators
        self.type = question_type
        self.check_convergence = check_convergence
        
        self.sample = sample
        
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds
        
        self.root_path = f"{optimized_path}/{self.dataset}"
        
        self.graph = None
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)
        
        self.workflow_graph = None
        self.evaluator = None
        self.max_steps = max_rounds
        
        self.optimizer_llm = optimizer_llm
        self.executor_llm = executor_llm
        
        
        self.action_graph = HumanEvalActionGraph(llm_config=action_graph_llm_config)
        
    def optimize(self, mode: OptimizerType = "Graph"):
        
        logger.info(f"mode is {mode}")
        
        if mode == "Test":
            # TODO
            test_n = 1
            for _ in tqdm(range(test_n)):
                loop = asyncio.get_event_loop()
                score = loop.run_until_complete(self.test())
            return None
        
        for _ in range(self.max_rounds):
            
            loop = asyncio.get_event_loop()
            
            retry_count = 0
            max_retries = 1
            
            while retry_count < max_retries:
                try:
                    # score = await self._optimize_graph()
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)
                
                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
            
            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")
            
            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)

            logger.info(f"converged is {converged}, convergence_round is {convergence_round}, final_round is {final_round}")
            
            
            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                break
        
        
    async def _optimize_graph(self):
        
        validation_n = self.validation_rounds
        graph_path = f"{self.root_path}/workflows"
        
        data = []
        
        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            
            logger.info(f"self.graph is {self.graph}")

            avg_score = await self.evaluation_utils.evaluate_graph_async(self, 
                                                                   directory, 
                                                                   validation_n, 
                                                                   data, 
                                                                   initial=True)
            
            logger.info(f"avg_score is {avg_score}")
            
        while True:
            
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)
            
            top_rounds = self.data_utils.get_top_rounds(self.sample)
            sample = self.data_utils.select_round(top_rounds)
            
            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)

            graph = self.graph_utils.extract_solve_graph(graph_load)
            
            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(processed_experience, 
                                                                 sample["round"])
            
            operator_description = self.graph_utils.load_operators_description(self.operators)
            log_data = self.data_utils.load_log(sample["round"])
            
            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, sample["score"], graph[0], prompt, operator_description,
                self.type, log_data
            )
            
            # This is where we make the LLM call - potentially async
            response = await self.action_graph.execute_async(graph_optimize_prompt)
        
            
            logger.info(f"response is {response}")
            
            check = self.experience_utils.check_modification(
                processed_experience,
                response['modification'],
                sample["round"]
            )
            
            logger.info(f"check is {check}")
            
            if check:
                break
        
        logger.info(f"response is {response}")
            
        # Save the graph and evaluate
        self.graph_utils.write_graph_files(
            directory,
            response,
            self.round + 1, 
            self.dataset
        )
        
        experience = self.experience_utils.create_experience_data(
            sample, 
            response['modification']
        )

        self.graph = self.graph_utils.load_graph(
            self.round + 1, 
            graph_path
        )
        
        avg_score = await self.evaluation_utils.evaluate_graph_async(
            self, directory,
            validation_n,
            data,
            initial=False
        )
    
        self.experience_utils.update_experience(
            directory,
            experience,
            avg_score
        )
        
        return avg_score
    
    async def test(self):
        
        logger.info(f"Testing...")
        
        rounds = [7, 8]
        data = []
        
        graph_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(graph_path)
        
        for round in tqdm(rounds, desc="Testing"):
            directory = self.graph_utils.create_round_directory(
                graph_path, round
            )
            self.graph = self.graph_utils.load_graph(
                round, graph_path
            )
            
            logger.info(f"self.graph is {self.graph}")
            logger.info(f"directory is {directory}")
            
            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test_async(self, 
                                                                                                directory, 
                                                                                                is_test=True)
            
            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost)
            data.append(new_data)
            
            self.data_utils.save_results(json_file_path, data)

        
        
        
        
        