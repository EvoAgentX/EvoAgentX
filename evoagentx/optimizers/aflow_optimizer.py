from .optimizer import Optimizer
from typing import Literal, List, Dict
from ..core.logging import logger
import time
from ..utils.aflow_utils import GraphUtils
from ..utils.aflow_evaluation_utils import EvaluationUtils
from ..models.base_model import BaseLLM
from evoagentx.ext.aflow.scripts.utils.data_utils import DataUtils
from evoagentx.ext.aflow.scripts.utils.experience_utils import ExperienceUtils
from pydantic import BaseModel, Field
from evoagentx.workflow.action_graph import HumanEvalActionGraph
from ..models.model_configs import LLMConfig


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
        
        self.workflow_graph = None
        self.evaluator = None
        self.max_steps = max_rounds
        
        self.optimizer_llm = optimizer_llm
        self.executor_llm = executor_llm
        
        
        self.action_graph = HumanEvalActionGraph(llm_config=action_graph_llm_config)
        
    def optimize(self, mode: OptimizerType = "Graph"):
        
        if mode == "Test":
            # TODO
            return None
        
        for _ in range(self.max_rounds):
 
            retry_count = 0
            max_retries = 1
            
            while retry_count < max_retries:
                try:
                    score = self._optimize_graph()
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)
            
            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")
        
        
    def _optimize_graph(self):
        
        validation_n = self.validation_rounds
        graph_path = f"{self.root_path}/workflows"
        
        data = []
        
        # We first set the check to True
        # check = True
        
        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            logger.info(f"self graph is {self.graph}")
            avg_score = self.evaluation_utils.evaluate_graph(self, 
                                                             directory, 
                                                             validation_n, 
                                                             data, 
                                                             initial=True)
            
        while True:
            
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)
            
            top_rounds = self.data_utils.get_top_rounds(self.sample)
            sample = self.data_utils.select_round(top_rounds)
            
            # logger.info(f"sample is {sample}")  
            # logger.info(f"graph_path is {graph_path}")
            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            # logger.info(f"graph_load is {graph_load}")
            graph = self.graph_utils.extract_solve_graph(graph_load)
            # logger.info(f"graph is {graph}")
            
            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(processed_experience, 
                                                                 sample["round"])
            
            # logger.info(f"experience is {experience}")
            operator_description = self.graph_utils.load_operators_description(self.operators)
            log_data = self.data_utils.load_log(sample["round"])
            
            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, sample["score"], graph[0], prompt, operator_description,
                self.type, log_data
            )
            
            # logger.info(f"graph_optimize_prompt is {graph_optimize_prompt}")
            
            response = self.action_graph.execute(graph_optimize_prompt)
            logger.info(f"response is {response}")
            
            check = self.experience_utils.check_modification(
                processed_experience,
                response['modification'],
                sample["round"]
            )
            
            logger.info(f"check is {check}")      
            
            if check:
                break
            
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
        
        logger.info(f"experience is {experience}")
        

        self.graph = self.graph_utils.load_graph(
            self.round + 1, 
            graph_path
        )
        
        logger.info(f"self.graph is {self.graph}")
        
        avg_score = self.evaluation_utils.evaluate_graph(
            self, directory,
            validation_n,
            data,
            initial=False
        )
    
        logger.info(f"Score for round {self.round + 1}: {avg_score}")
        
        self.experience_utils.update_experience(
            directory,
            experience,
            avg_score
        )
        
        return avg_score
            


        
        
        
        
        