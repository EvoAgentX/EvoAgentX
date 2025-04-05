from typing import Literal, List, Dict, Optional, Any, Type
from pydantic import BaseModel, Field
import asyncio
import time
from tqdm import tqdm

from .optimizer import Optimizer
from ..core.logging import logger
from ..models.base_model import BaseLLM
from ..models.model_configs import LLMConfig
from ..utils.data_utils import DataUtils
from ..utils.experience_utils import ExperienceUtils
from ..utils.aflow.aflow_evaluation_utils import EvaluationUtils
from ..utils.aflow.aflow_utils import GraphUtils
from ..utils.aflow.aflow_convergence_utils import ConvergenceUtils
from ..workflow.base_action_graph import BaseActionGraph
from ..workflow.action_graph import HumanEvalActionGraph
from ..config.optimizer_config import OptimizerConfig, OptimizerType

# Type definitions
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]
QuestionType = Literal["math", "code", "qa"]

class GraphOptimize(BaseModel):
    """Model for graph optimization results"""
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")

class AFlowOptimizer(Optimizer):
    """AFlow Optimizer for workflow optimization"""
    
    def __init__(self, config: OptimizerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all necessary components"""
        self.root_path = f"{self.config.optimized_path}/{self.config.dataset}"
        
        # Initialize utilities
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)
        
        # Initialize state
        self.graph = None
        self.round = self.config.initial_round
        
        # Initialize action graph
        self.action_graph = self.config.action_graph_class(
            llm_config=self.config.action_graph_llm_config
        )
        
    async def _execute_with_retry(self, func: callable, max_retries: int = 1) -> Any:
        """Execute a function with retry logic"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                return await func()
            except Exception as e:
                retry_count += 1
                logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                if retry_count == max_retries:
                    logger.info("Max retries reached.")
                    return None
                await asyncio.sleep(5 * retry_count)
        return None

    def optimize(self, mode: OptimizerType = "Graph"):
        """Main optimization loop"""
        logger.info(f"Starting optimization in {mode} mode")
        
        if mode == "Test":
            self._run_test_mode()
            return
            
        self._run_graph_mode()
        
    def _run_test_mode(self):
        """Run in test mode"""
        test_n = 3
        for _ in tqdm(range(test_n)):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.test())
            
    def _run_graph_mode(self):
        """Run in graph optimization mode"""
        for _ in range(self.config.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            score = loop.run_until_complete(
                self._execute_with_retry(self._optimize_graph)
            )
            
            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")
            
            if self._check_convergence():
                break
                
    def _check_convergence(self) -> bool:
        """Check if optimization has converged"""
        if not self.config.check_convergence:
            return False
            
        converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)
        if converged:
            logger.info(
                f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
            )
            self.convergence_utils.print_results()
            return True
        return False

    async def _optimize_graph(self):
        """Optimize the graph for one round"""
        validation_n = self.config.validation_rounds
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)
        
        if self.round == 1:
            return await self._handle_initial_round(graph_path, validation_n, data)
            
        return await self._handle_optimization_round(graph_path, validation_n, data)
        
    async def _handle_initial_round(self, graph_path: str, validation_n: int, data: List) -> float:
        """Handle the initial round of optimization"""
        directory = self.graph_utils.create_round_directory(graph_path, self.round)
        self.graph = self.graph_utils.load_graph(self.round, graph_path)
        return await self.evaluation_utils.evaluate_graph_async(
            self, directory, validation_n, data, initial=True
        )
        
    async def _handle_optimization_round(self, graph_path: str, validation_n: int, data: List) -> float:
        """Handle subsequent optimization rounds"""
        directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)
        
        # Get and process sample data
        sample = self._get_optimization_sample()
        response = await self._generate_optimization_response(sample, graph_path)
        
        # Save and evaluate results
        self._save_optimization_results(directory, response)
        return await self._evaluate_optimization_results(directory, validation_n, data)
        
    def _get_optimization_sample(self) -> Dict:
        """Get sample data for optimization"""
        top_rounds = self.data_utils.get_top_rounds(self.config.sample)
        return self.data_utils.select_round(top_rounds)
        
    async def _generate_optimization_response(self, sample: Dict, graph_path: str) -> Dict:
        """Generate optimization response using the action graph"""
        prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
        graph = self.graph_utils.extract_solve_graph(graph_load)
        
        processed_experience = self.experience_utils.load_experience()
        experience = self.experience_utils.format_experience(processed_experience, sample["round"])
        
        operator_description = self.graph_utils.load_operators_description(self.config.operators)
        log_data = self.data_utils.load_log(sample["round"])
        
        graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
            experience, sample["score"], graph[0], prompt, operator_description,
            self.config.question_type, log_data
        )
        
        while True:
            response = await self.action_graph.execute_async(graph_optimize_prompt)
            if self.action_graph.validate_response(response) and self.experience_utils.check_modification(
                processed_experience,
                response['modification'],
                sample["round"]
            ):
                break
                
        return response
        
    def _save_optimization_results(self, directory: str, response: Dict):
        """Save optimization results"""
        self.graph_utils.write_graph_files(
            directory,
            response,
            self.round + 1, 
            self.config.dataset
        )
        
        sample = self._get_optimization_sample()
        experience = self.experience_utils.create_experience_data(
            sample, 
            response['modification']
        )
        
        self.graph = self.graph_utils.load_graph(
            self.round + 1, 
            f"{self.root_path}/workflows"
        )
        
        self.experience_utils.update_experience(
            directory,
            experience,
            None  # Score will be updated after evaluation
        )
        
    async def _evaluate_optimization_results(self, directory: str, validation_n: int, data: List) -> float:
        """Evaluate optimization results"""
        return await self.evaluation_utils.evaluate_graph_async(
            self, directory,
            validation_n,
            data,
            initial=False
        )
        
    async def test(self):
        """Run test evaluation"""
        logger.info("Running test evaluation...")
        
        rounds = [21]
        data = []
        graph_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(graph_path)
        
        for round in tqdm(rounds, desc="Testing"):
            directory = self.graph_utils.create_round_directory(graph_path, round)
            self.graph = self.graph_utils.load_graph(round, graph_path)
            
            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test_async(
                self, directory, is_test=True
            )
            
            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost)
            data.append(new_data)
            
            self.data_utils.save_results(json_file_path, data)
        
        
        
        
        
        