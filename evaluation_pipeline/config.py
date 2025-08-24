"""
Configuration file for evaluation pipeline
"""
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional
from evoagentx.tools import (
    BrowserUseToolkit,
    FileToolkit,
    GoogleFreeSearchToolkit,
    RequestToolkit,
    RSSToolkit,
    Toolkit
)

tools = [
    FileToolkit(),
    GoogleFreeSearchToolkit(),
    BrowserUseToolkit(),
    RequestToolkit(),
    RSSToolkit()
]

@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline"""
    
    # Directory paths
    eval_data_dir: str = "evaluation_pipeline/workflow_generation_eval_data"
    eval_execution_data_dir: str = "evaluation_pipeline/workflow_execution_eval_data"  # directory of execution test data, containing input test data of each field for each workflow
    checkpoint_dir: str = "evaluation_pipeline/checkpoints"
    results_dir: str = "evaluation_pipeline/results"
    
    # Parallel execution settings
    max_processes: int = 4
    max_threads_per_process: int = 2
    batch_size: int = 10
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds
    llm_request_timeout: float = 60.0  # seconds
    
    # Layer execution control
    layers_to_run: List[int] = field(default_factory=lambda: [1, 2, 3])  # None means run all layers [1, 2, 3]
    
    # LLM settings for evaluation
    evaluation_llm_type: str = "OpenAILLM"  # TODO: add more llm types
    evaluation_llm_model: str = "gpt-4o"
    evaluation_llm_temperature: float = 0.7
    
    # Workflow execution settings
    workflow_execution_timeout: float = 300.0  # seconds

    # tools for workflow generator
    tools: List[Toolkit] = field(default_factory=lambda: tools)
    
    def __post_init__(self):

        # Create directories if they don't exist
        for dir_path in [self.checkpoint_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'EvaluationConfig':
        """Create config from environment variables"""
        return cls(
            max_processes=int(os.getenv('EVAL_MAX_PROCESSES', '4')),
            max_threads_per_process=int(os.getenv('EVAL_MAX_THREADS', '2')),
            batch_size=int(os.getenv('EVAL_BATCH_SIZE', '10')),
            max_retries=int(os.getenv('EVAL_MAX_RETRIES', '3')),
            layers_to_run=[int(x) for x in os.getenv('EVAL_LAYERS', '1,2,3').split(',')]
        )
    
    def recursive_to_dict(self, obj: Any) -> dict:
        """recursively convert the object to a dictionary"""
        if isinstance(obj, list):
            return [self.recursive_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.recursive_to_dict(v) for k, v in obj.items()}
        elif not isinstance(obj, (str, int, float, bool)):
            # try to use class name as string
            try:
                return obj.__class__.__name__
            except:
                return str(obj)
        else:
            return obj

    def to_dict(self) -> dict:
        """convert the config to a dictionary"""
        config_dict = self.__dict__
        # replace the field of non-serializable objects with their string representation
        return self.recursive_to_dict(config_dict)
        for key, value in config_dict.items():
            if not isinstance(value, (str, int, float, bool, list, dict)):
                config_dict[key] = str(value)
        return config_dict
