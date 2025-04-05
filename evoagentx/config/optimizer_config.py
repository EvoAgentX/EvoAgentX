from typing import Dict, List, Optional, Type
from pydantic import BaseModel, Field

from ..models.base_model import BaseLLM
from ..models.model_configs import LLMConfig
from ..workflow.base_action_graph import BaseActionGraph
from ..workflow.action_graph import HumanEvalActionGraph

# Type definitions
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]
QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]

class OptimizerConfig(BaseModel):
    """Configuration for the optimizer"""
    dataset: DatasetType
    operators: List[str]
    question_type: QuestionType
    sample: int = 1
    check_convergence: bool = False
    initial_round: int = 1
    max_rounds: int = 20
    validation_rounds: int = 5
    optimized_path: Optional[str] = None
    optimizer_llm: Optional[BaseLLM] = None
    executor_llm: Optional[BaseLLM] = None
    action_graph_llm_config: Optional[LLMConfig] = None
    action_graph_class: Type[BaseActionGraph] = HumanEvalActionGraph

    class Config:
        arbitrary_types_allowed = True

# Default configurations for different datasets
DEFAULT_CONFIGS: Dict[str, OptimizerConfig] = {
    "HumanEval": OptimizerConfig(
        dataset="HumanEval",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
        question_type="code",
        check_convergence=True,
        max_rounds=20,
        validation_rounds=5
    ),
    "MBPP": OptimizerConfig(
        dataset="MBPP",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
        question_type="code",
        check_convergence=True,
        max_rounds=20,
        validation_rounds=5
    ),
    "MATH": OptimizerConfig(
        dataset="MATH",
        operators=["Custom", "ScEnsemble", "Programmer"],
        question_type="math",
        check_convergence=True,
        max_rounds=20,
        validation_rounds=5
    ),
    "GSM8K": OptimizerConfig(
        dataset="GSM8K",
        operators=["Custom", "ScEnsemble", "Programmer"],
        question_type="math",
        check_convergence=True,
        max_rounds=20,
        validation_rounds=5
    ),
    "HotpotQA": OptimizerConfig(
        dataset="HotpotQA",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
        question_type="qa",
        check_convergence=True,
        max_rounds=20,
        validation_rounds=5
    ),
    "DROP": OptimizerConfig(
        dataset="DROP",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
        question_type="qa",
        check_convergence=True,
        max_rounds=20,
        validation_rounds=5
    )
}

def get_default_config(dataset: DatasetType) -> OptimizerConfig:
    """Get default configuration for a specific dataset"""
    if dataset not in DEFAULT_CONFIGS:
        raise ValueError(f"No default configuration found for dataset: {dataset}")
    return DEFAULT_CONFIGS[dataset]

def create_custom_config(
    dataset: DatasetType,
    operators: List[str],
    question_type: QuestionType,
    **kwargs
) -> OptimizerConfig:
    """Create a custom configuration with default values for unspecified parameters"""
    base_config = get_default_config(dataset)
    custom_config = OptimizerConfig(
        dataset=dataset,
        operators=operators,
        question_type=question_type,
        **kwargs
    )
    return custom_config 