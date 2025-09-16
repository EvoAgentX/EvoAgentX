"""
MASS Blocks Package

这个包包含了使用CustomizeAgent实现的各种MASS工作流组件，
对应原operators.py中的各种Operator。
"""

from .predictor_agent import Predictor, create_predictor_agent
from .aggregate import Aggregate, create_aggregate
from .debate import Debate, create_debate_agent
from .reflect import Reflect, create_reflect_agent
from .summarize import Summarize, create_summarize_agent
from .execute import Execute, create_execute_agent
from .utils import normalize_text, get_most_common_prediction, create_deep_copy

__all__ = [
    'Predictor',
    'Aggregate',
    'Debate',
    'Reflect', 
    'Summarize',
    'Execute',
    'create_predictor_agent',
    'create_aggregate',
    'create_debate_agent',
    'create_reflect_agent',
    'create_summarize_agent',
    'create_execute_agent',
    'normalize_text',
    'get_most_common_prediction',
    'create_deep_copy'
]
