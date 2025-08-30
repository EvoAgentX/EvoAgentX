"""
MASS Agents Package

这个包包含了使用CustomizeAgent实现的各种MASS工作流组件，
对应原operators.py中的各种Operator。
"""

from .predictor_agent import PredictorAgent, create_predictor_agent
from .summarizer_agent import SummarizerAgent, create_summarizer_agent
from .reflector_agent import ReflectorAgent, create_reflector_agent
from .debater_agent import DebaterAgent, create_debater_agent
from .code_reflector_agent import CodeReflectorAgent, create_code_reflector_agent

__all__ = [
    'PredictorAgent',
    'SummarizerAgent', 
    'ReflectorAgent',
    'DebaterAgent',
    'CodeReflectorAgent',
    'create_predictor_agent',
    'create_summarizer_agent',
    'create_reflector_agent',
    'create_debater_agent',
    'create_code_reflector_agent'
]
