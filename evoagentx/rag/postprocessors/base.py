import logging
from enum import Enum
from typing import List
from abc import ABC, abstractmethod

from evoagentx.rag.schema import RagQuery, RagResult


class RerankerType(str, Enum):
    SIMPLE = "simple"
    BGE = "bge"


class BasePostprocessor(ABC):
    """Base interface for post-processors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def postprocess(self, query: RagQuery, results: List[RagResult]) -> RagResult:
        """Post-process retrieval results."""
        pass