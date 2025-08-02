from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from evoagentx.rag.schema import Query, RagResult


class RerankerType(str, Enum):
    SIMPLE = "simple"
    BGE = "bge"


class BasePostprocessor(ABC):
    """Base interface for post-processors."""
    
    @abstractmethod
    def postprocess(self, query: Query, results: List[RagResult]) -> RagResult:
        """Post-process retrieval results."""
        pass