from abc import ABC, abstractmethod
from typing import Union, Optional, Dict

from evoagentx.rag.schema import Query


class BaseQueryTransform(ABC):

    @abstractmethod
    def _run(self, query: Query, metadata: Dict) -> Query:
        """The Main run logic for Transform"""
    
    def run(
        self,
        query_or_str: Union[str, Query],
        metadata: Optional[Dict] = None,
    ) -> Query:
        """Run query transform."""
        metadata = metadata or {}
        if isinstance(query_or_str, str):
            query = Query(
                query_str=query_or_str,
                custom_embedding_strs=[query_or_str],
            )
        else:
            query = query_or_str

        return self._run(query, metadata=metadata)

    def __call__(
        self,
        query_bundle_or_str: Union[str, Query],
        metadata: Optional[Dict] = None,
    ) -> Query:
        """Run query processor."""
        return self.run(query_bundle_or_str, metadata=metadata)