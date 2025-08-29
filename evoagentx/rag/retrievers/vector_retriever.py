from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import VectorIndexRetriever

from .base import BaseRetrieverWrapper
from evoagentx.core.logging import logger
from evoagentx.rag.schema import Query, RagResult, Corpus

class VectorRetriever(BaseRetrieverWrapper):
    """Wrapper for vector-based retrieval."""
    
    def __init__(self, index: BaseIndex, top_k: int = 5, chunk_class=None):
        super().__init__()
        self.index = index
        self.top_k = top_k
        self.chunk_class = chunk_class  # Use chunk class passed from RAGEngine based on config
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k
        )

    async def aretrieve(self, query: Query) -> RagResult:
        try:
            # config the top_k
            self.retriever.similarity_top_k = query.top_k
            nodes = await self.retriever.aretrieve(query.query_str)

            corpus = Corpus()
            scores = []
            
            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={"query": query.query_str, "retriever": "vector"})
            
            for score_node in nodes:
                if self.chunk_class is None:
                    raise ValueError("chunk_class not set - RAGEngine must pass chunk class based on config")
                chunk = self.chunk_class.from_llama_node(score_node.node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "retriever": "vector"}
            )
            logger.info(f"Vector retrieved {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            raise

    def retrieve(self, query: Query) -> RagResult:
        try:
            # config the top_k
            self.retriever.similarity_top_k = query.top_k
            nodes = self.retriever.retrieve(query.query_str)
            corpus = Corpus()
            scores = []

            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={"query": query.query_str, "retriever": "vector"})
            
            for score_node in nodes:
                if self.chunk_class is None:
                    raise ValueError("chunk_class not set - RAGEngine must pass chunk class based on config")
                chunk = self.chunk_class.from_llama_node(score_node.node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "retriever": "vector"}
            )
            logger.info(f"Vector retrieved {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            raise
    
    def get_retriever(self) -> VectorIndexRetriever:
        logger.debug("Returning vector retriever")
        return self.retriever