from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.indices.property_graph import VectorContextRetriever

from .base import BaseRetrieverWrapper
from evoagentx.core.logging import logger
from evoagentx.rag.schema import RagQuery, RagResult, Corpus


class GraphRetriever(BaseRetrieverWrapper):
    """Wrapper for graph-based retrieval."""
    
    def __init__(self, graph_store: GraphStore, embed_model: BaseEmbedding, top_k: int = 5):
        super().__init__()
        self.graph_store = graph_store
        self.embed_model = embed_model
        self.top_k = top_k
        self.retriever = VectorContextRetriever(
            graph_store=self.graph_store,
            embed_model=self.embed_model,
            similarity_top_k=self.top_k
        )
    
    def retrieve(self, query: RagQuery) -> RagResult:
        try:
            nodes = self.retriever.retrieve(query.query_str)
            corpus = Corpus.from_llama_nodes(nodes)
            scores = [node.score or 0.0 for node in nodes]
            
            for chunk, score in zip(corpus.chunks, scores):
                chunk.metadata.similarity_score = score
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "retriever": "graph"}
            )
            logger.info(f"Graph retrieved {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            logger.error(f"Graph retrieval failed: {str(e)}")
            raise
    
    def get_retriever(self) -> VectorContextRetriever:
        logger.debug("Returning graph retriever")
        return self.retriever