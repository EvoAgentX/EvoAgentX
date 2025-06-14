from typing import List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor

from .base import BasePostprocessor
from evoagentx.rag.schema import RagQuery, RagResult, Corpus, Chunk


class SimpleReranker(BasePostprocessor):
    """Post-processor for reranking retrieval results."""
    
    def __init__(
        self,
        similarity_cutoff: Optional[float] = None,
        keyword_filters: Optional[List[str]] = None
    ):
        super().__init__()
        self.postprocessors = []
        if similarity_cutoff:
            self.postprocessors.append(SimilarityPostprocessor(similarity_cutoff=similarity_cutoff))
        if keyword_filters:
            self.postprocessors.append(KeywordNodePostprocessor(required_keywords=keyword_filters))
    
    def postprocess(self, query: RagQuery, results: List[RagResult]) -> RagResult:
        try:
            nodes = []
            for result in results:
                for chunk, score in zip(result.corpus.chunks, result.scores):
                    node = chunk.to_llama_node()
                    nodes.append(NodeWithScore(node=node, score=score))
            
            for postprocessor in self.postprocessors:
                nodes = postprocessor.postprocess_nodes(nodes)

            corpus = Corpus()
            scores = []
            for score_node in nodes:
                chunk = Chunk.from_llama_node(score_node.node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "postprocessor": "reranker"}
            )
            self.logger.info(f"Reranked to {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            self.logger.error(f"Reranking failed: {str(e)}")
            raise