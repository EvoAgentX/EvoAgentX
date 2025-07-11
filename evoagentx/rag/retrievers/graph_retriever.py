import json
from typing import Any, Callable, Optional, List


from llama_index.core.schema import NodeWithScore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.property_graph import PGRetriever
from llama_index.core.vector_stores.types import  BasePydanticVectorStore
from llama_index.core.graph_stores.types import PropertyGraphStore, KG_SOURCE_REL
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.indices.property_graph.sub_retrievers.vector import VectorContextRetriever

from .base import BaseRetrieverWrapper
from evoagentx.models import BaseLLM
from evoagentx.core.logging import logger
from evoagentx.rag.schema import Query, RagResult, Corpus, Chunk
from evoagentx.prompts.rag.graph_synonym import DEFAULT_SYNONYM_EXPAND_TEMPLATE


class BasicLLMSynonymRetriever(BasePGRetriever):
    def __init__(
        self,
        graph_store: PropertyGraphStore,
        include_text: bool = True,
        include_properties: bool = False,
        synonym_prompt: str = DEFAULT_SYNONYM_EXPAND_TEMPLATE,
        max_keywords: int = 10,
        path_depth: int = 2,
        limit: int = 30,
        output_parsing_fn: Optional[Callable] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs: Any,
    ) -> None:

        self._llm = llm
        self._synonym_prompt = synonym_prompt
        self._output_parsing_fn = output_parsing_fn
        self._max_keywords = max_keywords
        self._path_depth = path_depth
        self._limit = limit

        super().__init__(
            graph_store=graph_store,
            include_text=include_text,
            include_properties=include_properties,
            **kwargs,
        )

    def _parse_llm_output(self, output: str) -> List[str]:
        if self._output_parsing_fn:
            matches = self._output_parsing_fn(output)
        else:
            matches = output.strip().split("^")

        # capitalize to normalize with ingestion
        return [x.strip().capitalize().replace(" ", "_") for x in matches if x.strip()]

    def _prepare_matches(
        self, matches: List[str], limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        kg_nodes = self._graph_store.get(ids=matches)
        triplets = self._graph_store.get_rel_map(
            kg_nodes,
            depth=self._path_depth,
            limit=limit or self._limit,
            ignore_rels=[KG_SOURCE_REL],
        )

        return self._get_nodes_with_score(triplets)

    async def _aprepare_matches(
        self, matches: List[str], limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        kg_nodes = await self._graph_store.aget(ids=matches)
        triplets = await self._graph_store.aget_rel_map(
            kg_nodes,
            depth=self._path_depth,
            limit=limit or self._limit,
            ignore_rels=[KG_SOURCE_REL],
        )

        return self._get_nodes_with_score(triplets)

    def retrieve_from_graph(
        self, query_bundle: Query, limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        
        # format the prompt
        synonym_prompt = self._synonym_prompt.format_map({"max_keywords": self._max_keywords, "query_str": query_bundle.query_str})
        response = self._llm.generate(
            prompt=synonym_prompt,
            parse_mode="str"
        )
        matches = self._parse_llm_output(response.content)
        logger.info(f"{self.__class__.__name__}, synonym words from llm: {matches}")

        return self._prepare_matches(matches, limit=limit or self._limit)

    async def aretrieve_from_graph(
        self, query_bundle: Query, limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        synonym_prompt = self._synonym_prompt.format_map({"max_keywords": self._limit, "query_str": query_bundle.query_str})
        response = await self._llm.async_generate(
            prompt=synonym_prompt,
            parse_mode="str"
        )
        matches = self._parse_llm_output(response.content)
        
        logger.info(f"{self.__class__.__name__}: query: {query_bundle.query_str} \nsynonym words from llm: {matches}")
        return await self._aprepare_matches(matches, limit=limit or self._limit)


class GraphRetriever(BaseRetrieverWrapper):
    """Wrapper for graph-based retrieval."""
    
    def __init__(self, llm: BaseLLM, graph_store: PropertyGraphStore, embed_model: Optional[BaseEmbedding], 
                 include_text: bool = True, _use_async: bool = True,
                 vector_store: Optional[BasePydanticVectorStore] = None,
                 top_k:int=5):
        super().__init__()
        self.graph_store = graph_store
        self._embed_model = embed_model
        self.vector_store = vector_store
        self._llm = llm
        
        sub_retrievers = [
            BasicLLMSynonymRetriever(graph_store=graph_store, include_text=include_text, llm=llm),
        ]

        if self._embed_model and (
                self.graph_store.supports_vector_queries or self.vector_store
        ):
            sub_retrievers.append(
                VectorContextRetriever(
                        graph_store=self.graph_store,
                        vector_store=self.vector_store,
                        include_text=include_text,
                        embed_model=self._embed_model,
                        similarity_top_k=top_k
                    )
            )

        self.retriever = PGRetriever(
            sub_retrievers, use_async=_use_async
        )
    
    async def aretrieve(self, query: Query) -> RagResult:
        try:
            # config the top_k
            subretriever_bool = [isinstance(sub, VectorContextRetriever) for sub in self.retriever.sub_retrievers]
            if any(subretriever_bool):
                ind = subretriever_bool.index(True) 
                self.retriever.sub_retrievers[ind]._similarity_top_k = query.top_k

            nodes = await self.retriever.aretrieve(query.query_str)

            corpus = Corpus()
            scores = []
            
            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={"query": query.query_str, "retriever": "graph"})
            
            for score_node in nodes:
                # parsed the metadata
                node = score_node.node
                node.metadata = json.loads(node.metadata.get('metadata', '{}'))

                chunk = Chunk.from_llama_node(node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            
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

    def retrieve(self, query: Query) -> RagResult:
        try:
            # config the top_k
            subretriever_bool = [isinstance(sub, VectorContextRetriever) for sub in self.retrieve.sub_retrievers]
            if any(subretriever_bool):
                ind = subretriever_bool.index(True) 
                self.retriever[ind].similarity_top_k = query.top_k
            nodes = self.retriever.retrieve(query.query_str)
            corpus = Corpus()
            scores = []

            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={"query": query.query_str, "retriever": "graph"})
            
            for score_node in nodes:
                # parsed the metadata
                node = score_node.node
                flattened_metadata = {}
                for key, value in node.metadata.items():
                        flattened_metadata[key] = json.loads(value)
                node.metadata = flattened_metadata
                
                chunk = Chunk.from_llama_node(score_node.node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "retriever": "graph"}
            )
            logger.info(f"Vector retrieved {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            raise
    
    def get_retriever(self) -> PGRetriever:
        logger.debug("Returning graph retriever")
        return self.retriever