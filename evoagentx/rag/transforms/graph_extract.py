import asyncio
from typing import Any, Optional, Sequence

from llama_index.core.async_utils import run_jobs
from llama_index.core.schema import TransformComponent, BaseNode, MetadataMode
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)

from evoagentx.core.logging import logger
from evoagentx.models.base_model import BaseLLM, LLMOutputParser
from evoagentx.prompts.rag.graph_extract import ENTITY_EXTRACT_PROMPT, RELATION_EXTRACT_PROMPT


class BasicGraphExtractLLM(TransformComponent):
    """
    A TransformComponent for extracting knowledge graph triplets using an LLM without tool-calling capabilities.

    This class performs two-stage extraction:
    1. Entity extraction: Identifies named entities and their types (e.g., Person, Organization).
    2. Relation extraction: Identifies directed relationships between extracted entities.

    The extracted entities and relations are stored in the node's metadata for use in LlamaIndex's PropertyGraphIndex.

    Attributes:
        llm (BaseLLM): The language model for entity and relation extraction.
        entity_extract_prompt (str): Prompt template for entity extraction.
        relation_extract_prompt (str): Prompt template for relation extraction.
        num_workers (int): Number of workers for parallel processing of nodes.
    """

    llm: BaseLLM
    entity_extract_prompt: str
    relation_extract_prompt: str
    num_workers: int

    def __init__(
        self,
        llm: BaseLLM,
        entity_extract_prompt: Optional[str] = None,
        relation_extract_prompt: Optional[str] = None,
        num_workers: int = 4,
    ):
        """
        Initialize the BasicGraphExtractLLM.

        Args:
            llm (BaseLLM): The language model to use for extraction.
            entity_extract_prompt (Optional[str]): Custom prompt for entity extraction. Defaults to ENTITY_EXTRACT_PROMPT.
            relation_extract_prompt (Optional[str]): Custom prompt for relation extraction. Defaults to RELATION_EXTRACT_PROMPT.
            num_workers (int): Number of workers for parallel node processing. Defaults to 4.
        """

        super().__init__(
            llm=llm,
            entity_extract_prompt=entity_extract_prompt or ENTITY_EXTRACT_PROMPT,
            relation_extract_prompt=relation_extract_prompt or RELATION_EXTRACT_PROMPT,
            num_workers=num_workers,
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """
        Asynchronously extract entities and relations from a single node.

        This method performs two LLM calls:
        1. Extracts entities and their types using the entity_extract_prompt.
        2. Extracts relations between entities using the relation_extract_prompt.

        The results are stored in the node's metadata under KG_NODES_KEY and KG_RELATIONS_KEY.

        Args:
            node (BaseNode): The node containing text to process.

        Returns:
            BaseNode: The node with updated metadata containing extracted entities and relations.

        Raises:
            AssertionError: If the node lacks a 'text' attribute.
            ValueError: If JSON parsing of LLM output fails (handled with empty fallback).
        """
        assert hasattr(node, "text"), "Node must have a 'text' attribute"

        text = node.get_content(metadata_mode=MetadataMode.LLM)

        try:
            # Step 1: Extract entities and their types
            extract_prompt = self.entity_extract_prompt.replace("{text}", text)
            llm_response = await self.llm.async_generate(
                prompt=extract_prompt,
                parse_mode="json",
            )
            # Parse entity results into a JSON string
            json_string = llm_response.content.strip()
            # Create a mapping of entity names to their types
            entity_label_mapping = {
                entity_dict["name"]: entity_dict["type"]
                for entity_dict in LLMOutputParser._parse_json_content(json_string)["entities"]
            }

            # Step 2: Extract relations between entities
            relation_extract_prompt = self.relation_extract_prompt.replace("{text}", text).replace(
                "{entities_json}", json_string
            )
            llm_response = self.llm.generate(
                prompt=relation_extract_prompt,
                parse_mode="json",
            )

            # Parse relation results into triplets
            triples = LLMOutputParser._parse_json_content(llm_response.content.strip())["graph"]

        except ValueError as e:
            logger.warning(f"Failed to parse LLM output for node {node.node_id}: {str(e)}. Returning empty triples.")
            entity_label_mapping = {}
            triples = []

        logger.info(f"Extracted triples from chunk: {triples}")

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        metadata = node.metadata.copy()

        # Convert extracted triplets into EntityNode and Relation objects
        for triple in triples:
            subj, rel, obj = triple["source"], triple["relation"], triple["target"]
            # Normalize entity and relation names to lowercase and replace spaces with underscores
            subj = subj.capitalize().replace(" ", "_")
            rel = rel.lower().replace(" ", "_")
            obj = obj.capitalize().replace(" ", "_")

            subj_node = EntityNode(
                name=subj,
                label=entity_label_mapping.get(subj, "entity"),
            )
            obj_node = EntityNode(
                name=obj,
                label=entity_label_mapping.get(obj, "entity"),
            )
            # Create relation between entities
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        # Update node metadata with extracted entities and relations
        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

    def __call__(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        """
        Synchronously extract triples from a sequence of nodes.

        This method wraps the asynchronous acall method for synchronous execution.

        Args:
            nodes (Sequence[BaseNode]): The nodes to process.
            show_progress (bool): Whether to display a progress bar. Defaults to False.
            **kwargs: Additional keyword arguments passed to acall.

        Returns:
            Sequence[BaseNode]: The processed nodes with updated metadata.
        """
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def acall(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        """
        Asynchronously extract triples from a sequence of nodes.

        This method processes nodes in parallel using run_jobs for efficiency.

        Args:
            nodes (Sequence[BaseNode]): The nodes to process.
            show_progress (bool): Whether to display a progress bar. Defaults to False.
            **kwargs: Additional keyword arguments passed to run_jobs.

        Returns:
            Sequence[BaseNode]: The processed nodes with updated metadata.
        """
        jobs = [self._aextract(node, **kwargs) for node in nodes]
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

    @classmethod
    def class_name(cls) -> str:
        return "BasicGraphExtractLLM"


if __name__ == "__main__":
    import os
    import dotenv

    dotenv.load_dotenv()

    from llama_index.core.schema import TextNode
    # from evoagentx.models import OpenRouterConfig, OpenRouterLLM

    # OPEN_ROUNTER_API_KEY = os.environ["OPEN_ROUNTER_API_KEY"]
    # config = OpenRouterConfig(
    #     openrouter_key=OPEN_ROUNTER_API_KEY,
    #     temperature=0.5,
    #     model="google/gemini-2.5-flash-lite-preview-06-17",
    # )
    # llm = OpenRouterLLM(config=config)

    from evoagentx.models import OpenAILLMConfig, OpenAILLM

    config = OpenAILLMConfig(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
        openai_key=os.environ["OPENAI_API_KEY"],
    )

    llm = OpenAILLM(config=config)

    trans = BasicGraphExtractLLM(llm=llm)
    node = TextNode(
        text="Satya Nadella, the CEO of Microsoft, announced a new partnership with OpenAI in 2023. Microsoft, headquartered in Redmond, Washington, will integrate OpenAI’s AI technologies into its Azure cloud platform. OpenAI, based in San Francisco, California, is known for developing ChatGPT."
    )
    graph_nodes = trans([node] * 10)