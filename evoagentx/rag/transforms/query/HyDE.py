from string import Template
from typing import Dict, Union, Optional

from evoagentx.rag.schema import Query
from evoagentx.models.base_model import BaseLLM
from evoagentx.rag.transforms.query.base import BaseQueryTransform
from evoagentx.prompts.rag.hyde import DEFAULT_HYDE_PROMPT, HYDE_SYSTEM_IMPLE_


class HyDETransform(BaseQueryTransform):
    """
    Hypothetical Document Embeddings (HyDE) query transform.

    This class implements the HyDE technique for improving dense retrieval, as described in
    `Precise Zero-Shot Dense Retrieval without Relevance Labels` (https://arxiv.org/abs/2212.10496).
    It uses a language model to generate a hypothetical document (answer) for a given query, which
    is then used to create embedding strings for enhanced retrieval.

    Attributes:
        _llm (BaseLLM): The language model used to generate hypothetical documents.
        _hyde_prompt (Union[str, Template]): The prompt template for generating hypothetical documents.
        _include_original (bool): Whether to include the original query's embedding strings in the output.
    """

    def __init__(
        self,
        llm: BaseLLM,
        hyde_prompt: Optional[Union[str, Template]] = None,
        include_original: bool = True,
    ) -> None:
        """
        Initialize the HyDETransform.

        Args:
            llm (BaseLLM): The language model for generating hypothetical documents.
            hyde_prompt (Optional[Union[str, Template]]): Custom prompt template for HyDE generation.
                Defaults to DEFAULT_HYDE_PROMPT if not provided.
            include_original (bool): Whether to include the original query's embedding strings
                alongside the hypothetical document. Defaults to True.
        """
        self._llm = llm
        self._hyde_prompt = hyde_prompt or DEFAULT_HYDE_PROMPT
        self._include_original = include_original

    def _run(self, query: Query, metadata: Dict) -> Query:
        """
        Transform a query by generating a hypothetical document and updating embedding strings.

        This method uses the LLM to generate a hypothetical answer to the query, which is then
        used as an embedding string for retrieval. If include_original is True, the original
        query's embedding strings are also retained.

        Args:
            query (Query): The input query to transform.
            metadata (Dict): Additional metadata associated with the query (not used in this implementation).

        Returns:
            Query: A new Query instance with updated embedding strings, including the hypothetical document.
        """
        query_str = query.query_str

        # Format the prompt by substituting the query string into the HyDE prompt template
        instruction = self._hyde_prompt.format_map({"query": query_str})

        hypothetical_doc = self._llm.generate(
            prompt=instruction,
            system_message=HYDE_SYSTEM_IMPLE_,
        ).content

        # Initialize embedding strings with the hypothetical document
        embedding_strs = [hypothetical_doc]
        # Append original embedding strings if specified
        if self._include_original:
            embedding_strs.extend(query.embedding_strs)

        # Create a deep copy of the input query to avoid modifying the original
        tmp_query = query.deepcopy()
        tmp_query.custom_embedding_strs = embedding_strs
        return tmp_query


if __name__ == "__main__":
    import dotenv
    import os

    dotenv.load_dotenv()

    from evoagentx.models import OpenAILLMConfig, OpenAILLM

    os.environ["SSL_CERT_FILE"] = r"D:\miniconda3\envs\envoagentx\Library\ssl\cacert.pem"

    config = OpenAILLMConfig(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
        openai_key=os.environ["OPENAI_API_KEY"],
    )

    llm = OpenAILLM(config=config)
    hyde_trans = HyDETransform(llm=llm)
    output_query = hyde_trans(Query(query_str="Were Scott Derrickson and Ed Wood of the same nationality?"))
    print(output_query)