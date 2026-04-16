"""
ResearchToolkit — wraps the research tools as a single Toolkit.

Compatibility note: the ``_compat`` module (imported via ``__init__.py``)
patches all missing symbols into the project before this module loads,
so we can use normal project imports here.
"""

import os

from ...core.logging import logger
from ..tool import Toolkit
from ..storage_handler import FileStorageHandler

from .paper_search import ArxivPaperSearchTool


class ResearchToolkit(Toolkit):
    """Unified research toolkit.

    Always registers :class:`ArxivPaperSearchTool`.
    When ``SERPAPI_KEY`` (and optionally ``OPENROUTER_API_KEY``) are set,
    also registers ``PaperSearchTool``, ``FetchPaperMetaDataTool``, and
    ``BibReferenceTool``.
    """

    def __init__(
        self,
        name: str = "ResearchToolkit",
        storage_handler: FileStorageHandler = None,
        llm=None,
        **kwargs,
    ):
        # Gather API keys
        serpapi_key = kwargs.get("serpapi_key") or os.getenv("SERPAPI_KEY")
        openrouter_key = kwargs.get("openrouter_key") or os.getenv("OPENROUTER_API_KEY")
        semantic_scholar_api_key = kwargs.get("semantic_scholar_api_key") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        pubmed_api_key = kwargs.get("pubmed_api_key") or os.getenv("PUBMED_API_KEY")

        # Default storage handler
        if storage_handler is None:
            from ..storage_handler import LocalStorageHandler
            storage_handler = LocalStorageHandler()

        tools = []

        # ArxivPaperSearchTool works without any API keys
        tools.append(ArxivPaperSearchTool())

        # Full-featured tools require SERPAPI_KEY
        if serpapi_key:
            try:
                from .paper_search import PaperSearchTool
                from .fetch_paper import FetchPaperMetaDataTool
                from .bib_reference import BibReferenceTool

                tools.append(
                    PaperSearchTool(
                        storage_handler=storage_handler,
                        serpapi_key=serpapi_key,
                        openrouter_key=openrouter_key,
                        semantic_scholar_api_key=semantic_scholar_api_key,
                        pubmed_api_key=pubmed_api_key,
                        llm=llm,
                    )
                )
                tools.append(
                    FetchPaperMetaDataTool(
                        serpapi_key=serpapi_key,
                        openrouter_key=openrouter_key,
                        semantic_scholar_api_key=semantic_scholar_api_key,
                        pubmed_api_key=pubmed_api_key,
                    )
                )
                tools.append(
                    BibReferenceTool(
                        serpapi_key=serpapi_key,
                        openrouter_key=openrouter_key,
                        semantic_scholar_api_key=semantic_scholar_api_key,
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialise full research tools "
                    f"(SerpAPI/OpenRouter may be misconfigured): {e}. "
                    f"Only ArxivPaperSearchTool is available."
                )
        else:
            logger.info(
                "SERPAPI_KEY not set — ResearchToolkit running in arXiv-only "
                "mode.  Set SERPAPI_KEY and OPENROUTER_API_KEY to enable full "
                "paper search, metadata fetching, and BibTeX lookup."
            )

        super().__init__(name=name, tools=tools)
        self.storage_handler = storage_handler