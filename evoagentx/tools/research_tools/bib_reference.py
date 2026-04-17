import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from ...core.logging import logger
from ...utils.utils import add_dict
from ..tool import Tool, ToolMetadata, ToolResult
from .prompts import BIB_NOT_FOUND_TEXT
from .sources import DBLP
from .google_scholar import GoogleScholar
from .metadata import ResearchToolResult
from .cache import CacheMixin, BibReferenceCacheKeyGenerator

BIBREFERENCE_TOOL_EXTRA_DESCRIPTION = """
Examples: 
Arguments:
{
    "titles_or_keywords": ["trace the evidence: constructing knowledge-grounded reasoning"]
}

Return a dictionary of BibTeX entries, where the key is the title or keyword, and the value is the BibTeX entry, such as: 
{
    "trace the evidence: constructing knowledge-grounded reasoning": "@inproceedings{fang2024trace,
  author={Fang, Jinyuan and Meng, Zaiqiao and MacDonald, Craig},
  title={TRACE the Evidence: Constructing Knowledge-Grounded Reasoning Chains for Retrieval-Augmented Generation},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP},
  pages={8472--8494},
  year={2024},
}",
... # potentially more bibtex entries ...
}

Note that sometimes the author information is missing, which is normal. You should not try to infer the author information from the title or keyword, or call any other tools to infer the author information. Just return the BibTeX entry with the author information as is.
"""

class BibReferenceTool(CacheMixin, Tool):

    name: str = "search_bibtex"
    description: str = "Perform a web search to automatically retrieve BibTeX reference(s) for academic paper(s) based on one or more provided titles or keywords, and return standardized BibTeX entry (or entries). You should use this tool when dealing with tasks related to references and citations."
    extra_description: str = BIBREFERENCE_TOOL_EXTRA_DESCRIPTION.strip()
    inputs: Dict[str, Dict] = {
        "titles_or_keywords": {
            "type": "array",
            "description": "An array of titles or keywords, used to search for corresponding academic paper references. Each provided title or keyword will be used to perform a web search, and the function will return a dictionary of BibTeX entries, where the key is the title or keyword, and the value is the BibTeX entry.",
            "items": {"type": "string"}
        }
    }
    required: Optional[List[str]] = ["titles_or_keywords"]

    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY", None)
        openrouter_key = openrouter_key or os.getenv("OPENROUTER_API_KEY", None)

        if not serpapi_key:
            raise ValueError("SERPAPI_KEY is not set")

        self.google_scholar = GoogleScholar(
            serpapi_key=serpapi_key,
            openrouter_key=openrouter_key,
            semantic_scholar_api_key=semantic_scholar_api_key
        )
        self.dblp = DBLP()

        # Initialize cache if enabled
        self.enable_cache = enable_cache
        if enable_cache:
            self._init_cache(
                tool_name="bib_reference",
                cache_dir=cache_dir,
                key_generator=BibReferenceCacheKeyGenerator(),
            )

    def __call__(self, titles_or_keywords: list) -> ToolResult:

        metadata = ToolMetadata(
            tool_name=self.name, 
            args={"titles_or_keywords": titles_or_keywords} 
        )

        bibtexts = {}

        if not titles_or_keywords:
            return ToolResult(
                metadata=metadata,
                result={
                    "success": True,
                    "bibtex_entries": bibtexts,
                },
            )

        # Parallel execution of _search_bibtext
        with ThreadPoolExecutor(max_workers=min(len(titles_or_keywords), 5)) as executor:
            futures = [
                executor.submit(self._search_bibtext, title_or_keyword)
                for title_or_keyword in titles_or_keywords
            ]

            failed = []
            for title_or_keyword, future in zip(titles_or_keywords, futures):
                try:
                    bibtext, cost_breakdown = future.result()
                    bibtexts[title_or_keyword] = bibtext
                    metadata.add_cost_breakdown(cost_breakdown)
                except Exception as e:
                    logger.warning(f"Error searching BibTeX for {title_or_keyword}: {e}")
                    failed.append(title_or_keyword)

        if failed:
            if len(failed) == len(titles_or_keywords):
                return ToolResult(
                    metadata=metadata,
                    result={
                        "success": False,
                        "error": "Failed to find BibTeX for all titles or keywords"
                    },
                )
            
            return ToolResult(
                metadata=metadata,
                result={
                    "failed": failed,
                    "bibtex_entries": bibtexts,
                },
            )
        
        return ToolResult(
            metadata=metadata, 
            result={
                "success": True, 
                "bibtex_entries": bibtexts 
            }
        )
    
    def _search_bibtext(self, title_or_keyword: str) -> Tuple[str, Dict[str, float]]:

        cost_breakdown: Dict[str, float] = {}

        # Check cache first
        if self.enable_cache:
            cached_result, hit = self._cache_get(title_or_keyword)
            if hit and cached_result is not None:
                logger.debug(f"Cache hit for BibTeX: {title_or_keyword[:50]}...")
                return cached_result, cost_breakdown

        # first search from DBLP
        try:
            dblp_bibtext_results: ResearchToolResult = self.dblp.search_bibtext(title_or_keyword)
            dblp_bibtext: dict = dblp_bibtext_results.result
            cost_breakdown = add_dict(cost_breakdown, dblp_bibtext_results.metadata.cost_breakdown)
            if dblp_bibtext.get("bibtex", BIB_NOT_FOUND_TEXT) != BIB_NOT_FOUND_TEXT:
                # Cache successful result (background write)
                if self.enable_cache:
                    self._cache_set(dblp_bibtext["bibtex"], dblp_bibtext.get("title", title_or_keyword))
                return dblp_bibtext["bibtex"], cost_breakdown
        except Exception as e:
            logger.warning(f"Error searching bibtex for {title_or_keyword} from DBLP: {e}")

        # then search from Google Scholar
        try:
            google_scholar_bibtext_results: ResearchToolResult = self.google_scholar.search_bibtext(title_or_keyword)
            cost_breakdown = add_dict(cost_breakdown, google_scholar_bibtext_results.metadata.cost_breakdown)
            bibtext: dict = google_scholar_bibtext_results.result
            if bibtext.get("bibtex", BIB_NOT_FOUND_TEXT) != BIB_NOT_FOUND_TEXT:
                # Cache successful result (background write)
                if self.enable_cache:
                    self._cache_set(bibtext["bibtex"], bibtext.get("title", title_or_keyword))
                return bibtext["bibtex"], cost_breakdown
        except Exception as e:
            logger.warning(f"Error searching bibtext for {title_or_keyword} from Google Scholar: {e}")

        # if no bibtext found, return BIB_NOT_FOUND_TEXT
        return BIB_NOT_FOUND_TEXT, cost_breakdown
