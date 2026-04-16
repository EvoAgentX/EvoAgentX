import os
from typing import Any, Dict, List, Optional, Tuple

from ...core.logging import logger
from ...core.module_utils import parse_json_from_llm_output
from ...utils.utils import add_dict, ContextualThreadPoolExecutor
from ..crawler_crawl4ai import Crawl4AICrawlTool
from ..tool import Tool, ToolMetadata, ToolResult
from .google_scholar import GoogleScholar
from .sources import DBLP, Arxiv, SemanticScholar, search_authors_from_databases
from .utils import (
    extract_arxiv_id,
    is_pdf_link,
    is_arxiv_link,
    is_link_accessible,
    validate_author_info
)
from .prompts import (
    INFER_PAPER_VENUE_PROMPT,
    PAPER_METADATA_EXTRACTION_PROMPT,
    EXTRACT_AUTHORS_FROM_CHICAGO_CITATION_PROMPT,
)
from .cache import CacheMixin, PaperMetadataCacheKeyGenerator


FETCH_PAPER_METADATA_TOOL_EXTRA_DESCRIPTION = """
Examples:
Arguments:
{
    "papers": [
        {"paper_title": "Attention is All You Need", "paper_link": "https://arxiv.org/abs/1706.03762"},
        {"paper_title": "BERT: Pre-training of Deep Bidirectional Transformers"}
    ]
}

Returns a dict with 'success' and 'paper_metadata', where each paper maps to its detailed metadata:
{
    "success": true,
    "paper_metadata": [
        {
            "paper_title": "Attention is All You Need",
            "paper_link": "https://papers.nips.cc/paper/7181-attention-is-all-you-need",
            "paper_pdf_link": "https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", ...],
            "abstract": "The dominant sequence transduction models are based on...",
            "year": 2017,
            "venue": "NeurIPS",
            "citation_count": 50000
        },
        ...
    ]
}

Note: The returned paper_link may differ from the input if the original link has anti-crawling protection. The tool will attempt to find an accessible link for the paper.
"""

class FetchPaperMetaDataTool(CacheMixin, Tool):

    name: str = "fetch_paper_metadata"
    description: str = (
        "Fetch detailed metadata for a list of academic papers. "
        "Given paper titles and optional links, this tool retrieves comprehensive information including: "
        "paper_title, paper_link, paper_pdf_link, authors, abstract, year, venue, and citation_count. "
        "Use this tool when you need detailed information about specific papers beyond what search results provide."
    )
    extra_description: str = FETCH_PAPER_METADATA_TOOL_EXTRA_DESCRIPTION.strip()
    inputs: Dict[str, Dict] = {
        "papers": {
            "type": "array",
            "description": (
                "A list of papers to fetch metadata for. Each paper should be a dict with 'paper_title' (required) "
                "and optionally 'paper_link'. Example: [{'paper_title': 'Attention is All You Need', 'paper_link': 'https://...'}]"
            ),
            "items": {
                "type": "object",
                "properties": {
                    "paper_title": {"type": "string", "description": "The title of the paper"},
                    "paper_link": {"type": "string", "description": "Optional link to the paper page"}
                },
                "required": ["paper_title"]
            }
        }
    }
    required: Optional[List[str]] = ["papers"]

    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        pubmed_api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY", None)
        openrouter_key = openrouter_key or os.getenv("OPENROUTER_API_KEY", None)
        semantic_scholar_api_key = semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", None)
        pubmed_api_key = pubmed_api_key or os.getenv("PUBMED_API_KEY", None)

        if not serpapi_key:
            raise ValueError("SERPAPI_KEY is not set")

        self.google_scholar = GoogleScholar(
            serpapi_key=serpapi_key,
            openrouter_key=openrouter_key,
            semantic_scholar_api_key=semantic_scholar_api_key
        )
        self.dblp = DBLP(openrouter_key=openrouter_key)
        self.arxiv = Arxiv()
        self.semantic_scholar = SemanticScholar(semantic_scholar_api_key=semantic_scholar_api_key)
        self.craw4ai = Crawl4AICrawlTool()
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.pubmed_api_key = pubmed_api_key
        self.serpapi_key = serpapi_key
        self.openrouter_key = openrouter_key

        # Initialize cache if enabled
        self.enable_cache = enable_cache
        if enable_cache:
            self._init_cache(
                tool_name="fetch_paper_metadata",
                cache_dir=cache_dir,
                key_generator=PaperMetadataCacheKeyGenerator(),
            )

    def __call__(self, papers: list) -> ToolResult:

        metadata = ToolMetadata(
            tool_name=self.name,
            args={"papers": papers}
        )

        paper_metadata_list = []

        if not papers:
            return ToolResult(
                metadata=metadata,
                result={
                    "success": True,
                    "paper_metadata": paper_metadata_list,
                },
            )

        # Parallel execution of _fetch_paper_metadata
        with ContextualThreadPoolExecutor(max_workers=min(len(papers), 5)) as executor:
            futures = [
                executor.submit(self._fetch_paper_metadata, paper)
                for paper in papers
            ]

            failed = []
            for paper, future in zip(papers, futures):
                try:
                    paper_metadata, cost_breakdown = future.result()
                    paper_metadata_list.append(paper_metadata)
                    metadata.add_cost_breakdown(cost_breakdown)
                except Exception as e:
                    paper_title = paper.get("paper_title", "Unknown")
                    logger.warning(f"Error fetching metadata for {paper_title}: {e}")
                    failed.append(paper_title)

        if failed:
            if len(failed) == len(papers):
                return ToolResult(
                    metadata=metadata,
                    result={
                        "success": False,
                        "error": "Failed to fetch metadata for all papers"
                    },
                )

            return ToolResult(
                metadata=metadata,
                result={
                    "failed": failed,
                    "paper_metadata": paper_metadata_list,
                },
            )

        return ToolResult(
            metadata=metadata,
            result={
                "success": True,
                "paper_metadata": paper_metadata_list
            }
        )

    def _init_paper_metadata(self, paper: Dict) -> Dict[str, Dict]:
        """
        Initialize paper metadata with status tracking.

        Status definitions:
        - "missing": No information available yet
        - "candidate": Information obtained but not verified (may be incomplete or inaccurate)
        - "verified": Information confirmed to be accurate and complete
        """
        paper_title = paper.get("paper_title", None)
        paper_link = paper.get("paper_link", None)

        # Determine if input link is a PDF link
        is_pdf = is_pdf_link(paper_link) if paper_link else False

        return {
            "paper_title": {"content": paper_title, "source": "input", "status": "candidate"},
            "paper_link": {
                "content": paper_link if paper_link else None,
                "source": "input" if paper_link else None,
                "status": "missing" if paper_link else "candidate"
            },
            "paper_pdf_link": {
                "content": paper_link if is_pdf else None,
                "source": "input" if is_pdf else None,
                "status": "candidate" if is_pdf else "missing"
            },
            "authors": {"content": None, "source": None, "status": "missing"},
            "abstract": {"content": None, "source": None, "status": "missing"},
            "year": {"content": None, "source": None, "status": "missing"},
            "venue": {"content": None, "source": None, "status": "missing"},
            "citation_count": {"content": None, "source": None, "status": "missing"}
        }

    def _update_metadata_field(
        self,
        paper_metadata: Dict[str, Dict],
        field_name: str,
        content: Optional[Any] = None,
        source: Optional[str] = None,
        status: Optional[str] = None,
        only_if_not_verified: bool = False
    ) -> Dict[str, Dict]:
        """
        Update a metadata field in paper_metadata.
        
        Args:
            paper_metadata: The metadata dict to update
            field_name: Name of the field to update (e.g., "paper_title", "authors")
            content: New content value (None means don't update)
            source: New source value (None means don't update)
            status: New status value (None means don't update)
            only_if_not_verified: If True, only update if current status is not "verified"
        
        Returns:
            Updated paper_metadata
        """
        if field_name not in paper_metadata:
            return paper_metadata
        
        # Check if we should skip update
        if only_if_not_verified and paper_metadata[field_name]["status"] == "verified":
            return paper_metadata
        
        # Update fields if provided
        if content is not None:
            paper_metadata[field_name]["content"] = content
        if source is not None:
            paper_metadata[field_name]["source"] = source
        if status is not None:
            paper_metadata[field_name]["status"] = status
        
        return paper_metadata

    def _flatten_paper_metadata(self, paper_metadata: Dict[str, Dict]) -> Dict:
        """Convert status-tracked metadata to simple key-value format."""
        return {
            "paper_title": paper_metadata["paper_title"]["content"],
            "paper_link": paper_metadata["paper_link"]["content"],
            "paper_pdf_link": paper_metadata["paper_pdf_link"]["content"],
            "authors": paper_metadata["authors"]["content"],
            "abstract": paper_metadata["abstract"]["content"],
            "year": paper_metadata["year"]["content"],
            "venue": paper_metadata["venue"]["content"],
            "citation_count": paper_metadata["citation_count"]["content"]
        }

    def _fetch_paper_metadata_from_arxiv(
        self,
        paper_link: Optional[str],
        paper_metadata: Dict[str, Dict],
        paper_pdf_link: Optional[str] = None
    ) -> Tuple[Dict[str, Dict], bool]:
        """
        Fetch paper metadata from arXiv API if either paper_link or paper_pdf_link is an arXiv link.
        
        Args:
            paper_link: The paper link to check
            paper_metadata: The paper metadata dict to update
            paper_pdf_link: Optional PDF link to check if paper_link is not arXiv

        Returns:
            Tuple of (updated paper_metadata, success_flag)
        """
        # Check if paper_link is an arXiv link
        arxiv_link = None
        if paper_link and is_arxiv_link(paper_link):
            arxiv_link = paper_link
        # If paper_link is not arXiv, check paper_pdf_link
        elif paper_pdf_link and is_arxiv_link(paper_pdf_link):
            arxiv_link = paper_pdf_link
        
        if not arxiv_link:
            return paper_metadata, False

        arxiv_id = extract_arxiv_id(arxiv_link)
        if not arxiv_id:
            return paper_metadata, False

        arxiv_result = self.arxiv.get_metadata_based_on_arxiv_id(arxiv_id)
        arxiv_paper_metadata = arxiv_result.result

        if not arxiv_paper_metadata:
            return paper_metadata, False

        # Update metadata from arXiv
        if arxiv_paper_metadata.get("title"):
            paper_metadata = self._update_metadata_field(
                paper_metadata, "paper_title",
                content=arxiv_paper_metadata["title"],
                source="arxiv",
                status="verified"
            )

        if arxiv_paper_metadata.get("paper_link"):
            paper_metadata = self._update_metadata_field(
                paper_metadata, "paper_link",
                content=arxiv_paper_metadata["paper_link"],
                source="arxiv",
                status="verified"
            )

        if arxiv_paper_metadata.get("pdf_link"):
            paper_metadata = self._update_metadata_field(
                paper_metadata, "paper_pdf_link",
                content=arxiv_paper_metadata["pdf_link"],
                source="arxiv",
                status="verified"
            )

        if arxiv_paper_metadata.get("authors"):
            paper_metadata = self._update_metadata_field(
                paper_metadata, "authors",
                content=arxiv_paper_metadata["authors"],
                source="arxiv",
                status="verified"
            )

        if arxiv_paper_metadata.get("abstract"):
            paper_metadata = self._update_metadata_field(
                paper_metadata, "abstract",
                content=arxiv_paper_metadata["abstract"],
                source="arxiv",
                status="verified"
            )

        if arxiv_paper_metadata.get("year"):
            paper_metadata = self._update_metadata_field(
                paper_metadata, "year",
                content=arxiv_paper_metadata["year"],
                source="arxiv",
                status="verified"
            )

        # Set venue to arXiv
        paper_metadata = self._update_metadata_field(
            paper_metadata, "venue",
            content="arXiv",
            source="arxiv",
            status="verified"
        )

        return paper_metadata, True

    def _extract_authors_from_chicago_citation(self, chicago_citation: str) -> Tuple[List[str], bool]:
        """
        Extract author names from a Chicago-style citation using LLM.

        Args:
            chicago_citation: The Chicago citation string.

        Returns:
            Tuple of (authors list, is_complete flag)
        """
        if not chicago_citation or not self.google_scholar.llm:
            return [], False

        prompt = EXTRACT_AUTHORS_FROM_CHICAGO_CITATION_PROMPT.format(chicago_citation=chicago_citation)

        try:
            llm_response = self.google_scholar.llm.generate(prompt=prompt, max_tokens=1024)
            result = parse_json_from_llm_output(llm_response.content)

            if result:
                authors = result.get("authors", [])
                is_complete = result.get("is_complete", False)
                return authors, is_complete
        except Exception as e:
            logger.warning(f"Failed to extract authors from Chicago citation: {e}")

        return [], False

    def _search_paper_authors(
        self,
        paper_metadata: Dict[str, Dict],
    ) -> Tuple[Dict[str, Dict], Dict[str, float]]:
        """
        Search for author information using open-source databases.
        """
        paper_title = paper_metadata["paper_title"]["content"]
        if not paper_title:
            return paper_metadata, {}

        try:
            # Extract last names from current authors for validation if available
            current_authors = paper_metadata["authors"]["content"]
            gs_authors = current_authors or None
            
            # Search from multiple databases
            authors, cost_breakdown = search_authors_from_databases(
                paper_title=paper_title,
                gs_authors=gs_authors,
                pubmed_api_key=self.pubmed_api_key,
                semantic_scholar_api_key=self.semantic_scholar_api_key,
            )

            if authors:
                paper_metadata = self._update_metadata_field(
                    paper_metadata, "authors",
                    content=authors,
                    source="open_source_databases",
                    status="verified"
                )
                
        except Exception as e:
            logger.warning(f"Failed to search paper authors from databases: {e}")

        return paper_metadata, cost_breakdown

    def _infer_paper_venue(
        self,
        paper_metadata: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Infer the canonical venue name from noisy venue information using LLM.
        """
        if not self.google_scholar.llm:
            return paper_metadata

        paper_title = paper_metadata["paper_title"]["content"]
        year = paper_metadata["year"]["content"]
        venue_info = paper_metadata["venue"]["content"]
        paper_link = paper_metadata["paper_link"]["content"]

        # Skip if no venue info to infer from
        if not venue_info and not paper_link:
            return paper_metadata

        prompt = INFER_PAPER_VENUE_PROMPT.format(
            paper_title=paper_title or "Unknown",
            year=year or "Unknown",
            venue_info=venue_info or "Unknown",
            paper_link=paper_link or "Unknown"
        )

        try:
            llm_response = self.google_scholar.llm.generate(prompt=prompt, max_tokens=256)
            result = parse_json_from_llm_output(llm_response.content)

            if result and result.get("venue"):
                paper_metadata = self._update_metadata_field(
                    paper_metadata, "venue",
                    content=result["venue"],
                    source="infer_paper_venue",
                    status="verified"
                )
        except Exception as e:
            logger.warning(f"Failed to infer paper venue: {e}")

        return paper_metadata

    def _get_metadata_from_paper_page(
        self,
        paper_url: str,
        paper_metadata: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Fetch and extract metadata from a paper's webpage.
        """
        if not paper_url or not self.google_scholar.llm:
            return paper_metadata

        try:
            # Use crawl4ai to fetch page content
            crawl_result = self.craw4ai.call_sync(url=paper_url, fetch_raw_content=True)
            
            if not crawl_result.result.get("success"):
                logger.warning(f"Crawl4AI failed for {paper_url}: {crawl_result.result.get('error', 'Unknown error')}")
                return paper_metadata
            
            # Get markdown content from crawl4ai
            page_content = crawl_result.result.get("content", "")
            if not page_content:
                logger.warning(f"No content extracted from {paper_url}")
                return paper_metadata

            prompt = PAPER_METADATA_EXTRACTION_PROMPT.format(context=page_content)
            llm_response = self.google_scholar.llm.generate(prompt=prompt, max_tokens=2048)
            result = parse_json_from_llm_output(llm_response.content)

            if result:
                # Update title if not verified
                if result.get("title"):
                    paper_metadata = self._update_metadata_field(
                        paper_metadata, "paper_title",
                        content=result["title"],
                        source="paper_page",
                        status="verified",
                        only_if_not_verified=True
                    )

                # Update authors if not verified
                if result.get("authors"):
                    current_authors = paper_metadata["authors"]["content"]
                    if current_authors:
                        # Validate against existing authors
                        gs_authors = [author.split()[-1] if author else "" for author in current_authors]
                        if validate_author_info(result["authors"], gs_authors):
                            paper_metadata = self._update_metadata_field(
                                paper_metadata, "authors",
                                content=result["authors"],
                                source="paper_page",
                                status="verified",
                                only_if_not_verified=True
                            )
                    else:
                        paper_metadata = self._update_metadata_field(
                            paper_metadata, "authors",
                            content=result["authors"],
                            source="paper_page",
                            status="verified",
                            only_if_not_verified=True
                        )

                # Update abstract if not verified
                if result.get("abstract"):
                    paper_metadata = self._update_metadata_field(
                        paper_metadata, "abstract",
                        content=result["abstract"],
                        source="paper_page",
                        status="verified",
                        only_if_not_verified=True
                    )
                
                # Update PDF link if available
                if result.get("pdf_link") and is_pdf_link(result["pdf_link"]):
                    paper_metadata = self._update_metadata_field(
                        paper_metadata, "paper_pdf_link",
                        content=result["pdf_link"],
                        source="paper_page",
                        status="verified",
                        only_if_not_verified=True
                    )
                
                # Update venue if paper is published
                if result.get("is_published") and result.get("venue"):
                    paper_metadata = self._update_metadata_field(
                        paper_metadata, "venue",
                        content=result["venue"],
                        source="paper_page",
                        status="verified",
                        only_if_not_verified=True
                    )

        except Exception as e:
            logger.warning(f"Failed to get metadata from paper page ({paper_url}): {e}")

        return paper_metadata

    def _fetch_paper_metadata(self, paper: Dict) -> Tuple[Dict, Dict[str, float]]:
        """
        Fetch detailed metadata for a single paper.

        Args:
            paper: A dict containing 'paper_title' and optionally 'paper_link'

        Returns:
            A tuple of (paper_metadata_dict, cost_breakdown)
            paper_metadata_dict contains: paper_title, paper_link, paper_pdf_link,
            authors, abstract, year, venue, citation_count
        """
        cost_breakdown: Dict[str, float] = {}
        paper_title = paper.get("paper_title", "")

        # Check cache first
        if self.enable_cache and paper_title:
            cached_result, hit = self._cache_get(paper_title)
            if hit and cached_result is not None:
                logger.debug(f"Cache hit for paper metadata: {paper_title[:50]}...")
                return cached_result, cost_breakdown

        # Cache for link checks to avoid redundant requests/computations
        link_accessibility_cache: Dict[str, bool] = {}
        pdf_link_cache: Dict[str, bool] = {}

        def check_link_accessible(url: str) -> bool:
            """Check if a link is accessible, using cache to avoid redundant checks."""
            if not url:
                return False
            if url not in link_accessibility_cache:
                link_accessibility_cache[url] = is_link_accessible(url)
            return link_accessibility_cache[url]

        def check_is_pdf_link(url: str) -> bool:
            """Check if a URL is a PDF link, using cache to avoid redundant checks."""
            if not url:
                return False
            if url not in pdf_link_cache:
                pdf_link_cache[url] = is_pdf_link(url)
            return pdf_link_cache[url]

        # Initialize paper metadata with status tracking
        paper_metadata = self._init_paper_metadata(paper)
        input_link = paper.get("paper_link")

        # Step 1: Perform Google Scholar search
        gs_result = self.google_scholar.get_metadata_based_on_title(paper_metadata["paper_title"]["content"])
        gs_metadata = gs_result.result
        cost_breakdown = add_dict(cost_breakdown, gs_result.metadata.cost_breakdown)

        if gs_metadata:
            # Update paper_title
            if gs_metadata.get("title"):
                paper_metadata = self._update_metadata_field(
                    paper_metadata, "paper_title",
                    content=gs_metadata["title"],
                    source="google_scholar",
                    status="verified"
                )

            # Update year
            if gs_metadata.get("year"):
                paper_metadata = self._update_metadata_field(
                    paper_metadata, "year",
                    content=gs_metadata["year"],
                    source="google_scholar",
                    status="verified"
                )

            # Update citation_count
            if gs_metadata.get("citation_count") is not None:
                paper_metadata = self._update_metadata_field(
                    paper_metadata, "citation_count",
                    content=gs_metadata["citation_count"],
                    source="google_scholar",
                    status="verified"
                )

            # Update venue (candidate, not verified yet)
            if gs_metadata.get("venue_info"):
                paper_metadata = self._update_metadata_field(
                    paper_metadata, "venue",
                    content=gs_metadata["venue_info"],
                    source="google_scholar",
                    status="candidate"
                )
            
            # Update paper_link and paper_pdf_link if input link is missing or is a PDF
            if not input_link or check_is_pdf_link(input_link) or not check_link_accessible(input_link):
                if gs_metadata.get("publication_page_link"):
                    paper_metadata = self._update_metadata_field(
                        paper_metadata, "paper_link",
                        content=gs_metadata["publication_page_link"],
                        source="google_scholar",
                        status="candidate"
                    )

                if gs_metadata.get("publication_pdf_link"):
                    paper_metadata = self._update_metadata_field(
                        paper_metadata, "paper_pdf_link",
                        content=gs_metadata["publication_pdf_link"],
                        source="google_scholar",
                        status="candidate"
                    )

            # Try to fetch from arXiv if either paper_link or paper_pdf_link is an arXiv link
            paper_metadata, success = self._fetch_paper_metadata_from_arxiv(
                paper_link=paper_metadata["paper_link"]["content"],
                paper_metadata=paper_metadata,
                paper_pdf_link=paper_metadata["paper_pdf_link"]["content"]
            )
            if success:
                result = self._flatten_paper_metadata(paper_metadata)
                # Cache successful result (background write)
                if self.enable_cache and paper_metadata["paper_title"]["content"]:
                    self._cache_set(result, paper_metadata["paper_title"]["content"])
                return result, cost_breakdown

            # Try to extract authors from Google Scholar Chicago citation
            chicago_citation, chicago_cost = self.google_scholar.get_chicago_citation(gs_metadata)
            cost_breakdown = add_dict(cost_breakdown, chicago_cost)

            if chicago_citation:
                # Extract authors from Chicago citation using LLM (cost will be automatically tracked)
                authors_from_chicago, is_complete = self._extract_authors_from_chicago_citation(chicago_citation)
                if authors_from_chicago:
                    paper_metadata = self._update_metadata_field(
                        paper_metadata, "authors",
                        content=authors_from_chicago,
                        source="chicago_citation",
                        status="verified" if is_complete else "candidate"
                    )

            # If authors not set from Chicago citation, use Google Scholar authors (may be incomplete)
            if not paper_metadata["authors"]["content"]:
                gs_authors = gs_metadata.get("authors", [])
                if gs_authors:
                    author_names = [author.get("name", "") for author in gs_authors]
                    paper_metadata = self._update_metadata_field(
                        paper_metadata, "authors",
                        content=author_names,
                        source="google_scholar",
                        status="candidate"
                    )

        # Step 2: Obtain publicly accessible paper link if needed
        current_link = paper_metadata["paper_link"]["content"]
        if not current_link or check_is_pdf_link(current_link) or not check_link_accessible(current_link):
            # Use Google Search to find accessible link
            try:
                gsearch_result = self.google_scholar.serp.get_metadata_based_on_title(
                    title = paper_metadata["paper_title"]["content"],
                    openrouter_key = self.openrouter_key
                )
                gsearch_metadata = gsearch_result.result
                cost_breakdown = add_dict(cost_breakdown, gsearch_result.metadata.cost_breakdown)

                if gsearch_metadata:
                    new_link = gsearch_metadata.get("paper_link")

                    if new_link:
                        paper_metadata = self._update_metadata_field(
                            paper_metadata, "paper_link",
                            content=new_link,
                            source="google_search",
                            status="verified"
                        )
                    
                    # Update year if not verified
                    if gsearch_metadata.get("year"):
                        paper_metadata = self._update_metadata_field(
                            paper_metadata, "year",
                            content=gsearch_metadata["year"],
                            source="google_search",
                            status="verified",
                            only_if_not_verified=True
                        )

                    # Update venue if not set
                    if gsearch_metadata.get("venue_info") and not paper_metadata["venue"]["content"]:
                        paper_metadata = self._update_metadata_field(
                            paper_metadata, "venue",
                            content=gsearch_metadata["venue_info"],
                            source="google_search",
                            status="candidate"
                        )
            except Exception as e:
                logger.warning(f"Failed to get metadata from Google Search: {e}")

        current_link = paper_metadata["paper_link"]["content"]
        current_pdf_link = paper_metadata["paper_pdf_link"]["content"]
        if current_link or current_pdf_link:
            # Try to fetch from arXiv if either paper_link or paper_pdf_link is an arXiv link
            paper_metadata, success = self._fetch_paper_metadata_from_arxiv(
                paper_link=current_link,
                paper_metadata=paper_metadata,
                paper_pdf_link=current_pdf_link
            )
            if success:
                result = self._flatten_paper_metadata(paper_metadata)
                # Cache successful result (background write)
                if self.enable_cache and paper_metadata["paper_title"]["content"]:
                    self._cache_set(result, paper_metadata["paper_title"]["content"])
                return result, cost_breakdown
            # If the status of any of paper_title, authors, abstract is not verified, try to obtain metadata from paper page
            if any(paper_metadata[k]["status"] != "verified" for k in ["paper_title", "authors", "abstract"]):
                current_link = paper_metadata["paper_link"]["content"]
                if current_link and not check_is_pdf_link(current_link):
                    # cost will be automatically tracked
                    paper_metadata = self._get_metadata_from_paper_page(current_link, paper_metadata)

        # Search for complete author info if not verified
        if paper_metadata["authors"]["status"] != "verified":
            paper_metadata, search_authors_cost_breakdown = self._search_paper_authors(paper_metadata)
            cost_breakdown = add_dict(cost_breakdown, search_authors_cost_breakdown)

        # Infer venue if not verified
        if paper_metadata["venue"]["status"] != "verified":
            paper_metadata = self._infer_paper_venue(paper_metadata)

        result = self._flatten_paper_metadata(paper_metadata)
        # Cache successful result (background write)
        if self.enable_cache and paper_metadata["paper_title"]["content"]:
            self._cache_set(result, paper_metadata["paper_title"]["content"])
        return result, cost_breakdown
    
