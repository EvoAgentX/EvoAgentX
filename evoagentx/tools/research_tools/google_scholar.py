import os
import requests
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..request_base import RequestBase
from ...core.logging import logger
# from ..tool_cost_registry import get_serpapi_cost 
from ...models.model_utils import track_cost 
from ...models import OpenRouterConfig, OpenRouterLLM
from ...utils.utils import add_dict

from .utils import (
    extract_doi,
    is_identifier,
    extract_arxiv_id,
    get_metadata_from_paper_page,
    PAPER_SEARCH_LLM_MODEL,
    DEFAULT_PAGE_HEADERS
)

from .prompts import (
    BIB_NOT_FOUND_TEXT,
    PAPER_BIBTEXT_FORMULATION_PROMPT,
    ARXIV_PAPER_BIBTEXT_FORMULATION_PROMPT
)
from .metadata import ResearchToolMetadata, ResearchToolResult
from .serp_utils import SerpAPIForResearch
from .sources import Arxiv, search_paper_author_info, get_metadata_based_on_doi
from .text_match import text_match


class GoogleScholar(RequestBase):

    def __init__(
        self, 
        serpapi_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        pubmed_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(timeout=10, max_retries=3)
        self.serpapi_key = serpapi_key if serpapi_key else os.getenv("SERPAPI_KEY")
        self.serp = SerpAPIForResearch(serpapi_key=self.serpapi_key)
        
        # Initialize Arxiv (Arxiv class is defined later in this file, but available at runtime)
        self.arxiv = Arxiv()
        
        # Initialize LLM
        openrouter_key = openrouter_key if openrouter_key else os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            self.llm = OpenRouterLLM(
                config=OpenRouterConfig(
                    openrouter_key=openrouter_key,
                    model=PAPER_SEARCH_LLM_MODEL,
                    stream=False,
                    output_response=False,
                    temperature=0.0,
                )
            )
        else:
            self.llm = None

        self.pubmed_api_key = pubmed_api_key 
        self.semantic_scholar_api_key = semantic_scholar_api_key 
        
    def search_publications(
        self, 
        title_or_keyword: str, 
        topk: int = 5,
        sort_by: str = "relevance",
        year_from: int = None,
        year_to: int = None,
    ) -> ResearchToolResult:

        """
        Search for publications on Google Scholar using a single title or keyword.

        Args:
            title_or_keyword: A single title or keyword to search for.
            topk: The number of top results to return.
            sort_by: The field to sort the results by. Valid values: ["relevance", "date"].
            year_from: The minimum year of the publication.
            year_to: The maximum year of the publication.
        
        """
        assert sort_by in ["relevance", "date"], "Invalid sort_by value. Valid values: [\"relevance\", \"date\"]"

        # obtain raw results from SerpAPI Google Scholar 
        results: ResearchToolResult = self.serp.search_publications(title_or_keyword, topk, sort_by, year_from, year_to)

        return results

    def get_metadata_based_on_title(self, title: str) -> ResearchToolResult:

        metadata = ResearchToolMetadata(tool_name="google_scholar_get_metadata_based_on_title")
        paper_metadata = {}

        title = title.strip()
        if not title:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        results: ResearchToolResult = self.search_publications(title)
        papers_info = results.result
        metadata.add_cost_breakdown(results.metadata.cost_breakdown)
        if not papers_info:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        # Check if the input is likely an identifier
        if is_identifier(title):
            # For identifiers, return the first result directly
            logger.info(f"Input appears to be an identifier: '{title}'. Returning first result.")
            paper_metadata = papers_info[0].copy()
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        # For regular titles, use text_match to find the best match
        for paper_info in papers_info:
            match_result = text_match(title, [paper_info["title"]])
            if match_result.get("match", False):
                logger.info(f"Google Scholar title matched: query='{title}', fetched_title='{paper_info['title']}'")
                paper_metadata = paper_info.copy()
                return ResearchToolResult(metadata=metadata, result=paper_metadata)

        return ResearchToolResult(metadata=metadata, result=paper_metadata)

    def get_followup_papers(self, title: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> ResearchToolResult:
        """
        Get followup papers (papers that cite the given paper) based on the paper title.

        Args:
            title: The title of the paper to find followup papers for.
            topk: The number of top results to return.
            sort_by: The field to sort the results by. Valid values: ["relevance", "date"].
            year_from: The minimum year of the publication.
            year_to: The maximum year of the publication.

        Returns:
            ResearchToolResult containing a list of papers that cite the given paper.
        """
        metadata = ResearchToolMetadata(tool_name="google_scholar_get_followup_papers")

        title = title.strip()
        if not title:
            return ResearchToolResult(metadata=metadata, result=[])

        # First, search for the paper to get its metadata including cited_by link
        paper_result: ResearchToolResult = self.get_metadata_based_on_title(title)
        paper_metadata = paper_result.result
        metadata.add_cost_breakdown(paper_result.metadata.cost_breakdown)

        if not paper_metadata:
            logger.warning(f"Paper not found on Google Scholar: {title}")
            return ResearchToolResult(metadata=metadata, result=[])

        # Get the cited_by_serpapi_link
        cited_by_serpapi_link = paper_metadata.get("cited_by_serpapi_link", None)

        if not cited_by_serpapi_link:
            logger.warning(f"No cited_by_serpapi_link found for paper: {title}")
            return ResearchToolResult(metadata=metadata, result=[])

        # Search for cited_by papers using the link
        cited_by_result: ResearchToolResult = self.serp.search_cited_by(
            cited_by_serpapi_link=cited_by_serpapi_link,
            topk=topk,
            sort_by=sort_by,
            year_from=year_from,
            year_to=year_to
        )
        metadata.add_cost_breakdown(cited_by_result.metadata.cost_breakdown)

        return ResearchToolResult(metadata=metadata, result=cited_by_result.result)

    def get_chicago_citation(self, pub: Dict) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Get the Chicago citation for a given publication.

        Args:
            pub: The publication dictionary containing 'serpapi_scholar_key'.

        Returns:
            A tuple containing (chicago_citation_string, cost_breakdown).
            chicago_citation_string is None if not found.
        """
        url = "https://serpapi.com/search?engine=google_scholar_cite"
        cost_breakdown: Dict[str, float] = {}

        key = pub.get("serpapi_scholar_key")
        if not key:
            return None, cost_breakdown

        params = {"q": key, "api_key": self.serpapi_key}
        try:
            response = self.request(url=url, params=params, headers=DEFAULT_PAGE_HEADERS)
            data = response.json()
            # cost_breakdown = add_dict(cost_breakdown, {"serpapi:google_scholar": get_serpapi_cost(1)})
        except Exception as e:
            logger.warning(f"Failed to get Chicago citation from Google Scholar: {e}, url: {url}, params: {params}")
            return None, cost_breakdown

        if "citations" not in data:
            return None, cost_breakdown 

        chicago_citation = None 
        for citation in data["citations"]:
            if citation.get("title", "").lower() == "chicago":
                chicago_citation = citation.get("snippet", None)
                break 

        return chicago_citation, cost_breakdown

    def get_bibtext(self, pub: Dict) -> Tuple[str, Dict[str, float]]:
        """
        Get the BibTeX text for a given key.
        Args:
            pub: The publication dictionary.
        Returns:
            A tuple containing the BibTeX text and the cost.
        """
        cost_breakdown: Dict[str, float] = {}

        # Get Chicago citation first
        chicago_citation, chicago_cost = self.get_chicago_citation(pub)
        cost_breakdown = add_dict(cost_breakdown, chicago_cost)

        if not chicago_citation:
            return None, cost_breakdown 
        
        try:
            if "et al." in chicago_citation.lower():
                # In this case, the author information is incomplete, try to find author information from reliable sources.
                gs_authors = [aut["name"] for aut in pub.get("authors", [])]
                authors, cost = search_paper_author_info(
                    gs_title=pub["title"], 
                    gs_link=pub["publication_page_link"], 
                    serpapi_key=self.serpapi_key,
                    gs_authors=gs_authors,
                    pubmed_api_key=self.pubmed_api_key,
                    semantic_scholar_api_key=self.semantic_scholar_api_key, 
                )
                cost_breakdown = add_dict(cost_breakdown, cost)
                if not authors: 
                    # If the author information is not found, use the author information from Google Scholar as a fallback.
                    authors = gs_authors 
            else:
                # In this case, the author information is complete. 
                authors = [] 
            # Use LLM to formulate the BibTeX text 
            prompt = PAPER_BIBTEXT_FORMULATION_PROMPT.format(context=chicago_citation, authors=authors) 
            with track_cost() as cost_tracker:
                llm_response = self.llm.generate(prompt=prompt, max_tokens=2048) 
                cost_breakdown = add_dict(cost_breakdown, {"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()}) 
            bibtext = llm_response.content.replace("```bibtext", "").replace("```bibtex", "").replace("```", "").strip()
            if "error" in bibtext.lower():
                return None, cost_breakdown 
        except Exception as e:
            logger.warning(f"Failed to complement paper info: {e}")
            return None, cost_breakdown 
        
        return bibtext, cost_breakdown 
    
    def get_bibtext_from_arxiv(self, title_or_keyword: str) -> Tuple[str, str, Dict[str, float]]:

        cost_breakdown: Dict[str, float] = {}
        if not title_or_keyword:
            return None, None, cost_breakdown  
        
        arxiv_search_results = self.arxiv.search_arxiv(search_query=title_or_keyword, max_results=20)
        if arxiv_search_results.get("success", False):
            arxiv_papers = arxiv_search_results.get("papers", [])
            # sorted_arxiv_papers = sorted(arxiv_papers, key=lambda x: f1_score(x["title"], title_or_keyword), reverse=True)
            match_results = text_match(title_or_keyword, [paper["title"] for paper in arxiv_papers]) 
            # if sorted_arxiv_papers:
            #     top_arxiv_paper = sorted_arxiv_papers[0]
            if match_results.get("match", False): 
                top_arxiv_paper = arxiv_papers[match_results["best_candidate_metrics"]["index"]]
                if 'summary' in top_arxiv_paper:
                    del top_arxiv_paper['summary'] 
                prompt = ARXIV_PAPER_BIBTEXT_FORMULATION_PROMPT + "\n" + str(top_arxiv_paper)
                with track_cost() as cost_tracker:
                    llm_response = self.llm.generate(prompt=prompt, max_tokens=2048)
                    cost_breakdown = add_dict(cost_breakdown, {"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()})
                bibtext = llm_response.content.replace("```bibtext", "").replace("```bibtex", "").replace("```", "").strip()
                if "error" in bibtext.lower():
                    return None, None, cost_breakdown
                return bibtext, top_arxiv_paper["title"], cost_breakdown
        
        return None, None, cost_breakdown 

    def search_bibtext(self, title_or_keyword: str) -> ResearchToolResult:
        """
        Search for BibTeX text for a given title or keyword.
        Args:
            title_or_keyword: A single title or keyword to search for.
        Returns:
            A string containing the BibTeX text for the search results.
        """
        metadata = ResearchToolMetadata(tool_name="google_scholar_search_bibtext")
        
        # search for publications from Google Scholar
        publications_results: ResearchToolResult = self.search_publications(title_or_keyword, topk=20)
        metadata.add_cost_breakdown(publications_results.metadata.cost_breakdown)
        publications = publications_results.result 

        if not publications:
            return ResearchToolResult(metadata=metadata, result={"bibtex": BIB_NOT_FOUND_TEXT}) 
        
        # use text_match to find the matching publication
        paper_title = None
        match_results = text_match(title_or_keyword, [pub["title"] for pub in publications])
        if match_results.get("match", False):
            # search Google Scholar to get the bibtext for the matched publication
            matched_publication = publications[match_results["best_candidate_metrics"]["index"]]
            bibtext, get_bibtext_cost_breakdown = self.get_bibtext(matched_publication)
            metadata.add_cost_breakdown(get_bibtext_cost_breakdown)
            paper_title = matched_publication["title"]
        else:
            # if no match found, set bibtext to None to trigger arXiv fallback
            bibtext = None

        # if search from Google Scholar fails, determine if this is an arXiv paper 
        if not bibtext:
            # try to formulate the bibtex from arXiv since some latest papers cannot be found in Google Scholar 
            bibtext, paper_title, get_bibtext_from_arxiv_cost_breakdown = self.get_bibtext_from_arxiv(title_or_keyword)
            metadata.add_cost_breakdown(get_bibtext_from_arxiv_cost_breakdown)

        if not bibtext:
            return ResearchToolResult(metadata=metadata, result={"bibtex": BIB_NOT_FOUND_TEXT}) 
        
        result = {"bibtex": bibtext, "title": paper_title}
        return ResearchToolResult(metadata=metadata, result=result)   

    def _merge_author_info(self, author_full_names: List[str], google_scholar_author_info: List[Dict] = None) -> List[str]: 
        """
        Merge full authors with google scholar author info.
        Returns only author names as a list of strings.
        """
        return author_full_names

    def _complement_paper_info(self, pub: Dict, include_abstract: bool) -> Tuple[List[str], str, Dict[str, float]]:
        """
        Complement paper information by extracting metadata from various sources.
        Returns: (authors, abstract, cost)
        """
        def _get_metadata_from_google_scholar() -> Tuple[List[str], str]:
            authors = [] 
            for aut in pub.get("authors", []):
                authors.append(aut.get("name", "Unknown Author"))
            abstract = pub.get("snippet", None) if include_abstract else None 
            return authors, abstract

        paper_page = pub.get("publication_page_link", None)
        google_scholar_author_info = pub.get("authors", []) 
        cost_breakdown: Dict[str, float] = {}

        if not paper_page:
            # Note: In this case, the author and abstract information are incomplete.
            authors, abstract = _get_metadata_from_google_scholar()
            return authors, abstract, cost_breakdown 
        
        # Case 1: For arXiv papers, try to extract the information from the arXiv API
        if "arxiv" in paper_page.lower() and self.arxiv:
            # extract information from the arXiv API for arXiv papers 
            arxiv_id = extract_arxiv_id(paper_page)
            if arxiv_id:
                arxiv_results: ResearchToolResult = self.arxiv.get_metadata_based_on_arxiv_id(arxiv_id)
                arxiv_paper_metadata = arxiv_results.result
                if arxiv_paper_metadata:
                    author_full_names = arxiv_paper_metadata["authors"]
                    abstract = arxiv_paper_metadata["abstract"] if include_abstract else None 
                    authors = self._merge_author_info(author_full_names, google_scholar_author_info)
                    return authors, abstract, cost_breakdown 

        # Case 2: Try to extract the DOI from the page and get the metadata from online resources. 
        doi = extract_doi(paper_page)
        if doi:
            paper_metadata_results: ResearchToolResult = get_metadata_based_on_doi(
                doi, pub.get("title", None), semantic_scholar_api_key=self.semantic_scholar_api_key
            )
            paper_metadata = paper_metadata_results.result 
            cost_breakdown = add_dict(cost_breakdown, paper_metadata_results.metadata.cost_breakdown)
            if paper_metadata and isinstance(paper_metadata, dict) and paper_metadata.get("authors", None):
                author_full_names = paper_metadata["authors"]
                authors = self._merge_author_info(author_full_names, google_scholar_author_info)
                if not include_abstract:
                    return authors, None, cost_breakdown 
                else:
                    abstract = paper_metadata.get("abstract", None)
                    if abstract:
                        # If both title and abstract are available, return the results. 
                        return authors, abstract, cost_breakdown
                    else:
                        # If abstract is not available, try if the page is reachable. 
                        try:
                            # If the page is reachable, use get_metadata_from_paper_page to get the abstract.
                            response = requests.head(paper_page, headers=DEFAULT_PAGE_HEADERS, timeout=10, allow_redirects=True)
                            response.raise_for_status()
                        except Exception:
                            # If the page is not reachable, return the results. 
                            abstract = pub.get("snippet", None)
                            return authors, abstract, cost_breakdown
        
        # Case 3: Visit the page to extract the author and abstract information 
        if self.llm:
            paper_metadata_results: ResearchToolResult = get_metadata_from_paper_page(paper_page, self.llm, include_abstract)
            paper_metadata = paper_metadata_results.result 
            cost_breakdown = add_dict(cost_breakdown, paper_metadata_results.metadata.cost_breakdown)
            if paper_metadata and isinstance(paper_metadata, dict) and paper_metadata.get("authors", None):
                author_full_names = paper_metadata["authors"]
                authors = self._merge_author_info(author_full_names, google_scholar_author_info)
                abstract = paper_metadata.get("abstract", None) if include_abstract else None 
                return authors, abstract, cost_breakdown
        
        # If all attempts fail, use the author information from Google Scholar and the snippet from the page. 
        authors, abstract = _get_metadata_from_google_scholar()
        return authors, abstract, cost_breakdown

    def _complement_paper_info_parallel(self, pubs: List[Dict], include_abstract: bool) -> List[Tuple[List[str], str, Dict[str, float]]]:
        """
        Parallel execution of _complement_paper_info
        Use ThreadPoolExecutor to parallel execute, it can work in both synchronous and asynchronous environments.
        """
        if not pubs:
            return []
        
        with ThreadPoolExecutor(max_workers=min(len(pubs), 5)) as executor:
            futures = [
                executor.submit(self._complement_paper_info, pub, include_abstract)
                for pub in pubs
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    # If a single paper info complement fails, fallback to Google Scholar data
                    # Extract pub from the future if possible, otherwise use empty data
                    logger.warning(f"Failed to complement paper info: {e}")
                    # Return fallback: empty authors list, None abstract, empty cost breakdown
                    results.append(([], None, {}))
            return results

