from typing import List, Dict

from ..request_base import RequestBase
from ...core.logging import logger
from ...core.module_utils import parse_json_from_llm_output 
# from ..tool_cost_registry import get_serpapi_cost 
from ...models.model_utils import track_cost 
from ...models import OpenRouterConfig, OpenRouterLLM
from .utils import (
    normalize_title,
    extract_year,
    DEFAULT_PAGE_HEADERS,
    PAPER_SEARCH_LLM_MODEL
)
from .prompts import EXTRACT_PAPER_METADATA_FROM_GOOGLE_SEARCH_RESULTS_PROMPT
from .text_match import text_match 
from .metadata import ResearchToolMetadata, ResearchToolResult


class SerpAPIForResearch(RequestBase):

    def __init__(self, serpapi_key: str, **kwargs):
        assert serpapi_key, "serpapi_key must be provided when initializing SerpAPIForResearch"
        super().__init__(timeout=10, max_retries=3)
        self.serpapi_key = serpapi_key 
        self.serpapi_google_scholar_url = "https://serpapi.com/search?engine=google_scholar"
        self.serpapi_google_search_url = "https://serpapi.com/search?engine=google" 
    
    def search_publications(self, title_or_keyword: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> ResearchToolResult:
        """
        Return the raw results from the SerpAPI Google Scholar search, i.e., a list of dictionaries containing the search results:
        [
            {
                "serpapi_scholar_key": "the key of the paper",
                "title": "paper title", 
                "authors": [
                    {
                        "name": "the name of the author", # abbreviated name, such as "M. Chen", "T. Li", "H. Zhou", etc.
                        "google_scholar_link": "the link of the author on google scholar, such as https://scholar.google.com/citations?user=...", 
                        "serpapi_scholar_author_id": "the author id in SerpAPI Google Scholar", 
                        "serpapi_scholar_link": "the link of the author on SerpAPI Google Scholar, such as https://serpapi.com/search?engine=google_scholar&q=author:...", 
                    }
                ], # NOTE: the authors might be incomplete since SerpAPI only parse the returned by the search engine, whcih typically includes the first 6-7 authors.  
                "year": "the year of the paper, None if not available", 
                "snippet": "the snippet of the paper", 
                "venue_info": "the venue information of the paper, None if not available", 
                "publication_page_link": "the link of the paper page, None if not available", 
                "publication_pdf_link": "the link of the paper pdf, None if not available", 
                "citation_count": "the citation count of the paper, None if not available", 
            },
            ...
        ]
        """

        assert sort_by in ["relevance", "date"], f"Invalid sort_by value: {sort_by}. Valid values: [\"relevance\", \"date\"]" 

        params = {
            "api_key": self.serpapi_key,
            "q": normalize_title(title_or_keyword), 
        }
        if year_from:
            params["as_ylo"] = year_from 
        if year_to:
            params["as_yhi"] = year_to 
        if sort_by == "date":
            params["scisbd"] = 1 

        metadata = ResearchToolMetadata(tool_name="serpapi_search_publications")
        
        max_page_size = 20  # SerpAPI caps Google Scholar responses at 20 results per call
        collected_results = []
        start = 0
        num_serapi_calls = 0 

        while len(collected_results) < topk:
            page_params = params.copy()
            page_params["num"] = min(max_page_size, topk - len(collected_results))
            if start:
                page_params["start"] = start

            try:
                response = self.request(url=self.serpapi_google_scholar_url, params=page_params, headers=DEFAULT_PAGE_HEADERS)
            except Exception as e:
                logger.warning(f"Failed to search publications via SerpAPI Google Scholar: {e}, page_params: {page_params}")
                # If the API call fails, return the collected results so far and update the cost of the tool call. 
                # if num_serapi_calls > 0:
                #     metadata.add_cost_breakdown({"serpapi:google_scholar": get_serpapi_cost(num_serapi_calls)})
                return ResearchToolResult(metadata=metadata, result=collected_results)

            data = response.json() 
            num_serapi_calls += 1 

            if "organic_results" not in data:
                break

            page_results = self._parse_google_scholar_results(data)
            if not page_results:
                break

            collected_results.extend(page_results)

            # Stop early if the API returned fewer results than requested (no more pages)
            if len(page_results) < page_params["num"]:
                break

            start += page_params["num"]
        
        # update the cost of the tool call 
        # if num_serapi_calls > 0:
        #     metadata.add_cost_breakdown({"serpapi:google_scholar": get_serpapi_cost(num_serapi_calls)})
        
        return ResearchToolResult(metadata=metadata, result=collected_results[:topk])

    def search_cited_by(self, cited_by_serpapi_link: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> ResearchToolResult:
        """
        Search for papers that cite a given paper using the cited_by_serpapi_link.

        Args:
            cited_by_serpapi_link: The SerpAPI scholar link for cited_by papers (e.g., "https://serpapi.com/search.json?as_sdt=5%2C41&cites=3387547533016043281&engine=google_scholar&hl=en")
            topk: The number of top results to return.
            sort_by: The field to sort the results by. Valid values: ["relevance", "date"].
            year_from: The minimum year of the publication.
            year_to: The maximum year of the publication.

        Returns:
            ResearchToolResult containing a list of papers that cite the given paper.
        """
        assert sort_by in ["relevance", "date"], f"Invalid sort_by value: {sort_by}. Valid values: [\"relevance\", \"date\"]"

        if not cited_by_serpapi_link:
            raise ValueError("cited_by_serpapi_link must be provided")

        # Parse the cited_by_serpapi_link to extract query parameters
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(cited_by_serpapi_link)
        query_params = parse_qs(parsed_url.query)

        # Build the params dictionary, starting with parameters from the link
        params = {
            "api_key": self.serpapi_key,
        }

        # Extract cites parameter (most important)
        if "cites" in query_params:
            params["cites"] = query_params["cites"][0]
        else:
            raise ValueError(f"cited_by_serpapi_link does not contain 'cites' parameter: {cited_by_serpapi_link}")

        # Extract other relevant parameters if present
        if "as_sdt" in query_params:
            params["as_sdt"] = query_params["as_sdt"][0]
        if "hl" in query_params:
            params["hl"] = query_params["hl"][0]

        # Apply year filters
        if year_from:
            params["as_ylo"] = year_from
        if year_to:
            params["as_yhi"] = year_to

        # Apply sort order
        if sort_by == "date":
            params["scisbd"] = 1

        metadata = ResearchToolMetadata(tool_name="serpapi_search_cited_by")

        max_page_size = 20  # SerpAPI caps Google Scholar responses at 20 results per call
        collected_results = []
        start = 0
        num_serapi_calls = 0

        while len(collected_results) < topk:
            page_params = params.copy()
            page_params["num"] = min(max_page_size, topk - len(collected_results))
            if start:
                page_params["start"] = start

            try:
                response = self.request(url=self.serpapi_google_scholar_url, params=page_params, headers=DEFAULT_PAGE_HEADERS)
            except Exception as e:
                logger.warning(f"Failed to search cited_by via SerpAPI Google Scholar: {e}, page_params: {page_params}")
                # If the API call fails, return the collected results so far and update the cost of the tool call.
                # if num_serapi_calls > 0:
                #     metadata.add_cost_breakdown({"serpapi:google_scholar": get_serpapi_cost(num_serapi_calls)})
                return ResearchToolResult(metadata=metadata, result=collected_results)

            data = response.json()
            num_serapi_calls += 1

            if "organic_results" not in data:
                break

            page_results = self._parse_google_scholar_results(data)
            if not page_results:
                break

            collected_results.extend(page_results)

            # Stop early if the API returned fewer results than requested (no more pages)
            if len(page_results) < page_params["num"]:
                break

            start += page_params["num"]

        # update the cost of the tool call
        # if num_serapi_calls > 0:
        #     metadata.add_cost_breakdown({"serpapi:google_scholar": get_serpapi_cost(num_serapi_calls)})

        return ResearchToolResult(metadata=metadata, result=collected_results[:topk])

    @staticmethod
    def _parse_google_scholar_results(data: Dict) -> List[Dict]:

        results = []
        for item in data.get("organic_results", []):

            serpapi_scholar_key = item.get("result_id", "")
            title = item.get("title", "Unknown Title")

            publication_info = item.get("publication_info", {})
            authors = [] 
            for author_item in publication_info.get("authors", []):
                authors.append(
                    {
                        "name": author_item.get("name", "Unknown Author"),
                        "google_scholar_link": author_item.get("link", None),
                        "serpapi_scholar_author_id": author_item.get("author_id", None), 
                        "serpapi_scholar_link": author_item.get("serpapi_scholar_link", None), 
                    }
                )
            
            # obtain year from the summary like "P Lewis, E Perez, A Piktus, F Petroni\u2026 - Advances in neural \u2026, 2020 - proceedings.neurips.cc" 
            summary = publication_info.get("summary", "")
            year = extract_year(summary) if summary else None 

            # obtain ciation count
            citation_count = item.get("inline_links", {}).get("cited_by", {}).get("total", None)

            # obtain cited_by serpapi scholar link
            cited_by_serpapi_link = item.get("inline_links", {}).get("cited_by", {}).get("serpapi_scholar_link", None)

            # obtain pdf link from the resources if available
            pdf_link = None
            resources = item.get("resources", [])
            for resource in resources:
                if resource.get("file_format", "").lower() == "pdf":
                    pdf_link = resource.get("link", None)
                    break

            results.append(
                {
                    "serpapi_scholar_key": serpapi_scholar_key,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "snippet": item.get("snippet", None),
                    "venue_info": summary,
                    "publication_page_link": item.get("link", None),
                    "publication_pdf_link": pdf_link,
                    "citation_count": citation_count,
                    "cited_by_serpapi_link": cited_by_serpapi_link,
                }
            )
        
        return results

    def google_search(self, query: str, topk: int = 5) -> ResearchToolResult:
        """
        Search for information from Google using SerpAPI.
        Args:
            query: The query to search for.
            topk: The number of top results to return.
        Returns:
            A list of dictionaries containing the search results.
        """
        params = {
            "api_key": self.serpapi_key,
            "q": query,
        }

        metadata = ResearchToolMetadata(tool_name="serpapi_google_search")
    
        max_page_size = 10  # SerpAPI Google Search typically returns up to 10 results per page
        collected_results = []
        start = 0
        num_serapi_calls = 0

        while len(collected_results) < topk:
            page_params = params.copy()
            if start > 0:
                page_params["start"] = start

            try:
                response = self.request(url=self.serpapi_google_search_url, params=page_params, headers=DEFAULT_PAGE_HEADERS)
            except Exception as e:
                logger.warning(f"Failed to search via SerpAPI Google Search: {e}, page_params: {page_params}")
                # If the API call fails, return the collected results so far and update the cost of the tool call.
                # if num_serapi_calls > 0:
                #     metadata.add_cost_breakdown({"serpapi:google_search": get_serpapi_cost(num_serapi_calls)})
                return ResearchToolResult(metadata=metadata, result=collected_results)

            data = response.json()
            num_serapi_calls += 1

            if "organic_results" not in data:
                break

            page_results = self._parse_google_search_results(data)
            if not page_results:
                break

            collected_results.extend(page_results)

            # Stop early if the API returned fewer results than max_page_size (no more pages)
            if len(page_results) < max_page_size:
                break

            start += max_page_size
        
        # update the cost of the tool call
        # if num_serapi_calls > 0:
        #     metadata.add_cost_breakdown({"serpapi:google_search": get_serpapi_cost(num_serapi_calls)})
        
        return ResearchToolResult(metadata=metadata, result=collected_results[:topk])
    
    @staticmethod
    def _parse_google_search_results(data: Dict) -> List[Dict]:
        results = []
        for item in data.get("organic_results", []):
            # position starts from 1  
            results.append(
                {
                    "position": item.get("position", None),
                    "title": item.get("title", "Unknown Title"),
                    "link": item.get("link", None),
                    "author": item.get("author", None), 
                    "date": item.get("date", None),
                    "snippet": item.get("snippet", None),
                    "source": item.get("source", None),
                }
            )
        
        return results

    def get_metadata_based_on_title(self, title: str, openrouter_key: str) -> ResearchToolResult:

        metadata = ResearchToolMetadata(tool_name="serpapi_get_metadata_based_on_title")
        paper_metadata = {} 

        title = title.strip() 
        if not title:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        google_search_results: ResearchToolResult = self.google_search(query=title, topk=5)
        google_search_papers_info = google_search_results.result 
        metadata.add_cost_breakdown(google_search_results.metadata.cost_breakdown)

        if not google_search_papers_info:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        # use LLM to extract the paper metadata from the google search results
        llm = OpenRouterLLM(OpenRouterConfig(openrouter_key=openrouter_key, model=PAPER_SEARCH_LLM_MODEL, stream=False, output_response=False, temperature=0.0))
        prompt = EXTRACT_PAPER_METADATA_FROM_GOOGLE_SEARCH_RESULTS_PROMPT.format(title=title, papers_info=google_search_papers_info)
        # parse the LLM response
        try:
            with track_cost() as cost_tracker:
                llm_response = llm.generate(prompt=prompt, max_tokens=512)
                metadata.add_cost_breakdown({"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()})
            
            if "error" in llm_response.content.lower():
                logger.info(f"Failed to extract the paper metadata from the Google Search results in SerpAPIForResearch.get_metadata_based_on_title: {llm_response.content}") 
                return ResearchToolResult(metadata=metadata, result={})
            paper_metadata = parse_json_from_llm_output(llm_response.content)
            # add paper_pdf_link & citation_count 
            paper_metadata["paper_pdf_link"] = None 
            paper_metadata["citation_count"] = None
        except Exception as e:
            logger.info(f"Failed to parse the LLM response in SerpAPIForResearch.get_metadata_based_on_title: {e}") 
            return ResearchToolResult(metadata=metadata, result={})
        
        # validate the extracted metadata
        if not paper_metadata.get("paper_title", None) or not paper_metadata.get("paper_link", None):
            logger.info(f"Failed to extract the paper metadata from the Google Search results for title: {title}, Google Search Paper Metadata: {paper_metadata}") 
            return ResearchToolResult(metadata=metadata, result={})
        
        # match the paper title with the google search results
        match_result = text_match(title, [paper_metadata["paper_title"]])
        if match_result["match"]:
            logger.info(f"Google Search Paper Title Matched: query='{title}', fetched_title='{paper_metadata['paper_title']}'") 
            # paper_metadata["paper_title"] = title  # use the original title as the paper title since the google search results may be truncated
        else:
            logger.info(f"Failed to match the paper title with the google search results for title: {title}, Google Search Paper Metadata: {paper_metadata}")
            return ResearchToolResult(metadata=metadata, result={})
        
        return ResearchToolResult(metadata=metadata, result=paper_metadata)
