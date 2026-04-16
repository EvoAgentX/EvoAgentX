import os 
import math 
import random
from datetime import datetime 
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor 

from ..tool import Tool, ToolMetadata, ToolResult
from ...core.logging import logger
from ..storage_handler import FileStorageHandler
from ...utils.utils import add_dict
from .utils import (
    extract_year,
    query_expansion_for_recall,
    find_seminal_papers_for_topic,
    normalize_title_aggressively,
    extract_related_work_references,
    extract_year_from_arxiv_dates
)
from .metadata import ResearchToolResult
from .google_scholar import GoogleScholar
from .sources import Arxiv, DBLP, resolve_paper_title
from .text_match import text_match, tokenize, jaccard_similarity, BM25


PAPER_SEARCH_TOOL_EXTRA_DESCRIPTION = """
Example 1: General search for papers on a given topic
Arguments:
{
    "queries": ["retrieval-augmented generation"],
    "search_mode": "general",
    "topk": 5
}

Returns a dict with 'success' and 'query_papers', where each query maps to a list of papers:
{
    "success": true,
    "query_papers": {
        "retrieval-augmented generation": [
            {
                "paper_title": "the title of the paper",
                "paper_link": "the link of the paper page",
                "snippet": "a brief snippet or description of the paper",
                "year": 2024,
                "venue_info": "the venue information",
                "citation_count": 100
            },
            ...
        ]
    }
}

Example 2: Search papers from a specific venue (e.g., arXiv)
Use `year_from` and `year_to` to specify the time range. For example, If the user wants the latest version, set `year_from` and `year_to` to the current year.
Arguments:
{
    "queries": ["arxiv: retrieval-augmented generation"],
    "search_mode": "venue_specific",
    "topk": 5
}

Returns papers from the specified venue. Note: Some fields may be null depending on the data source (e.g., arXiv papers lack citation_count and venue_info):
{
    "success": true,
    "query_papers": {
        "arxiv: retrieval-augmented generation": [
            {
                "paper_title": "the title of the paper",
                "paper_link": "the link of the paper page",
                "snippet": "the abstract of the paper",
                "year": 2024,
                "venue_info": null,
                "citation_count": null
            },
            ...
        ]
    }
}
"""

# KNOWN_VENUES = ["arxiv", "iclr", "neurips", "icml", "acl"]

VENUE_FILTER_PROMPT_TEMPLATE = """You are a research paper venue matcher. Given a target venue and a list of paper venue information, identify which papers are published at the target venue.

Target Venue: {venue}

Paper Venue Information:
{paper_list}

Task: Return ONLY the indices (0-based) of papers that match the target venue. 

Output format: Return a JSON array of integers representing the matching indices. For example: [0, 2, 5]

If no papers match, return an empty array: []

Output:"""

class ArxivPaperSearchTool(Tool):

    name: str = "search_papers"
    description: str = "Search arXiv for papers matching a query and return a list of top results with details like title, author(s), publication date, and a link to the paper."
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "A keyword phrase, natural-language topic, or paper title used to search for academic papers."
        }, 
        "topk": {
            "type": "integer", 
            "description": "An optional integer specifying how many top-ranked papers should be returned. When provided, only the top-matching papers up to this number will be included in the result. Default: 10" 
        }
    }
    required: Optional[List[str]] = ["query"]

    def __init__(self, **kwargs):
        super().__init__()
        self.arxiv = Arxiv()
    
    def __call__(self, query: str, topk: int = 10) -> ToolResult:

        metadata = ToolMetadata(
            tool_name=self.name, 
            args={"query": query, "topk": topk}
        )
        if not query:
            return ToolResult(
                metadata=metadata, 
                result={
                    "success": False, 
                    "error": "`query` must be provided" 
                }
            )
        
        result = self.arxiv.search_arxiv(search_query=query, max_results=topk)
        return ToolResult(metadata=metadata, result=result)

    
class PaperSearchTool(Tool):

    name: str = "search_papers"
    description: str = (
        "Search academic papers from multiple sources (e.g., arXiv, Google Scholar, DBLP) for a list of queries under a specified search mode "
        "(e.g., general query-based search, paper lookup, latest research, representative works, venue-specific search or follow-up studies), "
        "with optional filters on publication year and ranking preferences. "
        "Returns a dict with 'success' (bool) and 'query_papers' (dict mapping each query to a list of paper metadata). "
        "Each paper contains: paper_title, paper_link, snippet, year, venue_info, citation_count. "
        "You should use this tool when dealing with tasks related to paper search and retrieval. "
        "Note: This tool only returns outline/summary information. If you need detailed information about a specific paper (e.g., pdf link, full author list, abstract), use fetch_paper_metadata tool."
    )
    extra_description: str = PAPER_SEARCH_TOOL_EXTRA_DESCRIPTION.strip()
    inputs: Dict[str, Dict] = {
        "queries": {
            "type": "array",
            "description": "Required. A list of search queries. Each query will be searched separately under the same search mode",  
            "items": {"type": "string"} 
        }, 
        "search_mode": {
            "type": "string", 
            "description": (
                "Required. The search intent mode. Choose based on your search goal:\n"
                "- general: general query-based search (use Google Scholar) for papers, you should use this as a default search mode when there is no specific or strong search intent\n"
                "- lookup: Find a specific paper by its title, DOI, or arXiv ID. Returns at most 1 paper paer query.\n"
                "- latest: Find latest research papers in a specific field or topic\n"
                "- representative: Find representative/seminal works in a specific field or topic\n"
                "- venue_specific: Search papers from a specific venue, such as 'arxiv'. In this case, EACH QUERY MUST be a 'venue: topic/keyword/title' string, such as 'arxiv: retrieval-augmented generation'. To specify time range, use year_from and year_to parameters\n"
                "- followup: search follow-up works of a specific paper. In this case, each query should be a paper title, IDs or other paper-related information\n"
            ),
            "enum": ["general", "lookup", "latest", "representative", "venue_specific", "followup"]
        },
        "topk": {
            "type": "integer", 
            "description": "An optional integer specifying how many top-ranked papers should be returned. Default: 5.",
            "maximum": 20
        },
        "sort_by": {
            "type": "string",
            "description": "An optional ranking preference used to reorder candidate papers before returning them. Default: \"relevance\"",
            "enum": ["relevance", "date"]
        },
        "year_from": {
            "type": "integer", 
            "description": "An optional lower bound for the publication year filter. If supplied, only papers published in or after this year will be considered during the search." 
        }, 
        "year_to": {
            "type": "integer", 
            "description": f"An optional upper bound for the publication year filter. If supplied, only papers published in or before this year will be considered during the search. Current date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" 
        }
    }
    required: Optional[List[str]] = ["queries", "search_mode"]

    def __init__(
        self, 
        storage_handler: FileStorageHandler, 
        serpapi_key: Optional[str] = None,
        openrouter_key: Optional[str] = None, 
        semantic_scholar_api_key: Optional[str] = None,
        pubmed_api_key: Optional[str] = None, 
        **kwargs
    ):
        super().__init__()
        serpapi_key = serpapi_key if serpapi_key else os.getenv("SERPAPI_KEY")
        openrouter_key = openrouter_key if openrouter_key else os.getenv("OPENROUTER_API_KEY")
        semantic_scholar_api_key = semantic_scholar_api_key if semantic_scholar_api_key else os.getenv("SEMANTIC_SCHOLAR_API_KEY")  
        pubmed_api_key = pubmed_api_key if pubmed_api_key else os.getenv("PUBMED_API_KEY")
        self.google_scholar = GoogleScholar(
            serpapi_key=serpapi_key, 
            openrouter_key=openrouter_key, 
            semantic_scholar_api_key=semantic_scholar_api_key, 
            pubmed_api_key=pubmed_api_key
        )
        self.dblp = DBLP(openrouter_key=openrouter_key)
        self.openrouter_key = openrouter_key 
        self.semantic_scholar_api_key = semantic_scholar_api_key 
        self.pubmed_api_key = pubmed_api_key 
        self.storage_handler = storage_handler 

    def __call__(self, queries: list, search_mode: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> ToolResult:

        """
        Output Format:
        {
            "query_papers": {
                "query1": [
                    {
                        "paper_title": "the title of the paper",
                        "paper_link": "the link of the paper page",
                        "snippet": "the snippet of the paper",
                        "year": "the year of the paper",
                        "venue_info": "the venue of the paper",
                        "citation_count": "the citation count of the paper"
                    }
                ],
                "query2": [...]
            }
        }
        """
        metadata = ToolMetadata(
            tool_name=self.name,
            args={"queries": queries, "search_mode": search_mode, "topk": topk, "sort_by": sort_by, "year_from": year_from, "year_to": year_to}
        )
        query_papers = {} 
        
        if not queries:
            return ToolResult(
                metadata=metadata, 
                result={
                    "success": True, 
                    "query_papers": query_papers,  
                }
            )

        # parallel execution of _search_papers 
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor: 
            futures = [
                executor.submit(self._search_papers, query, search_mode, topk, sort_by, year_from, year_to)
                for query in queries
            ]

            failed = [] 
            for query, future in zip(queries, futures):
                try:
                    papers, cost_breakdown = future.result()
                    query_papers[query] = papers 
                    metadata.add_cost_breakdown(cost_breakdown) 
                except Exception as e:
                    logger.warning(f"Error searching papers for query: {query}: {e}")
                    failed.append(query)
            
        if failed:
            if len(failed) == len(queries):
                return ToolResult(
                    metadata=metadata, 
                    result={
                        "success": False, 
                        "error": "Failed to search papers for all queries"
                    }
                )
            
            return ToolResult(
                metadata=metadata, 
                result={
                    "failed": failed, 
                    "query_papers": query_papers,  
                }
            )
        
        return ToolResult(
            metadata=metadata, 
            result={
                "success": True, 
                "query_papers": query_papers,  
            }
        )

    def _search_papers(self, query: str, search_mode: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> Tuple[List[Dict], Dict[str, float]]:

        if search_mode == "general":
            return self._general_search(query, topk, sort_by, year_from, year_to)
        elif search_mode == "lookup":
            return self._lookup_papers(query, topk, sort_by, year_from, year_to)
        elif search_mode == "latest":
            return self._search_latest_papers(query, topk, sort_by, year_from, year_to) 
        elif search_mode == "representative":
            return self._representative_papers(query, topk, sort_by, year_from, year_to)
        elif search_mode == "venue_specific":
            return self._venue_specific_papers(query, topk, sort_by, year_from, year_to)
        elif search_mode == "followup":
            return self._followup_papers(query, topk, sort_by, year_from, year_to)
        else:
            raise ValueError(f"Unsupported search mode: {search_mode}. Valid search modes: ['general', 'lookup', 'latest', 'representative', 'venue_specific', 'followup']")

    def _format_google_scholar_paper_info(self, paper: Dict, default_title: Optional[str] = None) -> Dict:
        """
        Normalize google scholar / serp metadata to the tool output schema.
        """
        if not paper:
            return {
                "paper_title": default_title,
                "paper_link": None,
                "snippet": None,
                "year": None,
                "venue_info": None,
            }
        return {
            "paper_title": paper.get("title", default_title),
            "paper_link": paper.get("publication_page_link", None),
            "snippet": paper.get("snippet", None),
            "year": paper.get("year", None),
            "venue_info": paper.get("venue_info", None),
            "citation_count": paper.get("citation_count", None)
        }

    def _format_arxiv_paper_info(self, paper: Dict, default_title: Optional[str] = None) -> Dict:
        """
        Normalize arxiv metadata to the tool output schema.
        Fields that arxiv doesn't provide (citation_count, venue_info) are set to None.
        """
        if not paper:
            return {
                "paper_title": default_title,
                "paper_link": None,
                "snippet": None,
                "year": None,
                "venue_info": None,
                "citation_count": None
            }

        # Extract year from published date (format: "2025-03-19 16:00:00" or "2025-03-19T16:00:00Z")
        year = None
        published = paper.get("published_date", "")
        if published:
            year = extract_year_from_arxiv_dates(published)
        
        # Get links
        links = paper.get("links", {})
        paper_link = links.get("html", None)

        return {
            "paper_title": paper.get("title", default_title),
            "paper_link": paper_link,
            "snippet": paper.get("summary", None),
            "year": year,
            "venue_info": None,  # arxiv doesn't provide venue info
            "citation_count": None  # arxiv doesn't provide citation count
        }

    def _format_dblp_paper_info(self, paper: Dict, default_title: Optional[str] = None) -> Dict:
        """
        Normalize DBLP metadata (from search_venue_specific_publications) to the tool output schema.

        Expected input format from search_venue_specific_publications:
        {
            "title": "paper_title",
            "authors": ["author1", "author2", ...],
            "conference": "Conference Title 2024",
            "bm25_score": 5.123  # optional, only present when queried with title_or_keyword
        }
        """
        if not paper:
            return {
                "paper_title": default_title,
                "paper_link": None,
                "snippet": None,
                "year": None,
                "venue_info": None,
                "citation_count": None
            }

        # Extract year from conference title (e.g., "ICLR 2024" -> 2024)
        year = None
        conference = paper.get("conference", "")
        if conference:
            year = extract_year(conference)

        # Format authors as a string for snippet
        authors = paper.get("authors", [])
        authors_str = ", ".join(authors) if authors else None

        return {
            "paper_title": paper.get("title", default_title),
            "paper_link": None,  # DBLP venue search doesn't provide paper links
            "snippet": None,  # DBLP venue search does not provide snippet
            "year": year,
            "venue_info": f"{authors_str} - {conference}",
            "citation_count": None  # DBLP doesn't provide citation count in venue search
        }

    def _general_search(self, query: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> Tuple[List[Dict], Dict[str, float]]:

        query = query.strip() 
        if not query:
            raise ValueError("search query must be provided, but got empty string") 
        
        cost_breakdown: Dict[str, float] = {} 
        google_scholar_results: ResearchToolResult = self.google_scholar.search_publications(
            title_or_keyword=query, 
            topk=topk, 
            sort_by=sort_by, 
            year_from=year_from, 
            year_to=year_to
        )
        google_scholar_paper_metadata = google_scholar_results.result  
        cost_breakdown = add_dict(cost_breakdown, google_scholar_results.metadata.cost_breakdown)

        paper_info_list = [
            self._format_google_scholar_paper_info(paper)
            for paper in google_scholar_paper_metadata
        ]
        
        return paper_info_list, cost_breakdown  
    
    def _lookup_papers(self, query: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> Tuple[List[Dict], Dict[str, float]]:
        
        query = query.strip() 
        if not query:
            raise ValueError("search query must be provided, but got empty string") 
        
        cost_breakdown: Dict[str, float] = {} 
        resolved_paper_title = resolve_paper_title(query=query, semantic_scholar_api_key=self.semantic_scholar_api_key)

        # perform google scholar search to obtain the paper info 
        google_scholar_results: ResearchToolResult = self.google_scholar.get_metadata_based_on_title(resolved_paper_title)
        google_scholar_paper_metadata = google_scholar_results.result
        cost_breakdown = add_dict(cost_breakdown, google_scholar_results.metadata.cost_breakdown)

        if google_scholar_paper_metadata:
            paper_info = self._format_google_scholar_paper_info(
                google_scholar_paper_metadata,
                default_title=resolved_paper_title,
            )
            return [paper_info], cost_breakdown

        # if the paper is not found on google scholar, use google search to obtain the paper info
        google_search_results: ResearchToolResult = self.google_scholar.serp.get_metadata_based_on_title(
            resolved_paper_title,
            openrouter_key=self.openrouter_key
        )
        google_search_paper_metadata = google_search_results.result
        cost_breakdown = add_dict(cost_breakdown, google_search_results.metadata.cost_breakdown)
        if google_search_paper_metadata:
            return [google_search_paper_metadata], cost_breakdown
        else:
            raise ValueError(f"Failed to search paper info for query: {query}")

    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]: 
        # papers are expected to be results from google scholar with "serpapi_scholar_key" 
        if not papers:
            return [] 
        
        # deduplicate papers by title
        unique_papers = [] 
        seen_paper_keys = set()
        for paper in papers:
            key = paper.get("serpapi_scholar_key", None) or normalize_title_aggressively(paper.get("title", None))
            if not key or key in seen_paper_keys:
                continue
            seen_paper_keys.add(key)
            unique_papers.append(paper)
        
        return unique_papers

    def _get_google_scholar_paper_year(self, paper: Dict) -> Optional[int]:
        try:
            year = int(paper.get("year", None))
        except Exception:
            year = None 
        return year 

    def _get_google_scholar_paper_citation_count(self, paper: Dict) -> int:
        citation_count = paper.get("citation_count", None)
        if not citation_count:
            return 0 
        return citation_count 

    def _is_survey_paper(self, paper: Dict) -> bool:
        title = paper.get("title", "").lower()
        survey_markers = ["survey", "review", "tutorial", "overview", "a review of", "a survey of"]
        for marker in survey_markers:
            if marker in title:
                return True
        return False

    def _get_citation_count_cap(self, papers: List[Dict], percentile: int = 95, min_cap: int = 50) -> int:
        """
        Calculate the citation count cap at a given percentile.

        Args:
            papers: List of papers with citation counts
            percentile: The percentile to use for the cap (default: 95)
            min_cap: Minimum cap value (default: 50)

        Returns:
            The citation count cap value
        """
        citation_counts = sorted(
            [
                self._get_google_scholar_paper_citation_count(paper)
                for paper in papers
            ]
        )
        if not citation_counts:
            return min_cap
        k = int(math.ceil((percentile / 100.0) * len(citation_counts))) - 1
        k = max(0, min(k, len(citation_counts) - 1))
        cap = citation_counts[k]
        return max(cap, min_cap) 
    
    def _score_and_select_seed_papers(
        self, 
        topic: str, 
        candidates: List[Dict],
        years_back: int = 2,
        min_cited_by: int = 5,
        year_from: Optional[int] = None, 
        year_to: Optional[int] = None, 
        max_survey_papers: int = 2, 
        max_seed_papers: int = 20, 
        seminal_paper_titles: Optional[List[str]] = None
    ) -> List[Dict]:

        def _looks_like_published(paper: Dict) -> bool:
            venue_info = paper.get("venue_info", None)
            if not venue_info:
                return False 
            venue_info = venue_info.lower()
            venue_markers = [
                "proceedings", "conference", "journal", "transactions", "ieee", "acm", "springer", "wiley", "elsevier", 
                "neurips", "nips", "icml", "iclr", "acl", "emnlp", "naacl", "tacl", "coling", "cvpr", "eccv", "iccv", "ijcai", 
                "kdd", "sigir", "tmlr", "cikm", "www", "aaai", "nature", "science", "cell", "pnas", "lancet", "nejm", "physical",
                "clinical", "medical", "medicine" 
            ]
            for marker in venue_markers:
                if marker in venue_info:
                    return True
            if self._get_google_scholar_paper_citation_count(paper) >= 100:
                return True 
            return False 

        if not candidates:
            return []
        
        # topic_tokens = tokenize(topic)

        now_year = year_to or datetime.now().year 
        year_max = min(now_year, year_to or 9999)
        years_back = max(1, years_back) 
        year_min = max(year_max - years_back + 1, year_from or -1)
        if year_min > year_max:
            return [] 

        # Step 1: Filter candidates 
        filtered_candidates = [] 
        require_published_signal = (len(candidates) > max_seed_papers//2)
        for candidate in candidates:
            # filter by year 
            year = self._get_google_scholar_paper_year(candidate)
            if not year or year < year_min or year > year_max:
                continue
            # filter by citation count  
            if self._get_google_scholar_paper_citation_count(candidate) < min_cited_by:
                continue
            # filter by publication when the number of candidates is large enough 
            if require_published_signal and not _looks_like_published(candidate):
                continue 
            # jaccard_score = jaccard_similarity(topic_tokens, tokenize(candidate.get("title", None)))
            # if jaccard_score < 0.05:
            #     continue
            filtered_candidates.append(candidate)
        
        if not filtered_candidates:
            return [] 
        
        # Step 2: Score candidates
        # (1) relevance scores 
        paper_texts = [
            "{} {}".format(candidate.get("title", ""), candidate.get("snippet", "")).strip()
            for candidate in filtered_candidates
        ]
        bm25 = BM25(paper_texts)
        relevance_scores = [bm25.score(topic, text) for text in paper_texts]
        max_relevance_score = max(relevance_scores)
        relevance_scores = [score / (max_relevance_score + 1e-9) for score in relevance_scores] # rescale to [0, 1]
        # (2) impact scores
        citation_count_cap = self._get_citation_count_cap(filtered_candidates)
        impact_scores = [
            math.log1p(self._get_google_scholar_paper_citation_count(candidate)) / math.log1p(citation_count_cap)
            for candidate in filtered_candidates
        ]
        # (3) recency scores 
        recency_scores = [] 
        for candidate in filtered_candidates:
            year = self._get_google_scholar_paper_year(candidate) 
            recency_score = max(1.0 - min(now_year - year, 5) / 5.0, 0.0) if year else 0.0 
            recency_scores.append(recency_score) 
        # (4) combine scores 
        scores = [
            relevance_scores[i] * 0.6 + impact_scores[i] * 0.3 + recency_scores[i] * 0.1
            for i in range(len(filtered_candidates))
        ]
        # (5) seminal paper scores 
        if seminal_paper_titles:
            for seminal_paper_title in seminal_paper_titles:
                filtered_candidates_titles = [candidate.get("title", "") for candidate in filtered_candidates] 
                text_match_result = text_match(seminal_paper_title, filtered_candidates_titles) 
                if text_match_result["match"] and text_match_result["best_candidate_metrics"]["score"] > 0.8:
                    best_candidate_index = text_match_result["best_candidate_metrics"]["index"]
                    scores[best_candidate_index] += 0.4 
        sorted_candidates = sorted(zip(filtered_candidates, scores), key=lambda x: x[1], reverse=True)
        for candidate, score in sorted_candidates:
            candidate["score"] = score

        # Step 3: Light diversity: avoid all from same year; aim to spread across last 2-3 years 
        buckets = defaultdict(list) 
        for candidate, score in sorted_candidates:
            year = self._get_google_scholar_paper_year(candidate) or -1 
            buckets[year].append((candidate, score))
        
        years_priority = list(range(year_min, year_max + 1)) + [-1] # -1 because some papers may not have a year  
        per_year_cap = max(5, max_seed_papers // (len(years_priority) - 1))

        selected_candidates = [] 
        num_survey_paper = 0 
        idx = {year: 0 for year in years_priority} 
        year_count = {year: 0 for year in years_priority} 
        while len(selected_candidates) < max_seed_papers:
            progress = False 
            for year in years_priority:
                if len(selected_candidates) >= max_seed_papers:
                    break 
                if year not in buckets:
                    continue
                if year_count[year] >= per_year_cap:
                    continue
                i = idx[year] 
                if i >= len(buckets[year]):
                    continue 
                paper, _ = buckets[year][i] 
                is_survey_paper = self._is_survey_paper(paper)
                if is_survey_paper and num_survey_paper >= max_survey_papers:
                    continue 
                idx[year] += 1 
                selected_candidates.append(paper) 
                year_count[year] += 1 
                num_survey_paper += 1 if is_survey_paper else 0 
                progress = True 
            if not progress:
                break 
        
        if len(selected_candidates) < max_seed_papers:
            selected_titles = {paper.get("title", None) for paper in selected_candidates} 
            for candidate, score in sorted_candidates:
                if len(selected_candidates) >= max_seed_papers:
                    break
                is_survey_paper = self._is_survey_paper(candidate)
                if is_survey_paper and num_survey_paper >= max_survey_papers:
                    continue 
                candidate_title = candidate.get("title", None)
                if not candidate_title or candidate_title in selected_titles:
                    continue 
                selected_candidates.append(candidate)
                selected_titles.add(candidate_title)
                num_survey_paper += 1 if is_survey_paper else 0 
        
        selected_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return selected_candidates 

    def _find_seed_papers(
        self, 
        topic: str, 
        max_queries: int = 3, 
        per_query_topk: int = 20, 
        years_back: int = 2,
        min_cited_by: int = 5,
        year_from: Optional[int] = None, 
        year_to: Optional[int] = None, 
        max_seed_papers: int = 20,
        num_seminal_papers: int = 2 
    ) -> Tuple[List[Dict], Dict[str, float]]:

        cost_breakdown: Dict[str, float] = {} 

        # Step 1: Query expansion for the topic using LLM 
        query_list, query_expansion_cost_breakdown = query_expansion_for_recall(query=topic, llm=self.google_scholar.llm, max_queries=max(1, max_queries-num_seminal_papers)) 
        cost_breakdown = add_dict(cost_breakdown, query_expansion_cost_breakdown)

        if num_seminal_papers > 0:
            seminal_paper_titles, find_seminal_cost_breakdown = find_seminal_papers_for_topic(topic=topic, llm=self.google_scholar.llm, max_papers=num_seminal_papers)
            cost_breakdown = add_dict(cost_breakdown, find_seminal_cost_breakdown)
            # randomly drop some words from titles to avoid exact match
            actual_num_seminal_papers = 0 
            for paper_title in seminal_paper_titles:
                words = paper_title.lower().split()
                indices = list(range(len(words)))
                random.shuffle(indices)
                drop_indices = indices[:min(5, max(2, int(len(words) * 0.2)))]
                filtered_words = [word for i, word in enumerate(words) if i not in drop_indices] 
                new_title = " ".join(filtered_words).strip()
                if new_title:
                    query_list.append(new_title) 
                    actual_num_seminal_papers += 1 

        if not query_list:
            return [], cost_breakdown 

        candidate_papers = [] 
        for i, query in enumerate(query_list):
            if num_seminal_papers > 0:
                if i < len(query_list) - actual_num_seminal_papers:
                    topk = per_query_topk
                else:
                    topk = 10 # for the seminal papers, we only need to find the most relevant paper 
            else:
                topk = per_query_topk
            google_scholar_results: ResearchToolResult = self.google_scholar.search_publications(
                title_or_keyword=query, 
                topk=topk, 
                year_from = year_from, 
                year_to = year_to 
            )
            candidate_papers.extend(google_scholar_results.result)
            cost_breakdown = add_dict(cost_breakdown, google_scholar_results.metadata.cost_breakdown)
        
        # Step 2: Filter out the duplicate papers  
        candidate_papers = self._deduplicate_papers(candidate_papers)
        if not candidate_papers:
            return [], cost_breakdown 

        # Step 3: Select seed papers from the candidate papers
        seed_papers = self._score_and_select_seed_papers(
            topic = topic, 
            candidates = candidate_papers,
            years_back = years_back,  
            min_cited_by = min_cited_by,
            year_from = year_from, 
            year_to = year_to, 
            max_seed_papers = max_seed_papers,
            seminal_paper_titles = seminal_paper_titles 
        )

        return seed_papers, cost_breakdown 
    
    def _fetch_related_work_references(self, paper: dict) -> Tuple[List[Dict], Dict[str, float]]:

        pdf_url = paper.get("publication_pdf_link")
        if not pdf_url:
            return [], {}
        
        file_name = normalize_title_aggressively(paper["title"]).replace(" ", "_")
        if not self.storage_handler.exists(path=file_name):
            download_result = self.storage_handler._download_content(url = pdf_url, filename=file_name, max_retries=1)
            if not download_result["success"]:
                print("Failed to download PDF file: ", pdf_url)
                return [], {} 
            file_path = download_result["file_path"] 
        else:
            file_path = file_name 
        
        # file_path = "ragas_automated_evaluation_of_retrieval_augmented_generation_1768ebd3.pdf" # "parametric_retrieval_augmented_generation_57e9e994.pdf" 
        read_result = self.storage_handler.read(file_path=file_path)
        if not read_result["success"]:
            print("Failed to read PDF file: ", file_path) 
            self.storage_handler.delete(file_path=file_path)
            return [], {} 
        
        content = read_result["content"]
        related_work_references, cost_breakdown = extract_related_work_references(content, llm=self.google_scholar.llm)
        # todo: delete the PDF file after extracting the related work references 
        return related_work_references, cost_breakdown 

    def _select_representative_papers_from_seed_references(
        self, 
        query: str, 
        seed_papers: List[Dict], 
        related_work_references: List[Dict], 
        years_back: int = 5, 
        year_from: Optional[int] = None, 
        year_to: Optional[int] = None, 
        max_representative_papers: int = 5, 
    ) -> List[Dict]:

        # Step 1: obtain unique references 
        unique_references = {} 
        frequency_in_seed_papers = {} 
        for references in related_work_references:
            # references is a list of dicts, corresponding to the related work references of a seed paper 
            references_in_current_seed_paper = set() 
            for reference in references: 
                title = reference.get("title", None)
                tokens = tokenize(title)
                if not tokens:
                    continue
                try:
                    year = int(extract_year(reference.get("year", ""))) 
                except Exception:
                    continue 
                matched_key = None
                for existing_key, meta in unique_references.items():
                    if jaccard_similarity(tokens, meta["tokens"]) >= 0.95:
                        matched_key = existing_key
                        break
                if not matched_key:
                    matched_key = normalize_title_aggressively(title)
                    unique_references[matched_key] = {"title": title, "year": year, "tokens": tokens}
                references_in_current_seed_paper.add(matched_key) 
            
            for key in references_in_current_seed_paper:
                frequency_in_seed_papers[key] = frequency_in_seed_papers.get(key, 0) + 1 
        
        # Step 2: Filter references by year range
        now_year = datetime.now().year 
        year_max = min(now_year, year_to or 9999)
        years_back = max(1, years_back) 
        year_min = max(year_max - years_back + 1, year_from or -1)
        if year_min > year_max:
            return [] 
        
        filtered_reference_keys, filtered_reference_metas = [], [] 
        for key, meta in unique_references.items():
            if meta["year"] < year_min or meta["year"] > year_max:
                continue
            filtered_reference_keys.append(key)
            filtered_reference_metas.append(meta)
        
        if not filtered_reference_keys:
            return [] 
        
        # Step 3: Score each reference by its frequency in seed papers and relevance to the query 
        # (1) BM25 scores 
        bm25 = BM25([meta["title"] for meta in filtered_reference_metas])
        relevance_scores = [bm25.score(query, meta["title"]) for meta in filtered_reference_metas] 
        max_relevance_scores = max(relevance_scores) 
        relevance_scores = [score / (max_relevance_scores + 1e-9) for score in relevance_scores] # rescale to [0, 1] 
        # (2) frequency scores 
        num_seed_papers = max(1, len(seed_papers)) 
        frequency_scores = [frequency_in_seed_papers.get(key, 0) / num_seed_papers for key in filtered_reference_keys]
        # (3) combine scores 
        scores = [relevance_scores[i] * 0.25 + frequency_scores[i] * 0.75 for i in range(len(filtered_reference_keys))] 

        scored_references = [] 
        for meta, score in zip(filtered_reference_metas, scores):
            scored_references.append({"title": meta["title"], "year": meta["year"], "score": score})
        scored_references.sort(key=lambda x: x["score"], reverse=True)

        # select representative paper across all years 
        buckets = defaultdict(list)
        for reference in scored_references:
            buckets[reference["year"]].append(reference)
        years_range = list(range(year_min, year_max + 1))
        per_year_cap = max(5, max_representative_papers // len(years_range))

        selected_references = [] 
        idx = {year: 0 for year in years_range} 
        year_count = {year: 0 for year in years_range} 
        while len(selected_references) < max_representative_papers:
            progress = False 
            for year in years_range:
                if len(selected_references) >= max_representative_papers:
                    break 
                if year not in buckets:
                    continue 
                if year_count[year] >= per_year_cap:
                    continue 
                i = idx[year] 
                if i >= len(buckets[year]):
                    continue
                idx[year] += 1 
                selected_references.append(buckets[year][i])
                year_count[year] += 1 
                progress = True 
            if not progress:
                break 
            
        if len(selected_references) < max_representative_papers:
            selected_reference_titles = {reference["title"] for reference in selected_references}
            for reference in scored_references:
                if len(selected_references) >= max_representative_papers:
                    break 
                reference_title = reference["title"]
                if reference_title in selected_reference_titles:
                    continue 
                selected_references.append(reference)
                selected_reference_titles.add(reference_title) 
        
        # sort the selected references by socre 
        selected_references.sort(key=lambda x: x["score"], reverse=True)
        return selected_references 

    def _score_and_select_latest_papers(
        self,
        query: str,
        candidates: List[Dict],
        topk: int = 5,
        max_survey_papers: int = 2
    ) -> List[Dict]:
        """
        Score and select latest papers from candidates based on relevance and citation count.

        Args:
            query: The search query/topic
            candidates: List of candidate papers (raw Google Scholar format)
            topk: Number of top papers to select
            max_survey_papers: Maximum number of survey papers to include

        Returns:
            A list of selected papers sorted by combined score
        """
        if not candidates:
            return []

        # Step 1: Calculate BM25 relevance scores
        paper_texts = [
            "{} {}".format(candidate.get("title", ""), candidate.get("snippet", "")).strip()
            for candidate in candidates
        ]
        bm25 = BM25(paper_texts)
        relevance_scores = [bm25.score(query, text) for text in paper_texts]
        max_relevance_score = max(relevance_scores) if relevance_scores else 1.0
        relevance_scores = [score / (max_relevance_score + 1e-9) for score in relevance_scores]

        # Step 2: Calculate impact scores (citation-based)
        citation_count_cap = self._get_citation_count_cap(candidates)
        impact_scores = [
            math.log1p(self._get_google_scholar_paper_citation_count(candidate)) / math.log1p(citation_count_cap)
            for candidate in candidates
        ]

        # Step 3: Combine scores (higher weight on relevance for latest papers)
        scores = [
            relevance_scores[i] * 0.6 + impact_scores[i] * 0.4
            for i in range(len(candidates))
        ]

        # Step 4: Sort candidates by score
        sorted_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        for candidate, score in sorted_candidates:
            candidate["score"] = score

        # Step 5: Select top-k papers with survey limit
        selected_papers = []
        num_survey_papers = 0

        for candidate, score in sorted_candidates:
            if len(selected_papers) >= topk:
                break
            is_survey = self._is_survey_paper(candidate)
            # Skip survey papers if we already have max_survey_papers
            if is_survey and num_survey_papers >= max_survey_papers:
                continue
            selected_papers.append(candidate)
            if is_survey:
                num_survey_papers += 1

        return selected_papers

    def _representative_papers(self, query: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> Tuple[List[Dict], Dict[str, float]]:

        query = query.strip() 
        if not query:
            raise ValueError("search query must be provided, but got empty string") 
        
        cost_breakdown: Dict[str, float] = {} 

        # Step 1: Start with seed papers for the query 
        seed_papers, seed_paper_cost_breakdown = self._find_seed_papers(
            topic = query, 
            max_queries=5,  # max 3 queries for query expansion and seminal paper search 
            num_seminal_papers=2, 
            per_query_topk=20,   
            years_back = 5, # find representative papers from the last 5 years by default 
            year_from = year_from, 
            year_to = year_to, 
            max_seed_papers=topk 
        ) # obtain seed papers for the query 
        cost_breakdown = add_dict(cost_breakdown, seed_paper_cost_breakdown)
        if not seed_papers:
            raise ValueError(f"Failed to find seed papers for query: {query}")

        representative_papers = [
            self._format_google_scholar_paper_info(paper) 
            for paper in seed_papers
        ]

        # Step 2: Fetch related work references for each seed paper 
        # related_work_references = [] 
        # for seed_paper in seed_papers: 
        #     try:
        #         references, fetch_related_work_cost_breakdown = self._fetch_related_work_references(seed_paper)
        #         related_work_references.append(references)
        #         cost_breakdown = add_dict(cost_breakdown, fetch_related_work_cost_breakdown)
        #     except Exception: 
        #         continue 
        
        # # score and select representative papers from the seed references  
        # representative_paper_info = self._select_representative_papers_from_seed_references(
        #     query = query, 
        #     seed_papers = seed_papers, 
        #     related_work_references = related_work_references, 
        #     year_from = year_from, 
        #     year_to = year_to, 
        #     max_representative_papers = topk
        # )
        # print("Representative papers: ", representative_paper_info) 

        # # format the representative papers 
        # representative_papers = []
        # for pinfo in representative_paper_info:
        #     title = pinfo.get("title", None)
        #     if title:
        #         try:
        #             paper, lookup_cost_breakdown = self._lookup_papers(title)
        #             cost_breakdown = add_dict(cost_breakdown, lookup_cost_breakdown)
        #             if paper:
        #                 representative_papers.extend(paper) # _lookup_papers return a list of paper info with only one element 
        #         except Exception:
        #             pass 
        
        return representative_papers, cost_breakdown 

    def _followup_papers(self, query: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Find followup papers (papers that cite a given paper) based on the paper title or identifier.

        Args:
            query: The paper title or identifier (e.g., DOI, arXiv ID).
            topk: The number of top results to return.
            sort_by: The field to sort the results by. Valid values: ["relevance", "date"].
            year_from: The minimum year of the publication.
            year_to: The maximum year of the publication.

        Returns:
            A tuple containing a list of formatted paper information and cost breakdown.
        """
        query = query.strip()
        if not query:
            raise ValueError("search query must be provided, but got empty string")

        cost_breakdown: Dict[str, float] = {}

        # Step 1: Resolve the paper title from the query (could be title or identifier)
        resolved_paper_title = resolve_paper_title(query=query, semantic_scholar_api_key=self.semantic_scholar_api_key)

        # Step 2: Get followup papers using GoogleScholar's get_followup_papers method
        followup_results: ResearchToolResult = self.google_scholar.get_followup_papers(
            title=resolved_paper_title,
            topk=topk,
            sort_by=sort_by,
            year_from=year_from,
            year_to=year_to
        )
        followup_papers = followup_results.result
        cost_breakdown = add_dict(cost_breakdown, followup_results.metadata.cost_breakdown)

        if not followup_papers:
            logger.warning(f"No followup papers found for query: {query}")
            return [], cost_breakdown

        # Step 3: Format the papers using the same format as other functions
        paper_info_list = [
            self._format_google_scholar_paper_info(paper)
            for paper in followup_papers
        ]

        return paper_info_list, cost_breakdown

    def _search_latest_papers(self, query: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Search for latest papers in a specific topic or domain, typically from the last year.
        
        Args:
            query: The topic or domain to search for
            topk: Number of top results to return
            sort_by: Sorting preference (currently not used as we use combined scoring)
            year_from: Optional lower bound for publication year
            year_to: Optional upper bound for publication year

        Returns:
            A tuple of (paper_info_list, cost_breakdown)
        """
        query = query.strip()
        if not query:
            raise ValueError("search query must be provided, but got empty string")

        cost_breakdown: Dict[str, float] = {}

        # Determine the year range if not explicitly provided
        now = datetime.now()
        current_year = now.year
        current_month = now.month

        # If year_from is not provided, use default logic:
        # - If current month < 6, use past 2 years
        # - If current month >= 6, use current year
        if year_from is None:
            default_year_from = current_year - 1 if current_month < 6 else current_year
        else:
            default_year_from = year_from

        if year_to is None:
            default_year_to = current_year
        else:
            default_year_to = year_to

        # Step 1: Get top-5 representative papers for the query
        representative_papers, repr_cost = self._representative_papers(
            query=query,
            topk=5,
            sort_by="relevance"
        )
        cost_breakdown = add_dict(cost_breakdown, repr_cost)

        if not representative_papers:
            logger.warning(f"No representative papers found for query: {query}")
            return [], cost_breakdown

        # Step 2: For each representative paper, get top-20 followup papers with year filtering
        all_raw_followup_papers = []
        for repr_paper in representative_papers:
            paper_title = repr_paper.get("paper_title")
            if not paper_title:
                continue

            try:
                # Resolve paper title
                resolved_paper_title = resolve_paper_title(query=paper_title, semantic_scholar_api_key=self.semantic_scholar_api_key)

                # Get raw followup papers from Google Scholar
                followup_results: ResearchToolResult = self.google_scholar.get_followup_papers(
                    title=resolved_paper_title,
                    topk=20,
                    sort_by="relevance",
                    year_from=default_year_from,
                    year_to=default_year_to
                )
                raw_followup_papers = followup_results.result
                cost_breakdown = add_dict(cost_breakdown, followup_results.metadata.cost_breakdown)

                if raw_followup_papers:
                    all_raw_followup_papers.extend(raw_followup_papers)
            except Exception as e:
                logger.warning(f"Failed to get followup papers for {paper_title}: {e}")
                continue

        if not all_raw_followup_papers:
            logger.warning(f"No followup papers found for query: {query}")
            return [], cost_breakdown

        # Step 3: Deduplicate papers by title (using raw papers)
        unique_raw_papers = []
        seen_titles = set()
        for paper in all_raw_followup_papers:
            title = paper.get("title")
            if not title:
                continue
            normalized_title = normalize_title_aggressively(title)
            if normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)
            unique_raw_papers.append(paper)

        if not unique_raw_papers:
            logger.warning(f"No unique papers found after deduplication for query: {query}")
            return [], cost_breakdown

        # Step 4: Score, select and sort papers using combined relevance and citation metrics
        selected_papers = self._score_and_select_latest_papers(
            query=query,
            candidates=unique_raw_papers,
            topk=topk,
            max_survey_papers=2
        )

        # Step 5: Format the selected papers
        result_papers = [
            self._format_google_scholar_paper_info(paper)
            for paper in selected_papers
        ]

        return result_papers, cost_breakdown
    
    def _parse_venue_query(self, text: str) -> Tuple[str, str]:

        splits = text.split(":", 1)
        if len(splits) != 2:
            return None, text
        venue = splits[0].strip()
        query = splits[1].strip()
        return venue, query
    
    def _venue_specific_papers(self, query: str, topk: int = 5, sort_by: str = "relevance", year_from: int = None, year_to: int = None) -> Tuple[List[Dict], Dict[str, float]]:

        query = query.strip()
        if not query:
            raise ValueError("search query must be provided, but got empty string")

        # parse venue and query
        venue, query = self._parse_venue_query(query)
        if not venue:
            # fall back to general search
            # return self._general_search(query, topk, sort_by, year_from, year_to)
            # fall back to arXiv search 
            venue = "arxiv"

        cost_breakdown: Dict[str, float] = {}

        # If search from arxiv, directly use arXiv API
        if venue in ["arxiv"]:
            arxiv_results = self.google_scholar.arxiv.search_arxiv(search_query=query, max_results=topk)
            if not arxiv_results.get("success", False):
                error_msg = arxiv_results.get("error", "Unknown error")
                raise ValueError(f"Failed to search arxiv papers for query: {query}. Error: {error_msg}")
            papers = arxiv_results.get("papers", [])
            paper_info_list = [self._format_arxiv_paper_info(paper) for paper in papers]
            return paper_info_list, cost_breakdown 
        
        # TODO: for some well-known conference/journal, try to load from prebuilt index 

        # For CS conference/journal, try DBLP
        dblp_results: ResearchToolResult = self.dblp.search_venue_specific_publications(
            venue=venue,
            title_or_keyword=query, 
            topk=topk,
            year_from=year_from,
            year_to=year_to
        )
        cost_breakdown = add_dict(cost_breakdown, dblp_results.metadata.cost_breakdown)
        if dblp_results.result:
            papers = dblp_results.result
            paper_info_list = [self._format_dblp_paper_info(paper) for paper in papers]
            return paper_info_list, cost_breakdown
        
        # If all attemps fail, fall backs to google scholar
        paper_info_list, general_cost_breakdown = self._general_search(
            query = f"{venue}: {query}",
            topk = topk * 3,
            sort_by = "relevance",
            year_from = year_from,
            year_to = year_to
        )
        cost_breakdown = add_dict(cost_breakdown, general_cost_breakdown)

        return paper_info_list, cost_breakdown
        