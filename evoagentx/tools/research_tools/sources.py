
import os
import re
import urllib.parse
from bs4 import BeautifulSoup, Comment
from selectolax.parser import HTMLParser
from typing import List, Dict, Optional, Any, Tuple

from ..request_base import RequestBase
from ..request_arxiv import ArxivBase 
from ...core.logging import logger
from ...core.module_utils import parse_json_from_llm_output 
from ...models.model_utils import track_cost 
from ...models import OpenRouterConfig, OpenRouterLLM
from ...utils.utils import add_dict

from .utils import (
    normalize_title,
    extract_arxiv_id,
    validate_author_info,
    extract_year_from_arxiv_dates,
    DOI_REGEX,
    DBLP_LLM_MODEL,
    DEFAULT_PAGE_HEADERS,
)
from .prompts import (
    BIB_NOT_FOUND_TEXT,
    DBLP_BIBTEXT_POLISHING_PROMPT,
    DBLP_VENUE_SELECTION_PROMPT,
    DBLP_YEAR_FILTER_RANGE_PROMPT,
)
from .serp_utils import SerpAPIForResearch
from .text_match import text_match, BM25
from .metadata import ResearchToolMetadata, ResearchToolResult
from .cache import CacheMixin, ConferencePageCacheKeyGenerator


class Arxiv(ArxivBase):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def get_metadata_based_on_arxiv_id(self, arxiv_id: str) -> ResearchToolResult:

        metadata = ResearchToolMetadata(tool_name="arxiv_get_metadata_based_on_arxiv_id")
        paper_metadata = {}

        arxiv_id = arxiv_id.strip() 
        if not arxiv_id:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
         
        paper_info = self.search_arxiv(id_list=[arxiv_id.strip()])
        if paper_info.get("success", False):
            if not paper_info.get("papers", []):
                return ResearchToolResult(metadata=metadata, result=paper_metadata) # no paper found
            paper_metadata = self._convert_arxiv_metadata(paper_info.get("papers", [{}])[0])

        return ResearchToolResult(metadata=metadata, result=paper_metadata)

    def get_metadata_based_on_title(self, title: str) -> ResearchToolResult:
        """
        Get the metadata of a paper based on the title.
        Args:
            title: The title of the paper.
        Returns:
            A dictionary containing the metadata of the paper.
        """
        metadata = ResearchToolMetadata(tool_name="arxiv_get_metadata_based_on_title") 
        paper_metadata = {} 

        title = title.strip() 
        if not title:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        papers_info = self.search_arxiv(search_query=title, max_results=3)
        if papers_info.get("success", False) and papers_info.get("papers", []):
            for paper_info in papers_info.get("papers", []):
                match_result = text_match(title, [paper_info["title"]])
                if match_result.get("match", False):
                    logger.info(f"Arxiv title matched: query='{title}', fetched_title='{paper_info['title']}', arxiv_id='{paper_info['arxiv_id']}'")
                    paper_metadata = self._convert_arxiv_metadata(paper_info) 
                    return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        return ResearchToolResult(metadata=metadata, result=paper_metadata)

    def _convert_arxiv_metadata(self, paper_info: Dict) -> Dict:
        """
        Convert the metadata of a paper from arXiv to a dictionary.
        Args:
            paper_info: A dictionary containing the metadata of the paper.
        Returns:
            A dictionary containing the metadata of the paper.
        """
        paper_metadata = {} 
        paper_metadata["title"] = paper_info.get("title", "Unknown Title")
        paper_metadata["authors"] = paper_info.get("authors", None)
        paper_metadata["abstract"] = paper_info.get("summary", None)
        paper_metadata["year"] = extract_year_from_arxiv_dates(paper_info.get("published_date", None)) 
        paper_metadata["paper_link"] = paper_info.get("links", {}).get("html", None)
        paper_metadata["pdf_link"] = paper_info.get("links", {}).get("pdf", None)
        return paper_metadata


class DBLP(CacheMixin, RequestBase):

    def __init__(
        self,
        openrouter_key: Optional[str] = None,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(timeout=10, max_retries=1) 
        self.base_url = "https://dblp.org" 
        openrouter_key = openrouter_key if openrouter_key else os.getenv("OPENROUTER_API_KEY")
        self.llm = OpenRouterLLM(
            config=OpenRouterConfig(
                openrouter_key=openrouter_key,
                model=DBLP_LLM_MODEL,
                stream=False,
                output_response=False,
                temperature=0.0, 
            )
        )
        self.enable_cache = enable_cache
        self._cache_dir = cache_dir

    def _ensure_conference_page_cache(self) -> None:
        if self._tool_cache is None:
            self._init_cache(
                tool_name="dblp_conference_page",
                cache_dir=self._cache_dir,
                key_generator=ConferencePageCacheKeyGenerator(),
            )
    
    def _clean_author_name(self, name: str) -> str: 

        name = name.strip()
        if not name:
            return name 
        _re_dblp_suffix_num = re.compile(r"\s+\d+\s*$")
        name = _re_dblp_suffix_num.sub("", name).strip()
        return name

    def search_publications(self, title_or_keyword: str, topk: int = 10) -> List[Dict]:
        """
        Search for publications on DBLP using a single title or keyword.
        Args:
            title_or_keyword: A single title or keyword to search for.
            topk: The number of top results to return.
        Returns:
            A list of dictionaries containing the search results. The format is as follows:
            [
                {
                    "key": "the key of the paper", 
                    "title": "paper title", 
                    "authors": ["the authors of the paper"], 
                    "year": "the year of the paper",
                    "type": "the type of the paper", 
                    "publication_source": "the publication source (conference, journal, book, etc.)"
                }
            ]
        """
        url = f"{self.base_url}/search/publ/api"
        params = {"q": normalize_title(title_or_keyword), "format": "json", "h": topk}
        response = self.request(url=url, method='GET', params=params)
        data = response.json()["result"]["hits"]
        results = [] 

        if ("@sent" in data and data["@sent"] == "0") or "hit" not in data:
            # No results found 
            return results 
        
        hit_items = data["hit"]
        if not isinstance(hit_items, list):
            hit_items = [hit_items]
        
        for item in hit_items:
            author_items = item["info"]["authors"]["author"]
            if not isinstance(author_items, list):
                author_items = [author_items]
            authors = [author_item["text"] for author_item in author_items]
            title = item["info"]["title"].strip()
            if title.endswith("."):
                title = title[:-1]
            results.append({
                "key": item["info"]["key"], 
                "title": title,
                "authors": authors,
                "year": item["info"].get("year", ""), 
                "type": item["info"].get("type", ""), 
                "publication_source": item["info"].get("venue", "") or item["info"].get("ee", "")
            })
        return results

    def get_metadata_based_on_title(self, title: str) -> ResearchToolResult:

        metadata = ResearchToolMetadata(tool_name="dblp_get_metadata_based_on_title")
        paper_metadata = {} 

        title = title.strip() 
        if not title:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        publications = self.search_publications(title)
        duplicate_titles = self.get_duplicate_titles(publications)
        if duplicate_titles:
            publications = self.filter_duplicate_publications(publications, duplicate_titles)
        for publication in publications:
            match_result = text_match(title, [publication["title"]])
            if match_result.get("match", False):
                logger.info(f"DBLP title matched: query='{title}', fetched_title='{publication['title']}', dblp_key='{publication['key']}'")
                paper_metadata["title"] = publication["title"]
                paper_metadata["authors"] = publication["authors"]
                paper_metadata["year"] = publication["year"]
                return ResearchToolResult(metadata=metadata, result=paper_metadata)
        return ResearchToolResult(metadata=metadata, result=paper_metadata)

    def get_duplicate_titles(self, publications: List[Dict]) -> List[str]:
        """
        Get duplicate titles from a list of publications.
        Args:
            publications: A list of dictionaries containing the search results.
        Returns:
            A list of duplicate titles.
        """
        title_set = set()
        duplicate_titles = [] 
        for pub in publications:
            title = pub["title"] 
            if title in title_set:
                duplicate_titles.append(title)
            title_set.add(title)
        return duplicate_titles

    def filter_duplicate_publications(self, publications: List[Dict], duplicate_titles: List[str]) -> List[Dict]:
        """
        Filter duplicate publications from a list of publications.
        Args:
            publications: A list of dictionaries containing the search results.
            duplicate_titles: A list of duplicate titles.
        Returns:
            A list of publications with unique titles.
        """
        unique_publications = []
        for pub in publications:
            title = pub["title"] 
            if title not in duplicate_titles:
                unique_publications.append(pub)
            else:
                if pub["publication_source"] != "CoRR": # Remove duplicate publications from arXiv 
                    unique_publications.append(pub)
        return unique_publications

    def get_bibtext(self, key: str) -> str:
        """
        Get the BibTeX entry for a given key.
        Args:
            key: The key of the publication.
        Returns:
            A string containing the BibTeX entry.
        """
        url = f'{self.base_url}/rec/{key}.bib'
        response = self.request(url=url, params={'param': "1"}, headers=DEFAULT_PAGE_HEADERS)
        return response.text

    def polish_bibtext(self, bibtext: str) -> str: 
        """
        Polish the BibTeX entry to remove unnecessary fields and format the entry correctly.
        Args:
            bibtext: A string containing the BibTeX entry.
        Returns:
            A string containing the polished BibTeX entry.
        """
        prompt = DBLP_BIBTEXT_POLISHING_PROMPT + bibtext 
        response = self.llm.generate(prompt=prompt, max_tokens=2048)
        return response.content.strip()

    def search_bibtext(self, title_or_keyword: str) -> ResearchToolResult:
        """
        Search for BibTeX text for a given title or keyword.
        Args:
            title_or_keyword: A single title or keyword to search for.
        Returns:
            A ResearchToolResult containing the BibTeX text for the search results.
        """
        metadata = ResearchToolMetadata(tool_name="dblp_search_bibtext")

        publications = self.search_publications(title_or_keyword)

        # deduplicate publications by title
        duplicate_titles = self.get_duplicate_titles(publications)
        if duplicate_titles:
            publications = self.filter_duplicate_publications(publications, duplicate_titles)
        
        # match the title or keyword with the publication titles using text_match
        match_result = text_match(title_or_keyword, [pub["title"] for pub in publications]) # text_match already handles empty candidates
        if not match_result.get("match", False):
            return ResearchToolResult(metadata=metadata, result={"bibtex": BIB_NOT_FOUND_TEXT})

        # find the BibTeX entry for the most relevant publication
        matched_publication = publications[match_result["best_candidate_metrics"]["index"]]
        bibtext = self.get_bibtext(matched_publication["key"])

        if not bibtext:
            return ResearchToolResult(metadata=metadata, result={"bibtex": BIB_NOT_FOUND_TEXT})
        
        # polish bibtext 
        with track_cost() as cost_tracker:
            bibtext = self.polish_bibtext(bibtext)
            metadata.add_cost_breakdown({"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()})

        result = {"bibtex": bibtext, "title": matched_publication["title"]}
        return ResearchToolResult(metadata=metadata, result=result)

    def _parse_venue_page_for_conferences(self, html: str) -> List[Dict[str, str]]:
        """
        Parse the venue page HTML to extract conference/journal entries with their titles and URLs.

        Args:
            html: The HTML content of the venue page.

        Returns:
            A list of dictionaries containing conference/journal title and URL.
            Format: [{"title": "conference_title", "url": "conference_url"}, ...]
        """
        soup = BeautifulSoup(html, 'html.parser')
        conference_entries = []

        # Find all conference/journal entries
        entries = soup.find_all('li', class_='entry')

        for entry in entries:
            # Extract title from cite element
            cite_elem = entry.find('cite', class_='data')
            if not cite_elem:
                continue

            # Extract the title
            title_elem = cite_elem.find('span', class_='title')
            if title_elem:
                title = title_elem.get_text(strip=True)
            else:
                # Fallback: get text from cite
                title = cite_elem.get_text(strip=True)
                # Clean up the title
                title = re.sub(r'\[contents\]', '', title).strip()

            # Extract the URL to the table of contents
            toc_link = entry.find('a', href=re.compile(r'/db/conf/|/db/journals/'))
            if not toc_link:
                continue

            toc_url = toc_link.get('href')
            if not toc_url.startswith('http'):
                toc_url = f"https://dblp.org{toc_url}"

            conference_entries.append({
                "title": title,
                "url": toc_url
            })

        # If parsing failed, try to clean the HTML and parse again
        if not conference_entries:
            # Remove script, style, and comment elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Try parsing again
            entries = soup.find_all('li', class_='entry')
            for entry in entries:
                cite_elem = entry.find('cite')
                if not cite_elem:
                    continue

                title_elem = cite_elem.find('span', class_='title')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                else:
                    title = cite_elem.get_text(strip=True)
                    title = re.sub(r'\[contents\]', '', title).strip()

                toc_link = entry.find('a', href=re.compile(r'/db/'))
                if not toc_link:
                    continue

                toc_url = toc_link.get('href')
                if not toc_url.startswith('http'):
                    toc_url = f"https://dblp.org{toc_url}"

                conference_entries.append({
                    "title": title,
                    "url": toc_url
                })

        return conference_entries

    def _parse_publication_page_for_papers(self, html: str) -> List[Dict[str, Any]]:
        """
        Parse the publication page HTML to extract paper information.
        Uses selectolax for fast parsing of large HTML documents.

        Args:
            html: The HTML content of the publication page.

        Returns:
            A list of dictionaries containing paper title and authors.
            Format: [{"title": "paper_title", "authors": ["author1", "author2", ...]}, ...]
        """
        tree = HTMLParser(html)
        papers = []

        # Find all publication entries using CSS selector
        for pub_entry in tree.css('li.entry'):
            cite_elem = pub_entry.css_first('cite.data')
            if not cite_elem:
                continue

            # Extract title
            title_elem = cite_elem.css_first('span.title')
            if not title_elem:
                continue

            paper_title = title_elem.text(strip=True)

            # Extract authors
            authors = []
            for author_elem in cite_elem.css('span[itemprop="author"]'):
                author_name_elem = author_elem.css_first('span[itemprop="name"]')
                if author_name_elem:
                    author_name = author_name_elem.text(strip=True)
                    # Clean author name
                    author_name = self._clean_author_name(author_name)
                    authors.append(author_name)

            if paper_title and authors:
                papers.append({
                    "title": paper_title,
                    "authors": authors
                })

        return papers

    def search_venue_publications(self, venue: str, year_from: Optional[int] = None, year_to: Optional[int] = None) -> ResearchToolResult:
        """
        Search for publications in a specific venue (conference or journal) on DBLP.

        Args:
            venue: The name of the venue (conference or journal) to search for.
            year_from: The starting year for filtering conferences/journals (inclusive). If None, no lower bound.
            year_to: The ending year for filtering conferences/journals (inclusive). If None, no upper bound.

        Returns:
            A dictionary where keys are conference titles and values are lists of papers.
            Format: {
                "conference_title": [
                    {"title": "paper_title", "authors": ["author1", "author2", ...]},
                    ...
                ]
            }
        """
        metadata = ResearchToolMetadata(tool_name="search_venue_publications")
        if self.enable_cache:
            self._ensure_conference_page_cache()

        # Step 1: Search for venues using DBLP API
        venue_search_url = "https://dblp.org/search/venue/api"
        params = {"q": venue, "format": "json", "h": 10}

        try:
            response = self.request(url=venue_search_url, method='GET', params=params)
            venue_data = response.json()
        except Exception as e:
            logger.error(f"Failed to search for venue '{venue}': {e}")
            return ResearchToolResult(result={}, metadata=metadata)
        
        # Check if any venues were found
        hits = venue_data.get("result", {}).get("hits", {})

        if hits.get("@total", "0") == "0" or "hit" not in hits:
            logger.warning(f"No venues found for query: {venue}")
            return ResearchToolResult(result={}, metadata=metadata)

        # Step 2: Use LLM to identify the matching venue
        hit_items = hits["hit"]
        if not isinstance(hit_items, list):
            hit_items = [hit_items]

        # Prepare venue options for LLM
        venue_options = []
        for i, hit in enumerate(hit_items):
            info = hit.get("info", {})
            venue_options.append({
                "index": i,
                "name": info.get("venue", "Unknown Venue"),
                "acronym": info.get("acronym", ""),
                "type": info.get("type", ""),
                "url": info.get("url", "")
            })

        # Create prompt for LLM to select the best matching venue
        venue_list_str = "\n".join([
            f"{i+1}. Name: {v['name']}, Acronym: {v['acronym']}, Type: {v['type']}, URL: {v['url']}"
            for i, v in enumerate(venue_options)
        ])
        llm_prompt = DBLP_VENUE_SELECTION_PROMPT.format(venue=venue, venue_list=venue_list_str)

        try:
            with track_cost() as cost_tracker:
                llm_response = self.llm.generate(prompt=llm_prompt, max_tokens=500)
                metadata.add_cost_breakdown({"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()})
            venue_selection = parse_json_from_llm_output(llm_response.content)
            if not venue_selection:
                # LLM return empty json, indicating that no matching conference found
                return ResearchToolResult(result={}, metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to get LLM response for venue selection: {e}")
            # Fallback to first venue
            venue_selection = {
                "selected_index": 1,
                "venue_name": venue_options[0]["name"],
                "url": venue_options[0]["url"]
            }

        selected_venue_url = venue_selection.get("url") or venue_options[0].get("url")
        if not selected_venue_url:
            logger.warning(f"No valid venue URL found for query: {venue}")
            return ResearchToolResult(result={}, metadata=metadata)

        # Step 3: Parse the venue page to get conferences/journals by year
        try:
            venue_page_response = self.request(url=selected_venue_url, headers=DEFAULT_PAGE_HEADERS)
            venue_html = venue_page_response.text
        except Exception as e:
            logger.error(f"Failed to fetch venue page {selected_venue_url}: {e}")
            return ResearchToolResult(result={}, metadata=metadata)

        # Parse HTML to extract conference/journal entries
        conference_entries = self._parse_venue_page_for_conferences(venue_html)

        if not conference_entries:
            logger.warning(f"No conference entries found on venue page: {selected_venue_url}")
            return ResearchToolResult(result={}, metadata=metadata)

        # Step 3b: Use LLM to decide which conferences/journals to fetch based on year_from and year_to
        if year_from is None and year_to is None:
            # Default: return only the latest conference (first entry)
            selected_conferences = [conference_entries[0]]
        else:
            # Build conference list string for LLM prompt
            conference_list_str = "\n".join([
                f"{i+1}. Title: {conf['title']}, URL: {conf['url']}"
                for i, conf in enumerate(conference_entries)
            ])

            # Build year constraint
            if year_from is not None and year_to is not None:
                year_constraint = f"between {year_from} and {year_to} (inclusive)"
            elif year_from is not None:
                year_constraint = f"from {year_from} onwards"
            else:  # year_to is not None and year_from is None
                year_constraint = f"up to {year_to}"

            year_filter_prompt = DBLP_YEAR_FILTER_RANGE_PROMPT.format(
                conference_list=conference_list_str,
                year_constraint=year_constraint
            )
            try:
                with track_cost() as cost_tracker:
                    year_filter_response = self.llm.generate(prompt=year_filter_prompt, max_tokens=500)
                    metadata.add_cost_breakdown({"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()})
                year_filter_result = parse_json_from_llm_output(year_filter_response.content)
                if not year_filter_result or not year_filter_result.get("selected_indices"):
                    # LLM couldn't determine matches, fallback to latest conference
                    logger.warning("LLM returned empty result for year filtering, falling back to latest conference")
                    selected_indices = [1]
                else:
                    selected_indices = year_filter_result.get("selected_indices", [1])
                    # Limit results: max 5 if only year_to specified, max 10 otherwise
                    max_limit = 5 if year_from is None else 10
                    selected_indices = selected_indices[:max_limit]
            except Exception as e:
                logger.error(f"Failed to get LLM response for year filtering: {e}")
                selected_indices = [1]
            selected_conferences = [conference_entries[i-1] for i in selected_indices if 0 < i <= len(conference_entries)]

            # Check if any valid conferences were selected
            if not selected_conferences:
                logger.warning("No valid conferences matched the year filter criteria")
                return ResearchToolResult(result={}, metadata=metadata)

        # Step 4: Fetch publications from each selected conference using streaming
        result = {}
        for conf in selected_conferences:
            conf_title = conf["title"]
            conf_url = conf["url"]

            if self.enable_cache:
                cached_papers, hit = self._cache_get(conf_url)
                if hit:
                    if cached_papers:
                        result[conf_title] = cached_papers
                    continue

            try:
                # Use streaming request for large HTML pages
                conf_html = self.request_stream(url=conf_url, headers=DEFAULT_PAGE_HEADERS)
            except Exception as e:
                logger.error(f"Failed to fetch conference page {conf_url}: {e}")
                continue

            # Parse the conference page to extract papers (using selectolax)
            papers = self._parse_publication_page_for_papers(conf_html)
            if self.enable_cache:
                self._cache_set(papers, conf_url)
            if papers:
                result[conf_title] = papers

        return ResearchToolResult(result=result, metadata=metadata) 
    
    def search_venue_specific_publications(self, venue: str, title_or_keyword: Optional[str] = None, topk: Optional[int] = 10, year_from: Optional[int] = None, year_to: Optional[int] = None) -> ResearchToolResult:
        """
        Search for specific publications in a venue matching a title or keyword.

        This function first retrieves all publications from a venue using search_venue_publications,
        then uses BM25 to find the top-k most relevant papers matching the query.

        Args:
            venue: The name of the venue (conference or journal) to search for.
            title_or_keyword: The title or keyword to search for within the venue's publications.
                If None, returns all publications from the venue (up to topk).
            topk: The maximum number of papers to return. Defaults to 10.
            year_from: The starting year for filtering conferences/journals (inclusive). If None, no lower bound.
            year_to: The ending year for filtering conferences/journals (inclusive). If None, no upper bound.

        Returns:
            ResearchToolResult containing a list of matching papers with their titles and authors.
        """
        metadata = ResearchToolMetadata(tool_name="search_venue_specific_publications")

        # Step 1: Get all publications from the venue
        venue_result = self.search_venue_publications(venue=venue, year_from=year_from, year_to=year_to)
        metadata.add_cost_breakdown(venue_result.metadata.cost_breakdown)

        if not venue_result.result:
            logger.warning(f"No publications found for venue: {venue}")
            return ResearchToolResult(result=[], metadata=metadata)

        # Step 2: Flatten all papers from all conference years into a single list
        all_papers = []
        for conf_title, papers in venue_result.result.items():
            for paper in papers:
                # Add conference info to each paper for reference
                paper_with_conf = paper.copy()
                paper_with_conf["conference"] = conf_title
                all_papers.append(paper_with_conf)

        if not all_papers:
            logger.warning(f"No papers found in venue: {venue}")
            return ResearchToolResult(result=[], metadata=metadata)

        # Step 3: If no query provided, return top-k papers (most recent first based on order)
        if not title_or_keyword:
            result_papers = all_papers[:topk] if topk else all_papers
            return ResearchToolResult(result=result_papers, metadata=metadata)

        # Step 4: Use BM25 to rank papers by relevance to the query
        # Build corpus from paper titles
        corpus = [paper["title"] for paper in all_papers]

        try:
            bm25 = BM25(corpus=corpus)
            ranked_results = bm25.rank(query=title_or_keyword, topk=topk)
        except ValueError as e:
            logger.warning(f"BM25 ranking failed: {e}")
            return ResearchToolResult(result=[], metadata=metadata)
        
        # Step 5: Build result list with paper info and BM25 scores
        result_papers = []
        for idx, score, _ in ranked_results:
            paper_info = all_papers[idx].copy()
            paper_info["bm25_score"] = score
            result_papers.append(paper_info)

        return ResearchToolResult(result=result_papers, metadata=metadata)


class SemanticScholar(RequestBase):

    def __init__(self, semantic_scholar_api_key: Optional[str] = None, **kwargs):
        super().__init__(timeout=10, max_retries=1) 
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = semantic_scholar_api_key 

        self.headers = DEFAULT_PAGE_HEADERS.copy() 
        if self.api_key:
            self.headers["x-api-key"] = self.api_key 

    def get_metadata_based_on_doi(self, doi: str) -> ResearchToolResult:
        """
        Get the metadata of a paper based on the DOI.
        Args:
            doi: The DOI of the paper.
        Returns:
            A dictionary containing the metadata of the paper.
        """
        metadata = ResearchToolMetadata(tool_name="semantic_scholar_get_metadata_based_on_doi")
        paper_metadata = {}

        doi = doi.strip()
        if not doi:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        doi_enc = urllib.parse.quote(doi, safe="")
        url = (
            f"{self.base_url}/paper/DOI:{doi_enc}"
            "?fields=title,abstract,authors,year,publicationVenue,"
            "externalIds,citationCount,referenceCount,url,isOpenAccess,openAccessPdf"
        )

        try:
            response = self.request(url=url, headers=self.headers)
        except Exception as e:
            logger.warning(f"Failed to get metadata from the DOI ({doi}): {e}")
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        data = response.json()
        paper_metadata["title"] = data.get("title", "Unknown Title")
        paper_metadata["authors"] = [author.get("name", "Unknown Author") for author in data.get("authors", [])]
        paper_metadata["abstract"] = data.get("abstract", None) 

        return ResearchToolResult(metadata=metadata, result=paper_metadata)

    def get_metadata_based_on_title(self, title: str) -> ResearchToolResult:
        """
        Get the metadata of a paper based on the title.
        Args:
            title: The title of the paper.
        Returns:
            A dictionary containing the metadata of the paper, including:
            title, authors, abstract, year, paper_link, pdf_link (if available).
        """
        metadata = ResearchToolMetadata(tool_name="semantic_scholar_get_metadata_based_on_title")
        paper_metadata = {}

        title = title.strip()
        if not title:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        try:
            # Request top 5 results
            params = {
                "query": title,
                "limit": "5",
                "fields": "title,abstract,authors,year,publicationVenue,externalIds,citationCount,referenceCount,url,isOpenAccess,openAccessPdf"
            }
            
            response = self.request(
                url=f"{self.base_url}/paper/search",
                params=params,
                headers=self.headers
            )
            
            data = response.json() or {}
            papers = data.get("data", [])
            
            if not papers:
                return ResearchToolResult(metadata=metadata, result=paper_metadata)
            
            # Iterate through results and find the best match
            for paper in papers:
                if not paper:
                    continue
                    
                paper_title = paper.get("title", "").strip()
                if not paper_title:
                    continue
                
                # Use text_match to verify title match
                match_result = text_match(query=title, text_candidates=[paper_title])
                
                if match_result["match"]:
                    # Found a matching paper
                    logger.info(f"Semantic Scholar title matched: query='{title}', fetched_title='{paper_title}'")
                    
                    paper_metadata["title"] = paper.get("title", "Unknown Title")
                    paper_metadata["authors"] = [
                        author.get("name", "Unknown Author") 
                        for author in paper.get("authors", [])
                    ]
                    paper_metadata["abstract"] = paper.get("abstract", None)
                    paper_metadata["year"] = paper.get("year", None)
                    
                    # Extract paper link
                    paper_url = paper.get("url", None)
                    if paper_url:
                        paper_metadata["paper_link"] = paper_url
                    
                    # Extract PDF link if available
                    open_access_pdf = paper.get("openAccessPdf", {})
                    if open_access_pdf and isinstance(open_access_pdf, dict):
                        pdf_url = open_access_pdf.get("url", None)
                        if pdf_url:
                            paper_metadata["pdf_link"] = pdf_url
                    
                    return ResearchToolResult(metadata=metadata, result=paper_metadata)
            
            # No matching paper found
            logger.info(f"No matching paper found in Semantic Scholar for title: '{title}'")
        
        except Exception as e:
            logger.info(f"Failed to get metadata from Semantic Scholar for title ({title}): {e}")
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        return ResearchToolResult(metadata=metadata, result=paper_metadata)


class PubMed(RequestBase):

    def __init__(self, pubmed_api_key: Optional[str] = None, **kwargs):
        super().__init__(timeout=10, max_retries=1)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.headers = DEFAULT_PAGE_HEADERS.copy()
        self.api_key = pubmed_api_key  

    def _parse_pubmed_xml(self, xml_text: str, pmid: str) -> dict:
        """
        Parse PubMed XML response and extract metadata.
        Args:
            xml_text: The XML response text from PubMed efetch API.
            pmid: The PubMed ID.
        Returns:
            A dictionary containing the extracted metadata.
        """
        paper_metadata = {}
        
        # Parse XML response
        soup = BeautifulSoup(xml_text, 'xml')
        
        # Extract title
        title_elem = soup.find('ArticleTitle')
        if title_elem:
            paper_metadata["title"] = title_elem.get_text(strip=True)
        
        # Extract authors
        authors = []
        author_list = soup.find('AuthorList')
        if author_list:
            for author in author_list.find_all('Author'):
                last_name = author.find('LastName')
                first_name = author.find('ForeName')
                if last_name and first_name:
                    authors.append(f"{first_name.get_text(strip=True)} {last_name.get_text(strip=True)}")
                elif last_name:
                    authors.append(last_name.get_text(strip=True))
        paper_metadata["authors"] = authors if authors else []
        
        # Extract abstract (may have multiple AbstractText elements)
        abstract_parts = []
        for abstract_elem in soup.find_all('AbstractText'):
            text = abstract_elem.get_text(strip=True)
            if text:
                label = abstract_elem.get('Label', '')
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        if abstract_parts:
            paper_metadata["abstract"] = " ".join(abstract_parts)
        
        # Extract year
        pub_date = soup.find('PubDate')
        if pub_date:
            year_elem = pub_date.find('Year')
            if year_elem:
                paper_metadata["year"] = year_elem.get_text(strip=True)
        
        # Extract paper link (PubMed URL)
        paper_metadata["paper_link"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        
        # Extract PDF link (if available from PMC)
        pmc_id_elem = soup.find('ArticleId', IdType="pmc")
        if pmc_id_elem:
            pmc_id = pmc_id_elem.get_text(strip=True)
            paper_metadata["pdf_link"] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
        
        return paper_metadata

    def get_metadata_based_on_title(self, title: str) -> ResearchToolResult:
        """
        Get the metadata of a paper based on the title from PubMed.
        Args:
            title: The title of the paper.
        Returns:
            A dictionary containing the metadata of the paper, including:
            title, authors, abstract, year, paper_link, pdf_link (if available).
        """
        metadata = ResearchToolMetadata(tool_name="pubmed_get_metadata_based_on_title")
        paper_metadata = {}

        title = title.strip()
        if not title:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        try:
            # Step 1: Search for PMIDs using title
            params = {
                "db": "pubmed",
                "term": f'{title}[ti]',
                "retmode": "json",
                "retmax": "1", 
            }
            if self.api_key:
                params["api_key"] = self.api_key  
            response = self.request(url=f"{self.base_url}/esearch.fcgi", params=params, headers=self.headers)
            search_data = response.json()
            
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return ResearchToolResult(metadata=metadata, result=paper_metadata)
            
            # Step 2: Iterate through PMIDs to find the best match
            for pmid in id_list:
                try:
                    # Use esummary to quickly get title without fetching full XML
                    params = {
                        "db": "pubmed",
                        "id": pmid,
                        "retmode": "json",
                    }
                    if self.api_key:
                        params["api_key"] = self.api_key  
                    summary_response = self.request(url=f"{self.base_url}/esummary.fcgi", params=params, headers=self.headers)
                    summary_data = summary_response.json()
                    
                    # Extract title from summary
                    result = summary_data.get("result", {}).get(pmid, {})
                    fetched_title = result.get("title", "")
                    
                    if not fetched_title:
                        continue
                    
                    # Use text_match to verify title match
                    match_result = text_match(query=title, text_candidates=[fetched_title])

                    if match_result["match"]:
                        # Found a matching paper, now fetch full metadata
                        logger.info(f"PubMed title matched: query='{title}', fetched_title='{fetched_title}', PMID={pmid}")
                        fetch_url = f"{self.base_url}/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
                        fetch_response = self.request(url=fetch_url, headers=self.headers)
                        paper_metadata = self._parse_pubmed_xml(fetch_response.text, pmid)
                        return ResearchToolResult(metadata=metadata, result=paper_metadata)
                    
                except Exception as e:
                    logger.info(f"Failed to fetch/parse PMID {pmid}: {e}")
                    continue
            
            # No matching paper found
            logger.info(f"No matching paper found in PubMed for title: '{title}'")
            
        except Exception as e:
            logger.info(f"Failed to get metadata from PubMed for title ({title}): {e}")
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        return ResearchToolResult(metadata=metadata, result=paper_metadata)


class Crossref(RequestBase):

    def __init__(self, **kwargs):
        super().__init__(timeout=10, max_retries=1)
        self.base_url = "https://api.crossref.org"
    
    def get_metadata_based_on_doi(self, doi: str) -> ResearchToolResult:
        """
        Get the metadata of a paper based on the DOI.
        Args:
            doi: The DOI of the paper.
        Returns:
            A dictionary containing the metadata of the paper.
        """
        metadata = ResearchToolMetadata(tool_name="crossref_get_metadata_based_on_doi")
        paper_metadata = {}
        
        doi = doi.strip()
        if not doi:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        doi_enc = urllib.parse.quote(doi, safe="")
        url = f"{self.base_url}/works/{doi_enc}"
        try:
            response = self.request(url=url, headers=DEFAULT_PAGE_HEADERS)
        except Exception as e:
            logger.warning(f"Failed to get metadata from Crossref for DOI ({doi}): {e}")
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        try:
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to parse Crossref response for DOI ({doi}): {e}")
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        message = data.get("message") or {}
        if not isinstance(message, dict):
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        titles = message.get("title") or []
        paper_metadata["title"] = titles[0] if titles else "Unknown Title"
        authors = message.get("author") or []
        author_names: list[str] = []
        for a in authors:
            given = a.get("given") or ""
            family = a.get("family") or ""
            full_name = (given + " " + family).strip()
            if not full_name:
                full_name = a.get("name") or "Unknown Author"
            author_names.append(full_name)
        paper_metadata["authors"] = author_names 

        abstract = message.get("abstract")
        paper_metadata["abstract"] = abstract

        return ResearchToolResult(metadata=metadata, result=paper_metadata)
    
    def get_metadata_based_on_title(self, title: str) -> ResearchToolResult:
        """
        Get the metadata of a paper based on the title.
        Args:
            title: The title of the paper.
        Returns:
            A dictionary containing the metadata of the paper, including:
            title, authors, abstract, year, paper_link, pdf_link (if available).
        """
        metadata = ResearchToolMetadata(tool_name="crossref_get_metadata_based_on_title")
        paper_metadata = {}

        title = title.strip()
        if not title:
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        try:
            # Request top 5 results
            params = {
                "query.title": title,
                "rows": "5",
                "select": "title,author,abstract,DOI,URL,published,link"
            }
            
            response = self.request(
                url=f"{self.base_url}/works",
                params=params,
                headers=DEFAULT_PAGE_HEADERS
            )
            
            data = response.json()
            message = data.get("message", {})
            items = message.get("items", [])
            
            if not items:
                return ResearchToolResult(metadata=metadata, result=paper_metadata)
            
            # Iterate through results and find the best match
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                # Extract title
                titles = item.get("title") or []
                if not titles:
                    continue
                
                item_title = titles[0].strip() if isinstance(titles, list) else str(titles).strip()
                if not item_title:
                    continue
                
                # Use text_match to verify title match
                match_result = text_match(query=title, text_candidates=[item_title])
                
                if match_result["match"]:
                    # Found a matching paper
                    logger.info(f"Crossref title matched: query='{title}', fetched_title='{item_title}'")
                    
                    paper_metadata["title"] = item_title
                    
                    # Extract authors
                    authors = item.get("author") or []
                    author_names = []
                    for a in authors:
                        given = a.get("given") or ""
                        family = a.get("family") or ""
                        full_name = (given + " " + family).strip()
                        if not full_name:
                            full_name = a.get("name") or "Unknown Author"
                        author_names.append(full_name)
                    paper_metadata["authors"] = author_names
                    
                    # Extract abstract
                    abstract = item.get("abstract")
                    if abstract:
                        paper_metadata["abstract"] = abstract
                    
                    # Extract year
                    published = item.get("published", {}) or item.get("published-print", {}) or item.get("published-online", {})
                    if published and isinstance(published, dict):
                        date_parts = published.get("date-parts", [])
                        if date_parts and date_parts[0]:
                            year = date_parts[0][0] if len(date_parts[0]) > 0 else None
                            if year:
                                paper_metadata["year"] = str(year)
                    
                    # Extract paper link
                    doi = item.get("DOI")
                    if doi:
                        paper_metadata["paper_link"] = f"https://doi.org/{doi}"
                    else:
                        url = item.get("URL")
                        if url:
                            paper_metadata["paper_link"] = url
                    
                    # Extract PDF link (if available from link array)
                    links = item.get("link", [])
                    for link in links:
                        if isinstance(link, dict):
                            content_type = link.get("content-type", "")
                            if "pdf" in content_type.lower():
                                pdf_url = link.get("URL")
                                if pdf_url:
                                    paper_metadata["pdf_link"] = pdf_url
                                    break
                    
                    return ResearchToolResult(metadata=metadata, result=paper_metadata)
            
            # No matching paper found
            logger.info(f"No matching paper found in Crossref for title: '{title}'")
        
        except Exception as e:
            logger.info(f"Failed to get metadata from Crossref for title ({title}): {e}")
            return ResearchToolResult(metadata=metadata, result=paper_metadata)

        return ResearchToolResult(metadata=metadata, result=paper_metadata)


def get_metadata_based_on_doi(doi: str, title: str = None, semantic_scholar_api_key: Optional[str] = None) -> ResearchToolResult:
    """
    Get the metadata of a paper based on the DOI.
    Args:
        doi: The DOI of the paper.
    Returns:
        A dictionary containing the metadata of the paper.
    """
    metadata = ResearchToolMetadata(tool_name="get_metadata_based_on_doi")
    paper_metadata = {}

    # Case 1: Can use Semantic Scholar to get the metadata if the title is provided
    if title:
        semantic_scholar = SemanticScholar(semantic_scholar_api_key=semantic_scholar_api_key) 
        semantic_scholar_metadata_results = semantic_scholar.get_metadata_based_on_title(title)
        paper_metadata = semantic_scholar_metadata_results.result
        if paper_metadata and isinstance(paper_metadata, dict) and paper_metadata.get("authors", None):
            # propagate inner tool cost
            metadata.add_cost_breakdown(semantic_scholar_metadata_results.metadata.cost_breakdown)
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
    
    # Case 2: Use Crossref to get the metadata if the title is not provided
    crossref = Crossref()
    crossref_metadata_results = crossref.get_metadata_based_on_doi(doi)
    paper_metadata = crossref_metadata_results.result
    if paper_metadata and isinstance(paper_metadata, dict) and paper_metadata.get("authors", None):
        metadata.add_cost_breakdown(crossref_metadata_results.metadata.cost_breakdown)
        return ResearchToolResult(metadata=metadata, result=paper_metadata)

    # no provider returned valid metadata; cost stays as accumulated so far (likely 0)
    return ResearchToolResult(metadata=metadata, result=paper_metadata)


def search_authors_from_databases(
    paper_title: str,
    gs_authors: Optional[List[str]] = None,
    pubmed_api_key: Optional[str] = None,
    semantic_scholar_api_key: Optional[str] = None,
) -> Tuple[Optional[List[str]], Dict[str, float]]:
    """
    Search for paper author information from open-source databases: Arxiv, DBLP, PubMed, Semantic Scholar, Crossref.
    
    Args:
        paper_title: The title of the paper.
        gs_authors: Optional list of author last names from Google Scholar for validation.
        pubmed_api_key: Optional API key for PubMed.
        semantic_scholar_api_key: Optional API key for Semantic Scholar.
        
    Returns:
        A tuple of (authors list, cost breakdown dict). Authors is None if not found.
    """
    cost_breakdown: Dict[str, float] = {}
    authors = None
    
    pubmed_database = PubMed(pubmed_api_key=pubmed_api_key) if pubmed_api_key else PubMed()
    semantic_scholar_database = SemanticScholar(semantic_scholar_api_key=semantic_scholar_api_key) if semantic_scholar_api_key else SemanticScholar()
    
    opensource_databases = [Arxiv(), DBLP(), pubmed_database, Crossref(), semantic_scholar_database]
    for database in opensource_databases:
        database_results: ResearchToolResult = database.get_metadata_based_on_title(paper_title)
        paper_metadata = database_results.result
        if paper_metadata:
            authors = paper_metadata["authors"]
            cost_breakdown = add_dict(cost_breakdown, database_results.metadata.cost_breakdown)
            if gs_authors:
                # if gs_authors is provided, validate the author information
                if validate_author_info(authors, gs_authors):
                    return authors, cost_breakdown
            else:
                # if gs_authors is not provided, return the author information from the database
                return authors, cost_breakdown
    
    return authors, cost_breakdown


def search_paper_author_info(
    gs_title: str, 
    gs_link: str, 
    serpapi_key: str,
    gs_authors: Optional[List[str]] = None, 
    pubmed_api_key: Optional[str] = None, 
    semantic_scholar_api_key: Optional[str] = None,  
) -> Tuple[List[str], Dict[str, float]]: 

    """
    Search for paper author information from Google Scholar and SerpAPI.
    Args:
        gs_title: The title of the paper on Google Scholar.
        gs_link: The link of the paper on Google Scholar.
        serpapi_key: The API key for SerpAPI.
    Returns:
        A list of author names, or None if not found. 
    """

    serp = SerpAPIForResearch(serpapi_key=serpapi_key)
    paper_title = gs_title # default to the title on Google Scholar  
    paper_link = gs_link # default to the link on Google Scholar  
    authors = None # default to None  
    cost_breakdown: Dict[str, float] = {}

    # Step 1: Search for paper info from Google Search (if the paper is not from Arxiv)
    if "arxiv" not in paper_link.lower():
        google_search_results = serp.google_search(query=gs_title, topk=5)
        google_search_papers_info = google_search_results.result
        cost_breakdown = add_dict(cost_breakdown, google_search_results.metadata.cost_breakdown)
        for paper_info in google_search_papers_info:
            match_result = text_match(gs_title, [paper_info["title"]])
            if match_result["match"]:
                paper_link = paper_info["link"]
                break 
        
    # NOTE: paper_link can be paper page or pdf link 
    # Step 2: If the paper is from Arxiv, get the metadata from Arxiv based on the arxiv_id 
    if paper_link and "arxiv" in paper_link.lower():
        # handles both arxiv paper page and pdf link 
        arxiv = Arxiv() 
        arxiv_id = extract_arxiv_id(paper_link)
        if arxiv_id:
            arxiv_results: ResearchToolResult = arxiv.get_metadata_based_on_arxiv_id(arxiv_id)
            arxiv_paper_metadata = arxiv_results.result
            if arxiv_paper_metadata:
                authors = arxiv_paper_metadata["authors"]
                cost_breakdown = add_dict(cost_breakdown, arxiv_results.metadata.cost_breakdown)
                return authors, cost_breakdown 
    
    # Step 3: Search for paper info from OpenSource databases: Arxiv, DBLP, PubMed, Semantic Scholar, Crossref based on the paper title 
    authors, db_cost_breakdown = search_authors_from_databases(
        paper_title=paper_title,
        gs_authors=gs_authors,
        pubmed_api_key=pubmed_api_key,
        semantic_scholar_api_key=semantic_scholar_api_key,
    )
    cost_breakdown = add_dict(cost_breakdown, db_cost_breakdown)

    return authors, cost_breakdown


def search_paper_title_based_on_doi(doi: str, semantic_scholar_api_key: Optional[str] = None) -> str: 

    title = None

    crossref = Crossref()
    crossref_results: ResearchToolResult = crossref.get_metadata_based_on_doi(doi)
    paper_metadata = crossref_results.result
    if paper_metadata and paper_metadata["title"].lower() != "unknown title":
        return paper_metadata["title"]

    semantic_scholar = SemanticScholar(semantic_scholar_api_key=semantic_scholar_api_key)
    semantic_scholar_results: ResearchToolResult = semantic_scholar.get_metadata_based_on_doi(doi)
    paper_metadata = semantic_scholar_results.result
    if paper_metadata and paper_metadata["title"].lower() != "unknown title":
        return paper_metadata["title"]

    return title 


def resolve_paper_title(query: str, semantic_scholar_api_key: Optional[str] = None) -> str: 
    """
    Try to resolve a paper title from a query string.

    The query may be:
      - a paper title (possibly with extra info: venue/year/authors)
      - a DOI (e.g., 10.1145/3366423.3380211) or a DOI URL
      - an arXiv ID (e.g., 1706.03762, arxiv:2301.12345v2, cs.CL/9901001) or arXiv URL
      - a mixed citation snippet
    """

    _ARXIV_RE = re.compile(
        r"\b(?:arxiv:)?("
        r"(?:\d{4}\.\d{4,5})(?:v\d+)?"
        r"|"
        r"(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
        r")\b",
        re.IGNORECASE,
    )

    _DOI_RE_STRICT = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
    _DOI_RE_LOOSE = DOI_REGEX 

    match = _ARXIV_RE.search(query) 
    if match:
        arxiv = Arxiv() 
        arxiv_id = match.group(1)
        try:
            arxiv_results: ResearchToolResult = arxiv.get_metadata_based_on_arxiv_id(arxiv_id) 
            paper_metadata = arxiv_results.result
            if paper_metadata and paper_metadata["title"].lower() != "unknown title":
                return paper_metadata["title"]
        except Exception as e: 
            logger.info(f"Failed to get metadata from Arxiv for arxiv_id ({arxiv_id}): {e}. Falling back to original query.")

    doi = None 
    match = _DOI_RE_STRICT.search(query)
    if match:
        doi = match.group(1)
    if not doi:
        match = _DOI_RE_LOOSE.search(query)
        if match:
            doi = match.group(1)
    
    if doi:
        try:
            title = search_paper_title_based_on_doi(doi=doi, semantic_scholar_api_key=semantic_scholar_api_key)
            if title:
                return title  # successfully found the title 
        except Exception as e: 
            logger.info(f"Failed to get metadata based on DOI ({doi}): {e}. Falling back to original query.")
    
    return query  # no title found
