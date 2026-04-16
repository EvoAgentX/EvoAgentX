import re
import requests
import unicodedata
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse, unquote
from typing import List, Dict, Optional, Tuple

from ...core.logging import logger
from ...core.module_utils import parse_json_from_llm_output 
from ...models.model_utils import track_cost 
from ...models import BaseLLM
from ...utils.utils import add_dict

from .prompts import (
    PAPER_METADATA_EXTRACTION_PROMPT, 
    PAPER_METADATA_EXTRACTION_PROMPT_NO_ABSTRACT,
    QUERY_EXPANSION_FOR_RECALL_PROMPT, 
    MATCH_CITATIONS_PROMPT, 
    SEMINAL_PAPERS_FOR_TOPIC_PROMPT
)
from .metadata import ResearchToolMetadata, ResearchToolResult, PaperMetadataOutputParser


DEFAULT_PAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
}

DBLP_LLM_MODEL = "google/gemini-2.5-flash" # "openai/gpt-4o"
PAPER_SEARCH_LLM_MODEL = "google/gemini-2.5-flash" # "openai/gpt-4o" 
DOI_REGEX = re.compile(r'10\.\d{4,9}/[^\s"<>?#]+', re.IGNORECASE)

def normalize_title(title: str) -> str: 
    # remove extra space and - , suitable for search query. It will retain punctuation and special characters. 
    if not title:
        return "" # empty string if the title is None or empty 
    title = title.lower()
    title = title.replace('-', ' ')
    return re.sub(r'  +', ' ', title).strip()

def normalize_title_aggressively(title: str) -> str: 
    # remove all punctuation and special characters, suitable for file name or deduplication. 
    if not title: 
        return "" 
    title = title.lower() 
    title = re.sub(r'[^\w\s]+', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title 


def extract_year(text: str) -> str:

    """
    Extract the year from the summary of Google Scholar search result. 
    Args:
        summary: A string containing the summary of the publication, e.g., "Y Gao, Y Xiong, X Gao, K Jia, J Pan, Y Bi… - arXiv preprint arXiv …, 2023 - simg.baai.ac.cn". 
    Returns:
        A string containing the year of the publication.
    """
    match = re.search(r'\b(19|20)\d{2}\b', text)
    return match.group(0) if match else "" 

def extract_year_from_arxiv_dates(dates: str) -> str:
    """
    Extract the year from the published and updated dates of the arXiv paper.
    Args:
        dates: A string containing the published and updated dates of the arXiv paper, e.g., "2025-03-19 16:00:00".
    Returns:
        A string containing the year of the publication.
    """
    if not dates:
        return None 
    
    # find all four-digit years (conservative: 19xx or 20xx)
    # use non-capturing group (?:...) so findall returns the full match
    years = re.findall(r"\b(?:19|20)\d{2}\b", dates)
    if not years:
        return ""

    # take the earliest year (v1 release time usually appears first)
    return years[0]

def extract_arxiv_id(url: str) -> str:
    """
    Extract the arXiv ID from the arXiv link.
    Args:
        url: The link of the arXiv paper.
    Returns:
        The arXiv ID.
    
    Examples:
    https://arxiv.org/abs/2503.19470      → 2503.19470
    https://arxiv.org/abs/2503.19470v2    → 2503.19470v2
    https://arxiv.org/pdf/2503.19470.pdf  → 2503.19470
    https://arxiv.org/pdf/2503.19470v3.pdf → 2503.19470v3
    """

    # new-style  ID: 2503.19470  / 2503.19470v2
    match = re.search(r'arxiv\.org/(abs|pdf)/([0-9]{4}\.[0-9]{4,5}(v[0-9]+)?)', url, re.IGNORECASE)
    if match:
        return match.group(2)

    # old-style ID: cs/0112017
    match = re.search(r'arxiv\.org/(abs|pdf)/([a-z\-]+\/[0-9]{7})(v[0-9]+)?', url, re.IGNORECASE)
    if match:
        return match.group(2)

    return None

def _clean_doi(raw: str) -> str:
    """Clean the DOI string."""
    if not raw:
        return None
    doi = raw.strip()
    # remove the prefix doi:
    doi = re.sub(r'^(doi:|DOI:)\s*', '', doi)
    # if there are obvious parameters/fragment separators, truncate the DOI
    for sep in ['?', "&", "#", "%"]:
        if sep in doi:
            doi = doi.split(sep, 1)[0]
    # remove the common extra characters 
    doi = doi.rstrip(').,; ')
    return doi or None

def _extract_doi_from_text(text: str) -> str:
    """Extract the DOI from the text."""
    if not text:
        return None
    m = DOI_REGEX.search(text)
    if m:
        return _clean_doi(m.group(0))
    return None

def _extract_doi_from_known_path(parsed) -> str:
    """
    For some common domains, try to extract the DOI directly from the path.
    Here we only do the "looks like DOI" extraction, not make any guesses.
    """
    host = parsed.netloc.lower()
    path = parsed.path

    # Case 1: Directly extract from the path
    if host.endswith("doi.org") or host.endswith("dx.doi.org"):
        return _clean_doi(path.lstrip("/"))

    # Case 2: Springer link: /article/<doi> 或 /chapter/<doi>
    if "link.springer.com" in host:
        for prefix in ("/article/", "/chapter/", "/book/"):
            if path.startswith(prefix):
                candidate = path[len(prefix):]
                return _clean_doi(candidate)

    # Case 3: Wiley / Taylor & Francis / Sage / etc: /doi/<doi> 或 /doi/full/<doi> ...
    if any(d in host for d in (
        "onlinelibrary.wiley.com",
        "tandfonline.com",
        "sagepub.com",
        "emerald.com",
        "cambridge.org",
    )):
        if "/doi/" in path:
            candidate = path.split("/doi/", 1)[1]
            # sometimes it is /doi/full/10... or /doi/abs/10...
            parts = candidate.split("/")
            # find the first part that looks like DOI and join back
            for i in range(len(parts)):
                joined = "/".join(parts[i:])
                doi = _extract_doi_from_text(joined)
                if doi:
                    return doi

    # Case 4: ACM: /doi/10.1145/... 或 /doi/abs/10.1145/...
    if "dl.acm.org" in host:
        if "/doi/" in path:
            candidate = path.split("/doi/", 1)[1]
            # remove the intermediate layers like abs/pdf
            for prefix in ("abs/", "pdf/"):
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix):]
                    break
            doi = _extract_doi_from_text(candidate)
            if doi:
                return doi

    # Case 5: Other domains: /doi/10.1145/... 或 /doi/abs/10.1145/...
    return None

def _extract_doi_from_html(html: str) -> str:
    """
    Find the DOI in the HTML source code, such as <meta name="citation_doi" content="...">.
    Only use regex to scan, not do complete HTML parsing.
    """
    if not html:
        return None

    # common meta tags
    meta_patterns = [
        r'name=["\']citation_doi["\'][^>]*content=["\']([^"\']+)["\']',
        r'name=["\']dc.identifier["\'][^>]*content=["\']doi:([^"\']+)["\']',
        r'property=["\']og:doi["\'][^>]*content=["\']([^"\']+)["\']',
    ]
    for pat in meta_patterns:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            doi = _clean_doi(m.group(1))
            if doi:
                return doi

    # fallback: scan the entire page text for DOI regex
    return _extract_doi_from_text(html)

def extract_doi(url: str) -> str:
    """
    Extract the DOI from the the Paper Page link.
    Examples:
      https://dl.acm.org/doi/abs/10.1145/3626772.3657957 → 10.1145/3626772.3657957
      https://dl.acm.org/doi/10.1145/3626772.3657957     → 10.1145/3626772.3657957
      https://dl.acm.org/doi/abs/10.1145/12345           → 10.1145/12345
      https://link.springer.com/chapter/10.1007/978-3-031-88708-6_1 → 10.1007/978-3-031-88708-6_1
      https://ieeexplore.ieee.org/document/5541170             → 10.1109/ICCDA.2010.5541170
    Returns:
        The DOI.
    Returns None if the DOI is not found.
    """
    url = url.strip()
    if not url:
        return None 
    
    parsed_url = urlparse(url)

    # Case 1: Directly extract from the URL
    doi = _extract_doi_from_text(url)
    if doi:
        return doi 
    
    # Case 2: Extract from the path
    doi = _extract_doi_from_known_path(parsed_url)
    if doi:
        return doi 
    
    # Case 3: Extract from the HTML
    try:
        # if HEAD does not find the DOI, try GET (may be restricted by some sites)
        get_response = requests.get(url, headers=DEFAULT_PAGE_HEADERS, allow_redirects=True, timeout=10)
        final_url = get_response.url 
        doi = _extract_doi_from_text(final_url)
        if doi:
            return doi 
        
        # find the DOI in the HTML
        content_type = get_response.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            doi = _extract_doi_from_html(get_response.text)
            if doi:
                return doi
    except Exception as e:
        logger.warning(f"Failed to extract DOI from the paper page ({url}): {e}")
        return None

    return None

def is_pdf_link(url: str, check_header: bool = False, timeout: float = 3.0) -> bool:
    # Check if the URL is likely a PDF link.
    url = url.strip()
    if not url:
        return False

    try:
        parsed = urlparse(url)
    except Exception:
        return False
    
    if parsed.scheme not in ["http", "https"]:
        raise ValueError(f"is_pdf_link only receives http(s) url, but got {url}")

    path_lower = unquote(parsed.path).lower()
    query_lower = parsed.query.lower()
    host_lower = parsed.netloc.lower()

    # Case 1: Direct .pdf extension (handles .pdf?query=xxx and .pdf#section)
    if path_lower.endswith(".pdf"):
        return True

    # Case 2: Common PDF path patterns
    pdf_path_patterns = [
        "/pdf/",
        "/pdfs/",
        "/download/pdf/",
        "/viewpdf/",
        "/getpdf/",
        "/fullpdf/",
        "/pdf-download/",
        "/pdfdownload/",
    ]
    if any(pattern in path_lower for pattern in pdf_path_patterns):
        return True

    # Case 3: Known academic sources
    if "arxiv.org" in host_lower and "/pdf/" in path_lower:
        return True
    if "openreview.net" in host_lower and path_lower.endswith("/pdf"):
        return True
    if "aclanthology.org" in host_lower and "format=pdf" in query_lower:
        return True
    if "semanticscholar.org" in host_lower and "/pdf/" in path_lower:
        return True
    if "researchgate.net" in host_lower and ("/publication/" in path_lower and "/download" in path_lower):
        return True
    if ("nips.cc" in host_lower or "neurips.cc" in host_lower) and "/file/" in path_lower:
        return True
    if "ieeexplore.ieee.org" in host_lower and "/stamp/stamp.jsp" in path_lower:
        return True
    if path_lower.endswith(("/download", "/pdf", "/fulltext.pdf", "/full-text.pdf")):
        academic_hosts = [
            "springer", "wiley", "elsevier", "sciencedirect",
            "tandfonline", "sagepub", "nature.com", "science.org",
            "acm.org", "ieee.org", "aaai.org", "ijcai.org",
        ]
        if any(h in host_lower for h in academic_hosts):
            return True

    # Case 4: URL query parameters suggesting PDF
    pdf_query_indicators = [
        "format=pdf",
        "type=pdf",
        "download=pdf",
        "output=pdf",
        "filetype=pdf",
    ]
    if any(indicator in query_lower for indicator in pdf_query_indicators):
        return True    

    # Case 5: Send HEAD request to check Content-Type header
    if check_header:
        try:
            response = requests.head(
                url,
                headers=DEFAULT_PAGE_HEADERS,
                allow_redirects=True,
                timeout=timeout,
            )
            content_type = response.headers.get("Content-Type", "").lower()
            # Check for PDF MIME type
            if "pdf" in content_type:
                return True
            # Check Content-Disposition header for PDF filename
            content_disposition = response.headers.get("Content-Disposition", "").lower()
            if content_disposition and ".pdf" in content_disposition:
                return True
        except Exception:
            pass

    return False

def is_arxiv_link(url: str) -> bool:
    """Check if the URL is an arXiv link."""
    if not url:
        return False
    return "arxiv.org" in url.lower()

def is_link_accessible(url: str, timeout: float = 5.0) -> bool:

    # Check if the URL is publicly accessible.
    url = (url or "").strip()
    if not url:
        return False

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
    except Exception:
        return False

    def _is_success(status_code: int) -> bool:
        return 200 <= status_code < 300

    try:
        # Try HEAD first (faster, less bandwidth)
        response = requests.head(
            url,
            headers=DEFAULT_PAGE_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )

        # If HEAD succeeds with 2xx, the link is accessible
        if _is_success(response.status_code):
            return True

        # Some servers don't support HEAD or block it (405 Method Not Allowed, 403 Forbidden)
        # Fall back to GET with stream=True to avoid downloading entire content
        if response.status_code in (405, 403, 501):
            response = requests.get(
                url,
                headers=DEFAULT_PAGE_HEADERS,
                timeout=timeout,
                allow_redirects=True,
                stream=True,  # Don't download body, just check status
            )
            response.close()  # Close connection immediately
            return _is_success(response.status_code)

        return False

    except requests.exceptions.SSLError:
        # SSL certificate issues - try without verification as fallback
        try:
            response = requests.head(
                url,
                headers=DEFAULT_PAGE_HEADERS,
                timeout=timeout,
                allow_redirects=True,
                verify=False,
            )
            return _is_success(response.status_code)
        except Exception:
            return False
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.ConnectionError:
        return False
    except Exception:
        return False

def clean_html_content(html: str) -> str:
    """
    Clean HTML content by removing irrelevant tags and extracting only the text content.
    This helps reduce token consumption when processing paper pages.
    
    Args:
        html: Raw HTML content from the web page.
    
    Returns:
        Cleaned text content with minimal formatting.
    """
    if not html:
        return ""
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'iframe']):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Extract text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace and newlines
        # Replace multiple newlines with single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        # Remove empty lines
        lines = [line for line in lines if line]
        
        return '\n'.join(lines)
    
    except Exception as e:
        logger.warning(f"Failed to clean HTML content: {e}")
        # Fallback: simple regex-based cleaning
        # Remove script and style tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def get_metadata_from_paper_page(paper_url: str, llm: BaseLLM, include_abstract: bool=True) -> ResearchToolResult:

    # visit the paper page to get the metadata
    metadata = ResearchToolMetadata(tool_name="get_metadata_from_paper_page")
    paper_metadata = {} 

    try:
        response = requests.get(paper_url, headers=DEFAULT_PAGE_HEADERS, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower() 
        if "pdf" in content_type or "application/octet-stream" in content_type:
            logger.info(f"Content type '{content_type}' is unsupported for HTML parsing.")
            return ResearchToolResult(metadata=metadata, result=paper_metadata)
        
        # Clean HTML content to reduce token consumption
        page_context = clean_html_content(response.text)
        if include_abstract:
            prompt = PAPER_METADATA_EXTRACTION_PROMPT.format(context=page_context) 
        else:
            prompt = PAPER_METADATA_EXTRACTION_PROMPT_NO_ABSTRACT.format(context=page_context)
        with track_cost() as cost_tracker:
            llm_response = llm.generate(prompt=prompt, parser=PaperMetadataOutputParser, parse_mode="json", max_tokens=2048)
            metadata.add_cost_breakdown({"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()})
        for key in ["title", "authors", "abstract"]:
            value = getattr(llm_response, key, None)
            paper_metadata[key] = value
            if key == "authors" and isinstance(value, str) and "[]" in value:
                paper_metadata[key] = [] 
            if key == "abstract" and isinstance(value, str) and ("none" in value.lower() or "null" in value.lower()):
                paper_metadata[key] = None 
    except Exception as e: 
        logger.warning(f"Failed to get metadata from the paper URL ({paper_url}): {e}") 

    return ResearchToolResult(metadata=metadata, result=paper_metadata)


def validate_author_info(authors: List[str], gs_authors: List[str]) -> bool:

    match_list = [] 
    for author, gs_author in zip(authors, gs_authors):
        author_words = set(author.lower().split())
        gs_author_words = set(gs_author.lower().split())
        match_list.append(len(author_words.intersection(gs_author_words))>0)
    return all(match_list) 


def is_identifier(query: str) -> bool:
    """
    Check if the query string is likely a paper identifier (arXiv ID, DOI, or other identifier).

    Uses heuristics:
    - If the query has no spaces (all letters and numbers are continuous) AND
    - The number of digits is >= 3
    Then it's likely an identifier.

    Args:
        query: The query string to check.

    Returns:
        True if the query is likely an identifier, False otherwise.
    """
    query = query.strip()
    if not query:
        return False

    # Check if the query has no spaces (ignoring common separators like ., -, /, :)
    # Remove common separators to check if remaining characters are continuous
    normalized = query.replace('.', '').replace('-', '').replace('/', '').replace(':', '').replace('_', '')

    # If there are spaces in the normalized version, it's likely a title
    if ' ' in normalized:
        return False

    # Count the number of digits
    digit_count = sum(1 for c in query if c.isdigit())

    # If there are 3 or more digits and no spaces, likely an identifier
    if digit_count >= 3:
        return True

    return False


def query_expansion_for_recall(query: str, llm: BaseLLM, max_queries: int = 3) -> Tuple[List[str], Dict[str, float]]:

    def _normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'^[\-\*\•\d\.\)\s]+', '', s)  # strip bullets/numbering if any
        s = re.sub(r'\s+', ' ', s)
        return s

    query = query.strip()
    if not query:
        return [], {} 
    
    if max_queries <= 1:
        return [query], {} 

    cost_breakdown: Dict[str, float] = {}
    prompt = QUERY_EXPANSION_FOR_RECALL_PROMPT.format(query=query, max_queries=max_queries-1)
    with track_cost() as cost_tracker:
        llm_response = llm.generate(prompt=prompt, max_tokens=512) 
        cost_breakdown = add_dict(cost_breakdown, {"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()})

    if "error" in llm_response.content.lower():
        # if the query expansion fails, return the original query as the only query
        return [query], cost_breakdown 

    # parse the LLM results 
    lines = [line.strip() for line in llm_response.content.splitlines() if line.strip()]

    query_list = []
    seen = set() 

    # add the original query  
    query_list.append(query)
    seen.add(_normalize(query)) 

    for line in lines:
        candidate = re.sub(r'^[\-\*\•\d\.\)\s]+', '', line).strip()
        candidate = re.sub(r'\s+', ' ', candidate).strip() 
        if not candidate:
            continue 
        normalized_candidate = _normalize(candidate)
        if normalized_candidate in seen:
            continue
        if len(candidate.split()) > 8:
            continue
        query_list.append(candidate) 
        seen.add(normalized_candidate)
        if len(query_list) >= max_queries:
            break

    return query_list, cost_breakdown 


def find_seminal_papers_for_topic(topic: str, llm: BaseLLM, max_papers: int = 3) -> Tuple[List[Dict], Dict[str, float]]:

    topic = topic.strip()
    if not topic:
        return [], {}

    cost_breakdown: Dict[str, float] = {}
    prompt = SEMINAL_PAPERS_FOR_TOPIC_PROMPT.format(topic=topic, max_papers=max_papers)
    with track_cost() as cost_tracker:
        llm_response = llm.generate(prompt=prompt, max_tokens=512)
        cost_breakdown = add_dict(cost_breakdown, {"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()})
    
    try:
        seminal_papers = parse_json_from_llm_output(llm_response.content)
    except Exception:
        return [], cost_breakdown

    seminal_paper_titles = [paper["title"] for paper in seminal_papers]
    return seminal_paper_titles, cost_breakdown 

 
def extract_related_work_references(content: str, llm: BaseLLM) -> Tuple[List[Dict], Dict[str, float]]:

    if not content or len(content.strip()) < 100:
        return [], {} 
    
    # Step 1: Extract Introduction and Related Work sections
    related_work_text = _extract_introduction_related_work_section(content)
    if not related_work_text:
        return [], {} 
    
    # Step 2: Extract citations (numeric and author-year formats)
    citations = _extract_citations_from_text(related_work_text)
    if not citations:
        return [], {} 
    
    # Step 3: Parse the citation information from the References section
    references_text = _extract_references_section(content)
    if not references_text:
        return [], {} 
    
    # Step 4: Match the citations and extract the detailed information
    matched_references, cost_breakdown = _match_citations(citations, references_text, llm) 
    
    return matched_references, cost_breakdown 


def _extract_introduction_related_work_section(content: str, intro_words: int = 1500, related_words: int = 1000) -> str:
    """
    Extract Introduction and Related Work sections from paper content.
    Uses fixed word counts after finding section headers.
    Avoids overlap when Introduction is immediately followed by Related Work.
    Ensures Related Work content stops before References section.
    
    Args:
        content: Paper text content
        intro_words: Number of words to extract from Introduction
        related_words: Number of words to extract from Related Work
        
    Returns:
        Combined text from both sections
    """
    content_lower = content.lower()
    
    # Patterns for Introduction section
    intro_patterns = [
        r"(?im)^\s*introduction\s*$",
        r"(?im)^\s*\d+\.?\s+introduction\s*$",
    ]
    
    # Patterns for Related Work section
    related_patterns = [
        r"(?im)^\s*related\s+work(?:s)?\s*$",
        r"(?im)^\s*\d+\.?\s+related\s+work(?:s)?\s*$",
        r"(?im)^\s*background\s+(?:and|&)\s+related\s+work(?:s)?\s*$",
        r"(?im)^\s*literature\s+review(?:s)?\s*$",
        r"(?im)^\s*previous\s+work(?:s)?\s*$",
    ]
    
    # Patterns for References section (to find the boundary)
    references_patterns = [
        r"(?im)^\s*reference(?:s)?\s*:?\s*$",
        r"(?im)^\s*\d+\.?\s+reference(?:s)?\s*:?\s*$",
        r"(?im)^\s*bibliograph(?:y|ies)\s*:?\s*$",
    ]
    
    # Find Introduction start position
    intro_start = -1
    for pattern in intro_patterns:
        match = re.search(pattern, content_lower, re.IGNORECASE)
        if match:
            intro_start = match.end()
            break

    # Find Related Work start position
    related_start = -1
    for pattern in related_patterns:
        match = re.search(pattern, content_lower, re.IGNORECASE)
        if match:
            related_start = match.end()
            break
    
    # Find References start position (boundary for Related Work)
    references_start = -1
    for pattern in references_patterns:
        match = re.search(pattern, content_lower, re.IGNORECASE)
        if match:
            references_start = match.start()
            break
    
    combined_text = ""

    # Extract Introduction section
    if intro_start != -1:
        intro_text = content[intro_start:]

        if related_start != -1 and related_start > intro_start:
            intro_before_related = content[intro_start:related_start]
            intro_words_available = len(intro_before_related.split())
            
            if intro_words_available < intro_words:
                intro_extracted = intro_before_related
            else:
                intro_extracted = _extract_n_words(intro_text, intro_words)
        else:
            intro_extracted = _extract_n_words(intro_text, intro_words)
        
        combined_text += intro_extracted.strip() + "\n\n"
    
    # Extract Related Work section
    if related_start != -1:
        # Determine the end boundary for Related Work
        if references_start != -1 and references_start > related_start:
            # Related Work must end before References
            related_before_refs = content[related_start:references_start]
            related_words_available = len(related_before_refs.split())
            
            if related_words_available < related_words:
                # Take all available content before References
                related_extracted = related_before_refs
            else:
                # Take fixed number of words, but ensure it doesn't exceed References boundary
                related_text_limited = content[related_start:references_start]
                related_extracted = _extract_n_words(related_text_limited, related_words)
        else:
            # No References found, extract normally
            related_text = content[related_start:]
            related_extracted = _extract_n_words(related_text, related_words)
        
        combined_text += related_extracted.strip()
    
    # If neither section found, return empty
    if not combined_text.strip():
        return ""
    
    return combined_text.strip()


def _extract_n_words(text: str, n_words: int) -> str:
    """
    Extract first n words from text while preserving formatting (newlines, spaces).
    """
    if not text:
        return ""
    
    words = text.split()
    if len(words) <= n_words:
        return text
    
    # Count words and find the position of the n-th word in original text
    word_count = 0
    # current_pos = 0
    target_pos = len(text)
    
    # Use regex to find word boundaries while preserving original text
    word_pattern = re.compile(r'\S+')
    
    for match in word_pattern.finditer(text):
        word_count += 1
        if word_count == n_words:
            # Find the end of this word
            target_pos = match.end()
            break
    
    extracted_text = text[:target_pos]
    
    # Try to end at a sentence boundary for cleaner cutoff
    last_period = max(
        extracted_text.rfind('. '),
        extracted_text.rfind('! '),
        extracted_text.rfind('? ')
    )
    
    # If found a sentence ending in the last 20% of text, cut there
    if last_period > len(extracted_text) * 0.8:
        return extracted_text[:last_period + 1]
    
    return extracted_text


def _extract_citations_from_text(text: str) -> List[str]:
    """
    Extract all citations from text.
    Returns list of citation strings in format: "[1]" or "Author et al. (2025)"
    """

    # parse the text: replace newline with space and "-\n" -> ""
    text = unicodedata.normalize('NFKC', text)
    text = text.replace("\u00ad", "") 
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"(?i)\bet\s*\.?\s*al\.?(?=\W|$)", "et al.", text)

    citations = []
    # Pattern 1: Numeric citations [1], [1,2], [1-3], [1, 2, 3], [1-3, 4, 5-6] 
    numeric_pattern = r'\[\s*\d+(?:\s*-\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*-\s*\d+)?)*\s*\]'
    for match in re.finditer(numeric_pattern, text):
        citation_text = match.group(0)
        # Parse the numeric citations (handle ranges and lists)
        numbers = _parse_numeric_citation(citation_text)
        for n in numbers:
            candidate = f"[{n}]"
            if candidate not in citations:
                citations.append(candidate)

    # Pattern 2: Author-year citations (Author, 2025), Author et al. (2025), (Author 2025; Author2 2024), etc.
    author_year_citations = _extract_author_year_citations(text)
    citations.extend(author_year_citations)
    
    return citations 


def _expand_year_list(s: str, base_year_hint: Optional[str] = None) -> List[str]: 
    """ 
    Expand year expressions inside parentheses: 
    - "2024;2025" -> ["2024", "2025"] 
    - "2024, 2025" -> ["2024", "2025"] 
    - "2025a,b" -> ["2025a", "2025b"] 
    - "2025c;d" -> ["2025c", "2025d"] 
    - "2025b,a" -> ["2025b", "2025a"] (keep order) 
    - "2025c;d" -> ["2025c", "2025d"] 
    """ 
    s = s.strip() 
    s = re.sub(r"^[,;]\s*", "", s) 

    out = [] 
    last_base_year = base_year_hint 
    # split by ; first (common for separating different citations / years) 
    parts = [p.strip() for p in re.split(r"\s*;\s*", s) if p.strip()] 
    
    for part in parts: 
        # further split by comma, but keep things like "2025a,b" together by parsing below 
        tokens = [t.strip() for t in re.split(r"\s*,\s*", part) if t.strip()] 
        i = 0 
        while i < len(tokens): 
            tok = tokens[i] 
            # Full year with optional suffix, e.g. 2025 / 2025a 
            m = re.fullmatch(r"((?:19|20)\d{2})([a-z]?)", tok, flags=re.I) 
            if m: 
                base = m.group(1) 
                suf = m.group(2) 
                last_base_year = base 
                out.append(base + suf if suf else base) 
                i += 1 
                continue 
        
            # Suffix-only token, e.g. "b" in "2025a,b" or "d" in "2025c;d" 
            m2 = re.fullmatch(r"([a-z])", tok, flags=re.I) 
            if m2 and last_base_year: 
                out.append(last_base_year + m2.group(1)) 
                i += 1 
                continue 
            
            # Sometimes PDF gives "2025a,b" as a single token with comma already removed earlier; handle raw patterns too 
            m3 = re.fullmatch(r"((?:19|20)\d{2})([a-z])\s*[,/]\s*([a-z])", tok, flags=re.I) 
            if m3: 
                base = m3.group(1) 
                a = m3.group(2) 
                b = m3.group(3) 
                last_base_year = base 
                out.extend([base + a, base + b]) 
                i += 1 
                continue 
            # Otherwise ignore token (page numbers etc.)
            i += 1 
    
    return out 


def _extract_author_year_citations(text: str) -> List[str]:
    """
    Extract author-year citations using robust patterns. 
    Returns list of strings in format "Author et al. (2025)"
    """
    YEAR = r"(?:19|20)\d{2}[a-z]?"
    
    # Author patterns:
    # SURNAME = r"[A-Z][A-Za-z''\-]+"
    SURNAME = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+"
    AUTHOR_LIST = rf"{SURNAME}(?:\s*,\s*{SURNAME})+\s*,?\s*(?:and|&)\s+{SURNAME}"
    # AUTHOR_CORE = rf"{SURNAME}(?:\s+(?:and|&)\s+{SURNAME})?(?:\s+et\s+al\.?)?"
    AUTHOR_CORE = rf"(?:{AUTHOR_LIST}|{SURNAME}(?:\s+(?:and|&)\s+{SURNAME})?(?:\s+et\s+al\.?)?)"
    NEXT_AUTHOR = re.compile(rf"(?i)(?:^|[,;]\s*)({AUTHOR_CORE})\s*,?\s*{YEAR}\b")
    
    # Two full surnames: "Chen and Mueller"
    TWO_AUTHORS = rf"{SURNAME}\s+(?:and|&)\s+{SURNAME}"
    
    # Parenthetical groups: (...)
    PAREN_GROUP = re.compile(r"\(([^()]{0,250})\)")
    
    # Author-year inside parentheses: Author, 2025 / Author 2025
    AUTH_YEAR = re.compile(rf"(?i)\b({AUTHOR_CORE})\s*,?\s*({YEAR})\b")
    
    # Two full surnames in parentheses: (Chen and Mueller 2023)
    TWO_AUTH_YEAR = re.compile(rf"(?i)\b({TWO_AUTHORS})\s+({YEAR})\b")
    
    # Narrative citation: Author et al. (2025)
    NARRATIVE = re.compile(rf"(?i)\b({AUTHOR_CORE})\s*\(\s*([^(){{}}]{{1,40}})\s*\)")
    
    results = []
    seen = set()
    
    # A) Narrative citations: Author et al. (2025)
    for m in NARRATIVE.finditer(text):
        author = " ".join(m.group(1).split())
        inside = m.group(2).strip()

        # only treat as citation if inside contains a 4-digit year
        if not re.match(r"(?i)^\s*(?:19|20)\d{2}", inside):
            continue

        # Parse inside like: "2024;2025" / "2024, 2025" / "2025a,b" / "2025c;d"
        years = _expand_year_list(inside)

        for y in years:
            key = (author.lower(), y.lower())
            if key not in seen:
                seen.add(key)
                results.append(f"{author} ({y})")
    
    # B) Parenthetical citations
    for gm in PAREN_GROUP.finditer(text):

        chunk = gm.group(1)

        # Check if this looks like a citation list
        looks_like_citation_list = (
            re.search(r"(?i)\bet\s+al\.?(?=\W|$)", chunk) is not None
            or re.search(rf"(?i)(?:and|&)\s+{SURNAME}\s+{YEAR}", chunk) is not None
            or re.search(rf"(?i),\s*{YEAR}\b", chunk) is not None
            or ";" in chunk
        )
        if not looks_like_citation_list:
            continue

        # First, try to match two-author patterns: (Chen and Mueller 2023)
        two_auth_matches = list(TWO_AUTH_YEAR.finditer(chunk))
        for m in two_auth_matches:
            author = " ".join(m.group(1).split())
            year = m.group(2)
            key = (author.lower(), year.lower())
            if key not in seen:
                seen.add(key)
                results.append(f"{author} ({year})")

        # Then match standard author-year patterns
        pairs = list(AUTH_YEAR.finditer(chunk))
        last_author = None

        for m in pairs:
            author = " ".join(m.group(1).split())
            year = m.group(2)
            last_author = author
            key = (author.lower(), year.lower())
            if key not in seen:
                seen.add(key)
                results.append(f"{author} ({year})")
            
            # Check for comma-separated years after this author-year pair
            tail = chunk[m.end():]
            m_next = NEXT_AUTHOR.search(tail)
            if m_next:
                tail = tail[:m_next.start()]
            if len(tail) > 20:
                continue
            extra_years = _expand_year_list(tail, base_year_hint=year[:4])
            for y in extra_years:
                k2 = (author.lower(), y.lower())
                if k2 not in seen:
                    seen.add(k2)
                    results.append(f"{author} ({y})")
        
        # Handle semicolon-separated inherited years
        if last_author:
            last_match_end = pairs[-1].end() if pairs else 0
            remaining_chunk = chunk[last_match_end:]
            for ym in re.finditer(rf"(?i);\s*({YEAR})\b", remaining_chunk):
                y = ym.group(1)
                key = (last_author.lower(), y.lower())
                if key not in seen:
                    seen.add(key)
                    results.append(f"{last_author} ({y})")
    
    return results


def _parse_numeric_citation(citation_text: str) -> List[int]:

    numbers = []
    parts = re.split(r'\s*[,;]\s*', citation_text.strip("[]"))
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle ranges "1-3"
            try:
                start_num, end_num = map(int, re.split(r"\s*-\s*", part))
                numbers.extend(range(start_num, end_num + 1))
            except Exception:
                pass
        else:
            # Single number
            try:
                numbers.append(int(part))
            except Exception:
                pass
    
    return sorted(set(numbers))


def _remove_line_numbers(content: str) -> str:
    """
    Remove review line numbers that appear as single numbers per line forming an arithmetic sequence.
    Example: \n9\n\n\n486\n487\n488\n...\n
    """
    lines = content.splitlines(keepends=True)
    if not lines:
        return content
    
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if current line is a single number
        if line.isdigit():
            # Start collecting consecutive number lines
            number_lines = []
            j = i
            
            while j < len(lines):
                stripped = lines[j].strip()
                if stripped.isdigit():
                    number_lines.append((j, int(stripped)))
                    j += 1
                elif not stripped:  # Empty line, continue
                    j += 1
                else:
                    break
            
            # Check if we have enough numbers to form a sequence (at least 3)
            if len(number_lines) >= 3:
                # Extract the numbers
                numbers = [num for _, num in number_lines]
                
                # Check if they form an arithmetic sequence
                if len(numbers) >= 2:
                    diffs = [numbers[k+1] - numbers[k] for k in range(len(numbers)-1)]
                    # All differences should be the same (common difference)
                    if len(set(diffs)) == 1 and diffs[0] > 0:
                        # This is a line number sequence, skip these lines
                        i = j
                        continue
            
            # Not a valid sequence, keep the lines
            result_lines.append(lines[i])
            i += 1
        else:
            result_lines.append(lines[i])
            i += 1
    
    return ''.join(result_lines)


def _extract_references_section(content: str) -> str:

    # Remove review line numbers first
    content = _remove_line_numbers(content)
    
    patterns = [
        r"(?im)^\s*reference(?:s)?\s*:?\s*$",
        r"(?im)^\s*\d+\.?\s+reference(?:s)?\s*:?\s*$",
        r"(?im)^\s*bibliograph(?:y|ies)\s*:?\s*$",
    ]
    
    content_lower = content.lower()
    
    # Find the start position of the References section
    start_pos = -1
    for pattern in patterns:
        match = re.search(pattern, content_lower, re.IGNORECASE)
        if match:
            start_pos = match.end()
            break
    
    if start_pos == -1:
        return ""

    tail = content[start_pos:]          # use original content to preserve indices/format
    tail_lower = content_lower[start_pos:]

    YEAR_RE = re.compile(r"(?<![A-Za-z0-9/\.])(?:19|20)\d{2}[a-z]?(?![A-Za-z0-9])")
    lines_lower = tail_lower.splitlines(keepends=True)
    lines_orig = tail.splitlines(keepends=True)

    # Determine single-column vs double-column based on average word count per line
    sample_lines = lines_orig[:min(50, len(lines_orig))]
    if sample_lines:
        word_counts = []
        for line in sample_lines:
            words = line.strip().split()
            if words:
                word_counts.append(len(words))
        if word_counts:
            avg_words_per_line = sum(word_counts) / len(word_counts)
            is_double_column = avg_words_per_line < 20 
            max_no_year_streak = 50 if is_double_column else 25
        else:
            max_no_year_streak = 25  # default to single-column
    else:
        max_no_year_streak = 25  # default to single-column

    no_year_streak = 0
    seen_any_year = False
    end_offset = len(tail)  # default: to end

    running_offset = 0
    for i, line in enumerate(lines_lower):
        if YEAR_RE.search(line):
            seen_any_year = True
            no_year_streak = 0
        else:
            # only start counting after we've seen at least one year in references
            if seen_any_year:
                no_year_streak += 1
                if no_year_streak >= max_no_year_streak:
                    # end at the start of this streak
                    cut_idx = i - (max_no_year_streak - 1)
                    end_offset = sum(len(x) for x in lines_orig[:cut_idx])
                    break

        running_offset += len(lines_orig[i])

    end_pos = start_pos + end_offset

    return content[start_pos:end_pos]


def _match_citations(citations: str, references: str, llm: BaseLLM) -> Tuple[List[Dict], Dict[str, float]]:

    if not citations or not references:
        return [], {}
    
    references = unicodedata.normalize("NFKC", references)
    references = references.replace("\u00ad", "") 
    references = re.sub(r"\s*\n\s*", " ", references) 
    references = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", references)
    references = references.replace("–", "-").replace("—", "-")

    cost_breakdown = {}
    try:
        with track_cost() as cost_tracker:
            llm_response = llm.generate(
                prompt = MATCH_CITATIONS_PROMPT.format(
                    citations=citations, 
                    references=references
                ), 
                parse_mode = "str",
                max_tokens = 2048, 
            ).content 
            matched_citations = parse_json_from_llm_output(llm_response) if llm_response else []
            cost_breakdown = add_dict(cost_breakdown, {"openrouter:" + k: v for k, v in cost_tracker.cost_per_model.items()})
    except Exception as e:
        logger.warning(f"Failed to match citations: {e}, return empty list")
        matched_citations = [] 
    
    for citation in matched_citations:
        title = citation.get("title")
        if title:
            citation["title"] = title.strip().rstrip(".")
    return matched_citations, cost_breakdown 
