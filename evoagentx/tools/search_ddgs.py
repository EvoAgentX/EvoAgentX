## Dux Distributed Global Search

from .search_base import SearchBase
from .tool import Tool,Toolkit
from ddgs import DDGS
from typing import Dict, Any, List, Optional

class SearchDDGS(SearchBase):
    """
    DDGS (Dux Distributed Global Search) tool that aggregates results from multiple search engines.
    Supports DuckDuckGo, Google, Bing, Brave, Yahoo, and other backends.
    """
    
    def __init__(
        self, 
        name: str = "SearchDDGS",
        num_search_pages: Optional[int] = 5, 
        max_content_words: Optional[int] = None,
        backend: str = "auto",
        region: str = "us-en",
        **kwargs 
    ):
        """
        Initialize the DDGS Search tool.
        
        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            backend (str): Search backend(s) to use. Options: "auto", "duckduckgo", "google", "bing", "brave", "yahoo", etc.
            region (str): Search region (e.g., "us-en", "uk-en", "ru-ru")
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(name=name, num_search_pages=num_search_pages, max_content_words=max_content_words, **kwargs)
        self.backend = backend
        self.region = region

    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None, backend: str = None, region: str = None) -> Dict[str, Any]:
        """
        Searches using DDGS for the given query and retrieves content from multiple pages.

        Args:
            query (str): The search query.
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, None means no limit
            backend (str): Search backend to use (overrides instance default)
            region (str): Search region to use (overrides instance default)

        Returns:
            Dict[str, Any]: Contains a list of search results and optional error message.
        """
        # Use class defaults
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words 
        backend = backend or self.backend
        region = region or self.region
            
        results = []
        try:
            # Step 1: Get search results using DDGS
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    query, 
                    max_results=num_search_pages,
                    backend=backend,
                    region=region
                ))
            
            if not search_results:
                return {"results": [], "error": "No search results found."}
            
            # Step 2: Process each search result
            for result in search_results:
                try:
                    title = result.get('title', 'No Title')
                    url = result.get('href', '') or result.get('link', '') or result.get('url', '')
                    
                    # Always try to scrape the actual page content
                    if url and url.startswith(('http://', 'https://')):
                        try:
                            scraped_title, scraped_content = self._scrape_page(url)
                            if scraped_content:
                                title = scraped_title or title
                                content = scraped_content
                            else:
                                # Fall back to snippet if scraping fails
                                content = result.get('body', '')
                        except Exception:
                            # Fall back to snippet if scraping fails
                            content = result.get('body', '')
                    else:
                        # No valid URL, use snippet
                        content = result.get('body', '')
                    
                    if content:  # Ensure valid content exists
                        # Use the base class's content truncation method
                        display_content = self._truncate_content(content, max_content_words)
                            
                        results.append({
                            "title": title,
                            "content": display_content,
                            "url": url,
                        })
                        
                except Exception as e:
                    continue  # Skip results that cannot be processed

            return {"results": results, "error": None}
        
        except Exception as e:
            return {"results": [], "error": str(e)}
    

class DDGSSearchTool(Tool):
    name: str = "ddgs_search"
    description: str = "Search using DDGS (Dux Distributed Global Search) which aggregates results from multiple search engines including DuckDuckGo, Google, Bing, and others"
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "The search query to execute"
        },
        "num_search_pages": {
            "type": "integer",
            "description": "Number of search results to retrieve. Default: 5"
        },
        "max_content_words": {
            "type": "integer",
            "description": "Maximum number of words to include in content per result. None means no limit. Default: None"
        },
        "backend": {
            "type": "string",
            "description": "Search backend to use. Options: 'auto', 'duckduckgo', 'google', 'bing', 'brave', 'yahoo'. Default: 'auto'"
        },
        "region": {
            "type": "string",
            "description": "Search region (e.g., 'us-en', 'uk-en', 'ru-ru'). Default: 'us-en'"
        }
    }
    required: Optional[List[str]] = ["query"]
    
    def __init__(self, search_ddgs: SearchDDGS = None):
        super().__init__()
        self.search_ddgs = search_ddgs
    
    def __call__(self, query: str, num_search_pages: int = None, max_content_words: int = None, backend: str = None, region: str = None) -> Dict[str, Any]:
        """Execute DDGS search using the SearchDDGS instance."""
        if not self.search_ddgs:
            raise RuntimeError("DDGS search instance not initialized")
        
        try:
            return self.search_ddgs.search(query, num_search_pages, max_content_words, backend, region)
        except Exception as e:
            return {"results": [], "error": f"Error executing DDGS search: {str(e)}"}


class DDGSSearchToolkit(Toolkit):
    def __init__(
        self,
        name: str = "DDGSSearchToolkit",
        num_search_pages: Optional[int] = 5,
        max_content_words: Optional[int] = None,
        backend: str = "auto",
        region: str = "us-en",
        **kwargs
    ):
        # Create the shared DDGS search instance
        search_ddgs = SearchDDGS(
            name="DDGSSearch",
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            backend=backend,
            region=region,
            **kwargs
        )
        
        # Create tools with the shared search instance
        tools = [
            DDGSSearchTool(search_ddgs=search_ddgs)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store search_ddgs as instance variable
        self.search_ddgs = search_ddgs
    

