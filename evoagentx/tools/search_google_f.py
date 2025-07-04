from .search_base import SearchBase
from .tool import Tool,Toolkit
from googlesearch import search as google_f_search
from typing import Dict, Any, List, Optional
from evoagentx.core.logging import logger

class SearchGoogleFree(SearchBase):
    """
    Free Google Search tool that doesn't require API keys.
    """
    
    def __init__(
        self, 
        name: str = "GoogleFreeSearch",
        num_search_pages: Optional[int] = 5, 
        max_content_words: Optional[int] = None,
       **kwargs 
    ):
        """
        Initialize the Free Google Search tool.
        
        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(name=name, num_search_pages=num_search_pages, max_content_words=max_content_words, **kwargs)

    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None) -> Dict[str, Any]:
        """
        Searches Google for the given query and retrieves content from multiple pages.

        Args:
            query (str): The search query.
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, None means no limit

        Returns:
            Dict[str, Any]: Contains a list of search results and optional error message.
        """
        # Use class defaults
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words 
            
        results = []
        try:
            # Step 1: Get top search result links
            logger.info(f"Searching Google (Free) for: {query}, num_results={num_search_pages}, max_content_words={max_content_words}")
            search_results = list(google_f_search(query, num_results=num_search_pages))
            if not search_results:
                return {"results": [], "error": "No search results found."}

            logger.info(f"Found {len(search_results)} search results")
            
            # Step 2: Fetch content from each page
            for url in search_results:
                try:
                    title, content = self._scrape_page(url)
                    if content:  # Ensure valid content exists
                        # Use the base class's content truncation method
                        display_content = self._truncate_content(content, max_content_words)
                            
                        results.append({
                            "title": title,
                            "content": display_content,
                            "url": url,
                        })
                except Exception as e:
                    logger.warning(f"Error processing URL {url}: {str(e)}")
                    continue  # Skip pages that cannot be processed

            return {"results": results, "error": None}
        
        except Exception as e:
            logger.error(f"Error in free Google search: {str(e)}")
            return {"results": [], "error": str(e)}
    

class GoogleFreeSearchTool(Tool):
    name: str = "google_free_search"
    description: str = "Search Google without requiring an API key and retrieve content from search results"
    inputs: Dict[str, Dict[str, str]] = {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute on Google"
                    },
                    "num_search_pages": {
                        "type": "integer",
                        "description": "Number of search results to retrieve. Default: 5"
                    },
                    "max_content_words": {
                        "type": "integer",
                        "description": "Maximum number of words to include in content per result. None means no limit. Default: None"
                    }
    }
    required: Optional[List[str]] = ["query"]
    
    def __init__(self, search_google_free: SearchGoogleFree = None):
        super().__init__()
        self.search_google_free = search_google_free
    
    def __call__(self, query: str, num_search_pages: int = None, max_content_words: int = None) -> Dict[str, Any]:
        """Execute Google free search using the SearchGoogleFree instance."""
        if not self.search_google_free:
            raise RuntimeError("Google free search instance not initialized")
        
        try:
            return self.search_google_free.search(query, num_search_pages, max_content_words)
        except Exception as e:
            return {"results": [], "error": f"Error executing Google free search: {str(e)}"}


class GoogleFreeSearchToolkit(Toolkit):
    def __init__(
        self,
        name: str = "GoogleFreeSearchToolkit",
        num_search_pages: Optional[int] = 5,
        max_content_words: Optional[int] = None,
        **kwargs
    ):
        # Create the shared Google free search instance
        search_google_free = SearchGoogleFree(
            name="GoogleFreeSearch",
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            **kwargs
        )
        
        # Create tools with the shared search instance
        tools = [
            GoogleFreeSearchTool(search_google_free=search_google_free)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store search_google_free as instance variable
        self.search_google_free = search_google_free
    

