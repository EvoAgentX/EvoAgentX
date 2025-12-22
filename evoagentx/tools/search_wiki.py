import wikipedia
from .search_base import SearchBase
from .tool import Tool,Toolkit
from typing import Dict, Any, Optional, List
from pydantic import Field
from ..core.logging import logger


class SearchWiki(SearchBase):

    max_summary_sentences: Optional[int] = Field(default=None, description="Maximum number of sentences in the summary. Default None means return all available content.")
    
    def __init__(
        self, 
        name: str = 'SearchWiki',
        num_search_pages: Optional[int] = 5, 
        max_content_words: Optional[int] = None,
        max_summary_sentences: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the Wikipedia Search tool.
        
        Args:
            name (str): The name of the search tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int, optional): Maximum number of words to include in content, None means no limit
            max_summary_sentences (int, optional): Maximum number of sentences in the summary, None means no limit
            **kwargs: Additional data to pass to the parent class
        """

        super().__init__(
            name=name,
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            max_summary_sentences=max_summary_sentences,
            **kwargs
        )

    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None, max_summary_sentences: int = None) -> Dict[str, Any]:
        """
        Searches Wikipedia for the given query and returns the summary and truncated full content.

        Args:
            query (str): The search query.
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, None means no limit
            max_summary_sentences (int): Maximum number of sentences in the summary, None means no limit

        Returns:
            dict: A dictionary with the title, summary, truncated content, and Wikipedia page link.
        """
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words
        max_summary_sentences = max_summary_sentences or self.max_summary_sentences
            
        try:
            logger.info(f"Searching wikipedia: {query}, num_results={num_search_pages}, max_content_words={max_content_words}, max_summary_sentences={max_summary_sentences}")
            # Search for top matching titles
            search_results = wikipedia.search(query, results=num_search_pages)
            logger.info(f"Search results: {search_results}")
            if not search_results:
                return {"results": [], "error": "No search results found."}

            # Try fetching the best available page
            results = []
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Handle the max_summary_sentences parameter
                    if max_summary_sentences is not None and max_summary_sentences > 0:
                        summary = wikipedia.summary(title, sentences=max_summary_sentences)
                    else:
                        # Get the full summary without limiting sentences
                        summary = wikipedia.summary(title)

                    # Use the base class's content truncation method
                    display_content = self._truncate_content(page.content, max_content_words)
                    
                    results.append({
                        "title": page.title,
                        "summary": summary,
                        "content": display_content,
                        "url": page.url,
                    })
                except wikipedia.exceptions.DisambiguationError:
                    # Skip ambiguous results and try the next
                    continue
                except wikipedia.exceptions.PageError:
                    # Skip non-existing pages and try the next
                    continue
            
            # logger.info(f"get results from wikipedia: {results}")
            return {"results": results, "error": None}
        
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {str(e)}")
            return {"results": [], "error": str(e)}
    

class WikipediaSearchTool(Tool):
    name: str = "wikipedia_search"
    description: str = "Search Wikipedia for relevant articles and content"
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "The search query to look up on Wikipedia"
        },
        "num_search_pages": {
            "type": "integer",
            "description": "Number of search results to retrieve. Default: 5"
        },
        "max_content_words": {
            "type": "integer",
            "description": "Maximum number of words to include in content per result. None means no limit. Default: None"
        },
        "max_summary_sentences": {
            "type": "integer",
            "description": "Maximum number of sentences in the summary. None means no limit. Default: None"
        }
    }
    required: Optional[List[str]] = ["query"]
    
    def __init__(self, search_wiki: SearchWiki = None):
        super().__init__()
        self.search_wiki = search_wiki
    
    def __call__(self, query: str, num_search_pages: int = None, max_content_words: int = None, max_summary_sentences: int = None) -> Dict[str, Any]:
        """Execute Wikipedia search using the SearchWiki instance."""
        if not self.search_wiki:
            raise RuntimeError("Wikipedia search instance not initialized")
        
        try:
            return self.search_wiki.search(query, num_search_pages, max_content_words, max_summary_sentences)
        except Exception as e:
            return {"results": [], "error": f"Error executing Wikipedia search: {str(e)}"}


class WikipediaSearchToolkit(Toolkit):
    def __init__(
        self,
        name: str = "WikipediaSearchToolkit",
        num_search_pages: Optional[int] = 5,
        max_content_words: Optional[int] = None,
        max_summary_sentences: Optional[int] = None,
        **kwargs
    ):
        # Create the shared Wikipedia search instance
        search_wiki = SearchWiki(
            name="SearchWiki",
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            max_summary_sentences=max_summary_sentences,
            **kwargs
        )
        
        # Create tools with the shared search instance
        tools = [
            WikipediaSearchTool(search_wiki=search_wiki)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store search_wiki as instance variable
        self.search_wiki = search_wiki
    

