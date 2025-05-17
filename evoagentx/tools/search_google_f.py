from .search_base import SearchBase
from googlesearch import search as google_f_search
from typing import Dict, Any, Optional, List, Callable
from evoagentx.core.logging import logger

class SearchGoogleFree(SearchBase):
    """
    Free Google Search tool that doesn't require API keys.
    """
    
    def __init__(
        self, 
        name: str = "Free Google Search",
        schemas: Optional[List[dict]] = None,
        descriptions: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        num_search_pages: int = 5, 
        max_content_words: int = None,
        **kwargs 
    ):
        """
        Initialize the Free Google Search tool.
        
        Args:
            name (str): Name of the tool
            schemas (Optional[List[dict]]): Tool schemas
            descriptions (Optional[List[str]]): Tool descriptions
            tools (Optional[List[Callable]]): Tool functions
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            **kwargs: Additional keyword arguments for parent class initialization
        """
        # name = kwargs.get('name', "FreeGoogleSearch")
        schemas = schemas or self.get_tool_schemas()
        descriptions = descriptions or self.get_tool_descriptions()
        tools = tools or self.get_tools()
        
        # Pass to parent class initialization
        super().__init__(
            name=name,
            schemas=schemas,
            descriptions=descriptions,
            tools=tools,
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            **kwargs
        )

    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None) -> Dict[str, Any]:
        """
        Searches Google for the given query and retrieves content from multiple pages.

        Args:
            query (str): The search query.
            num_search_pages (int, optional): Number of search results to retrieve. 
                                            If None, uses the class default.
            max_content_words (int, optional): Maximum words to include in content.
                                             If None, uses the class default.

        Returns:
            Dict[str, Any]: Contains a list of search results and optional error message.
        """
        # Use class defaults if parameters are not provided
        if num_search_pages is None:
            num_search_pages = self.num_search_pages
        if max_content_words is None:
            max_content_words = self.max_content_words
            
        results = []
        try:
            # Step 1: Get top search result links
            logger.info(f"Searching Google (Free) for: {query}, num_results={num_search_pages}")
            search_results = list(google_f_search(query, num_results=num_search_pages))
            if not search_results:
                return {"results": [], "error": "No search results found."}

            logger.info(f"Found {len(search_results)} search results")
            
            # Step 2: Fetch content from each page
            for url in search_results:
                try:
                    title, content = self._scrape_page(url)
                    if content:  # Ensure valid content exists
                        # Truncate content while preserving structure
                        if max_content_words is not None and max_content_words > 0:
                            # This preserves the original spacing while limiting word count
                            words = content.split()
                            is_truncated = len(words) > max_content_words
                            word_count = 0
                            truncated_content = ""
                            
                            # Rebuild the content preserving original whitespace
                            for i, char in enumerate(content):
                                if char.isspace():
                                    if i > 0 and not content[i-1].isspace():
                                        word_count += 1
                                    if word_count >= max_content_words:
                                        break
                                truncated_content += char
                                
                            # Add ellipsis only if truncated
                            display_content = truncated_content
                            if is_truncated:
                                display_content += " ..."
                        else:
                            # Use full content if max_content_words is None or <= 0
                            display_content = content
                            
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
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for the free Google search tool.
        
        Returns:
            list[Dict[str, Any]]: Function schema in OpenAI format
        """
        return [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search Google without requiring an API key and retrieve content from search results.",
                "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute on Google"
                    },
                    "num_search_pages": {
                        "type": "integer",
                        "description": "The number of search results to retrieve. If not provided, uses the default value."
                    },
                    "max_content_words": {
                        "type": "integer",
                        "description": "The maximum number of words to retain from the content. If not provided, uses the default value. Set to None for unlimited words."
                    }
                },
                "required": ["query"]
                }
            }
        }]
        
    def get_tools(self) -> List[Callable]:
        return [self.search]

    def get_tool_descriptions(self) -> List[str]:
        return [
            "Free Google Search Tool that queries Google without requiring an API key."
        ]

