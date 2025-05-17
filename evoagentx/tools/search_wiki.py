import wikipedia
from .search_base import SearchBase
from typing import Dict, Any, Optional, List, Callable
from pydantic import Field
from evoagentx.core.logging import logger


class SearchWiki(SearchBase):
    max_sentences: int = Field(default=10, description="Maximum number of sentences in the summary. Default 0 means return all available content.")
    
    def __init__(
        self, 
        name: str = 'Wikipedia Search',
        schemas: Optional[List[dict]] = None,
        descriptions: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        num_search_pages: int = 5, 
        max_content_words: int = None,
        max_sentences: int = 10,
        **kwargs
    ):
        # Set default name if not provided
        # name = data.get('name', 'WikipediaSearch')
        schemas = schemas or self.get_tool_schemas()
        descriptions = descriptions or self.get_tool_descriptions()
        tools = tools or self.get_tools()
        # Pass these to the parent class initialization
        super().__init__(
            name=name,
            schemas=schemas,
            descriptions=descriptions,
            tools=tools,
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            **kwargs
        )
        self.max_sentences = max_sentences
        # self.max_sentences = data.get('max_sentences', 10)

    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None, max_sentences: int = None) -> Dict[str, Any]:
        """
        Searches Wikipedia for the given query and returns the summary and truncated full content.

        Args:
            query (str): The search query.
            num_search_pages (int): Number of search results to retrieve.
            max_content_words (int): Maximum number of words to include in the content.

        Returns:
            dict: A dictionary with the title, summary, truncated content, and Wikipedia page link.
        """
        try:
            logger.info(f"Searching wikipedia: {query}, max_sentences={self.max_sentences}, num_results={num_search_pages}")
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
                    
                    # Handle the max_sentences parameter
                    if self.max_sentences > 0:
                        summary = wikipedia.summary(title, sentences=self.max_sentences)
                    else:
                        # Get the full summary without limiting sentences
                        summary = wikipedia.summary(title)

                    # Truncate the full content to the first max_content_words words
                    words = page.content.split()
                    is_truncated = len(words) > max_content_words
                    truncated_content = ' '.join(words[:max_content_words])
                    content = truncated_content + (" ..." if is_truncated else "")

                    results.append({
                        "title": page.title,
                        "summary": summary,
                        "content": content,
                        "url": page.url,
                    })
                except wikipedia.exceptions.DisambiguationError:
                    # Skip ambiguous results and try the next
                    continue
                except wikipedia.exceptions.PageError:
                    # Skip non-existing pages and try the next
                    continue
            
            logger.info(f"get results from wikipedia: {results}")
            return {"results": results}
        
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {str(e)}")
            return {"results": [], "error": str(e)}
    
    def get_tools(self):
        return [self.search]

    def get_tool_schemas(self) -> list[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for the Wikipedia search tool.
        
        Returns:
            list[Dict[str, Any]]: Function schema in OpenAI format
        """
        return [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search Wikipedia for relevant articles and extract key information including titles, summaries, and content.",
                "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on Wikipedia."
                    },
                    "num_search_pages": {
                        "type": "integer",
                        "description": "The number of Wikipedia search results to retrieve (default is 5)."
                    },
                    "max_content_words": {
                        "type": "integer",
                        "description": "The maximum number of words to retain from the full Wikipedia page content (default is 500)."
                    }
                },
                "required": ["query"]
                }
            }
        }]
        
    def get_tool_descriptions(self) -> list[str]:
        return [
            "Search Wikipedia for relevant articles and extract key information including titles, summaries, and content."
        ]

