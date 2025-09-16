from debug.tool.example_load import output
from typing import List, Optional, Dict, Any
from pydantic import Field


from .collection_base import ToolCollection
from .search_wiki import WikipediaSearchTool
from .search_google import GoogleSearchTool
from .search_google_f import GoogleFreeSearchTool
from .search_ddgs import DDGSSearchTool
from .search_serpapi import SerpAPIToolkit
from .search_serperapi import SerperAPIToolkit
from .request_arxiv import ArxivToolkit
from .rss_feed import RSSToolkit
from .request import RequestToolkit



def wiki_argue_convert_func(inputs):
    try:
        query = inputs.get("query")
        num_search_pages = inputs.get("num_search_pages", None)
        max_content_words = inputs.get("max_content_words", None)
        page_content_handler = inputs.get("page_content_handler", None)
    except:
        raise ValueError("query is required")
    return {"query": query, "num_search_pages": num_search_pages, "max_content_words": max_content_words, "page_content_handler": page_content_handler}

def wiki_output_convert_func(outputs):
    try:
        if outputs.get("error", None):
            raise f"Error while running wiki search: {outputs.get("error")}"
        result = output.get("result")
        return result
    except Exception as e:
        raise f"Error while running wiki search: {e}"

class SearchCollection(ToolCollection):
    """
    Input & format:
        - query: str
        - num_search_pages: int = None
        - max_content_words: int = None
        - page_content_handler: PageContentHandler = AutoPageContentHandler()
    
    Output & format:
        - results: List[Any] = []
        - error: str = None
        - success: bool = True
        - tool_name: str = None
    
    """
    default_argument_mapping_function: Optional[Dict[str, Dict[Any, Any]]] = Field(default={
        "wikipedia_search": wiki_argue_convert_func,
        }, description="Default argument mapping dictionary that convert the unified input into correct format for each tool")
    default_output_mapping_function: Optional[Dict[str, Dict[Any, Any]]] = Field(default={
        "wikipedia_search": wiki_output_convert_func,
        }, description="Default output mapping dictionary that convert the unified output into correct format for each tool")
    
    def __init__(self, name: str = "SearchCollection"):
        super().__init__(name=name, tools=[
            WikipediaSearchTool(),
            GoogleSearchTool(),
            GoogleFreeSearchTool(),
            DDGSSearchTool(),
            SerpAPIToolkit(),
            SerperAPIToolkit()
        ])
    
    def _get_next_execute(self, execution_history: List[str]) -> str:
        pass

