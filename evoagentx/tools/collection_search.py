from typing import Dict, Any, List, Optional
from .tool import Toolkit
from .collection_base import ToolCollection
from .search_ddgs import DDGSSearchToolkit
from .search_wiki import WikipediaSearchToolkit
from .search_serpapi import SerpAPIToolkit
from .search_google import GoogleSearchToolkit
from .search_google_f import GoogleFreeSearchToolkit
from .search_serperapi import SerperAPIToolkit
from .crawler_base import PageContentHandler, AutoPageContentHandler
# Import logger for error handling
from ..core.logging import logger


class SearchCollection(ToolCollection):
    """
    A comprehensive search tool collection that orchestrates multiple search engines.
    
    This collection includes DDGS, Wikipedia, SerpAPI, Google, Google Free, and SerperAPI
    search tools, executing them in a prioritized order with proper fallback mechanisms.
    """
    
    name: str = "search_collection"
    description: str = "Comprehensive search using multiple search engines with intelligent fallback"
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "The search query to execute across multiple search engines"
        },
        "num_results": {
            "type": "integer", 
            "description": "Number of search results to retrieve per engine (default: 5)"
        },
        "max_content_words": {
            "type": "integer",
            "description": "Maximum words per result content (default: 300)"
        },
        "language": {
            "type": "string",
            "description": "Language preference (ISO 639-1 code, e.g., 'en', 'es', 'fr')"
        },
        "region": {
            "type": "string", 
            "description": "Geographic region for localized results (e.g., 'us', 'uk', 'de')"
        }
    }
    required: Optional[List[str]] = ["query"]
    
    # Default execution order - prioritize serper, serp, google, ddgs, google_free, wiki
    default_order: List[str] = ["serperapi_search", "serpapi_search", "google_search", "ddgs_search", "google_free_search", "wikipedia_search"]

    def __init__(
        self,
        name: str = "search_collection",
        max_content_words: Optional[int] = 300,
        num_results: Optional[int] = 5,
        language: Optional[str] = "en",
        region: Optional[str] = "us",
        page_content_handler: Optional[PageContentHandler] = None,
        **kwargs
    ):
        """
        Initialize the SearchCollection with all search toolkits.
        
        Args:
            name (str): Name of the collection
            max_content_words (int): Maximum words per search result content
            num_results (int): Number of search results to retrieve
            language (str): Language for search results
            region (str): Region for search results
            page_content_handler (PageContentHandler): Optional shared handler for processing page content
            **kwargs: Additional arguments passed to ToolCollection base class
        """
        # Initialize shared page content handler if not provided
        if page_content_handler is None:
            page_content_handler = AutoPageContentHandler(max_length=max_content_words)
        
        # Initialize all search toolkits with their specific parameters and shared handler
        toolkits = [
            SerperAPIToolkit(
                num_search_pages=num_results,
                max_content_words=max_content_words,
                page_content_handler=page_content_handler
            ),
            WikipediaSearchToolkit(
                num_search_pages=num_results,
                max_content_words=max_content_words,
                page_content_handler=page_content_handler
            ),
            GoogleSearchToolkit(
                num_search_pages=num_results,
                max_content_words=max_content_words,
                page_content_handler=page_content_handler
            ),
            GoogleFreeSearchToolkit(
                num_search_pages=num_results,
                max_content_words=max_content_words,
                page_content_handler=page_content_handler
            ),
            DDGSSearchToolkit(
                num_search_pages=num_results,
                max_content_words=max_content_words,
                region=f"{region}-{language}",
                page_content_handler=page_content_handler
            ),
            SerpAPIToolkit(
                num_search_pages=num_results,
                max_content_words=max_content_words,
                default_language=language,
                default_country=region,
                page_content_handler=page_content_handler
            )
        ]
        
        # Updated execution order to match expected sequence: serper, serp, google, ddgs, google_free, wiki
        execution_order = ["serperapi_search", "serpapi_search", "google_search", "ddgs_search", "google_free_search", "wikipedia_search"]
        
        # Initialize parent class with toolkits first
        super().__init__(
            name=name,
            description="Comprehensive search using multiple search engines with intelligent fallback",
            kits=toolkits,
            execution_order=execution_order,
            argument_mapping_function={
                "wikipedia_search": self.wiki_argument_mapping_func,
                "google_search": self.google_argument_mapping_func,
                "google_free_search": self.google_free_argument_mapping_func,
                "ddgs_search": self.ddgs_argument_mapping_func,
                "serpapi_search": self.serpapi_argument_mapping_func,
                "serperapi_search": self.serperapi_argument_mapping_func
            },
            output_mapping_function={
                "wikipedia_search": self.wiki_output_convert_func,
                "google_search": self.google_output_convert_func,
                "google_free_search": self.google_free_output_convert_func,
                "ddgs_search": self.ddgs_output_convert_func,
                "serpapi_search": self.serpapi_output_convert_func,
                "serperapi_search": self.serperapi_output_convert_func
            },
            **kwargs
        )
        
        # Store instance attributes
        self.default_max_content_words = max_content_words
        self.default_num_results = num_results
        self.default_language = language
        self.default_region = region
        self.page_content_handler = page_content_handler
    
    def ddgs_argument_mapping_func(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map unified inputs to DDGS search tool format.
        
        Maps from unified schema to DDGSSearchTool inputs:
        - query (string, required): The search query to execute
        - num_search_pages (integer, optional): Number of search results to retrieve. Default: 5
        - max_content_words (integer, optional): Maximum number of words to include in content per result. None means no limit. Default: None
        - backend (string, optional): Search backend to use. Options: 'auto', 'duckduckgo', 'google', 'bing', 'brave', 'yahoo'. Default: 'auto'
        - region (string, optional): Search region (e.g., 'us-en', 'uk-en', 'ru-ru'). Default: 'us-en'
        """
        mapped_args = {
            "query": inputs.get("query", ""),
            "num_search_pages": inputs.get("num_results", 5),
            "max_content_words": inputs.get("max_content_words", 300)
        }
        
        # Map region to DDGS format (e.g., 'us' -> 'us-en')
        if "region" in inputs and inputs["region"]:
            region = inputs["region"]
            language = inputs.get("language", "en")
            mapped_args["region"] = f"{region}-{language}"
            
        return mapped_args
    
    def wiki_argument_mapping_func(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map unified inputs to Wikipedia search tool format.
        
        Maps from unified schema to WikipediaSearchTool inputs:
        - query (string, required): The search query to execute
        - num_search_pages (integer, optional): Number of search results to retrieve. Default: 5
        - max_content_words (integer, optional): Maximum number of words to include in content per result. None means no limit. Default: None
        - language (string, optional): Wikipedia language code (e.g., 'en', 'es', 'fr', 'de'). Default: 'en'
        """
        return {
            "query": inputs.get("query", ""),
            "num_search_pages": inputs.get("num_results", 5),
            "max_content_words": inputs.get("max_content_words", 300),
            "language": inputs.get("language", "en")
        }
    
    def serpapi_argument_mapping_func(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map unified inputs to SerpAPI search tool format.
        
        Maps from unified schema to SerpAPITool inputs:
        - query (string, required): The search query to execute
        - num_search_pages (integer, optional): Number of search results to retrieve. Default: 10
        - max_content_words (integer, optional): Maximum number of words to include in content per result. None means no limit. Default: None
        - location (string, optional): Geographic location for localized results (e.g., 'New York, NY', 'London, UK')
        - language (string, optional): Interface language code (e.g., 'en', 'es', 'fr', 'de'). Default: en
        - country (string, optional): Country code for country-specific results (e.g., 'us', 'uk', 'ca'). Default: us
        """
        return {
            "query": inputs.get("query", ""),
            "num_search_pages": inputs.get("num_results", 5),
            "max_content_words": inputs.get("max_content_words", 300),
            "language": inputs.get("language", "en"),
            "country": inputs.get("region", "us")
        }
    
    def ddgs_output_convert_func(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert DDGS search tool outputs to unified format.
        
        Unified output format:
        - results: List of search results with standardized fields
        - metadata: Information about the search execution
        - error: Error message if search failed, None if successful
        """
        if isinstance(outputs, dict) and outputs.get("error", None):
            return {
                "results": [],
                "metadata": {
                    "source": "ddgs_search",
                    "success": False,
                    "total_results": 0
                },
                "error": outputs.get("error")
            }
        
        # Extract results and standardize format
        results = outputs.get("results", []) if isinstance(outputs, dict) else []
        standardized_results = []
        
        for result in results:
            standardized_results.append({
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "site_content": result.get("site_content"),
                "url": result.get("url", ""),
                "source": "ddgs_search"
            })
        
        return {
            "results": standardized_results,
            "metadata": {
                "source": "ddgs_search",
                "success": True,
                "total_results": len(standardized_results)
            },
            "error": None
        }
    
    def wiki_output_convert_func(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert Wikipedia search tool outputs to unified format.
        
        Unified output format:
        - results: List of search results with standardized fields
        - metadata: Information about the search execution
        - error: Error message if search failed, None if successful
        """
        if isinstance(outputs, dict) and outputs.get("error", None):
            return {
                "results": [],
                "metadata": {
                    "source": "wiki_search",
                    "success": False,
                    "total_results": 0
                },
                "error": outputs.get("error")
            }
        
        # Extract results and standardize format
        results = outputs.get("results", []) if isinstance(outputs, dict) else []
        standardized_results = []
        
        for result in results:
            standardized_results.append({
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "site_content": result.get("site_content"),
                "url": result.get("url", ""),
                "source": "wiki_search"
            })
        
        return {
            "results": standardized_results,
            "metadata": {
                "source": "wiki_search",
                "success": True,
                "total_results": len(standardized_results)
            },
            "error": None
        }
    
    def serpapi_output_convert_func(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert SerpAPI search tool outputs to unified format.
        
        Unified output format:
        - results: List of search results with standardized fields
        - metadata: Information about the search execution
        - error: Error message if search failed, None if successful
        """
        if isinstance(outputs, dict) and outputs.get("error", None):
            return {
                "results": [],
                "metadata": {
                    "source": "serpapi_search",
                    "success": False,
                    "total_results": 0
                },
                "error": outputs.get("error")
            }
        
        # Extract results and standardize format
        results = outputs.get("results", []) if isinstance(outputs, dict) else []
        standardized_results = []
        
        for result in results:
            standardized_results.append({
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "site_content": result.get("site_content"),
                "url": result.get("url", ""),
                "source": "serpapi_search"
            })
        
        return {
            "results": standardized_results,
            "metadata": {
                "source": "serpapi_search",
                "success": True,
                "total_results": len(standardized_results)
            },
            "error": None
        }
    
    def google_argument_mapping_func(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map unified inputs to Google search tool format.
        
        Maps from unified schema to GoogleSearchTool inputs:
        - query (string, required): The search query to execute
        - num_search_pages (integer, optional): Number of search results to retrieve. Default: 10
        - max_content_words (integer, optional): Maximum number of words to include in content per result. None means no limit. Default: None
        - language (string, optional): Interface language code (e.g., 'en', 'es', 'fr', 'de'). Default: en
        - country (string, optional): Country code for country-specific results (e.g., 'us', 'uk', 'ca'). Default: us
        - location (string, optional): Geographic location for localized results
        """
        return {
            "query": inputs.get("query", ""),
            "num_search_pages": inputs.get("num_results", 5),
            "max_content_words": inputs.get("max_content_words", 300),
            "language": inputs.get("language", "en"),
            "country": inputs.get("region", "us")
        }
    
    def google_free_argument_mapping_func(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map unified inputs to Google Free search tool format.
        
        Maps from unified schema to GoogleFreeSearchTool inputs:
        - query (string, required): The search query to execute
        - num_search_pages (integer, optional): Number of search results to retrieve. Default: 10
        - max_content_words (integer, optional): Maximum number of words to include in content per result. None means no limit. Default: None
        """
        return {
            "query": inputs.get("query", ""),
            "num_search_pages": inputs.get("num_results", 5),
            "max_content_words": inputs.get("max_content_words", 300)
        }
    
    def serperapi_argument_mapping_func(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map unified inputs to SerperAPI search tool format.
        
        Maps from unified schema to SerperAPITool inputs:
        - query (string, required): The search query to execute
        - num_search_pages (integer, optional): Number of search results to retrieve. Default: 10
        - max_content_words (integer, optional): Maximum number of words to include in content per result. None means no limit. Default: None
        - location (string, optional): Geographic location for localized results (e.g., 'New York, NY', 'London, UK')
        - language (string, optional): Interface language code (e.g., 'en', 'es', 'fr', 'de'). Default: en
        - country (string, optional): Country code for country-specific results (e.g., 'us', 'uk', 'ca'). Default: us
        """
        return {
            "query": inputs.get("query", ""),
            "num_search_pages": inputs.get("num_results", 5),
            "max_content_words": inputs.get("max_content_words", 300),
            "language": inputs.get("language", "en"),
            "country": inputs.get("region", "us")
        }
    
    def google_output_convert_func(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert Google search tool outputs to unified format.
        
        Unified output format:
        - results: List of search results with standardized fields
        - metadata: Information about the search execution
        - error: Error message if search failed, None if successful
        """
        if isinstance(outputs, dict) and outputs.get("error", None):
            return {
                "results": [],
                "metadata": {
                    "source": "google_search",
                    "success": False,
                    "total_results": 0
                },
                "error": outputs.get("error")
            }
        
        # Extract results and standardize format
        results = outputs.get("results", []) if isinstance(outputs, dict) else []
        standardized_results = []
        
        for result in results:
            standardized_results.append({
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "site_content": result.get("site_content"),
                "url": result.get("url", ""),
                "source": "google_search"
            })
        
        return {
            "results": standardized_results,
            "metadata": {
                "source": "google_search",
                "success": True,
                "total_results": len(standardized_results)
            },
            "error": None
        }
    
    def google_free_output_convert_func(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert Google Free search tool outputs to unified format.
        
        Unified output format:
        - results: List of search results with standardized fields
        - metadata: Information about the search execution
        - error: Error message if search failed, None if successful
        """
        if isinstance(outputs, dict) and outputs.get("error", None):
            return {
                "results": [],
                "metadata": {
                    "source": "google_free_search",
                    "success": False,
                    "total_results": 0
                },
                "error": outputs.get("error")
            }
        
        # Extract results and standardize format
        results = outputs.get("results", []) if isinstance(outputs, dict) else []
        standardized_results = []
        
        for result in results:
            standardized_results.append({
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "site_content": result.get("site_content"),
                "url": result.get("url", ""),
                "source": "google_free_search"
            })
        
        return {
            "results": standardized_results,
            "metadata": {
                "source": "google_free_search",
                "success": True,
                "total_results": len(standardized_results)
            },
            "error": None
        }
    
    def serperapi_output_convert_func(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert SerperAPI search tool outputs to unified format.
        
        Unified output format:
        - results: List of search results with standardized fields
        - metadata: Information about the search execution
        - error: Error message if search failed, None if successful
        """
        if isinstance(outputs, dict) and outputs.get("error", None):
            return {
                "results": [],
                "metadata": {
                    "source": "serperapi_search",
                    "success": False,
                    "total_results": 0
                },
                "error": outputs.get("error")
            }
        
        # Extract results and standardize format
        results = outputs.get("results", []) if isinstance(outputs, dict) else []
        standardized_results = []
        
        for result in results:
            standardized_results.append({
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "site_content": result.get("site_content"),
                "url": result.get("url", ""),
                "source": "serperapi_search"
            })
        
        return {
            "results": standardized_results,
            "metadata": {
                "source": "serperapi_search",
                "success": True,
                "total_results": len(standardized_results)
            },
            "error": None
        }

    def __call__(self, query: str, num_results: int = None, max_content_words: int = None, language: str = None, region: str = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the search collection with unified parameters.
        
        Args:
            query: The search query to execute across multiple search engines
            num_results: Number of search results to retrieve per engine (default: 5)
            max_content_words: Maximum words per result content (default: 300)
            language: Language preference (ISO 639-1 code, e.g., 'en', 'es', 'fr')
            region: Geographic region for localized results (e.g., 'us', 'uk', 'de')
            
        Returns:
            Dict containing the unified results from executed search engines
        """
        # Extract required query parameter (now provided explicitly)
        if not query:
            raise ValueError("Missing required parameter 'query'")
            
        num_results = num_results or self.default_num_results
        max_content_words = max_content_words or self.default_max_content_words
        language = language or self.default_language
        region = region or self.default_region
        
        inputs = {
            "query": query,
            "num_results": num_results,
            "max_content_words": max_content_words,
            "language": language,
            "region": region,
            **{k: v for k, v in kwargs.items() if k not in ["query", "num_results", "max_content_words", "language", "region"]}
        }
        outputs = {}
        
        while True:
            next_tool_name = self._get_next_execute(inputs, outputs)
            if not next_tool_name:
                break
                
            tool = self.tool_mappings.get(next_tool_name)
            if not tool:
                break
                
            try:
                mapped_args = self.argument_mapping_function[next_tool_name](inputs, outputs)
                result = tool(**mapped_args)
                outputs[next_tool_name] = self.output_mapping_function[next_tool_name](result)
            except Exception as e:
                outputs[next_tool_name] = {
                    "results": [],
                    "metadata": {
                        "source": next_tool_name,
                        "success": False,
                        "total_results": 0
                    },
                    "error": str(e)
                }
                
        return outputs

    def _get_next_execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[str]:
        """
        Determine the next tool to execute based on current outputs and success criteria.
        
        Args:
            inputs: The input parameters
            outputs: Current outputs from executed tools
            
        Returns:
            Name of the next tool to execute, or None if done
        """
        for tool_name in self.execution_order:
            if tool_name not in outputs:
                return tool_name
            # Check if the tool succeeded and has results
            tool_output = outputs[tool_name]
            if (isinstance(tool_output, dict) and 
                tool_output.get("error") is None and 
                len(tool_output.get("results", [])) > 0):
                # Found successful results, stop here
                return None
        return None


class SearchCollectionToolkit(Toolkit):
    """
    A toolkit that wraps the SearchCollection tool to provide a unified interface
    for comprehensive search functionality across multiple search engines.
    """
    
    def __init__(
        self,
        name: str = "search_collection_toolkit",
        description: str = "Toolkit providing comprehensive search across multiple engines",
        **kwargs
    ):
        # Create the SearchCollection tool
        search_collection = SearchCollection(**kwargs)
        
        super().__init__(
            name=name,
            description=description,
            tools=[search_collection],
            **kwargs
        )

