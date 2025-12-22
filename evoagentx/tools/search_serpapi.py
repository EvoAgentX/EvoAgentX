import os
import requests
from typing import Dict, Any, Optional, List
from pydantic import Field
from .search_base import SearchBase
from .tool import Tool, Toolkit
from evoagentx.core.logging import logger
import dotenv

dotenv.load_dotenv()

class SearchSerpAPI(SearchBase):
    """
    SerpAPI search tool that provides access to multiple search engines including
    Google, Bing, Baidu, Yahoo, and DuckDuckGo through a unified interface.
    """
    
    api_key: Optional[str] = Field(default=None, description="SerpAPI authentication key")
    default_engine: Optional[str] = Field(default="google", description="Default search engine")
    default_location: Optional[str] = Field(default=None, description="Default geographic location")
    default_language: Optional[str] = Field(default="en", description="Default interface language")
    default_country: Optional[str] = Field(default="us", description="Default country code")
    enable_content_scraping: Optional[bool] = Field(default=True, description="Enable full content scraping")
    
    def __init__(
        self,
        name: str = "SearchSerpAPI",
        num_search_pages: Optional[int] = 5,
        max_content_words: Optional[int] = None,
        api_key: Optional[str] = None,
        default_engine: Optional[str] = "google",
        default_location: Optional[str] = None,
        default_language: Optional[str] = "en",
        default_country: Optional[str] = "us",
        enable_content_scraping: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize the SerpAPI Search tool.
        
        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            api_key (str): SerpAPI authentication key (can also use SERPAPI_KEY env var)
            default_engine (str): Default search engine (google, bing, baidu, yahoo, duckduckgo)
            default_location (str): Default geographic location for searches
            default_language (str): Default interface language
            default_country (str): Default country code
            enable_content_scraping (bool): Whether to scrape full page content
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(
            name=name,
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            api_key=api_key,
            default_engine=default_engine,
            default_location=default_location,
            default_language=default_language,
            default_country=default_country,
            enable_content_scraping=enable_content_scraping,
            **kwargs
        )
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('SERPAPI_KEY', '')
        self.base_url = "https://serpapi.com/search.json"
        
        if not self.api_key:
            logger.warning("SerpAPI key not found. Set SERPAPI_KEY environment variable or pass api_key parameter.")

    def _build_serpapi_params(self, query: str, engine: str = None, location: str = None, 
                             language: str = None, country: str = None, search_type: str = None,
                             num_results: int = None) -> Dict[str, Any]:
        """
        Build SerpAPI request parameters.
        
        Args:
            query (str): Search query
            engine (str): Search engine to use
            location (str): Geographic location
            language (str): Interface language
            country (str): Country code
            search_type (str): Type of search (web, images, news, shopping, maps)
            num_results (int): Number of results to retrieve
            
        Returns:
            Dict[str, Any]: SerpAPI request parameters
        """
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results or self.num_search_pages,
        }
        
        # Add optional parameters if provided
        if location or self.default_location:
            params["location"] = location or self.default_location
            
        if language or self.default_language:
            params["hl"] = language or self.default_language
            
        if country or self.default_country:
            params["gl"] = country or self.default_country
            
        # Handle different search types for Google
        if search_type and search_type != "web":
            search_type_map = {
                "images": "isch",
                "news": "nws", 
                "shopping": "shop",
                "maps": "lcl"
            }
            if search_type in search_type_map:
                params["tbm"] = search_type_map[search_type]
        
        return params

    def _execute_serpapi_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search using direct HTTP requests to SerpAPI.
        
        Args:
            params (Dict[str, Any]): Search parameters
            
        Returns:
            Dict[str, Any]: SerpAPI response data
            
        Raises:
            Exception: For API errors
        """
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for SerpAPI errors in response
            if "error" in data:
                raise Exception(f"SerpAPI error: {data['error']}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"SerpAPI request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"SerpAPI search failed: {str(e)}")

    def _process_serpapi_results(self, serpapi_data: Dict[str, Any], max_content_words: int = None) -> Dict[str, Any]:
        """
        Process SerpAPI results into structured format with processed results + raw data.
        
        Args:
            serpapi_data (Dict[str, Any]): Raw SerpAPI response
            max_content_words (int): Maximum words per result content
            
        Returns:
            Dict[str, Any]: Structured response with processed results and raw data
        """
        processed_results = []
        
        # 1. Process Knowledge Graph (highest priority)
        if knowledge_graph := serpapi_data.get("knowledge_graph", {}):
            if description := knowledge_graph.get("description"):
                title = knowledge_graph.get("title", "Unknown")
                content = f"**{title}**"
                
                # Add type if available
                if kg_type := knowledge_graph.get("type"):
                    content += f" ({kg_type})"
                content += f"\n\n{description}"
                
                # Add key attributes if available
                if kg_list := knowledge_graph.get("list", {}):
                    content += "\n\n**Key Information:**"
                    for key, value in list(kg_list.items())[:5]:  # Limit to 5 attributes
                        if isinstance(value, list) and value:
                            formatted_key = key.replace('_', ' ').title()
                            formatted_value = ', '.join(str(v) for v in value[:3])  # Max 3 values
                            content += f"\n• {formatted_key}: {formatted_value}"
                
                processed_results.append({
                    "title": f"Knowledge: {title}",
                    "content": self._truncate_content(content, max_content_words or 200),
                    "url": knowledge_graph.get("source", {}).get("link", ""),
                    "type": "knowledge_graph",
                    "priority": 1
                })
        
        # 2. Process Organic Results with scraping
        for item in serpapi_data.get("organic_results", []):
            url = item.get("link", "")
            title = item.get("title", "No Title")
            snippet = item.get("snippet", "")
            position = item.get("position", 0)
            
            # Prepare the result dict
            result = {
                "title": title,
                "content": self._truncate_content(snippet, max_content_words or 400),
                "url": url,
                "type": "organic",
                "priority": 2,
                "position": position
            }
            
            # Try to scrape full content if enabled and add as site_content
            if self.enable_content_scraping and url and url.startswith(('http://', 'https://')):
                try:
                    scraped_title, scraped_content = self._scrape_page(url)
                    if scraped_content and scraped_content.strip():
                        # Update title if scraped title is better
                        if scraped_title and scraped_title.strip():
                            result["title"] = scraped_title
                        # Add scraped content as site_content
                        result["site_content"] = self._truncate_content(scraped_content, max_content_words or 400)
                    else:
                        result["site_content"] = None
                except Exception as e:
                    logger.debug(f"Content scraping failed for {url}: {str(e)}")
                    result["site_content"] = None
            else:
                result["site_content"] = None
            
            # Only include results that have either snippet or scraped content
            if snippet or result.get("site_content"):
                processed_results.append(result)
        
        # 3. Collect raw data sections for LLM processing
        raw_data = {}
        raw_sections = [
            "local_results", "news_results", "shopping_results", 
            "related_questions", "recipes_results", "images_results"
        ]
        
        for section in raw_sections:
            if section in serpapi_data and serpapi_data[section]:
                # Limit raw data to prevent overwhelming response
                if section == "local_results":
                    # Local results have nested structure
                    places = serpapi_data[section].get("places", [])[:3]
                    if places:
                        raw_data[section] = {"places": places}
                else:
                    # Other sections are arrays
                    raw_data[section] = serpapi_data[section][:3]
        
        # 4. Extract search metadata
        search_metadata = {}
        if search_meta := serpapi_data.get("search_metadata", {}):
            search_metadata = {
                "query": search_meta.get("query", ""),
                "location": search_meta.get("location", ""),
                "total_results": search_meta.get("total_results", ""),
                "search_time": search_meta.get("total_time_taken", "")
            }
        
        # Sort processed results by priority and position
        processed_results.sort(key=lambda x: (x.get("priority", 999), x.get("position", 0)))
        
        return {
            "results": processed_results,
            "raw_data": raw_data if raw_data else None,
            "search_metadata": search_metadata if search_metadata else None,
            "error": None
        }

    def _handle_api_errors(self, error: Exception) -> str:
        """
        Handle SerpAPI specific errors with appropriate messages.
        
        Args:
            error (Exception): The exception that occurred
            
        Returns:
            str: User-friendly error message
        """
        error_str = str(error).lower()
        
        if "api key" in error_str or "unauthorized" in error_str:
            return "Invalid or missing SerpAPI key. Please set SERPAPI_KEY environment variable."
        elif "rate limit" in error_str or "too many requests" in error_str:
            return "SerpAPI rate limit exceeded. Please try again later."
        elif "quota" in error_str or "credit" in error_str:
            return "SerpAPI quota exceeded. Please check your plan limits."
        elif "timeout" in error_str:
            return "SerpAPI request timeout. Please try again."
        else:
            return f"SerpAPI error: {str(error)}"

    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None,
               engine: str = None, location: str = None, language: str = None, 
               country: str = None, search_type: str = None) -> Dict[str, Any]:
        """
        Search using SerpAPI with comprehensive parameter support.
        
        Args:
            query (str): The search query
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            engine (str): Search engine (google, bing, baidu, yahoo, duckduckgo)
            location (str): Geographic location for localized results
            language (str): Interface language (e.g., 'en', 'es', 'fr')
            country (str): Country code for country-specific results (e.g., 'us', 'uk')
            search_type (str): Type of search (web, images, news, shopping, maps)
            
        Returns:
            Dict[str, Any]: Contains search results and optional error message
        """
        # Use instance defaults if parameters not provided
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words
        
        if not self.api_key:
            error_msg = (
                "SerpAPI key is required. Please set SERPAPI_KEY environment variable "
                "or pass api_key parameter. Get your key from: https://serpapi.com/"
            )
            logger.error(error_msg)
            return {"results": [], "raw_data": None, "search_metadata": None, "error": error_msg}
        
        try:
            search_engine = engine or self.default_engine
            logger.info(f"Searching {search_engine} via SerpAPI: {query}, "
                       f"num_results={num_search_pages}, max_content_words={max_content_words}")
            
            # Build request parameters
            params = self._build_serpapi_params(
                query=query,
                engine=search_engine,
                location=location,
                language=language,
                country=country,
                search_type=search_type,
                num_results=num_search_pages
            )
            
            # Execute search using direct HTTP request
            serpapi_data = self._execute_serpapi_search(params)
            
            # Process results
            response_data = self._process_serpapi_results(serpapi_data, max_content_words)
            
            logger.info(f"Successfully retrieved {len(response_data['results'])} processed results")
            return response_data
            
        except Exception as e:
            error_msg = self._handle_api_errors(e)
            logger.error(f"SerpAPI search failed: {error_msg}")
            return {"results": [], "raw_data": None, "search_metadata": None, "error": error_msg}


class SerpAPITool(Tool):
    name: str = "serpapi_search"
    description: str = "Search multiple search engines using SerpAPI with comprehensive result processing and content scraping"
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
        "engine": {
            "type": "string",
            "description": "Search engine to use: google, bing, baidu, yahoo, duckduckgo. Default: google"
        },
        "location": {
            "type": "string", 
            "description": "Geographic location for localized results (e.g., 'New York, NY', 'London, UK')"
        },
        "language": {
            "type": "string",
            "description": "Interface language code (e.g., 'en', 'es', 'fr', 'de'). Default: en"
        },
        "country": {
            "type": "string",
            "description": "Country code for country-specific results (e.g., 'us', 'uk', 'ca'). Default: us"
        },
        "search_type": {
            "type": "string",
            "description": "Type of search: web, images, news, shopping, maps. Default: web"
        }
    }
    required: Optional[List[str]] = ["query"]
    
    def __init__(self, search_serpapi: SearchSerpAPI = None):
        super().__init__()
        self.search_serpapi = search_serpapi
    
    def __call__(self, query: str, num_search_pages: int = None, max_content_words: int = None,
                 engine: str = None, location: str = None, language: str = None, 
                 country: str = None, search_type: str = None) -> Dict[str, Any]:
        """Execute SerpAPI search using the SearchSerpAPI instance."""
        if not self.search_serpapi:
            raise RuntimeError("SerpAPI search instance not initialized")
        
        try:
            return self.search_serpapi.search(
                query=query,
                num_search_pages=num_search_pages,
                max_content_words=max_content_words,
                engine=engine,
                location=location,
                language=language,
                country=country,
                search_type=search_type
            )
        except Exception as e:
            return {"results": [], "error": f"Error executing SerpAPI search: {str(e)}"}


class SerpAPIToolkit(Toolkit):
    def __init__(
        self,
        name: str = "SerpAPIToolkit",
        api_key: Optional[str] = None,
        num_search_pages: Optional[int] = 5,
        max_content_words: Optional[int] = None,
        default_engine: Optional[str] = "google",
        default_location: Optional[str] = None,
        default_language: Optional[str] = "en",
        default_country: Optional[str] = "us",
        enable_content_scraping: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize SerpAPI Toolkit.
        
        Args:
            name (str): Name of the toolkit
            api_key (str): SerpAPI authentication key
            num_search_pages (int): Default number of search results to retrieve
            max_content_words (int): Default maximum words per result content
            default_engine (str): Default search engine
            default_location (str): Default geographic location
            default_language (str): Default interface language
            default_country (str): Default country code
            enable_content_scraping (bool): Whether to enable content scraping
            **kwargs: Additional keyword arguments
        """
        # Create the shared SerpAPI search instance
        search_serpapi = SearchSerpAPI(
            name="SearchSerpAPI",
            api_key=api_key,
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            default_engine=default_engine,
            default_location=default_location,
            default_language=default_language,
            default_country=default_country,
            enable_content_scraping=enable_content_scraping,
            **kwargs
        )
        
        # Create tools with the shared search instance
        tools = [
            SerpAPITool(search_serpapi=search_serpapi)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store search_serpapi as instance variable
        self.search_serpapi = search_serpapi


