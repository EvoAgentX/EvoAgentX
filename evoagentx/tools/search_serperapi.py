import os
import requests
from typing import Dict, Any, Optional, List
from pydantic import Field
from .search_base import SearchBase
from .tool import Tool, Toolkit
from evoagentx.core.logging import logger
import dotenv

dotenv.load_dotenv()

class SearchSerperAPI(SearchBase):
    """
    SerperAPI search tool that provides access to Google search results
    through a simple and efficient API interface.
    """
    
    api_key: Optional[str] = Field(default=None, description="SerperAPI authentication key")
    default_location: Optional[str] = Field(default=None, description="Default geographic location")
    default_language: Optional[str] = Field(default="en", description="Default interface language")
    default_country: Optional[str] = Field(default="us", description="Default country code")
    enable_content_scraping: Optional[bool] = Field(default=True, description="Enable full content scraping")
    
    def __init__(
        self,
        name: str = "SearchSerperAPI",
        num_search_pages: Optional[int] = 10,
        max_content_words: Optional[int] = None,
        api_key: Optional[str] = None,
        default_location: Optional[str] = None,
        default_language: Optional[str] = "en",
        default_country: Optional[str] = "us",
        enable_content_scraping: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize the SerperAPI Search tool.
        
        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            api_key (str): SerperAPI authentication key (can also use SERPERAPI_KEY env var)
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
            default_location=default_location,
            default_language=default_language,
            default_country=default_country,
            enable_content_scraping=enable_content_scraping,
            **kwargs
        )
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('SERPERAPI_KEY', '')
        self.base_url = "https://google.serper.dev/search"
        
        if not self.api_key:
            logger.warning("SerperAPI key not found. Set SERPERAPI_KEY environment variable or pass api_key parameter.")

    def _build_serperapi_payload(self, query: str, location: str = None, 
                                language: str = None, country: str = None,
                                num_results: int = None) -> Dict[str, Any]:
        """
        Build SerperAPI request payload.
        
        Args:
            query (str): Search query
            location (str): Geographic location
            language (str): Interface language
            country (str): Country code
            num_results (int): Number of results to retrieve
            
        Returns:
            Dict[str, Any]: SerperAPI request payload
        """
        payload = {
            "q": query
        }
        
        # Add optional parameters if provided
        if num_results:
            payload["num"] = num_results
            
        if location or self.default_location:
            payload["location"] = location or self.default_location
            
        if language or self.default_language:
            payload["hl"] = language or self.default_language
            
        if country or self.default_country:
            payload["gl"] = country or self.default_country
        
        return payload

    def _execute_serperapi_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search using direct HTTP POST requests to SerperAPI.
        
        Args:
            payload (Dict[str, Any]): Search payload
            
        Returns:
            Dict[str, Any]: SerperAPI response data
            
        Raises:
            Exception: For API errors
        """
        try:
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for SerperAPI errors in response
            if "error" in data:
                raise Exception(f"SerperAPI error: {data['error']}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"SerperAPI request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"SerperAPI search failed: {str(e)}")

    def _process_serperapi_results(self, serperapi_data: Dict[str, Any], max_content_words: int = None) -> Dict[str, Any]:
        """
        Process SerperAPI results into structured format with processed results + raw data.
        
        Args:
            serperapi_data (Dict[str, Any]): Raw SerperAPI response
            max_content_words (int): Maximum words per result content
            
        Returns:
            Dict[str, Any]: Structured response with processed results and raw data
        """
        processed_results = []
        
        # 1. Process Knowledge Graph (highest priority)
        if knowledge_graph := serperapi_data.get("knowledgeGraph", {}):
            if description := knowledge_graph.get("description"):
                title = knowledge_graph.get("title", "Unknown")
                content = f"**{title}**\n\n{description}"
                
                # Add attributes if available
                if attributes := knowledge_graph.get("attributes", {}):
                    content += "\n\n**Key Information:**"
                    for key, value in list(attributes.items())[:5]:  # Limit to 5 attributes
                        formatted_key = key.replace('_', ' ').title()
                        content += f"\n• {formatted_key}: {value}"
                
                processed_results.append({
                    "title": f"Knowledge: {title}",
                    "content": self._truncate_content(content, max_content_words or 200),
                    "url": knowledge_graph.get("descriptionLink", ""),
                    "type": "knowledge_graph",
                    "priority": 1
                })
        
        # 2. Process Organic Results with scraping
        for item in serperapi_data.get("organic", []):
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
        raw_sections = ["relatedSearches"]  # SerperAPI specific sections
        
        for section in raw_sections:
            if section in serperapi_data and serperapi_data[section]:
                raw_data[section] = serperapi_data[section][:5]  # Limit to 5 items
        
        # 4. Extract search metadata
        search_metadata = {}
        if search_params := serperapi_data.get("searchParameters", {}):
            search_metadata = {
                "query": search_params.get("q", ""),
                "engine": search_params.get("engine", ""),
                "type": search_params.get("type", ""),
                "credits": serperapi_data.get("credits", 0)
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
        Handle SerperAPI specific errors with appropriate messages.
        
        Args:
            error (Exception): The exception that occurred
            
        Returns:
            str: User-friendly error message
        """
        error_str = str(error).lower()
        
        if "api key" in error_str or "unauthorized" in error_str:
            return "Invalid or missing SerperAPI key. Please set SERPERAPI_KEY environment variable."
        elif "rate limit" in error_str or "too many requests" in error_str:
            return "SerperAPI rate limit exceeded. Please try again later."
        elif "quota" in error_str or "credit" in error_str:
            return "SerperAPI quota exceeded. Please check your plan limits."
        elif "timeout" in error_str:
            return "SerperAPI request timeout. Please try again."
        else:
            return f"SerperAPI error: {str(error)}"

    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None,
               location: str = None, language: str = None, country: str = None) -> Dict[str, Any]:
        """
        Search using SerperAPI with comprehensive parameter support.
        
        Args:
            query (str): The search query
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            location (str): Geographic location for localized results
            language (str): Interface language (e.g., 'en', 'es', 'fr')
            country (str): Country code for country-specific results (e.g., 'us', 'uk')
            
        Returns:
            Dict[str, Any]: Contains search results and optional error message
        """
        # Use instance defaults if parameters not provided
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words
        
        if not self.api_key:
            error_msg = (
                "SerperAPI key is required. Please set SERPERAPI_KEY environment variable "
                "or pass api_key parameter. Get your key from: https://serper.dev/"
            )
            logger.error(error_msg)
            return {"results": [], "raw_data": None, "search_metadata": None, "error": error_msg}
        
        try:
            logger.info(f"Searching SerperAPI: {query}, "
                       f"num_results={num_search_pages}, max_content_words={max_content_words}")
            
            # Build request payload
            payload = self._build_serperapi_payload(
                query=query,
                location=location,
                language=language,
                country=country,
                num_results=num_search_pages
            )
            
            # Execute search using direct HTTP request
            serperapi_data = self._execute_serperapi_search(payload)
            
            # Process results
            response_data = self._process_serperapi_results(serperapi_data, max_content_words)
            
            logger.info(f"Successfully retrieved {len(response_data['results'])} processed results")
            return response_data
            
        except Exception as e:
            error_msg = self._handle_api_errors(e)
            logger.error(f"SerperAPI search failed: {error_msg}")
            return {"results": [], "raw_data": None, "search_metadata": None, "error": error_msg}


class SerperAPITool(Tool):
    name: str = "serperapi_search"
    description: str = "Search Google using SerperAPI with comprehensive result processing and content scraping"
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "The search query to execute"
        },
        "num_search_pages": {
            "type": "integer", 
            "description": "Number of search results to retrieve. Default: 10"
        },
        "max_content_words": {
            "type": "integer",
            "description": "Maximum number of words to include in content per result. None means no limit. Default: None"
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
        }
    }
    required: Optional[List[str]] = ["query"]
    
    def __init__(self, search_serperapi: SearchSerperAPI = None):
        super().__init__()
        self.search_serperapi = search_serperapi
    
    def __call__(self, query: str, num_search_pages: int = None, max_content_words: int = None,
                 location: str = None, language: str = None, country: str = None) -> Dict[str, Any]:
        """Execute SerperAPI search using the SearchSerperAPI instance."""
        if not self.search_serperapi:
            raise RuntimeError("SerperAPI search instance not initialized")
        
        try:
            return self.search_serperapi.search(
                query=query,
                num_search_pages=num_search_pages,
                max_content_words=max_content_words,
                location=location,
                language=language,
                country=country
            )
        except Exception as e:
            return {"results": [], "error": f"Error executing SerperAPI search: {str(e)}"}


class SerperAPIToolkit(Toolkit):
    def __init__(
        self,
        name: str = "SerperAPIToolkit",
        api_key: Optional[str] = None,
        num_search_pages: Optional[int] = 10,
        max_content_words: Optional[int] = None,
        default_location: Optional[str] = None,
        default_language: Optional[str] = "en",
        default_country: Optional[str] = "us",
        enable_content_scraping: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize SerperAPI Toolkit.
        
        Args:
            name (str): Name of the toolkit
            api_key (str): SerperAPI authentication key
            num_search_pages (int): Default number of search results to retrieve
            max_content_words (int): Default maximum words per result content
            default_location (str): Default geographic location
            default_language (str): Default interface language
            default_country (str): Default country code
            enable_content_scraping (bool): Whether to enable content scraping
            **kwargs: Additional keyword arguments
        """
        # Create the shared SerperAPI search instance
        search_serperapi = SearchSerperAPI(
            name="SearchSerperAPI",
            api_key=api_key,
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            default_location=default_location,
            default_language=default_language,
            default_country=default_country,
            enable_content_scraping=enable_content_scraping,
            **kwargs
        )
        
        # Create tools with the shared search instance
        tools = [
            SerperAPITool(search_serperapi=search_serperapi)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store search_serperapi as instance variable
        self.search_serperapi = search_serperapi
