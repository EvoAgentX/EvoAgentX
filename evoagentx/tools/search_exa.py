import os
from typing import Dict, Any, Optional, List
from pydantic import Field
from .search_base import SearchBase
from .tool import Tool, Toolkit
from evoagentx.core.logging import logger
import dotenv

dotenv.load_dotenv()


class SearchExa(SearchBase):
    """
    Exa search tool that provides AI-powered web search and content retrieval
    through the Exa API. Supports multiple search types including auto, fast,
    instant, and deep variants with built-in content extraction
    (highlights, full text, and summaries).
    """

    api_key: Optional[str] = Field(default=None, description="Exa API authentication key")
    search_type: Optional[str] = Field(default="auto", description="Search type: 'auto', 'fast', 'instant', 'neural', 'deep-lite', 'deep', or 'deep-reasoning'")
    content_mode: Optional[str] = Field(default="highlights", description="Content mode: 'highlights', 'text', 'summary', or 'none'")

    def __init__(
        self,
        name: str = "SearchExa",
        num_search_pages: Optional[int] = 10,
        max_content_words: Optional[int] = None,
        api_key: Optional[str] = None,
        search_type: Optional[str] = "auto",
        content_mode: Optional[str] = "highlights",
        **kwargs
    ):
        """
        Initialize the Exa Search tool.

        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            api_key (str): Exa API authentication key (can also use EXA_API_KEY env var)
            search_type (str): Search type - 'auto' (default), 'fast', 'instant', 'neural', 'deep-lite', 'deep', or 'deep-reasoning'
            content_mode (str): Content retrieval mode - 'highlights' (default), 'text', 'summary', or 'none'
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(
            name=name,
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            api_key=api_key,
            search_type=search_type,
            content_mode=content_mode,
            **kwargs
        )

        self.api_key = api_key or os.getenv("EXA_API_KEY", "")
        self.search_type = search_type
        self.content_mode = content_mode

        if not self.api_key:
            logger.warning("Exa API key not found. Set EXA_API_KEY environment variable or pass api_key parameter.")

    def _get_client(self):
        """
        Create and return an Exa client instance with integration tracking header.

        Returns:
            Exa: Configured Exa client
        """
        from exa_py import Exa

        client = Exa(api_key=self.api_key)
        client.headers["x-exa-integration"] = "evoagentx"
        return client

    def _build_contents_param(self, content_mode: str = None) -> Optional[Dict[str, Any]]:
        """
        Build the contents parameter for Exa API calls based on the content mode.

        Args:
            content_mode (str): Content mode override

        Returns:
            Optional[Dict[str, Any]]: Contents parameter dict, or None for 'none' mode
        """
        mode = content_mode or self.content_mode

        if mode == "highlights":
            return {"highlights": {"max_characters": 4000}}
        elif mode == "text":
            return {"text": {"max_characters": 10000}}
        elif mode == "summary":
            return {"summary": True}
        elif mode == "none":
            return None
        else:
            return {"highlights": {"max_characters": 4000}}

    def _process_exa_results(self, response, max_content_words: int = None, content_mode: str = None) -> Dict[str, Any]:
        """
        Process Exa API response into the standard result format.

        Args:
            response: Exa search response object
            max_content_words (int): Maximum words per result content
            content_mode (str): Content mode used for the request

        Returns:
            Dict[str, Any]: Structured response with processed results
        """
        processed_results = []
        mode = content_mode or self.content_mode

        for result in response.results:
            title = getattr(result, "title", "No Title") or "No Title"
            url = getattr(result, "url", "") or ""

            # Extract content based on mode
            content = ""
            if mode == "highlights":
                highlights = getattr(result, "highlights", None)
                if highlights:
                    content = "\n".join(highlights)
            elif mode == "text":
                content = getattr(result, "text", "") or ""
            elif mode == "summary":
                content = getattr(result, "summary", "") or ""

            if content and max_content_words:
                content = self._truncate_content(content, max_content_words)

            entry = {
                "title": title,
                "content": content,
                "url": url,
            }

            # Add optional metadata if available
            published_date = getattr(result, "published_date", None)
            if published_date:
                entry["published_date"] = published_date

            author = getattr(result, "author", None)
            if author:
                entry["author"] = author

            processed_results.append(entry)

        return {
            "results": processed_results,
            "error": None
        }

    def _handle_api_errors(self, error: Exception) -> str:
        """
        Handle Exa API specific errors with appropriate messages.

        Args:
            error (Exception): The exception that occurred

        Returns:
            str: User-friendly error message
        """
        error_str = str(error).lower()

        if "api key" in error_str or "unauthorized" in error_str or "401" in error_str:
            return "Invalid or missing Exa API key. Please set EXA_API_KEY environment variable."
        elif "rate limit" in error_str or "429" in error_str:
            return "Exa API rate limit exceeded. Please try again later."
        elif "timeout" in error_str:
            return "Exa API request timeout. Please try again."
        else:
            return f"Exa API error: {str(error)}"

    def search(
        self,
        query: str,
        num_search_pages: int = None,
        max_content_words: int = None,
        search_type: str = None,
        content_mode: str = None,
        category: str = None,
        include_domains: List[str] = None,
        exclude_domains: List[str] = None,
        include_text: List[str] = None,
        exclude_text: List[str] = None,
        user_location: str = None,
        start_published_date: str = None,
        end_published_date: str = None,
    ) -> Dict[str, Any]:
        """
        Search using the Exa API with AI-powered search capabilities.

        Args:
            query (str): The search query
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            search_type (str): Search type - 'auto' (default), 'fast', 'instant', 'neural',
                              'deep-lite', 'deep', or 'deep-reasoning'
            content_mode (str): Content mode - 'highlights', 'text', 'summary', or 'none'
            category (str): Filter by category (e.g., 'company', 'research paper', 'news',
                           'personal site', 'financial report', 'people')
            include_domains (List[str]): Only include results from these domains
            exclude_domains (List[str]): Exclude results from these domains
            include_text (List[str]): Strings that must appear in page text
            exclude_text (List[str]): Strings to exclude from results
            user_location (str): Two-letter ISO country code for localized results
            start_published_date (str): ISO 8601 date; only results published after this
            end_published_date (str): ISO 8601 date; only results published before this

        Returns:
            Dict[str, Any]: Contains search results and optional error message
        """
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words
        search_type = search_type or self.search_type
        content_mode = content_mode or self.content_mode

        if not self.api_key:
            error_msg = (
                "Exa API key is required. Please set EXA_API_KEY environment variable "
                "or pass api_key parameter. Get your key from: https://exa.ai"
            )
            logger.error(error_msg)
            return {"results": [], "error": error_msg}

        try:
            logger.info(
                f"Searching Exa: {query}, type={search_type}, "
                f"num_results={num_search_pages}, content_mode={content_mode}"
            )

            client = self._get_client()

            # Build search kwargs
            search_kwargs = {
                "query": query,
                "num_results": num_search_pages,
                "type": search_type,
            }

            # Add content retrieval parameters
            contents = self._build_contents_param(content_mode)
            if contents is not None:
                search_kwargs["contents"] = contents

            # Add optional filters
            if category:
                search_kwargs["category"] = category
            if include_domains:
                search_kwargs["include_domains"] = include_domains
            if exclude_domains:
                search_kwargs["exclude_domains"] = exclude_domains
            if include_text:
                search_kwargs["include_text"] = include_text
            if exclude_text:
                search_kwargs["exclude_text"] = exclude_text
            if user_location:
                search_kwargs["user_location"] = user_location
            if start_published_date:
                search_kwargs["start_published_date"] = start_published_date
            if end_published_date:
                search_kwargs["end_published_date"] = end_published_date

            response = client.search(**search_kwargs)

            result = self._process_exa_results(response, max_content_words, content_mode)

            logger.info(f"Successfully retrieved {len(result['results'])} results from Exa")
            return result

        except Exception as e:
            error_msg = self._handle_api_errors(e)
            logger.error(f"Exa search failed: {error_msg}")
            return {"results": [], "error": error_msg}


class ExaSearchTool(Tool):
    name: str = "exa_search"
    description: str = (
        "Search the web using Exa's AI-powered search engine. "
        "Supports multiple search types (auto, fast, instant, deep) with "
        "built-in content extraction including highlights, full text, and summaries."
    )
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
        "search_type": {
            "type": "string",
            "description": "Search type: 'auto' (default), 'fast', 'instant', 'neural', 'deep-lite', 'deep', or 'deep-reasoning'"
        },
        "content_mode": {
            "type": "string",
            "description": "Content retrieval mode: 'highlights' (default), 'text' (full), 'summary', or 'none'"
        },
        "category": {
            "type": "string",
            "description": "Filter by category: 'company', 'research paper', 'news', 'personal site', 'financial report', 'people'"
        },
        "include_domains": {
            "type": "string",
            "description": "Comma-separated list of domains to include (e.g., 'arxiv.org,github.com')"
        },
        "exclude_domains": {
            "type": "string",
            "description": "Comma-separated list of domains to exclude"
        },
        "include_text": {
            "type": "string",
            "description": "Text that must appear in page content"
        },
        "exclude_text": {
            "type": "string",
            "description": "Text to exclude from results"
        },
        "user_location": {
            "type": "string",
            "description": "Two-letter ISO country code for localized results (e.g., 'US', 'GB')"
        },
        "start_published_date": {
            "type": "string",
            "description": "ISO 8601 date; only results published after this date (e.g., '2024-01-01T00:00:00.000Z')"
        },
        "end_published_date": {
            "type": "string",
            "description": "ISO 8601 date; only results published before this date"
        }
    }
    required: Optional[List[str]] = ["query"]

    def __init__(self, search_exa: SearchExa = None):
        super().__init__()
        self.search_exa = search_exa

    def __call__(
        self,
        query: str,
        num_search_pages: int = None,
        max_content_words: int = None,
        search_type: str = None,
        content_mode: str = None,
        category: str = None,
        include_domains: str = None,
        exclude_domains: str = None,
        include_text: str = None,
        exclude_text: str = None,
        user_location: str = None,
        start_published_date: str = None,
        end_published_date: str = None,
    ) -> Dict[str, Any]:
        """Execute Exa search using the SearchExa instance."""
        if not self.search_exa:
            raise RuntimeError("Exa search instance not initialized")

        try:
            # Parse comma-separated domain lists from string inputs
            include_domain_list = [d.strip() for d in include_domains.split(",")] if include_domains else None
            exclude_domain_list = [d.strip() for d in exclude_domains.split(",")] if exclude_domains else None
            include_text_list = [include_text] if include_text else None
            exclude_text_list = [exclude_text] if exclude_text else None

            return self.search_exa.search(
                query=query,
                num_search_pages=num_search_pages,
                max_content_words=max_content_words,
                search_type=search_type,
                content_mode=content_mode,
                category=category,
                include_domains=include_domain_list,
                exclude_domains=exclude_domain_list,
                include_text=include_text_list,
                exclude_text=exclude_text_list,
                user_location=user_location,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
            )
        except Exception as e:
            return {"results": [], "error": f"Error executing Exa search: {str(e)}"}


class ExaSearchToolkit(Toolkit):
    def __init__(
        self,
        name: str = "ExaSearchToolkit",
        api_key: Optional[str] = None,
        num_search_pages: Optional[int] = 10,
        max_content_words: Optional[int] = None,
        search_type: Optional[str] = "auto",
        content_mode: Optional[str] = "highlights",
        **kwargs
    ):
        """
        Initialize Exa Search Toolkit.

        Args:
            name (str): Name of the toolkit
            api_key (str): Exa API authentication key
            num_search_pages (int): Default number of search results to retrieve
            max_content_words (int): Default maximum words per result content
            search_type (str): Default search type - 'auto', 'fast', 'instant', 'neural', 'deep-lite', 'deep', or 'deep-reasoning'
            content_mode (str): Default content mode - 'highlights', 'text', 'summary', or 'none'
            **kwargs: Additional keyword arguments
        """
        # Create the shared Exa search instance
        search_exa = SearchExa(
            name="SearchExa",
            api_key=api_key,
            num_search_pages=num_search_pages,
            max_content_words=max_content_words,
            search_type=search_type,
            content_mode=content_mode,
            **kwargs
        )

        # Create tools with the shared search instance
        tools = [
            ExaSearchTool(search_exa=search_exa)
        ]

        # Initialize parent with tools
        super().__init__(name=name, tools=tools)

        # Store search_exa as instance variable
        self.search_exa = search_exa
