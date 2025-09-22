from typing import Dict, Any, Optional, List

from .tool import Tool, Toolkit
from .crawler_base import CrawlerBase, PageContentHandler, AutoPageContentHandler
from .request_base import RequestBase


class RequestCrawler(CrawlerBase):
    """
    Basic HTTP request-based crawler with integrated content processing.
    
    Uses RequestBase for simple web page retrieval without browser automation.
    Suitable for straightforward web scraping tasks.
    """
    
    def __init__(self, page_content_handler: PageContentHandler, **kwargs):
        super().__init__(page_content_handler=page_content_handler, **kwargs)
        self.request_base = RequestBase()
    
    def crawl(self, url: str, query: Optional[str] = None, page_content_handler: Optional[PageContentHandler] = None) -> Dict[str, Any]:
        """
        Crawl a URL using HTTP requests and return processed content.
        
        Args:
            url: The URL to crawl
            query: Optional query for content filtering
            page_content_handler: Optional handler override for this crawl
            
        Returns:
            Dictionary containing crawl results and metadata
        """
        if not page_content_handler:
            page_content_handler = self.page_content_handler
        
        try:
            # Get raw response from RequestBase
            response = self.request_base.request_and_process(url=url)
            
            # Handle different response formats
            if isinstance(response, dict):
                # Extract content from response dict
                raw_content = response.get('content', '') or response.get('text', '')
                success = response.get('success', True)
                status_code = response.get('status_code')
                error_message = response.get('error')
            else:
                # Handle string response
                raw_content = str(response) if response else ''
                success = bool(raw_content)
                status_code = 200 if success else None
                error_message = None if success else "Empty response"
            
            # Process content with handler
            processed_content = self.handle_page_content(raw_content, query, page_content_handler)
            
            return {
                "success": success,
                "url": url,
                "content": processed_content,
                "raw_content": raw_content,
                "status_code": status_code,
                "error_message": error_message,
                "stats": {
                    "content_length": len(processed_content),
                    "raw_content_length": len(raw_content)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "content": "",
                "raw_content": "",
                "error_message": f"Crawling failed: {str(e)}",
                "stats": {"content_length": 0, "raw_content_length": 0}
            }


class RequestCrawlTool(Tool):
    """Simple HTTP-based web crawling with content processing."""
    
    name: str = "request_crawl"
    description: str = "Simple HTTP-based web crawling with automatic content processing. Best for basic web pages without complex JavaScript."
    inputs: Dict[str, Dict[str, str]] = {
        "url": {
            "type": "string",
            "description": "The URL to crawl and extract content from"
        },
        "query": {
            "type": "string",
            "description": "Optional query for content filtering and context-aware processing"
        },
        "max_content_words": {
            "type": "integer",
            "description": "Maximum number of words to include in processed content"
        },
        "output_format": {
            "type": "string",
            "description": "Output format preference: 'text', 'html', 'markdown' (default: auto-detected)"
        },
        "preferred_handler": {
            "type": "string",
            "description": "Preferred content handler: 'html2text', 'llm', 'disabled', or 'auto' (default: auto)"
        }
    }
    required: Optional[List[str]] = ["url"]
    
    def __init__(
        self,
        max_content_words: Optional[int] = None,
        preferred_handler: str = "auto",
        enable_llm: bool = True
    ):
        super().__init__()
        
        # Create auto page content handler with configuration
        self.page_content_handler = AutoPageContentHandler(
            max_length=max_content_words * 6 if max_content_words else None,  # Rough word-to-char conversion
            preferred_handler=preferred_handler if preferred_handler != "auto" else None,
            enable_llm=enable_llm,
            enable_html2text=True  # Always enabled
        )
        
        # Create the underlying crawler
        self.crawler = RequestCrawler(
            page_content_handler=self.page_content_handler
        )
    
    def __call__(self, url: str, query: str = None, max_content_words: int = None,
                 output_format: str = None, preferred_handler: str = None) -> Dict[str, Any]:
        """
        Crawl a web page using HTTP requests and return processed content.
        
        Args:
            url: The URL to crawl
            query: Optional query for content filtering
            max_content_words: Override max content words for this request
            output_format: Preferred output format (informational)
            preferred_handler: Override preferred handler for this request
            
        Returns:
            Dictionary containing crawled content and metadata
        """
        if not url:
            raise ValueError("URL is required")
        
        # Create handler override if needed
        page_content_handler = None
        if max_content_words or preferred_handler:
            page_content_handler = AutoPageContentHandler(
                max_length=max_content_words * 6 if max_content_words else None,
                preferred_handler=preferred_handler if preferred_handler and preferred_handler != "auto" else None,
                enable_llm=self.page_content_handler.enable_llm,
                enable_html2text=True  # Always enabled
            )
        
        return self.crawler.crawl(
            url=url,
            query=query,
            page_content_handler=page_content_handler
        )


class RequestCrawlToolkit(Toolkit):
    """Simple HTTP-based crawling toolkit with configurable content processing."""
    
    def __init__(
        self,
        name: str = "RequestCrawlToolkit",
        max_content_words: Optional[int] = None,
        preferred_handler: str = "auto",
        enable_llm: bool = True,
        **kwargs
    ):
        """
        Initialize Request crawling toolkit with shared configuration.
        
        Args:
            name: Name of the toolkit
            max_content_words: Maximum number of words in processed content
            preferred_handler: Preferred content handler ('html2text', 'llm', 'disabled', 'auto')
            enable_llm: Whether to enable LLM-based content processing
        """
        
        # Create request crawl tool with configuration
        request_crawl_tool = RequestCrawlTool(
            max_content_words=max_content_words,
            preferred_handler=preferred_handler,
            enable_llm=enable_llm
        )
        
        # Initialize parent with tools
        super().__init__(name=name, tools=[request_crawl_tool])