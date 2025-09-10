from typing import Dict, Any, Optional, List
from .tool import Tool, Toolkit
from .crawler_base import Crawl4AICrawler, DisabledPageContentHandler


class WebCrawlTool(Tool):
    """Simple and powerful web crawling tool using Crawl4AI."""
    
    name: str = "web_crawl"
    description: str = "Crawl web pages and extract clean, LLM-ready content. Converts web pages to Markdown with intelligent content filtering and stealth capabilities."
    inputs: Dict[str, Dict[str, str]] = {
        "url": {
            "type": "string",
            "description": "The URL to crawl and extract content from"
        },
        "output_format": {
            "type": "string",
            "description": "Output format: 'markdown' (default), 'html', 'text'"
        },
        "css_selector": {
            "type": "string",
            "description": "CSS selector to focus on specific content areas"
        },
        "word_count_threshold": {
            "type": "integer",
            "description": "Minimum word count for content extraction (default: 10)"
        },
        "include_images": {
            "type": "boolean",
            "description": "Extract and include image information (default: true)"
        },
        "include_links": {
            "type": "boolean",
            "description": "Extract and include link information (default: true)"
        },
        "wait_for": {
            "type": "string",
            "description": "Wait condition: CSS selector, time in seconds, or 'load' event"
        },
        "wait_until": {
            "type": "string",
            "description": "Wait condition for page load: 'networkidle', 'load', 'domcontentloaded' (default: networkidle)"
        },
        "wait_for_images": {
            "type": "boolean",
            "description": "Whether to wait for images to load (default: true)"
        },
        "scan_full_page": {
            "type": "boolean",
            "description": "Whether to scroll and scan the full page (default: true)"
        },
        "scroll_delay": {
            "type": "number",
            "description": "Delay between scroll actions in seconds (default: 0.5)"
        },
        "cache_mode": {
            "type": "string",
            "description": "Cache strategy: 'enabled', 'disabled', 'bypass' (default: enabled)"
        }
    }
    required: List[str] = ["url"]
    
    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        verbose: bool = False,
        user_agent: str = None,
        proxy: str = None,
        timeout: int = 30
    ):
        super().__init__()
        
        # Create the underlying crawler
        self.crawler = Crawl4AICrawler(
            page_content_handler=DisabledPageContentHandler(),
            browser_type=browser_type,
            headless=headless,
            verbose=verbose,
            user_agent=user_agent,
            proxy=proxy,
            timeout=timeout
        )
    
    def __call__(
        self, 
        url: str, 
        output_format: str = "markdown",
        css_selector: str = None,
        word_count_threshold: int = 10,
        include_images: bool = True,
        include_links: bool = True,
        take_screenshot: bool = False,
        wait_for: str = None,
        cache_mode: str = "enabled",
        wait_until: str = "networkidle",
        page_timeout: int = 3,
        wait_for_images: bool = True,
        scan_full_page: bool = True,
        scroll_delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Crawl a web page and extract clean, LLM-ready content.
        
        Returns:
            Dictionary containing crawled content and metadata
        """
        return self.crawler.crawl(
            url=url,
            query=None,  # No query filtering for basic crawling
            page_content_handler=None,  # Use the default handler
            output_format=output_format,
            css_selector=css_selector,
            word_count_threshold=word_count_threshold,
            include_images=include_images,
            include_links=include_links,
            take_screenshot=take_screenshot,
            wait_for=wait_for,
            cache_mode=cache_mode,
            wait_until=wait_until,
            page_timeout=page_timeout,
            wait_for_images=wait_for_images,
            scan_full_page=scan_full_page,
            scroll_delay=scroll_delay
        )


class Crawl4AIToolkit(Toolkit):
    """Simple and powerful toolkit for web crawling using Crawl4AI."""
    
    def __init__(
        self,
        name: str = "Crawl4AIToolkit",
        browser_type: str = "chromium",
        headless: bool = True,
        verbose: bool = False,
        user_agent: str = None,
        proxy: str = None,
        timeout: int = 30,
        **kwargs
    ):
        """Initialize Crawl4AI toolkit with shared configuration."""
        
        # Create web crawl tool with configuration
        web_crawl_tool = WebCrawlTool(
            browser_type=browser_type,
            headless=headless,
            verbose=verbose,
            user_agent=user_agent,
            proxy=proxy,
            timeout=timeout
        )
        
        # Initialize parent with tools
        super().__init__(name=name, tools=[web_crawl_tool])