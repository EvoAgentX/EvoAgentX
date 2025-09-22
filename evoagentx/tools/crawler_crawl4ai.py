from typing import Dict, Any, Optional, List
import asyncio
import json

from .tool import Tool, Toolkit
from .crawler_base import CrawlerBase, PageContentHandler, DisabledPageContentHandler, AutoPageContentHandler


class Crawl4AICrawler(CrawlerBase):
    """Crawler using Crawl4AI for advanced web crawling with browser automation."""
    
    def __init__(
        self, 
        page_content_handler: Optional[PageContentHandler] = None,
        browser_type: str = "chromium",
        headless: bool = True,
        verbose: bool = False,
        user_agent: Optional[str] = None,
        proxy: Optional[str] = None,
        timeout: int = 30,
        **kwargs
    ):
        if not page_content_handler:
            page_content_handler = AutoPageContentHandler()
        super().__init__(page_content_handler=page_content_handler, **kwargs)
        
        # Handle optional crawl4ai import
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
            self.crawl4ai_available = True
        except ImportError:
            self.crawl4ai_available = False
            raise ImportError(
                "crawl4ai is not installed. Please install it with: pip install crawl4ai"
            )
        
        self.browser_type = browser_type
        self.headless = headless
        self.verbose = verbose
        self.user_agent = user_agent
        self.proxy = proxy
        self.timeout = timeout
    
    def crawl(
        self, 
        url: str, 
        query: Optional[str] = None, 
        page_content_handler: Optional[PageContentHandler] = None,
        output_format: str = "markdown",
        css_selector: Optional[str] = None,
        word_count_threshold: int = 10,
        include_images: bool = True,
        include_links: bool = True,
        take_screenshot: bool = False,
        wait_for: Optional[str] = None,
        cache_mode: str = "enabled",
        wait_until: str = "networkidle",
        page_timeout: int = 3,
        wait_for_images: bool = True,
        scan_full_page: bool = True,
        scroll_delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Crawl a web page using Crawl4AI.
        
        Args:
            url: The URL to crawl
            query: Optional query for content filtering
            page_content_handler: Handler for processing the content
            output_format: Output format ('markdown', 'html', 'text')
            css_selector: CSS selector to focus on specific content
            word_count_threshold: Minimum word count for content extraction
            include_images: Whether to include image information
            include_links: Whether to include link information
            take_screenshot: Whether to take a screenshot
            wait_for: Wait condition for dynamic content (CSS selector, time in seconds, or 'load')
            cache_mode: Cache strategy ('enabled', 'disabled', 'bypass')
            wait_until: Wait condition for page load ('networkidle', 'load', 'domcontentloaded')
            page_timeout: Maximum time to wait for page load in seconds
            wait_for_images: Whether to wait for images to load
            scan_full_page: Whether to scroll and scan the full page
            scroll_delay: Delay between scroll actions in seconds
            
        Returns:
            Dictionary containing crawled content and metadata
        """
        if not self.crawl4ai_available:
            return self._error_result(url, "Crawl4AI is not available")
        
        try:
            # Check if we're already in an async context
            import asyncio
            asyncio.get_running_loop()
            # We're in an async context, run in a separate thread
            return self._run_in_thread(
                url, query, page_content_handler, output_format, css_selector,
                word_count_threshold, include_images, include_links, take_screenshot,
                wait_for, cache_mode, wait_until, page_timeout, wait_for_images,
                scan_full_page, scroll_delay
            )
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self._async_crawl(
                url, query, page_content_handler, output_format, css_selector,
                word_count_threshold, include_images, include_links, take_screenshot,
                wait_for, cache_mode, wait_until, page_timeout, wait_for_images,
                scan_full_page, scroll_delay
            ))
    
    def _run_in_thread(self, url, query, page_content_handler, output_format, css_selector,
                      word_count_threshold, include_images, include_links, take_screenshot,
                      wait_for, cache_mode, wait_until, page_timeout, wait_for_images,
                      scan_full_page, scroll_delay):
        """Run the async crawl in a separate thread to avoid event loop conflicts."""
        import concurrent.futures
        import asyncio
        
        def run_async():
            return asyncio.run(self._async_crawl(
                url, query, page_content_handler, output_format, css_selector,
                word_count_threshold, include_images, include_links, take_screenshot,
                wait_for, cache_mode, wait_until, page_timeout, wait_for_images,
                scan_full_page, scroll_delay
            ))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async)
            try:
                return future.result(timeout=60)
            except concurrent.futures.TimeoutError:
                return self._error_result(url, "Crawling timed out after 60 seconds")
            except Exception as e:
                return self._error_result(url, f"Crawling failed: {str(e)}")
    
    async def _async_crawl(
        self, 
        url: str, 
        query: str,
        page_content_handler: PageContentHandler,
        output_format: str,
        css_selector: str,
        word_count_threshold: int,
        include_images: bool,
        include_links: bool,
        take_screenshot: bool,
        wait_for: str,
        cache_mode: str,
        wait_until: str,
        page_timeout: int,
        wait_for_images: bool,
        scan_full_page: bool,
        scroll_delay: float
    ) -> Dict[str, Any]:
        """Async implementation of web crawling."""
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
            
            # Create browser config
            config_dict = {
                "browser_type": self.browser_type,
                "headless": self.headless,
                "verbose": self.verbose
            }
            
            if self.user_agent:
                config_dict["user_agent"] = self.user_agent
            if self.proxy:
                config_dict["proxy"] = self.proxy
                
            browser_config = BrowserConfig(**config_dict)
            
            # Create crawler run config
            cache_mode_mapping = {
                "enabled": "enabled",
                "disabled": "disabled", 
                "bypass": "bypass"
            }
            
            cache_enum_value = getattr(CacheMode, cache_mode_mapping.get(cache_mode, "enabled").upper())
            
            run_config_dict = {
                "cache_mode": cache_enum_value,
                "word_count_threshold": word_count_threshold,
                "screenshot": take_screenshot,
                "process_iframes": True,
                "remove_overlay_elements": True,
                "wait_until": wait_until,
                "page_timeout": page_timeout,  # Convert to milliseconds
                "wait_for_images": wait_for_images,
                "scan_full_page": scan_full_page,
                "scroll_delay": scroll_delay
            }
            
            if css_selector:
                run_config_dict["css_selector"] = css_selector
            if wait_for:
                run_config_dict["wait_for"] = wait_for
                
            run_config = CrawlerRunConfig(**run_config_dict)
            
            # Create and use crawler
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)
                processed_result = self._process_crawl4ai_result(result, output_format, include_images, include_links)
                
                # Return the processed result directly
                return processed_result
                
        except Exception as e:
            return self._error_result(url, f"Crawling failed: {str(e)}")
    
    def _process_crawl4ai_result(self, result, output_format: str, include_images: bool, include_links: bool) -> Dict[str, Any]:
        """Process Crawl4AI result into standardized format."""
        processed = {
            "success": result.success,
            "url": result.url,
            "title": getattr(result, 'title', ''),
            "status_code": getattr(result, 'status_code', None),
            "content": "",
            "raw_html": result.html or "",
            "cleaned_html": result.cleaned_html or "",
            "markdown": result.markdown or "",
            "links": result.links or {} if include_links else {},
            "media": result.media or {} if include_images else {},
            "metadata": getattr(result, 'metadata', {}),
            "error_message": result.error_message if not result.success else None
        }
        
        # Set primary content based on format
        if output_format == "markdown":
            processed["content"] = result.markdown or ""
        elif output_format == "html":
            processed["content"] = result.cleaned_html or result.html or ""
        elif output_format == "text":
            # Extract text from markdown by removing markdown syntax
            import re
            text = result.markdown or ""
            # Remove markdown syntax
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
            text = re.sub(r'[#*`_~]', '', text)  # Formatting
            processed["content"] = text
        else:
            processed["content"] = result.markdown or ""
        
        # Add extracted content if available
        if hasattr(result, 'extracted_content') and result.extracted_content:
            try:
                processed["extracted_content"] = json.loads(result.extracted_content)
            except (json.JSONDecodeError, TypeError):
                processed["extracted_content"] = result.extracted_content
        
        # Add media files if available
        if hasattr(result, 'screenshot') and result.screenshot:
            processed["screenshot"] = result.screenshot
        
        if hasattr(result, 'pdf') and result.pdf:
            processed["pdf"] = result.pdf
            
        # Add statistics
        processed["stats"] = {
            "content_length": len(processed["content"]),
            "html_length": len(processed["raw_html"]),
            "links_count": len(processed["links"].get('internal', [])) + len(processed["links"].get('external', [])),
            "images_count": len(processed["media"].get('images', []))
        }
        
        return processed
    
    def _error_result(self, url: str, error_message: str) -> Dict[str, Any]:
        """Create a standardized error result."""
        return {
            "success": False,
            "url": url,
            "content": "",
            "error_message": error_message,
            "stats": {"content_length": 0, "html_length": 0, "links_count": 0, "images_count": 0}
        }


class Crawl4AICrawlTool(Tool):
    """Advanced browser-based crawling with Crawl4AI."""
    
    name: str = "crawl4ai_crawl"
    description: str = "Advanced browser-based web crawling using Crawl4AI. Supports JavaScript rendering, dynamic content, screenshots, and comprehensive content extraction."
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
        },
        "take_screenshot": {
            "type": "boolean",
            "description": "Whether to take a screenshot of the page (default: false)"
        }
    }
    required: Optional[List[str]] = ["url"]
    
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
        output_format: str = 'markdown',
        css_selector: str = None,
        word_count_threshold: int = 10,
        include_images: bool = True,
        include_links: bool = True,
        take_screenshot: bool = False,
        wait_for: str = None,
        cache_mode: str = 'enabled',
        wait_until: str = 'networkidle',
        page_timeout: int = 3,
        wait_for_images: bool = True,
        scan_full_page: bool = True,
        scroll_delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Crawl a web page using Crawl4AI and extract comprehensive content.
        
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


class Crawl4AICrawlToolkit(Toolkit):
    """Advanced browser-based crawling toolkit using Crawl4AI."""
    
    def __init__(
        self,
        name: str = "Crawl4AICrawlToolkit",
        browser_type: str = "chromium",
        headless: bool = True,
        verbose: bool = False,
        user_agent: str = None,
        proxy: str = None,
        timeout: int = 30
    ):
        """
        Initialize Crawl4AI crawling toolkit with shared configuration.
        
        Args:
            name: Name of the toolkit
            browser_type: Browser type to use ('chromium', 'firefox', etc.)
            headless: Whether to run browser in headless mode
            verbose: Whether to enable verbose logging
            user_agent: Custom user agent string
            proxy: Proxy configuration
            timeout: Request timeout in seconds
        """
        
        # Create crawl4ai crawl tool with configuration
        crawl4ai_crawl_tool = Crawl4AICrawlTool(
            browser_type=browser_type,
            headless=headless,
            verbose=verbose,
            user_agent=user_agent,
            proxy=proxy,
            timeout=timeout
        )
        
        # Initialize parent with tools
        super().__init__(name=name, tools=[crawl4ai_crawl_tool])