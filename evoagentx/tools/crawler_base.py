from typing import Dict, Any, Optional, List

from .tool import Tool, Toolkit
from .request_base import RequestBase
import os
from ..core.module import BaseModule
from pydantic import Field
from ..models.base_model import BaseLLM, LLMOutputParser
from ..models.model_configs import LLMConfig, OpenAILLMConfig
from ..models.openai_model import OpenAILLM
from ..models.model_utils import create_llm_instance
from ..prompts.web_agent import SEARCH_RESULT_CONTENT_EXTRACTION_PROMPT

class CrawlerResultOutput(LLMOutputParser):
    title: str = Field(description="The title of the crawling result.")
    description: str = Field(description="The description of the crawling result.")
    content: str = Field(description="The content of the crawling result.")
    links: List[str] = Field(description="The links of the crawling result.")


class PageContentHandler(BaseModule):
    def handle(self, content: str, query: str = None) -> str:
        pass

class DisabledPageContentHandler(PageContentHandler):
    def handle(self, content: str, query: str = None) -> str:
        return content

class LLMPageContentHandler(PageContentHandler):
    llm: Optional[BaseLLM] = Field(default=None, description="The LLM to use for page content handling.")
    llm_config: Optional[LLMConfig] = Field(default=None, description="The LLM config to use for page content handling.")
    
    def __init__(self, llm: Optional[BaseLLM] = None, llm_config: Optional[LLMConfig] = None, **kwargs):
        super().__init__(**kwargs)
        if not self.llm or not self.llm_config:
            try:
                llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=os.getenv("OPENAI_API_KEY"))
                openai_llm = OpenAILLM(config=llm_config)
                self.llm = openai_llm
            except Exception as e:
                raise ValueError(f"Error initializing LLM: {str(e)}")
        else:
            if self.llm:
                self.llm = llm
            else:
                self.llm = create_llm_instance(llm_config)
    
    def handle(self, content: str, query: str = None) -> str:
        message = []
        message.append({"role": "system", "content": "You are a helpful assistant that can help with page content handling."})
        message.append({"role": "user", "content": SEARCH_RESULT_CONTENT_EXTRACTION_PROMPT.format(crawling_result=content, query=query)})
        
        return self.llm.generate(messages=message, parse_mode="title", parser=CrawlerResultOutput)


class CrawlerBase(BaseModule):
    """
    Base class for crawlers that retrieve information from various sources.
    Provides common functionality for crawling operations.
    """
    page_content_handler: PageContentHandler = Field(description="The handler for page content.")
    
    def __init__(self, page_content_handler: PageContentHandler, **kwargs):
        super().__init__(page_content_handler=page_content_handler, **kwargs)
    
    def crawl(self, url: str, query: str = None, page_content_handler: PageContentHandler = None) -> Dict[str, Any]:
        pass
    
    def handle_page_content(self, content: str, query: str = None, page_content_handler: PageContentHandler = None) -> str:
        if not page_content_handler:
            page_content_handler = self.page_content_handler
        
        return page_content_handler.handle(content, query)


class RequestCrawler(CrawlerBase):
    def __init__(self, page_content_handler: PageContentHandler, **kwargs):
        super().__init__(page_content_handler=page_content_handler, **kwargs)
        self.request_base = RequestBase()
    
    def crawl(self, url: str, query: str = None, page_content_handler: PageContentHandler = None) -> Dict[str, Any]:
        if not page_content_handler:
            page_content_handler = self.page_content_handler
        
        response = self.request_base.request_and_process(url=url)
        return self.handle_page_content(response, query, page_content_handler)


class Crawl4AICrawler(CrawlerBase):
    """Crawler using Crawl4AI for advanced web crawling with browser automation."""
    
    def __init__(
        self, 
        page_content_handler: PageContentHandler = None,
        browser_type: str = "chromium",
        headless: bool = True,
        verbose: bool = False,
        user_agent: str = None,
        proxy: str = None,
        timeout: int = 30,
        **kwargs
    ):
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
        query: str = None, 
        page_content_handler: PageContentHandler = None,
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
                import json
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
