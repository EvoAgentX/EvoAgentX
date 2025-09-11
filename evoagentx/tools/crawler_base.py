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
    """
    Base class for page content handlers.
    
    Provides common functionality for processing web page content, including
    optional truncation capabilities that can be used by all subclasses.
    """
    max_length: Optional[int] = Field(default=None, description="Maximum length of content before truncation")
    suffix: str = Field(default="...", description="Suffix to add when truncating content")
    
    def __init__(self, max_length: Optional[int] = None, suffix: str = "...", **kwargs):
        super().__init__(max_length=max_length, suffix=suffix, **kwargs)
    
    def handle(self, content: str, query: str = None) -> str:
        """
        Process page content. Subclasses should override this method.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            
        Returns:
            Processed content string
        """
        pass
    
    def _truncate_content(self, content: str) -> str:
        """
        Truncate content if max_length is specified.
        
        Args:
            content: Content to potentially truncate
            
        Returns:
            Truncated content if needed, otherwise original content
        """
        if self.max_length is None or len(content) <= self.max_length:
            return content
        
        return content[:self.max_length - len(self.suffix)] + self.suffix

class DisabledPageContentHandler(PageContentHandler):
    """
    A no-op page content handler that returns content unchanged.
    
    Useful when you want to skip content processing and work with raw content.
    Supports optional truncation if max_length is specified.
    """
    def handle(self, content: str, query: str = None) -> str:
        return self._truncate_content(content)


class HTML2TextPageContentHandler(PageContentHandler):
    """
    Page content handler that converts HTML to clean text using html2text.
    
    This handler is useful for extracting readable text content from HTML pages,
    with options to control the conversion process and optional truncation.
    """
    ignore_links: bool = Field(default=False, description="Whether to ignore links in the conversion")
    ignore_images: bool = Field(default=True, description="Whether to ignore images in the conversion")
    body_width: int = Field(default=0, description="Width for text wrapping (0 = no wrapping)")
    unicode_snob: bool = Field(default=True, description="Whether to use unicode characters")
    escape_snob: bool = Field(default=True, description="Whether to escape special characters")
    
    def __init__(
        self, 
        ignore_links: bool = False, 
        ignore_images: bool = True, 
        body_width: int = 0,
        unicode_snob: bool = True,
        escape_snob: bool = True,
        **kwargs
    ):
        super().__init__(
            ignore_links=ignore_links, 
            ignore_images=ignore_images, 
            body_width=body_width,
            unicode_snob=unicode_snob,
            escape_snob=escape_snob,
            **kwargs
        )
        
        # Initialize html2text converter
        import html2text
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = self.ignore_links
        self.html_converter.ignore_images = self.ignore_images
        self.html_converter.body_width = self.body_width
        self.html_converter.unicode_snob = self.unicode_snob
        self.html_converter.escape_snob = self.escape_snob
    
    def handle(self, content: str, query: str = None) -> str:
        """
        Convert HTML content to clean text.
        
        Args:
            content: HTML content to convert
            query: Optional query for context (not used in this handler)
            
        Returns:
            Clean text content, optionally truncated
        """
        # Convert HTML to text
        text_content = self.html_converter.handle(content)
        
        # Apply truncation if specified
        return self._truncate_content(text_content)


class LLMPageContentHandler(PageContentHandler):
    """
    Page content handler that uses an LLM to extract and process content.
    
    This handler can be configured with either a pre-initialized LLM instance
    or an LLM configuration that will be used to create the LLM instance.
    """
    llm: Optional[BaseLLM] = Field(default=None, description="The LLM to use for page content handling.")
    llm_config: Optional[LLMConfig] = Field(default=None, description="The LLM config to use for page content handling.")
    
    def __init__(self, llm: Optional[BaseLLM] = None, llm_config: Optional[LLMConfig] = None, **kwargs):
        super().__init__(**kwargs)
        
        # Priority: provided llm > provided llm_config > default fallback
        if llm is not None:
            self.llm = llm
        elif llm_config is not None:
            self.llm = create_llm_instance(llm_config)
        else:
            # Default fallback to OpenAI
            try:
                default_config = OpenAILLMConfig(
                    model="gpt-4o-mini", 
                    openai_key=os.getenv("OPENAI_API_KEY")
                )
                self.llm = OpenAILLM(config=default_config)
            except Exception as e:
                raise ValueError(f"Error initializing default LLM: {str(e)}")
    
    def handle(self, content: str, query: str = None) -> str:
        """
        Process content using LLM and return structured result.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            
        Returns:
            Processed content with title, description, content, and links
        """
        try:
            message = []
            message.append({"role": "system", "content": "You are a helpful assistant that can help with page content handling."})
            message.append({"role": "user", "content": SEARCH_RESULT_CONTENT_EXTRACTION_PROMPT.format(crawling_result=content, query=query)})
        
            # Try to get structured result first
            try:
                result = self.llm.generate(messages=message, parse_mode="title", parser=CrawlerResultOutput)
                
                # Format the result as a readable string
                formatted_result = f"Title: {result.title}\n\nDescription: {result.description}\n\nContent: {result.content}"
                
                if result.links:
                    formatted_result += f"\n\nLinks: {', '.join(result.links)}"
                
                return self._truncate_content(formatted_result)
                
            except Exception as parse_error:
                # If structured parsing fails, try to get raw text response
                print(f"Structured parsing failed: {str(parse_error)}, trying raw text response")
                
                # Get raw text response without parser
                raw_result = self.llm.generate(messages=message)
                
                # Extract basic information from raw response
                title = "Extracted Content"
                description = ""
                content_text = str(raw_result)
                
                # Try to extract title from content
                if '<title>' in content.lower():
                    import re
                    title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip()
                
                # Try to extract description from first paragraph
                lines = content.split('\n')
                for line in lines:
                    if line.strip() and not line.strip().startswith('<') and not line.strip().startswith('#'):
                        description = line.strip()[:200] + "..." if len(line.strip()) > 200 else line.strip()
                        break
                
                formatted_result = f"Title: {title}\n\nDescription: {description}\n\nContent: {content_text[:500]}{'...' if len(content_text) > 500 else ''}"
                
                return self._truncate_content(formatted_result)
            
        except Exception as e:
            # If all LLM processing fails, fall back to simple text extraction
            print(f"LLM processing failed: {str(e)}, falling back to simple extraction")
            
            # Simple fallback: extract basic information
            lines = content.split('\n')
            title = ""
            description = ""
            
            # Try to find title
            for line in lines:
                if '<title>' in line.lower():
                    title = line.replace('<title>', '').replace('</title>', '').strip()
                    break
                elif line.strip().startswith('# '):
                    title = line.strip()[2:].strip()
                    break
            
            # Try to find description
            for line in lines:
                if line.strip() and not line.strip().startswith('<') and not line.strip().startswith('#'):
                    description = line.strip()[:200] + "..." if len(line.strip()) > 200 else line.strip()
                    break
            
            formatted_result = f"Title: {title}\n\nDescription: {description}\n\nContent: {content[:500]}{'...' if len(content) > 500 else ''}"
            
            return self._truncate_content(formatted_result)


class AutoPageContentHandler(PageContentHandler):
    """
    Simple auto page content handler that intelligently selects the best handler.
    
    Automatically chooses between available handlers based on content type and query.
    Falls back gracefully if the preferred handler fails.
    """
    preferred_handler: Optional[str] = Field(
        default=None, 
        description="Preferred handler to use (html2text, llm, disabled). If None, auto-selects."
    )
    enable_llm: bool = Field(default=True, description="Whether to enable LLM processing")
    enable_html2text: bool = Field(default=True, description="Whether to enable HTML2Text processing")
    
    def __init__(
        self,
        preferred_handler: str = None,
        enable_llm: bool = True,
        enable_html2text: bool = True,
        **kwargs
    ):
        super().__init__(
            preferred_handler=preferred_handler,
            enable_llm=enable_llm,
            enable_html2text=enable_html2text,
            **kwargs
        )
        
        # Initialize available handlers
        self._handlers = {}
        self._initialize_handlers()
    
    def _initialize_handlers(self):
        """Initialize available handlers based on configuration and environment."""
        # Always available handlers
        self._handlers["disabled"] = DisabledPageContentHandler(
            max_length=self.max_length,
            suffix=self.suffix
        )
        
        # HTML2Text handler
        if self.enable_html2text:
            try:
                self._handlers["html2text"] = HTML2TextPageContentHandler(
                    max_length=self.max_length,
                    suffix=self.suffix,
                    ignore_images=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize HTML2Text handler: {e}")
        
        # LLM handler - only if API key is available
        if self.enable_llm and os.getenv("OPENAI_API_KEY"):
            try:
                self._handlers["llm"] = LLMPageContentHandler(
                    max_length=self.max_length,
                    suffix=self.suffix
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM handler: {e}")
    
    def _generate_handler_order(self, content: str, query: str = None) -> List[str]:
        """Generate the default execution order based on content and query."""
        # If preferred handler is specified and available, use it first
        if self.preferred_handler and self.preferred_handler in self._handlers:
            return [self.preferred_handler] + [h for h in self._handlers.keys() if h != self.preferred_handler]
        
        # Default order based on content analysis
        is_html = any(tag in content.lower() for tag in ['<html', '<body', '<div', '<p', '<h1'])
        has_query = bool(query and query.strip())
        is_long = len(content) > 1000
        
        # Build order based on content type and available handlers
        order = []
        
        if is_html and has_query and is_long and "llm" in self._handlers:
            # Complex HTML with query - prefer LLM
            order.append("llm")
        
        if is_html and "html2text" in self._handlers:
            # HTML content - use HTML2Text
            order.append("html2text")
        
        if has_query and "llm" in self._handlers and "llm" not in order:
            # Plain text with query - use LLM
            order.append("llm")
        
        # Always add disabled as final fallback
        order.append("disabled")
        
        return order
    
    def _should_handle(self, handler_name: str, content: str, query: str = None) -> bool:
        """Determine if we should use this handler based on content and query."""
        if handler_name not in self._handlers:
            return False
        
        # Always allow disabled handler as fallback
        if handler_name == "disabled":
            return True
        
        # LLM handler: use for complex content with queries
        if handler_name == "llm":
            has_query = bool(query and query.strip())
            is_complex = len(content) > 500 or any(tag in content.lower() for tag in ['<title', '<meta', '<h1', '<h2'])
            return has_query and is_complex
        
        # HTML2Text handler: use for HTML content
        if handler_name == "html2text":
            return any(tag in content.lower() for tag in ['<html', '<body', '<div', '<p', '<h1'])
        
        # For any other handler, allow it
        return True
    
    def handle(self, content, query: str = None):
        """
        Process content using the best available handler with automatic fallback.
        
        Args:
            content: Content to process (str or dict)
            query: Optional query for context
            
        Returns:
            Processed content (str or dict)
        """
        # Handle dictionary content with DisabledHandler
        if isinstance(content, dict):
            return self._handlers["disabled"].handle(str(content), query)
        
        # Generate handler order
        handler_order = self._generate_handler_order(content, query)
        
        # Try handlers in order
        for handler_name in handler_order:
            if not self._should_handle(handler_name, content, query):
                continue
            
            try:
                handler = self._handlers[handler_name]
                result = handler.handle(content, query)
                return self._truncate_content(result)
            except Exception as e:
                print(f"AutoPageHandler: Handler {handler_name} failed: {str(e)}, trying next...")
                continue
        
        # If all handlers failed, use disabled handler as final fallback
        return self._handlers["disabled"].handle(content, query)
    
    def get_available_handlers(self) -> List[str]:
        """Get list of available handler names."""
        return list(self._handlers.keys())
    
    def is_handler_available(self, handler_name: str) -> bool:
        """Check if a specific handler is available."""
        return handler_name in self._handlers


class CrawlerBase(BaseModule):
    """
    Base class for crawlers that retrieve information from various sources.
    
    This class provides a common interface and shared functionality for different
    types of web crawlers. It implements the template method pattern, allowing
    subclasses to define specific crawling implementations while maintaining
    consistent content processing through the PageContentHandler system.
    
    Attributes:
        page_content_handler: The handler used to process crawled content
    """
    page_content_handler: PageContentHandler = Field(description="The handler for page content.")
    
    def __init__(self, page_content_handler: PageContentHandler, **kwargs):
        """
        Initialize the crawler with a content handler.
        
        Args:
            page_content_handler: Handler for processing crawled content
            **kwargs: Additional arguments passed to BaseModule
        """
        super().__init__(page_content_handler=page_content_handler, **kwargs)
    
    def crawl(self, url: str, query: str = None, page_content_handler: PageContentHandler = None) -> Dict[str, Any]:
        """
        Crawl a URL and return processed content.
        
        This method should be implemented by subclasses to define the specific
        crawling behavior. The returned dictionary should contain at minimum:
        - success: bool indicating if crawling was successful
        - url: str the crawled URL
        - content: str the processed content
        
        Args:
            url: The URL to crawl
            query: Optional query for content filtering
            page_content_handler: Optional handler override for this crawl
            
        Returns:
            Dictionary containing crawl results and metadata
        """
        pass
    
    def handle_page_content(self, content: str, query: str = None, page_content_handler: PageContentHandler = None) -> str:
        """
        Process page content using the specified or default handler.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            page_content_handler: Optional handler override
            
        Returns:
            Processed content string
        """
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
