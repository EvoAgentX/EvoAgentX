import asyncio
import json
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import base64
from urllib.parse import urlparse

from .tool import Tool, Toolkit
from ..core.module import BaseModule

# Handle optional crawl4ai import
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai import AdaptiveCrawler, AdaptiveConfig
    from crawl4ai import JsonCssExtractionStrategy, LLMExtractionStrategy
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    # Fallback for type checking - these won't be used at runtime
    AsyncWebCrawler = None  # type: ignore
    BrowserConfig = None  # type: ignore
    CrawlerRunConfig = None  # type: ignore
    CacheMode = None  # type: ignore
    AdaptiveCrawler = None  # type: ignore
    AdaptiveConfig = None  # type: ignore
    JsonCssExtractionStrategy = None  # type: ignore
    LLMExtractionStrategy = None  # type: ignore


class Crawl4AIBase(BaseModule):
    """
    Base class for Crawl4AI functionality providing core web crawling capabilities.
    Handles AsyncWebCrawler management, configuration, and common operations.
    """
    
    def __init__(
        self, 
        browser_type: str = "chromium",
        headless: bool = True,
        verbose: bool = False,
        user_agent: str = None,
        proxy: str = None,
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize Crawl4AI base with browser configuration.
        
        Args:
            browser_type: Browser to use (chromium, firefox, webkit)
            headless: Run browser in headless mode
            verbose: Enable verbose logging
            user_agent: Custom user agent string
            proxy: Proxy configuration
            timeout: Request timeout in seconds
        """
        super().__init__()
        
        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "crawl4ai is not installed. Please install it with: pip install crawl4ai"
            )
        
        # Import at runtime to handle optional dependency
        from crawl4ai import BrowserConfig
        
        # Build config dict, excluding None values
        config_dict = {
            "browser_type": browser_type,
            "headless": headless,
            "verbose": verbose,
            **kwargs
        }
        
        if user_agent is not None:
            config_dict["user_agent"] = user_agent
        if proxy is not None:
            config_dict["proxy"] = proxy
        
        self.browser_config = BrowserConfig(**config_dict)
        self.timeout = timeout
        self._crawler = None
    
    async def _get_crawler(self):
        """Get or create AsyncWebCrawler instance."""
        if self._crawler is None:
            from crawl4ai import AsyncWebCrawler
            self._crawler = AsyncWebCrawler(config=self.browser_config)
            await self._crawler.start()
        return self._crawler
    
    async def _cleanup_crawler(self):
        """Cleanup crawler resources with proper event loop handling."""
        if self._crawler is not None:
            try:
                # First, try to gracefully close the crawler
                # Set a short timeout to avoid hanging
                await asyncio.wait_for(self._crawler.close(), timeout=1.0)
            except (asyncio.TimeoutError, RuntimeError, Exception):
                # If any error occurs, force cleanup
                try:
                    await self._force_cleanup_crawler()
                except Exception:
                    pass  # Ignore force cleanup errors
            finally:
                # Always set crawler to None
                self._crawler = None
                
    
    async def _force_cleanup_crawler(self):
        """Force cleanup of crawler resources when normal cleanup fails."""
        try:
            if hasattr(self._crawler, 'browser') and self._crawler.browser:
                # Force close all browser contexts and pages
                if hasattr(self._crawler.browser, 'contexts'):
                    for context in self._crawler.browser.contexts:
                        try:
                            # Close all pages in this context
                            for page in context.pages:
                                try:
                                    await asyncio.wait_for(page.close(), timeout=0.3)
                                except:
                                    pass
                            # Close the context
                            await asyncio.wait_for(context.close(), timeout=0.3)
                        except:
                            pass
                
                # Finally close the browser
                try:
                    await asyncio.wait_for(self._crawler.browser.close(), timeout=0.5)
                except:
                    pass
                    
                # Set browser to None to prevent further access
                if hasattr(self._crawler, 'browser'):
                    self._crawler.browser = None
                    
        except Exception:
            pass  # Ignore all errors during force cleanup
    
    def _process_result(self, result, output_format: str = "markdown") -> Dict[str, Any]:
        """
        Process crawl result into standardized format.
        
        Args:
            result: CrawlResult from crawl4ai
            output_format: Desired output format
            
        Returns:
            Standardized result dictionary
        """
        processed = {
            "success": result.success,
            "url": result.url,
            "title": getattr(result, 'title', ''),
            "status_code": getattr(result, 'status_code', None),
            "content": "",
            "raw_html": result.html or "",
            "cleaned_html": result.cleaned_html or "",
            "markdown": result.markdown or "",
            "links": result.links or {},
            "media": result.media or {},
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


class WebCrawlTool(Tool):
    """Advanced web crawling tool with LLM-friendly output using Crawl4AI."""
    
    name: str = "web_crawl"
    description: str = "Advanced web crawling that converts web pages to clean, LLM-ready Markdown. Extracts content, images, links, and metadata with intelligent content filtering and stealth capabilities. Perfect for research, content analysis, and AI data preparation."
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
        "cache_mode": {
            "type": "string",
            "description": "Cache strategy: 'enabled', 'disabled', 'bypass' (default: enabled)"
        }
    }
    required: List[str] = ["url"]
    
    def __init__(self, crawl4ai_base: 'Crawl4AIBase' = None):
        super().__init__()
        self.crawl4ai_base = crawl4ai_base or Crawl4AIBase()
    
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
        cache_mode: str = "enabled"
    ) -> Dict[str, Any]:
        """
        Crawl a web page and extract clean, LLM-ready content.
        
        Returns:
            Dictionary containing crawled content and metadata
        """
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self._async_call(
                url, output_format, css_selector, word_count_threshold,
                include_images, include_links, take_screenshot, wait_for, cache_mode
            ))
        else:
            # Event loop is already running, create a task
            import concurrent.futures
            import threading
            
            def run_in_thread():
                return asyncio.run(self._async_call(
                    url, output_format, css_selector, word_count_threshold,
                    include_images, include_links, take_screenshot, wait_for, cache_mode
                ))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
    
    async def _async_call(
        self, 
        url: str, 
        output_format: str,
        css_selector: str,
        word_count_threshold: int,
        include_images: bool,
        include_links: bool,
        take_screenshot: bool,
        wait_for: str,
        cache_mode: str
    ) -> Dict[str, Any]:
        """Async implementation of web crawling."""
        try:
            if not CRAWL4AI_AVAILABLE:
                raise ImportError("crawl4ai is not installed. Please install it with: pip install crawl4ai")
            
            # Map cache mode strings to CacheMode enum values
            cache_mode_mapping = {
                "enabled": "enabled",
                "disabled": "disabled", 
                "bypass": "bypass"
            }
            
            # Create crawler run config with dynamic imports
            from crawl4ai import CrawlerRunConfig, CacheMode
            
            # Map string to actual enum
            cache_enum_value = getattr(CacheMode, cache_mode_mapping.get(cache_mode, "enabled").upper())
            
            # Build config dict, excluding None values
            config_dict = {
                "cache_mode": cache_enum_value,
                "word_count_threshold": word_count_threshold,
                "screenshot": take_screenshot,
                "process_iframes": True,
                "remove_overlay_elements": True
            }
            
            # Add optional parameters only if they're not None
            if css_selector is not None:
                config_dict["css_selector"] = css_selector
            if wait_for is not None:
                config_dict["wait_for"] = wait_for
            
            config = CrawlerRunConfig(**config_dict)
            
            crawler = await self.crawl4ai_base._get_crawler()
            result = await crawler.arun(url=url, config=config)
            
            processed_result = self.crawl4ai_base._process_result(result, output_format)
            
            # Filter out images/links if requested
            if not include_images:
                processed_result["media"] = {}
            if not include_links:
                processed_result["links"] = {} 
                
            return processed_result
            
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "content": "",
                "error_message": f"Crawling failed: {str(e)}",
                "stats": {"content_length": 0, "html_length": 0, "links_count": 0, "images_count": 0}
            }


class Crawl4AIToolkit(Toolkit):
    """Comprehensive toolkit for advanced web crawling using Crawl4AI."""
    
    def __init__(
        self,
        name: str = "Crawl4AIToolkit",
        browser_type: str = "chromium",
        headless: bool = True,
        verbose: bool = False,
        user_agent: str = None,
        proxy: str = None,
        **kwargs
    ):
        """Initialize Crawl4AI toolkit with shared configuration."""
        
        # Create shared Crawl4AI base instance
        crawl4ai_base = Crawl4AIBase(
            browser_type=browser_type,
            headless=headless,
            verbose=verbose,
            user_agent=user_agent,
            proxy=proxy,
            **kwargs
        )
        
        # Create tools with shared base
        tools = [
            WebCrawlTool(crawl4ai_base=crawl4ai_base)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store shared base for cleanup
        self.crawl4ai_base = crawl4ai_base
    
    async def cleanup(self):
        """Cleanup shared resources with timeout."""
        if hasattr(self, 'crawl4ai_base'):
            try:
                await asyncio.wait_for(self.crawl4ai_base._cleanup_crawler(), timeout=2.0)
            except (asyncio.TimeoutError, RuntimeError, Exception):
                # Force set crawler to None on any error
                if hasattr(self.crawl4ai_base, '_crawler'):
                    self.crawl4ai_base._crawler = None
    
    def __del__(self):
        """Cleanup on deletion with proper event loop handling."""
        try:
            if hasattr(self, 'crawl4ai_base') and self.crawl4ai_base._crawler is not None:
                # Simply set crawler to None to avoid event loop issues
                # The browser will be cleaned up by the OS when the process exits
                self.crawl4ai_base._crawler = None
        except Exception:
            pass  # Ignore all destructor errors
    
    def _schedule_cleanup(self):
        """Schedule cleanup to run in the current event loop."""
        try:
            if hasattr(self, 'crawl4ai_base') and self.crawl4ai_base._crawler is not None:
                # Just set to None if we can't properly cleanup
                self.crawl4ai_base._crawler = None
        except Exception:
            pass

