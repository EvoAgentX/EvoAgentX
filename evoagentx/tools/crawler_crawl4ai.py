import asyncio
import os
import threading
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

from .crawler_base import AutoPageContentHandler, CrawlerBase, CrawlingResult
from .tool import Tool, Toolkit
from ..core.logging import logger

# ToolMetadata / ToolResult may be patched in by research_tools._compat.
# Provide an inline fallback so this module is importable on its own.
try:
    from .tool import ToolMetadata, ToolResult
except (ImportError, AttributeError):
    from pydantic import BaseModel as _BM, Field as _F
    from typing import Any as _Any, Dict as _Dict

    class ToolMetadata(_BM):
        tool_name: str = ""
        args: _Dict[str, _Any] = _F(default_factory=dict)
        cost_breakdown: _Dict[str, float] = _F(default_factory=dict)

        def add_cost_breakdown(self, cost_breakdown):
            for k, v in cost_breakdown.items():
                self.cost_breakdown[k] = self.cost_breakdown.get(k, 0.0) + v

    class ToolResult(_BM):
        metadata: ToolMetadata = _F(default_factory=ToolMetadata)
        result: _Any = None

# Other toolkit settings
TOOLKIT_SETTING: Dict[str, Any] = {
    "output_format": "handler",  # 'markdown' | 'html' | 'text' | 'handler'
    "openrouter_key": None,
    "llm_chunk_threshold": 8000,
    "chunk_size_words": 8000,
    "max_chunks_returned": 4,
}

# Default configuration for BrowserConfig
DEFAULT_BROWSER_CONFIG: Dict[str, Any] = {
    "browser_type": "chromium",
    "headless": True,
    "verbose": True,
    "enable_stealth": True,
    "light_mode": True,
}

# Default configuration for CrawlerRunConfig
DEFAULT_RUN_CONFIG: Dict[str, Any] = {
    "word_count_threshold": 20,
    "screenshot": False,
    "wait_for": None,  # CSS selector or 'load' or seconds
    "cache_mode": "disabled",  # 'enabled' | 'disabled' | 'bypass'
    "wait_until": "domcontentloaded",  # 'networkidle' | 'load' | 'domcontentloaded'
    "page_timeout": 60000,  # seconds
    "scan_full_page": False,
    "scroll_delay": 0.2,
}

DEFAULT_CRAWL4AI_MARKDOWN_GENERATOR_CONFIG: Dict[str, Any] = {
    "content_source": "cleaned_html", # raw_html, clean_html, fit_html
    "options": {
        "ignore_links": False,
        "ignore_images": True,
        "escape_html": True,
        "body_width": 80,
        "skip_internal_links": True,
        "include_sup_sub": True,
    }
}

CRAWL4AI_PRUNINGCONTENTFILTER_CONTENTFILTER_CONFIG = {
    "threshold":0.48,
    "threshold_type":"fixed",  # or "dynamic"
    "min_word_threshold":20
}


class CrawlerManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._max_instances = 5
        # Pool state:
        # _active_crawlers: set of (crawler, config_key)
        # _idle_crawlers: list of (crawler, config_key)
        self._active_crawlers = set()
        self._idle_crawlers = []
        self._pool_lock = None # Initialized in loop
        self._condition = None # Initialized in loop

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                cls._instance.start()
            return cls._instance

    def start(self):
        if self._thread is not None:
            return
        
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._pool_lock = asyncio.Lock()
            self._condition = asyncio.Condition(self._pool_lock)
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()

    def submit_crawl(self, url: str, run_config: Any, browser_config: Any, browser_config_dict: Dict[str, Any]) -> concurrent.futures.Future:
        """Submit a crawl task to the background loop."""
        return asyncio.run_coroutine_threadsafe(
            self._execute_crawl(url, run_config, browser_config, browser_config_dict),
            self._loop
        )

    async def _execute_crawl(self, url: str, run_config: Any, browser_config: Any, browser_config_dict: Dict[str, Any]):
        crawler = await self._acquire_crawler(browser_config, browser_config_dict)
        try:
            # We assume crawler is started and ready
            result = await crawler.arun(url=url, config=run_config)
            return result
        except Exception:
            raise
        finally:
            await self._release_crawler(crawler, browser_config_dict)

    async def _acquire_crawler(self, browser_config: Any, browser_config_dict: Dict[str, Any]):
        from crawl4ai import AsyncWebCrawler
        
        # Config key for matching
        key = self._make_config_key(browser_config_dict)
        
        async with self._condition:
            while True:
                # 1. Try to find idle crawler with matching config
                for i, (crawler, c_key) in enumerate(self._idle_crawlers):
                    if c_key == key:
                        self._idle_crawlers.pop(i)
                        self._active_crawlers.add((crawler, key))
                        return crawler
                
                # 2. If no match, check if we can create new
                total_count = len(self._active_crawlers) + len(self._idle_crawlers)
                if total_count < self._max_instances:
                    # Create new
                    crawler = AsyncWebCrawler(config=browser_config)
                    await crawler.start()
                    self._active_crawlers.add((crawler, key))
                    return crawler
                
                # 3. If limit reached, but we have idle crawlers with DIFFERENT config
                # We can close one idle crawler to make room
                if self._idle_crawlers:
                    crawler_to_close, _ = self._idle_crawlers.pop(0) # FIFO
                    try:
                        await crawler_to_close.close()
                    except Exception as e:
                        logger.error(f"CrawlerManager | Error closing crawler: {e}")
                    
                    # Create new
                    crawler = AsyncWebCrawler(config=browser_config)
                    await crawler.start()
                    self._active_crawlers.add((crawler, key))
                    return crawler
                
                # 4. If limit reached and NO idle crawlers (all busy)
                # Wait for someone to release
                await self._condition.wait()

    async def _release_crawler(self, crawler: Any, browser_config_dict: Dict[str, Any]):
        key = self._make_config_key(browser_config_dict)
        async with self._condition:
            # Remove from active
            if (crawler, key) in self._active_crawlers:
                self._active_crawlers.remove((crawler, key))
                # Add to idle
                self._idle_crawlers.append((crawler, key))
                # Notify waiters
                self._condition.notify()

    def _make_config_key(self, config_dict: Dict[str, Any]) -> Tuple:
        return (
            config_dict.get("browser_type"),
            config_dict.get("headless"),
            config_dict.get("enable_stealth"),
            config_dict.get("light_mode")
        )


class Crawl4AICrawler(CrawlerBase):
    """Crawler using Crawl4AI for advanced web crawling with browser automation."""
    
    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        openrouter_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        base_cfg = {}
        base_cfg.update(DEFAULT_BROWSER_CONFIG)
        base_cfg.update(DEFAULT_RUN_CONFIG)
        base_cfg.update(TOOLKIT_SETTING)
        
        base_cfg.update({
            "browser_type": browser_type,
            "headless": headless,
            "openrouter_key": openrouter_key,
        })
        if config:
            base_cfg.update({k: v for k, v in config.items() if v is not None})

        page_content_handler = AutoPageContentHandler(
            openrouter_key=openrouter_key,
            llm_chunk_threshold=base_cfg.get("llm_chunk_threshold", 8000),
            chunk_size_words=base_cfg.get("chunk_size_words", 8000),
            max_chunks_returned=base_cfg.get("max_chunks_returned", 4),
        )
        
        super().__init__(page_content_handler=page_content_handler, **kwargs)
        self.config: Dict[str, Any] = base_cfg

        
        try:
            from crawl4ai import BrowserConfig, CrawlerRunConfig, PruningContentFilter, DefaultMarkdownGenerator, CacheMode
            self.crawl4ai_available = True

            # Build rowserConfig statically at init, omit None values
            bc_dict = {
                "browser_type": self.config.get("browser_type"),
                "headless": self.config.get("headless"),
                "verbose": self.config.get("verbose"),
                # enable_stealth may not be supported depending on installed playwright_stealth variant
                "enable_stealth": self.config.get("enable_stealth", True),
                "light_mode": self.config.get("light_mode", True),
            }

            # Detect stealth support early; certain distributions (e.g. tf-playwright-stealth)
            # do not expose a Stealth class and cause import errors when enable_stealth=True.
            stealth_supported = True
            try:
                import importlib
                mod = importlib.import_module("playwright_stealth")
                # playwright-stealth exposes Stealth; tf-playwright-stealth exposes stealth_async/stealth_sync only
                stealth_supported = hasattr(mod, "Stealth")
            except Exception:
                stealth_supported = False

            if not stealth_supported:
                # Auto-disable stealth to avoid import/runtime errors
                bc_dict["enable_stealth"] = False

            try:
                self.browser_config = BrowserConfig(**bc_dict)
            except Exception:
                # As a final fallback, strip optional flags and proceed
                bc_dict.pop("enable_stealth", None)
                bc_dict.pop("light_mode", None)
                self.browser_config = BrowserConfig(**bc_dict)

            self.browser_config_dict = bc_dict

            # Normalize page_timeout to milliseconds
            try:
                pt = self.config.get("page_timeout", DEFAULT_RUN_CONFIG["page_timeout"])  # seconds by default
                self.page_timeout_ms = int(pt * 1000) if pt <= 600 else int(pt)
            except Exception:
                self.page_timeout_ms = 30000

            # Map cache_mode string to CacheMode enum
            cache_mode_mapping = {
                "enabled": "ENABLED",
                "disabled": "DISABLED",
                "bypass": "BYPASS",
            }
            try:
                cm = self.config.get("cache_mode", DEFAULT_RUN_CONFIG["cache_mode"]) or "disabled"
                self.cache_mode_enum = getattr(CacheMode, cache_mode_mapping.get(str(cm).lower(), "DISABLED"))
            except Exception:
                self.cache_mode_enum = CacheMode.DISABLED

            markdown_generator_config = DEFAULT_CRAWL4AI_MARKDOWN_GENERATOR_CONFIG.copy()
            markdown_generator_config["content_filter"]=PruningContentFilter(**CRAWL4AI_PRUNINGCONTENTFILTER_CONTENTFILTER_CONFIG)
            self.run_config = CrawlerRunConfig(
                cache_mode=self.cache_mode_enum,
                word_count_threshold=self.config.get("word_count_threshold"),
                screenshot=self.config.get("screenshot"),
                wait_until=self.config.get("wait_until"),
                page_timeout=self.page_timeout_ms,
                scan_full_page=self.config.get("scan_full_page"),
                scroll_delay=self.config.get("scroll_delay"),
                wait_for=self.config.get("wait_for"),
                markdown_generator=DefaultMarkdownGenerator(**markdown_generator_config),
            )
        except ImportError:
            self.crawl4ai_available = False
            raise ImportError(
                "crawl4ai is not installed. Please install it with: pip install crawl4ai"
            )
        
        # Convenience attributes derived from unified config
        self.browser_type = self.config.get("browser_type")
        self.headless = self.config.get("headless")
        self.verbose = self.config.get("verbose")

        # try:
        #     self._ensure_playwright_browsers_installed()
        # except Exception:
        #     pass
        
    def crawl(
        self,
        url: str,
        query: str = None,
        fetch_raw_content: bool = False,
    ) -> CrawlingResult:
        """
        Crawl a web page using Crawl4AI.

        Args:
            url: The URL to crawl
            query: Optional query to extract specific information
            fetch_raw_content: If True, return raw content without LLM processing

        Returns:
            Dictionary containing crawled content and metadata
        """
        if not self.crawl4ai_available:
            return self._error_result(url, "Crawl4AI is not available")

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop:
                # Run in thread to avoid blocking the event loop if called from async context
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(lambda: asyncio.run(self._async_crawl(url, query, fetch_raw_content))).result()
            else:
                return asyncio.run(self._async_crawl(url, query, fetch_raw_content))
        except Exception as e:
            return self._error_result(url, f"Crawling failed: {str(e)}")
    
    async def _async_crawl(
        self,
        url: str,
        query: str = None,
        fetch_raw_content: bool = False,
    ) -> CrawlingResult:
        """Async implementation of web crawling returning CrawlingResult."""

        try:
            future = CrawlerManager.get_instance().submit_crawl(
                url=url,
                run_config=self.run_config,
                browser_config=self.browser_config,
                browser_config_dict=self.browser_config_dict
            )
            raw_result = await asyncio.wrap_future(future)
            processed_result = await self._process_crawl4ai_result(raw_result, query, fetch_raw_content)
            return processed_result

        except Exception as e:
            msg = str(e)

            # Retry logic for stealth errors
            stealth_error = any(
                s in msg for s in (
                    "cannot import name 'Stealth' from 'playwright_stealth'",
                    "playwright_stealth",
                    "Stealth",
                    "enable_stealth"
                )
            )
            if stealth_error and self.browser_config_dict.get("enable_stealth", True):
                try:
                    from crawl4ai import BrowserConfig
                    new_dict = self.browser_config_dict.copy()
                    new_dict["enable_stealth"] = False
                    new_config = BrowserConfig(**new_dict)

                    future = CrawlerManager.get_instance().submit_crawl(
                        url=url,
                        run_config=self.run_config,
                        browser_config=new_config,
                        browser_config_dict=new_dict
                    )
                    raw_result = await asyncio.wrap_future(future)
                    processed_result = await self._process_crawl4ai_result(raw_result, query, fetch_raw_content)
                    return processed_result
                except Exception as e2:
                    return self._error_result(url, f"Crawling failed after disabling stealth: {str(e2)}")

            # Retry logic for navigation timeouts
            if ("ACS-GOTO" in msg) or ("Page.goto" in msg) or ("Timeout" in msg):
                try:
                    future = CrawlerManager.get_instance().submit_crawl(
                        url=url,
                        run_config=self.run_config,
                        browser_config=self.browser_config,
                        browser_config_dict=self.browser_config_dict
                    )
                    raw_result = await asyncio.wrap_future(future)
                    processed_result = await self._process_crawl4ai_result(raw_result, query, fetch_raw_content)
                    return processed_result
                except Exception as e2:
                    return self._error_result(url, f"Crawling failed after retry: {str(e2)}")

            return self._error_result(url, f"Crawling failed: {str(e)}")

    async def _process_crawl4ai_result(self, result, query: str = None, fetch_raw_content: bool = False) -> CrawlingResult:
        """Process Crawl4AI result and return CrawlingResult."""
        if not result.success:
            return CrawlingResult(
                result="",
                error=result.error_message or "Unknown error",
                cost=0.0,
                cost_breakdown={},
                handler_name=None,
            )

        # If fetch_raw_content is True, return raw content without any processing
        if fetch_raw_content:
            return CrawlingResult(
                result=result.markdown or result.text or result.cleaned_html or "",
                error=None,
                cost=0.0,
                cost_breakdown={},
                handler_name=None,
            )

        output_format = self.config.get("output_format", TOOLKIT_SETTING["output_format"])

        if output_format == "handler":
            result_text = result.markdown or result.text or result.cleaned_html or ""
            handler_result = await self.async_handle_page_content(result_text, query)
            return CrawlingResult(
                result=handler_result.result,
                error=None,
                cost=handler_result.cost,
                cost_breakdown=handler_result.cost_breakdown,
                handler_name=handler_result.handler_name,
            )
        elif output_format == "markdown":
            return CrawlingResult(result=result.markdown or "", error=None, cost=0.0, cost_breakdown={}, handler_name=None)
        elif output_format == "html":
            content_html = result.cleaned_html or result.html or ""
            return CrawlingResult(result=content_html, error=None, cost=0.0, cost_breakdown={}, handler_name=None)
        elif output_format == "text":
            import re
            text = result.markdown or ""
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            text = re.sub(r'[#*`_~]', '', text)
            return CrawlingResult(result=text, error=None, cost=0.0, cost_breakdown={}, handler_name=None)
        else:
            return CrawlingResult(result=result.markdown or "", error=None, cost=0.0, cost_breakdown={}, handler_name=None)
    
    def _error_result(self, url: str, error_message: str) -> CrawlingResult:
        """Create a standardized error result."""
        return CrawlingResult(result="", error=error_message, cost=0.0, cost_breakdown={}, handler_name=None)


class Crawl4AICrawlTool(Tool):
    """Advanced browser-based crawling with Crawl4AI."""
    
    name: str = "fetch_web_content"
    description: str = "Retrieve content from a web page."
    inputs: Dict[str, Dict] = {
        "url": {
            "type": "string", 
            "description": "Required. URL of the web page to retrieve content from."
        },
        "query": {
            "type": "string",
            "description": "Optional. Specify what information to extract from the page (e.g., 'pricing details', 'contact information'). If not provided, the entire page content will be returned."
        },
        "fetch_raw_content": {
            "type": "boolean",
            "description": "Optional. If true, returns unprocessed page content without LLM filtering or summarization, even for long pages. Default: false"
        }
    }
    required: Optional[List[str]] = ["url"]
    
    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        openrouter_key: str = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        if not openrouter_key:
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        # Create the underlying crawler
        self.crawler = Crawl4AICrawler(
            browser_type=browser_type,
            headless=headless,
            openrouter_key=openrouter_key,
            config=config,
            **kwargs,
        )

    def _build_tool_result(self, crawl_result: CrawlingResult, metadata: ToolMetadata) -> ToolResult:
        if getattr(crawl_result, "cost_breakdown", None):
            metadata.add_cost_breakdown(crawl_result.cost_breakdown)

        if crawl_result.error:
            return ToolResult(
                result={
                    "success": False,
                    "error": crawl_result.error,
                },
                metadata=metadata
            )

        return ToolResult(
            result={
                "success": True,
                "content": crawl_result.result,
            },
            metadata=metadata
        )
    
    async def __call__(self, url: str, query: str = None, fetch_raw_content: bool = False) -> ToolResult:
        """
        Crawl a web page using Crawl4AI and extract comprehensive content.

        Returns:
            Dictionary containing crawled content and metadata
        """
        metadata = ToolMetadata(
            tool_name=self.name,
            args={"url": url, "query": query, "fetch_raw_content": fetch_raw_content}
        )
        crawl_result = await self.crawler._async_crawl(url=url, query=query, fetch_raw_content=fetch_raw_content)
        return self._build_tool_result(crawl_result, metadata)

    def call_sync(self, url: str, query: str = None, fetch_raw_content: bool = False) -> ToolResult:
        """
        Synchronous compatibility entrypoint.
        Use this in synchronous callers that cannot `await` tool calls.
        """
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(lambda: asyncio.run(self.__call__(url=url, query=query, fetch_raw_content=fetch_raw_content))).result()
        except RuntimeError:
            return asyncio.run(self.__call__(url=url, query=query, fetch_raw_content=fetch_raw_content))


class Crawl4AICrawlToolkit(Toolkit):
    """Advanced browser-based crawling toolkit using Crawl4AI."""
    
    def __init__(
        self,
        name: str = "Crawl4AICrawlToolkit",
        browser_type: str = "chromium",
        headless: bool = True,
        openrouter_key: str = None,
    ):
        """
        Initialize Crawl4AI crawling toolkit with shared configuration.
        
        Args:
            name: Name of the toolkit
            browser_type: Browser type to use ('chromium', 'firefox', etc.)
            headless: Whether to run browser in headless mode
            openrouter_key: API key for OpenRouter
        """
        
        # Create crawl4ai crawl tool with configuration
        crawl4ai_crawl_tool = Crawl4AICrawlTool(
            browser_type=browser_type,
            headless=headless,
            openrouter_key=openrouter_key,
        )
        
        # Initialize parent with tools
        super().__init__(name=name, tools=[crawl4ai_crawl_tool])
