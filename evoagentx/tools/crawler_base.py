from typing import Dict, Any, Optional, List

import os
from ..core.module import BaseModule
from pydantic import Field
from ..models.base_model import BaseLLM, LLMOutputParser
from ..models.model_configs import LLMConfig, OpenAILLMConfig
from ..models.openai_model import OpenAILLM
from ..models.model_utils import create_llm_instance
from ..prompts.web_agent import SEARCH_RESULT_CONTENT_EXTRACTION_PROMPT

# Configuration constants
LLM_CONTENT_THRESHOLD = 50000

class PageContentOutput(LLMOutputParser):
    report: str = Field(description="The report of the page content.")


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
    
    def handle(self, content: str, query: Optional[str] = None) -> str:
        """
        Process page content. Subclasses should override this method.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            
        Returns:
            Processed content string
        """
        raise NotImplementedError("Subclasses must implement handle method")
    
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
    def handle(self, content: str, query: Optional[str] = None) -> str:
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
    
    def handle(self, content: str, query: Optional[str] = None) -> str:
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
    
    def handle(self, content: str, query: Optional[str] = None) -> str:
        """
        Process content using LLM and return structured result.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            
        Returns:
            Processed content with title, description, content, and links
        """
        try:
            # Pre-process HTML content with html2text before sending to LLM
            processed_content = content
            if any(tag in content.lower() for tag in ['<html', '<body', '<div', '<p', '<h1']):
                try:
                    import html2text
                    html_converter = html2text.HTML2Text()
                    html_converter.ignore_links = False
                    html_converter.ignore_images = True
                    html_converter.body_width = 0
                    html_converter.unicode_snob = True
                    html_converter.escape_snob = True
                    processed_content = html_converter.handle(content)
                except Exception as e:
                    print(f"HTML preprocessing failed: {str(e)}, using original content")
            
            message = []
            message.append({"role": "system", "content": "You are a helpful assistant that can help with page content handling."})
            message.append({"role": "user", "content": SEARCH_RESULT_CONTENT_EXTRACTION_PROMPT.format(crawling_result=processed_content, query=query)})
        
            # Try to get structured result first
            try:
                if self.llm is not None:
                    result = self.llm.generate(messages=message, parse_mode="title", parser=PageContentOutput)
                    
                    # Format the result as a readable string
                    if hasattr(result, 'report') and isinstance(result, PageContentOutput):
                        final_report = result.report
                    else:
                        final_report = str(result)
                    
                    return self._truncate_content(final_report)
                else:
                    raise ValueError("LLM not available")
                
            except Exception as parse_error:
                # If structured parsing fails, try to get raw text response
                print(f"Structured parsing failed: {str(parse_error)}, trying raw text response")
                
                # Get raw text response without parser
                if self.llm is not None:
                    raw_result = self.llm.generate(messages=message)
                else:
                    raw_result = "LLM not available"
                
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
        preferred_handler: Optional[str] = None,
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
    
    def _generate_handler_order(self, content: str, query: Optional[str] = None) -> List[str]:
        """Generate the default execution order based on content and query."""
        # If preferred handler is specified and available, use it first
        if self.preferred_handler and self.preferred_handler in self._handlers:
            return [self.preferred_handler] + [h for h in self._handlers.keys() if h != self.preferred_handler]
        
        # Default order based on content analysis
        is_html = any(tag in content.lower() for tag in ['<html', '<body', '<div', '<p', '<h1'])
        has_query = bool(query and query.strip())
        is_long = len(content) > LLM_CONTENT_THRESHOLD  # Very long content: ~8,000+ words
        
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
    
    def _should_handle(self, handler_name: str, content: str, query: Optional[str] = None) -> bool:
        """Determine if we should use this handler based on content and query."""
        if handler_name not in self._handlers:
            return False
        
        # Always allow disabled handler as fallback
        if handler_name == "disabled":
            return True
        
        # LLM handler: use for very long content with queries
        if handler_name == "llm":
            has_query = bool(query and query.strip())
            is_long = len(content) > LLM_CONTENT_THRESHOLD
            return has_query and is_long
        
        # HTML2Text handler: use for HTML content
        if handler_name == "html2text":
            return any(tag in content.lower() for tag in ['<html', '<body', '<div', '<p', '<h1'])
        
        # For any other handler, allow it
        return True
    
    def handle(self, content, query: Optional[str] = None):
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
    
    def crawl(self, url: str, query: Optional[str] = None, page_content_handler: Optional[PageContentHandler] = None) -> Dict[str, Any]:
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
        raise NotImplementedError("Subclasses must implement crawl method")
    
    def handle_page_content(self, content: str, query: Optional[str] = None, page_content_handler: Optional[PageContentHandler] = None) -> str:
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