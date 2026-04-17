import asyncio
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional
from pydantic import Field

from ..core.logging import logger
from ..core.module import BaseModule
from ..models.base_model import BaseLLM, LLMOutputParser
from ..models.model_configs import LLMConfig, OpenRouterConfig
from ..models.model_utils import create_llm_instance

# track_cost may not exist in the project yet; the research_tools._compat
# shim patches it in at import time when the research_tools package loads.
# If crawler_base is imported standalone (outside research_tools), provide
# a no-op fallback so the module still loads.
try:
    from ..models.model_utils import track_cost
except ImportError:
    from contextlib import contextmanager as _cm
    from dataclasses import dataclass as _dc, field as _f

    @_dc
    class _CostTracker:
        cost_per_model: dict = _f(default_factory=dict)
        total_llm_cost: float = 0.0

    @_cm
    def track_cost():
        yield _CostTracker()

# The web_agent prompts module does not exist in this project.
# Define the two prompts inline.
SEARCH_RESULT_CONTENT_EXTRACTION_PROMPT = """\
You are a helpful research assistant. Your task is to extract the most relevant and important information from the following web page content based on the user's query.

**User Query:** {query}

**Web Page Content:**
{crawling_result}

**Instructions:**
1. Extract key information that is relevant to the user's query.
2. Preserve important details, numbers, dates, and facts.
3. Organize the information in a clear and structured format.
4. If the content is not relevant to the query, say so briefly.
5. Include any useful links or references mentioned in the content.

**Extracted Information:**
"""

SEARCH_RESULT_CONTENT_SUMMARIZATION_PROMPT = """\
You are a helpful research assistant. Summarize the following extracted information into a concise, accurate, and well-organized answer for the user's query.

**User Query:** {query}

**Extracted Content from Multiple Pages:**
{processed_page_content}

**Instructions:**
1. Synthesize the information from all sources into a coherent summary.
2. Prioritize the most relevant and important information.
3. Remove redundant information across sources.
4. Maintain factual accuracy — do not add information not present in the sources.
5. Use clear formatting with sections if appropriate.

**Summary:**
"""

try:
    from ..utils.utils import add_dict
except ImportError:
    def add_dict(base, update):
        result = dict(base)
        for k, v in update.items():
            result[k] = result.get(k, 0.0) + v
        return result

# Configuration constants
# Converted to word-based measurements for better semantic accuracy
LLM_CONTENT_THRESHOLD = 8000 
CHUNK_SIZE_WORDS = 8000
MAX_CHUNKS_RETURNED = 3
NO_QUERY_MAX_CHUNKS_RETURNED = MAX_CHUNKS_RETURNED
CHUNK_OVERLAP_WORDS = 1600
DEFAULT_QUERY = """
Extract the key information from this web page, focusing on what a reader would need to understand and use it.

For the short summary: provide a compact, information-dense snapshot of the page's purpose and most important takeaways; include who/what/when/where if present, plus the highest-impact numbers or constraints.

For the content: extract detailed information organized by meaning, including as applicable:
- Page type and intent (what kind of page this is, what it's trying to accomplish)
- Metadata (title, site/organization, author/owner, publication/updated dates, primary topic/keywords, jurisdiction/location if relevant)
- Key points (main claims/arguments, decisions/policies/terms, what changed, what matters)
- Data/Numbers (prices, specs, limits, thresholds, metrics, timelines, versions; keep units and qualifiers)
- Procedures/How-to (steps, prerequisites/dependencies, inputs/outputs, configurations, edge cases, troubleshooting/FAQs)
- Risks/Caveats (exceptions, constraints, ambiguities, missing details)

For possible useful links: list important outbound links that advance understanding or next steps, each labeled with its purpose (docs, download, pricing, API reference, citation/source, contact, related).

For evidence: include exact, unmodified sentences from the page that support each important claim, requirement, number, or policy you extracted. If the page is thin, ambiguous, or irrelevant, say so and explain what is missing.
"""

LLM_HANDLER_CONFIGS = {
    "short_content_threshold_words": None,
    "short_content_mode": "pass", # pass | llm_summary | two_rounds
    "apply_preprocessing": True,
    "default_query": DEFAULT_QUERY,
    "preprocess_remove_newlines": True,
    "preprocess_normalize_whitespace": True,
    "preprocess_dropout_ratio": 0.0,
    "preprocess_dropout_seed": None,
}

DEFAULT_PAGEHANDLER_LLM_CONFIG={
    "temperature": 0.3,
    "model": "google/gemini-2.5-flash",
    "max_tokens": 16000
    # "model": "google/gemini-2.5-flash-lite-preview-09-2025"
    # "model": "openai/gpt-4o-mini"
}


def _count_words(text: str) -> int:
    """Count words in text by splitting on whitespace."""
    return len(text.split()) if text else 0

class PageContentOutput(LLMOutputParser):
    report: str = Field(description="The report of the page content.")


class PageHandlingResult(BaseModule):
    result: str
    cost: Optional[float] = Field(default=0.0)
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)
    handler_name: Optional[str] = Field(default=None)

class CrawlingResult(BaseModule):
    result: str
    error: Optional[str] = Field(default=None)
    cost: Optional[float] = Field(default=0.0)
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)
    handler_name: Optional[str] = Field(default=None)


class PageContentHandler(BaseModule):
    """
    Base class for page content handlers.
    
    Provides common functionality for processing web page content, including
    optional truncation capabilities that can be used by all subclasses.
    """
    max_words: Optional[int] = Field(default=None, description="Maximum number of words before truncation")
    suffix: str = Field(default="...", description="Suffix to add when truncating content")
    
    def __init__(self, max_words: Optional[int] = None, suffix: str = "...", **kwargs):
        super().__init__(max_words=max_words, suffix=suffix, **kwargs)
    
    def handle(self, content: str, query: Optional[str] = None):
        """
        Process page content. Subclasses should override this method.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            
        Returns:
            Processed content result
        """
        raise NotImplementedError("Subclasses must implement handle method")
    
    async def async_handle(self, content: str, query: Optional[str] = None):
        """
        Async version of handle method.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            
        Returns:
            Processed content result
        """
        return self.handle(content, query)
    
    def _truncate_content(self, content: str) -> str:
        """
        Truncate content if max_words is specified.
        
        Args:
            content: Content to potentially truncate
            
        Returns:
            Truncated content if needed, otherwise original content
        """
        if self.max_words is None or _count_words(content) <= self.max_words:
            return content
        
        words = content.split()
        return ' '.join(words[:self.max_words]) + self.suffix

class DisabledPageContentHandler(PageContentHandler):
    """
    A no-op page content handler that returns content unchanged.
    
    Useful when you want to skip content processing and work with raw content.
    Supports optional truncation if max_length is specified.
    """
    def handle(self, content: str, query: Optional[str] = None):
        return PageHandlingResult(result=self._truncate_content(content), cost=0.0, cost_breakdown={}, handler_name="disabled")


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
    
    def handle(self, content: str, query: Optional[str] = None):
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
        return PageHandlingResult(result=self._truncate_content(text_content), cost=0.0, cost_breakdown={}, handler_name="html2text")


class LLMPageContentHandler(PageContentHandler):
    """
    Page content handler that uses an LLM to extract and process content.
    
    This handler can be configured with either a pre-initialized LLM instance
    or an LLM configuration that will be used to create the LLM instance.
    """
    llm: Optional[BaseLLM] = Field(default=None, description="The LLM to use for page content handling.")
    llm_config: Optional[LLMConfig] = Field(default=None, description="The LLM config to use for page content handling.")
    llm_chunk_threshold: int = Field(default=LLM_CONTENT_THRESHOLD, description="Word count threshold to enable LLM chunking")
    chunk_size_words: int = Field(default=CHUNK_SIZE_WORDS, description="Chunk size in words for LLM processing")
    chunk_overlap_words: int = Field(default=CHUNK_OVERLAP_WORDS, description="Overlap length in words between adjacent chunks")
    max_chunks_returned: int = Field(default=MAX_CHUNKS_RETURNED, description="Maximum number of chunks to return and merge")
    no_query_max_chunks_returned: int = Field(default=NO_QUERY_MAX_CHUNKS_RETURNED, description="Maximum number of chunks to process when query is not provided")
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        llm_config: Optional[LLMConfig] = None,
        llm_chunk_threshold: int = LLM_CONTENT_THRESHOLD,
        chunk_size_words: int = CHUNK_SIZE_WORDS,
        chunk_overlap_words: int = CHUNK_OVERLAP_WORDS,
        max_chunks_returned: int = MAX_CHUNKS_RETURNED,
        no_query_max_chunks_returned: int = NO_QUERY_MAX_CHUNKS_RETURNED,
        handler_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_chunk_threshold = llm_chunk_threshold
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.max_chunks_returned = max_chunks_returned

        effective_handler_config = dict(LLM_HANDLER_CONFIGS)
        if handler_config:
            effective_handler_config.update({k: v for k, v in handler_config.items() if v is not None})

        short_threshold = effective_handler_config.get("short_content_threshold_words")
        if short_threshold is None:
            short_threshold = self.llm_chunk_threshold
        self.short_content_threshold_words = int(short_threshold)
        short_mode = str(effective_handler_config.get("short_content_mode") or "pass").strip().lower()
        if short_mode not in ("pass", "llm_summary", "two_rounds"):
            short_mode = "pass"
        self.short_content_mode = short_mode
        self.default_query = str(effective_handler_config.get("default_query") or "").strip() or "Extract the key information from this page."
        self.preprocess_remove_newlines = bool(effective_handler_config.get("preprocess_remove_newlines", True))
        self.preprocess_normalize_whitespace = bool(effective_handler_config.get("preprocess_normalize_whitespace", True))
        self.preprocess_dropout_ratio = float(effective_handler_config.get("preprocess_dropout_ratio", 0.0) or 0.0)
        self.preprocess_dropout_seed = effective_handler_config.get("preprocess_dropout_seed", None)
        self.apply_preprocessing = bool(effective_handler_config.get("apply_preprocessing", True))
        self.handler_config = effective_handler_config
        self.no_query_max_chunks_returned = no_query_max_chunks_returned
        
        # Priority: provided llm > provided llm_config > default fallback
        if llm is not None:
            self.llm = llm
        elif llm_config is not None:
            self.llm = create_llm_instance(llm_config)
        else:
            # Default fallback to OpenAI
            if os.getenv("SUPABASE_URL_STORAGE") is None:
                try:
                    llm_conf = DEFAULT_PAGEHANDLER_LLM_CONFIG.copy()
                    llm_conf["openrouter_key"] = os.getenv("OPENROUTER_API_KEY")
                    self.llm_config = OpenRouterConfig(**llm_conf)
                except Exception as e:
                    raise ValueError(f"Error initializing default LLM: {str(e)}")
            else:
                raise ValueError("Error setup LLMPageContentHandler: SUPABASE_URL_STORAGE environment variable detected, you are probably testing the server. Please ensure you pass in the customer's openrouter key to enable the LLM."
                                 "You might need check the Crawl4AICrawler for passing openrouter_key. If not applied, please check Crawler that uese AutoPageContentHandler")
                
    def _preprocess_input_text(self, text: str) -> str:
        processed = text or ""
        if self.preprocess_remove_newlines:
            processed = processed.replace("\r\n", "\n").replace("\r", "\n")
            processed = processed.replace("\n", " ")
        if self.preprocess_normalize_whitespace:
            processed = " ".join(processed.split())

        ratio = self.preprocess_dropout_ratio
        if ratio <= 0.0:
            return processed

        try:
            import random
            rng = random.Random(self.preprocess_dropout_seed)
            words = processed.split()
            if not words:
                return processed
            keep_prob = max(0.0, min(1.0, 1.0 - ratio))
            def _is_link_token(token: str) -> bool:
                t = (token or "").strip().lower()
                if not t:
                    return False
                if t.startswith(("http://", "https://")):
                    return True
                if t.startswith("www."):
                    return True
                if "://" in t:
                    return True
                if "." in t and not t.startswith(".") and not t.endswith("."):
                    if any(suffix in t for suffix in (".com", ".org", ".net", ".ai", ".io", ".edu", ".gov")):
                        return True
                return False

            def _has_digits(token: str) -> bool:
                return any(ch.isdigit() for ch in (token or ""))

            kept = [
                w
                for w in words
                if _is_link_token(w) or _has_digits(w) or (rng.random() < keep_prob)
            ]
            if len(kept) < max(1, int(0.1 * len(words))):
                return processed
            return " ".join(kept)
        except Exception:
            return processed

    async def _async_handle_short_content(self, processed_content: str, query: Optional[str]) -> PageHandlingResult:
        mode = str(getattr(self, "short_content_mode", "pass") or "pass").strip().lower()
        q = (query or "").strip()
        if not q:
            q = (self.default_query or "").strip()

        if mode == "pass" or self.llm is None:
            return PageHandlingResult(result=self._truncate_content(processed_content), cost=0.0, cost_breakdown={}, handler_name="llm")

        if mode == "llm_summary":
            summary_text, summarize_cost, summarize_breakdown = await self._async_summarize_extractions([processed_content], q)
            return PageHandlingResult(
                result=self._truncate_content(summary_text),
                cost=summarize_cost,
                cost_breakdown=summarize_breakdown,
                handler_name="llm",
            )
        
        if mode == "two_rounds":
            outputs, chunks_cost, chunks_breakdown = await self._async_process_chunks([processed_content], q)
            summary_text, summarize_cost, summarize_breakdown = await self._async_summarize_extractions(outputs, q)
            total_cost = (chunks_cost or 0.0) + (summarize_cost or 0.0)
            return PageHandlingResult(
                result=self._truncate_content(summary_text),
                cost=total_cost,
                cost_breakdown=add_dict(chunks_breakdown, summarize_breakdown),
                handler_name="llm",
            )
        
        return PageHandlingResult(result="", cost=0.0, cost_breakdown={}, error="in _async_handle_short_content: unknown mode", handler_name="llm")

    async def _async_handle_without_query(self, processed_content: str) -> PageHandlingResult:
        return await self._async_handle_with_query(
            processed_content,
            self.default_query,
            max_chunks_returned=self.no_query_max_chunks_returned,
        )

    async def _async_handle_with_query(self, processed_content: str, query: str, max_chunks_returned: Optional[int] = None) -> PageHandlingResult:
        chunks = self._chunk_text(processed_content, max_chunks_returned=max_chunks_returned)
        outputs, chunks_cost, chunks_breakdown = await self._async_process_chunks(chunks, query)
        summary_text, summarize_cost, summarize_breakdown = await self._async_summarize_extractions(outputs, query)
        total_cost = (chunks_cost or 0.0) + (summarize_cost or 0.0)
        return PageHandlingResult(
            result=self._truncate_content(summary_text),
            cost=total_cost,
            cost_breakdown=add_dict(chunks_breakdown, summarize_breakdown),
            handler_name="llm",
        )

    async def async_handle(self, content: str, query: Optional[str] = None):
        """
        Process content using LLM and return structured result (Async version).
        
        Args:
            content: Raw content to process
            query: Optional query for context
            
        Returns:
            Processed content result
        """
        try:
            raw_content = content or ""
            raw_word_count = _count_words(raw_content)
            has_query = bool(isinstance(query, str) and query.strip())
            preprocess_applied = False

            processed_content = raw_content
            if self.apply_preprocessing and raw_word_count > self.llm_chunk_threshold:
                processed_content = self._preprocess_input_text(raw_content)
                preprocess_applied = True
            
            processed_word_count = _count_words(processed_content)

            try:
                logger.info(
                    f"LLMPageContentHandler processing: raw_word_count={raw_word_count} processed_word_count={processed_word_count} "
                    f"preprocess_applied={preprocess_applied} "
                    f"chunk_threshold={self.llm_chunk_threshold} chunk_size_words={self.chunk_size_words} "
                    f"chunk_overlap_words={self.chunk_overlap_words} max_chunks_returned={self.max_chunks_returned} "
                    f"no_query_max_chunks_returned={self.no_query_max_chunks_returned}"
                )
            except Exception:
                pass

            if processed_word_count <= self.short_content_threshold_words and not has_query:
                return await self._async_handle_short_content(processed_content, query)

            if not has_query:
                return await self._async_handle_without_query(processed_content)
            
            return await self._async_handle_with_query(processed_content, query.strip())
            
        except Exception as e:
            # If all LLM processing fails, fall back to simple text extraction
            logger.warning(f"LLM processing failed: {str(e)}, falling back to simple extraction")
            
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
            
            return PageHandlingResult(result=self._truncate_content(formatted_result), cost=0.0, cost_breakdown={}, handler_name="llm")

    def handle(self, content: str, query: Optional[str] = None):
        """
        Process content using LLM and return structured result.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            
        Returns:
            Processed content result
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(self.async_handle(content, query)))
                    return future.result()
            else:
                return asyncio.run(self.async_handle(content, query))
        except RuntimeError:
            return asyncio.run(self.async_handle(content, query))

    def _chunk_text(self, content: str, max_chunks_returned: Optional[int] = None) -> List[str]:
        if not content:
            return []
        words = content.split()
        size = max(1, int(self.chunk_size_words))
        overlap = int(getattr(self, "chunk_overlap_words", 0))
        overlap = max(0, min(overlap, size - 1))
        step = max(1, size - overlap)
        chunks = [' '.join(words[i:i+size]) for i in range(0, len(words), step)]
        effective_max_chunks_returned = self.max_chunks_returned if max_chunks_returned is None else max_chunks_returned
        if effective_max_chunks_returned is not None:
            chunks = chunks[:effective_max_chunks_returned]
        try:
            logger.info(
                f"LLMPageContentHandler chunking: page_word_count={len(words)} "
                f"chunk_size_words={size} chunk_overlap_words={overlap} step={step} "
                f"llm_chunk_threshold={self.llm_chunk_threshold} "
                f"max_chunks_returned={effective_max_chunks_returned} chunks_returned={len(chunks)}"
            )
        except Exception:
            pass
        return chunks

    async def _async_process_chunk(self, ch: str, query: Optional[str]) -> tuple:
        ch_msg = []
        ch_msg.append({"role": "system", "content": "You are a helpful assistant that can help with page content handling."})
        ch_msg.append({"role": "user", "content": SEARCH_RESULT_CONTENT_EXTRACTION_PROMPT.format(crawling_result=ch, query=query)})
        if self.llm is not None:
            from ..models.model_utils import track_cost
            with track_cost() as tracker:
                raw = await self.llm.async_generate(messages=ch_msg)
                llm_cost_breakdown = {"openrouter:" + k: v for k, v in tracker.cost_per_model.items()}
                return str(raw), tracker.total_llm_cost, llm_cost_breakdown
        return "LLM not available", 0.0, {}

    async def _async_process_chunks(self, chunks: List[str], query: Optional[str]) -> tuple:
        tasks = [self._async_process_chunk(ch, query) for ch in chunks]
        results = await asyncio.gather(*tasks)
        outputs = [r[0] for r in results]
        total_cost = sum(r[1] for r in results)
        total_breakdown: Dict[str, float] = {}
        for r in results:
            total_breakdown = add_dict(total_breakdown, r[2])
        return outputs, total_cost, total_breakdown

    async def _async_summarize_extractions(self, extractions: List[str], query: Optional[str]) -> tuple:
        if not extractions:
            return "", 0.0, {}
        
        if self.llm is None:
            return "\n\n---\n\n".join(extractions), 0.0, {}
        prompt = []
        prompt.append({"role": "system", "content": "You are a helpful assistant that summarizes extracted content into a concise, accurate answer for the query."})
        prompt.append({"role": "user", "content": SEARCH_RESULT_CONTENT_SUMMARIZATION_PROMPT.format(query=query, processed_page_content=extractions)})
        with track_cost() as tracker:
            res = await self.llm.async_generate(messages=prompt, parse_mode="str", parser=LLMOutputParser)
            llm_cost_breakdown = {"openrouter:" + k: v for k, v in tracker.cost_per_model.items()}
            return str(res), tracker.total_llm_cost, llm_cost_breakdown
        
    def _summarize_extractions(self, extractions: List[str], query: Optional[str]) -> str:
        if not extractions:
            return ""
        
        if self.llm is None:
            return "\n\n---\n\n".join(extractions)
        prompt = []
        prompt.append({"role": "system", "content": "You are a helpful assistant that summarizes extracted content into a concise, accurate answer for the query."})
        prompt.append({"role": "user", "content": SEARCH_RESULT_CONTENT_SUMMARIZATION_PROMPT.format(query=query, processed_page_content=extractions)})
        try:
            res = self.llm.generate(messages=prompt, parse_mode="str", parser=LLMOutputParser)
            return str(res)
        except Exception:
            try:
                raw = self.llm.generate(messages=prompt)
                return str(raw)
            except Exception:
                return "\n\n---\n\n".join(extractions)

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
    handler_config: Optional[Dict[str, Any]] = Field(default=None, description="Handler configuration")
    enable_llm: bool = Field(default=True, description="Whether to enable LLM processing")
    enable_html2text: bool = Field(default=True, description="Whether to enable HTML2Text processing")
    openrouter_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM configuration")
    llm_chunk_threshold: int = Field(default=LLM_CONTENT_THRESHOLD, description="Word count threshold to enable LLM chunking")
    chunk_size_words: int = Field(default=CHUNK_SIZE_WORDS, description="Chunk size in words for LLM processing")
    chunk_overlap_words: int = Field(default=CHUNK_OVERLAP_WORDS, description="Overlap length in words between adjacent chunks")
    max_chunks_returned: int = Field(default=MAX_CHUNKS_RETURNED, description="Maximum number of chunks to return and merge")

    def init_module(self):
        self._initialize_handlers()
    
    def _initialize_handlers(self):
        """Initialize available handlers based on configuration and environment."""

        if not getattr(self, "_handlers", None):
            self._handlers = dict()

        # Always available handlers
        self._handlers["disabled"] = DisabledPageContentHandler(
            max_words=self.max_words,
            suffix=self.suffix
        )
        
        # HTML2Text handler
        if self.enable_html2text:
            try:
                self._handlers["html2text"] = HTML2TextPageContentHandler(
                    max_words=self.max_words,
                    suffix=self.suffix,
                    ignore_images=True
                )
            except Exception as e:
                logger.warning(f"Warning: Could not initialize HTML2Text handler: {e}")
        
        # LLM handler - only if API key is available
        if self.enable_llm:
            try:
                if self.openrouter_key is None and self.llm_config is None:
                    logger.warning("Warning: Could not initialize LLM handler: must provide `openrouter_key` or `llm_config`")

                else:                 
                    if self.openrouter_key is not None and self.llm_config is None:
                        llm_conf = deepcopy(DEFAULT_PAGEHANDLER_LLM_CONFIG)
                        llm_conf["openrouter_key"] = self.openrouter_key
                        self.llm_config = OpenRouterConfig(**llm_conf)

                    self._handlers["llm"] = LLMPageContentHandler(
                        max_words=self.max_words,
                        suffix=self.suffix,
                        llm_config=self.llm_config,
                        handler_config=self.handler_config,
                        llm_chunk_threshold=self.llm_chunk_threshold,
                        chunk_size_words=self.chunk_size_words,
                        chunk_overlap_words=self.chunk_overlap_words,
                        max_chunks_returned=self.max_chunks_returned
                    )
            except Exception as e:
                logger.warning(f"Warning: Could not initialize LLM handler: {e}")
    
    def _generate_handler_order(self, content: str, query: Optional[str] = None) -> List[str]:
        """Generate the default execution order based on content and query."""
        content = content or ""
        
        # If preferred handler is specified and available, use it first
        if self.preferred_handler and self.preferred_handler in self._handlers:
            others = [h for h in self._handlers.keys() if h != self.preferred_handler and h != "disabled"]
            return [self.preferred_handler] + others + ["disabled"]
        
        # Default order based on content analysis
        is_html = any(tag in content.lower() for tag in ['<html', '<body', '<div', '<p', '<h1'])
        has_query = bool(query and query.strip())
        is_long = _count_words(content) > self.llm_chunk_threshold  # Very long content: ~20,000+ words
        
        # Build order based on content type and available handlers
        order = []
        
        if is_long and "llm" in self._handlers:
            # Long content - prefer LLM summarization even without query
            order.append("llm")
        
        if is_html and "html2text" in self._handlers:
            # HTML content - use HTML2Text
            order.append("html2text")
        
        if has_query and "llm" in self._handlers and "llm" not in order:
            # Plain text with query - use LLM
            order.append("llm")
        
        # Always add disabled as final fallback
        if "disabled" in order:
            order.remove("disabled")
        order.append("disabled")
        
        return order
    
    def _should_handle(self, handler_name: str, content: str, query: Optional[str] = None) -> bool:
        """Determine if we should use this handler based on content and query."""
        if handler_name not in self._handlers:
            return False
        
        # Always allow disabled handler as fallback
        if handler_name == "disabled":
            return True
        
        # LLM handler: use for very long content; prefer when query present but not required
        if handler_name == "llm":
            has_query = bool(query and query.strip())
            is_long = _count_words(content) > self.llm_chunk_threshold
            return is_long or has_query
        
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
            Processed content result
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
                handler_result = handler.handle(content, query)
                return PageHandlingResult(
                    result=self._truncate_content(handler_result.result),
                    cost=handler_result.cost,
                    cost_breakdown=handler_result.cost_breakdown,
                    error=handler_result.error,
                    handler_name=handler_name,
                )
            except Exception as e:
                logger.warning(f"AutoPageHandler: Handler {handler_name} failed: {str(e)}, trying next...")
                continue
        
        # If all handlers failed, use disabled handler as final fallback
        fallback_result = self._handlers["disabled"].handle(content, query)
        # If LLM is available and content is long, still attempt summarization without query
        if self.enable_llm and "llm" in self._handlers and _count_words(content) > self.llm_chunk_threshold:
            try:
                llm_handler = self._handlers["llm"]
                llm_result = llm_handler.handle(content, query)
                return PageHandlingResult(
                    result=self._truncate_content(llm_result.result),
                    cost=llm_result.cost,
                    cost_breakdown=llm_result.cost_breakdown,
                    error=llm_result.error,
                    handler_name="llm",
                )
            except Exception:
                pass
        return fallback_result

    async def async_handle(self, content: str, query: Optional[str] = None):
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
                if hasattr(handler, "async_handle"):
                    handler_result = await handler.async_handle(content, query)
                else:
                    handler_result = handler.handle(content, query)
                
                return PageHandlingResult(
                    result=self._truncate_content(handler_result.result),
                    cost=handler_result.cost,
                    cost_breakdown=handler_result.cost_breakdown,
                    error=handler_result.error,
                    handler_name=handler_name,
                )
            except Exception as e:
                logger.warning(f"AutoPageHandler: Handler {handler_name} failed: {str(e)}, trying next...")
                continue
        
        # If all handlers failed, use disabled handler as final fallback
        fallback_result = self._handlers["disabled"].handle(content, query)
        # If LLM is available and content is long, still attempt summarization without query
        # Use safe word count check
        word_count = _count_words(content or "")
        if self.enable_llm and "llm" in self._handlers and word_count > self.llm_chunk_threshold:
            try:
                llm_handler = self._handlers["llm"]
                if hasattr(llm_handler, "async_handle"):
                    llm_result = await llm_handler.async_handle(content, query)
                else:
                    llm_result = llm_handler.handle(content, query)

                return PageHandlingResult(
                    result=self._truncate_content(llm_result.result),
                    cost=llm_result.cost,
                    cost_breakdown=llm_result.cost_breakdown,
                    error=llm_result.error,
                    handler_name="llm",
                )
            except Exception:
                pass
        return fallback_result
    
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
    
    def handle_page_content(self, content: str, query: Optional[str] = None, page_content_handler: Optional[PageContentHandler] = None):
        """
        Process page content using the specified or default handler.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            page_content_handler: Optional handler override
            
        Returns:
            PageHandlingResult
        """
        if not page_content_handler:
            page_content_handler = self.page_content_handler
        
        res = page_content_handler.handle(content, query)
        return res
    
    async def async_handle_page_content(self, content: str, query: Optional[str] = None, page_content_handler: Optional[PageContentHandler] = None):
        """
        Process page content using the specified or default handler asynchronously.
        
        Args:
            content: Raw content to process
            query: Optional query for context
            page_content_handler: Optional handler override
            
        Returns:
            PageHandlingResult
        """
        if not page_content_handler:
            page_content_handler = self.page_content_handler
        
        if hasattr(page_content_handler, "async_handle"):
            res = await page_content_handler.async_handle(content, query)
        else:
            res = page_content_handler.handle(content, query)
        return res
