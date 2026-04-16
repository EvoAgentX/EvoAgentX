"""
Cache module for research tools.

Provides file-based caching with:
- Thread-safe in-memory cache with lazy loading
- File-based persistence with file locking for concurrent access
- Background async write operations to avoid blocking the main thread
- Extensible design for adding caching to any tool
- BM25-based fuzzy matching for cache key lookup
"""

import atexit
import hashlib
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from ...core.logging import logger
from .text_match import BM25, text_match
from .utils import normalize_title_aggressively

# Try to import filelock for cross-process file locking
try:
    from filelock import FileLock
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False
    logger.warning("filelock not available - file locking disabled. Install with: pip install filelock")


# Global background writer executor (shared across all cached tools)
_background_writer: Optional[ThreadPoolExecutor] = None
_writer_lock = threading.Lock()

# Global registry of BM25CacheKeyGenerator instances for shutdown cleanup
_bm25_generators: List["BM25CacheKeyGenerator"] = []
_generators_lock = threading.Lock()


def _register_bm25_generator(generator: "BM25CacheKeyGenerator") -> None:
    """Register a BM25CacheKeyGenerator for shutdown cleanup."""
    with _generators_lock:
        _bm25_generators.append(generator)


def get_background_writer() -> ThreadPoolExecutor:
    """Get or create the global background writer executor."""
    global _background_writer
    if _background_writer is None:
        with _writer_lock:
            if _background_writer is None:
                _background_writer = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache_writer")
    return _background_writer


class CacheKeyGenerator(ABC):
    """Abstract base class for generating cache keys."""

    @abstractmethod
    def generate_key_for_get(self, *args, **kwargs) -> Optional[str]:
        """
        Generate a cache key for GET operation.
        May use fuzzy matching to find existing keys.

        Returns:
            Cache key string if found, None if no match.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_key_for_set(self, *args, **kwargs) -> str:
        """
        Generate a cache key for SET operation.
        Should use normalized form for consistent storage.

        Returns:
            Cache key string.
        """
        raise NotImplementedError

    def on_cache_set(self, key: str, *args, **kwargs) -> None:
        """
        Called after a value is set in cache.
        Override to update any indexes (e.g., BM25).
        """
        pass

    # For backward compatibility
    def generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key (default to set behavior)."""
        return self.generate_key_for_set(*args, **kwargs)


class DefaultCacheKeyGenerator(CacheKeyGenerator):
    """Default cache key generator using MD5 hash of pickle-serialized arguments."""

    def generate_key_for_get(self, *args, **kwargs) -> Optional[str]:
        """Generate a cache key by hashing the pickle representation of arguments."""
        return self.generate_key_for_set(*args, **kwargs)

    def generate_key_for_set(self, *args, **kwargs) -> str:
        """Generate a cache key by hashing the pickle representation of arguments."""
        key_data = {"args": args, "kwargs": kwargs}
        key_bytes = pickle.dumps(key_data)
        return hashlib.md5(key_bytes).hexdigest()


class BM25CacheKeyGenerator(CacheKeyGenerator):
    """
    Cache key generator with BM25-based fuzzy matching for lookup.

    Features:
    - Uses BM25 index for fuzzy title matching during GET
    - Uses normalized title for consistent key generation during SET
    - Thread-safe index updates with batched rebuilding
    - Persists BM25 index to disk
    """

    # Threshold for rebuilding BM25 index after new items added
    INDEX_REBUILD_THRESHOLD = 20

    def __init__(
        self,
        index_name: str,
        cache_dir: Optional[str] = None,
        top_k: int = 5,
    ):
        """
        Initialize the BM25-based cache key generator.

        Args:
            index_name: Name for the BM25 index file
            cache_dir: Directory to store index files
            top_k: Number of top candidates to consider for matching
        """
        self.index_name = index_name
        self.top_k = top_k

        # Set up cache directory
        if cache_dir is None:
            current_file_abspath = os.path.abspath(__file__)
            cache_dir = os.path.join(Path(current_file_abspath).parent.parent, "assets", "research_tools", "cache")
            cache_dir = os.path.normpath(cache_dir)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Index file paths
        self.index_file = os.path.join(self.cache_dir, f"{index_name}_bm25.pkl")
        self.corpus_file = os.path.join(self.cache_dir, f"{index_name}_corpus.pkl")

        # In-memory state
        self._corpus: List[str] = []  # List of normalized titles
        self._key_map: Dict[str, str] = {}  # normalized_title -> cache_key
        self._bm25 = None  # Lazy loaded BM25 index
        self._pending_items: List[Tuple[str, str]] = []  # Items waiting to be indexed
        self._loaded = False

        # Thread safety
        self._lock = threading.RLock()
        self._index_lock = threading.RLock()

        # Register for shutdown cleanup
        _register_bm25_generator(self)

    def _normalize_title(self, title: str) -> str:
        """
        Normalize title for consistent key generation.
        Uses aggressive normalization: lowercase, remove punctuation.
        """
        return normalize_title_aggressively(title)

    def _generate_cache_key(self, normalized_title: str) -> str:
        """Generate MD5 hash key from normalized title."""
        return hashlib.md5(normalized_title.encode()).hexdigest()

    def _ensure_loaded(self) -> None:
        """Ensure BM25 index and corpus are loaded from disk."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            # Load corpus and key map
            if os.path.exists(self.corpus_file):
                try:
                    with open(self.corpus_file, "rb") as f:
                        data = pickle.load(f)
                        self._corpus = data.get("corpus", [])
                        self._key_map = data.get("key_map", {})
                except Exception as e:
                    logger.warning(f"Failed to load corpus from {self.corpus_file}: {e}")
                    self._corpus = []
                    self._key_map = {}

            # Load or rebuild BM25 index
            if os.path.exists(self.index_file) and self._corpus:
                try:
                    self._bm25 = BM25.from_index(self.index_file)
                except Exception as e:
                    logger.warning(f"Failed to load BM25 index from {self.index_file}: {e}")
                    self._rebuild_bm25_index()
            elif self._corpus:
                self._rebuild_bm25_index()

            self._loaded = True

    def _rebuild_bm25_index(self, save_async: bool = True) -> None:
        """Rebuild BM25 index from corpus."""
        if not self._corpus:
            self._bm25 = None
            return

        try:
            with self._index_lock:
                self._bm25 = BM25(self._corpus)
                if save_async:
                    # Save index to disk in background
                    get_background_writer().submit(self._save_index)
                else:
                    # Save index synchronously (used for flush/shutdown)
                    self._save_index()
        except Exception as e:
            logger.warning(f"Failed to rebuild BM25 index: {e}")
            self._bm25 = None

    def _save_index(self) -> None:
        """Save BM25 index and corpus to disk."""
        try:
            with self._lock:
                corpus_data = {
                    "corpus": list(self._corpus),
                    "key_map": dict(self._key_map),
                }

            # Save corpus
            temp_corpus = self.corpus_file + ".tmp"
            with open(temp_corpus, "wb") as f:
                pickle.dump(corpus_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_corpus, self.corpus_file)

            # Save BM25 index
            with self._index_lock:
                if self._bm25 is not None:
                    self._bm25.save_index(self.index_file)

            logger.debug(f"BM25 index saved for {self.index_name}")
        except Exception as e:
            logger.warning(f"Failed to save BM25 index: {e}")

    def _find_matching_key(self, query_title: str) -> Optional[str]:
        """
        Use BM25 to find a matching cache key for the query title.

        Args:
            query_title: The title to search for

        Returns:
            Cache key if a match is found, None otherwise.
        """
        self._ensure_loaded()

        if not self._corpus or self._bm25 is None:
            return None

        try:
            # Get top-k candidates from BM25
            with self._index_lock:
                results = self._bm25.rank(query_title, topk=self.top_k)

            if not results:
                return None

            # Extract candidate titles
            candidates = [r[2] for r in results]  # (idx, score, text)

            # Use text_match to verify the best candidate
            match_result = text_match(query_title, candidates)

            if match_result["match"] and match_result["best_candidate"]:
                matched_title = match_result["best_candidate"]
                # Look up the cache key for this title
                with self._lock:
                    return self._key_map.get(matched_title)

            return None
        except Exception as e:
            logger.warning(f"BM25 matching failed for '{query_title[:50]}...': {e}")
            return None

    def _add_to_index(self, normalized_title: str, cache_key: str) -> None:
        """
        Add a new title to the corpus and schedule index rebuild if needed.

        Args:
            normalized_title: Normalized title to add
            cache_key: Associated cache key
        """
        with self._lock:
            # Check if already exists
            if normalized_title in self._key_map:
                return

            # Add to corpus and key map
            self._corpus.append(normalized_title)
            self._key_map[normalized_title] = cache_key
            self._pending_items.append((normalized_title, cache_key))

            # Check if we need to rebuild index
            pending_count = len(self._pending_items)

        if pending_count >= self.INDEX_REBUILD_THRESHOLD:
            self._flush_pending_items()

    def _flush_pending_items(self, rebuild_index: bool = True) -> None:
        """
        Flush pending items and optionally rebuild index.

        Args:
            rebuild_index: If True, rebuild BM25 index. If False, only save corpus.
        """
        with self._lock:
            if not self._pending_items:
                return
            self._pending_items.clear()

        if rebuild_index:
            # Rebuild index in background (also saves corpus)
            get_background_writer().submit(self._rebuild_bm25_index)
        else:
            # Only save corpus without rebuilding index
            get_background_writer().submit(self._save_corpus_only)

    def _save_corpus_only(self) -> None:
        """Save corpus and key_map to disk without rebuilding BM25 index."""
        try:
            with self._lock:
                corpus_data = {
                    "corpus": list(self._corpus),
                    "key_map": dict(self._key_map),
                }

            # Save corpus
            temp_corpus = self.corpus_file + ".tmp"
            with open(temp_corpus, "wb") as f:
                pickle.dump(corpus_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_corpus, self.corpus_file)

            logger.debug(f"Corpus saved for {self.index_name} (without index rebuild)")
        except Exception as e:
            logger.warning(f"Failed to save corpus: {e}")

    def generate_key_for_get(self, title: str) -> Optional[str]:
        """
        Generate cache key for GET using BM25 fuzzy matching.

        First tries exact match (normalized), then falls back to BM25 search.
        """
        normalized = self._normalize_title(title)
        exact_key = self._generate_cache_key(normalized)

        # First check if we have an exact match in key_map
        self._ensure_loaded()
        with self._lock:
            if normalized in self._key_map:
                return exact_key

        # Try BM25 fuzzy matching
        matched_key = self._find_matching_key(title)
        if matched_key:
            return matched_key

        # Return exact key (may or may not exist in cache)
        return exact_key

    def generate_key_for_set(self, title: str) -> str:
        """
        Generate cache key for SET using normalized title.
        """
        normalized = self._normalize_title(title)
        return self._generate_cache_key(normalized)

    def on_cache_set(self, key: str, title: str, *args, **kwargs) -> None:
        """
        Called after cache set to update BM25 index.
        """
        normalized = self._normalize_title(title)
        self._add_to_index(normalized, key)

    def flush(self) -> None:
        """
        Force flush any pending index updates.
        Saves corpus even if pending items count is below threshold.
        This is a synchronous operation that saves immediately.
        """
        with self._lock:
            if not self._pending_items:
                return
            # Clear pending items
            self._pending_items.clear()

        # Save corpus synchronously (not in background) to ensure data is saved before shutdown
        self._rebuild_bm25_index(save_async=False)


class ToolCache:
    """
    Thread-safe cache with file persistence and background writes.

    Features:
    - In-memory cache with lazy loading from file
    - Thread-safe operations using locks
    - File-based persistence with file locking for concurrent access
    - Background async writes to avoid blocking main thread
    - Configurable cache directory and TTL
    """

    def __init__(
        self,
        tool_name: str,
        cache_dir: Optional[str] = None,
        ttl: Optional[float] = None,  # Time-to-live in seconds, None = no expiry
        key_generator: Optional[CacheKeyGenerator] = None,
    ):
        """
        Initialize the tool cache.

        Args:
            tool_name: Name of the tool (used for cache file naming)
            cache_dir: Directory to store cache files. Defaults to ~/.evoagentx/cache/research_tools/
            ttl: Time-to-live in seconds. None means no expiry.
            key_generator: Custom key generator. Defaults to DefaultCacheKeyGenerator.
        """
        self.tool_name = tool_name
        self.ttl = ttl
        self.key_generator = key_generator or DefaultCacheKeyGenerator()

        # Set up cache directory
        if cache_dir is None:
            # Default path: evoagentx/tools/assets/research_tools/cache
            current_file_abspath = os.path.abspath(__file__)
            cache_dir = os.path.join(Path(current_file_abspath).parent.parent, "assets", "research_tools", "cache")
            cache_dir = os.path.normpath(cache_dir)
        self.cache_dir = cache_dir

        # Cache file path
        self.cache_file = os.path.join(self.cache_dir, f"{tool_name}_cache.pkl")
        self.lock_file = os.path.join(self.cache_dir, f"{tool_name}_cache.lock")

        # In-memory cache and state
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._dirty = False  # Track if memory cache has unsaved changes

        # Thread safety
        self._lock = threading.RLock()

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_file_lock(self) -> Optional["FileLock"]:
        """Get a file lock for thread/process-safe file operations."""
        if FILELOCK_AVAILABLE:
            return FileLock(self.lock_file, timeout=10)
        return None

    def _load_from_file(self) -> Dict[str, Dict[str, Any]]:
        """Load cache data from file. Thread-safe with file locking."""
        if not os.path.exists(self.cache_file):
            return {}

        file_lock = self._get_file_lock()
        try:
            if file_lock:
                with file_lock:
                    return self._read_cache_file()
            else:
                return self._read_cache_file()
        except Exception as e:
            logger.warning(f"Failed to load cache from {self.cache_file}: {e}")
            return {}

    def _read_cache_file(self) -> Dict[str, Dict[str, Any]]:
        """Read and parse the cache file."""
        if not os.path.exists(self.cache_file):
            return {}
        with open(self.cache_file, "rb") as f:
            return pickle.load(f)

    def _save_to_file(self, data: Dict[str, Dict[str, Any]]) -> bool:
        """Save cache data to file. Thread-safe with file locking."""
        file_lock = self._get_file_lock()
        try:
            if file_lock:
                with file_lock:
                    self._write_cache_file(data)
            else:
                self._write_cache_file(data)
            return True
        except Exception as e:
            logger.warning(f"Failed to save cache to {self.cache_file}: {e}")
            return False

    def _write_cache_file(self, data: Dict[str, Dict[str, Any]]):
        """Write cache data to file."""
        # Write to temp file first, then rename for atomic operation
        temp_file = self.cache_file + ".tmp"
        with open(temp_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_file, self.cache_file)

    def _ensure_loaded(self) -> None:
        """Ensure cache is loaded from file. Only loads once (lazy loading)."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._memory_cache = self._load_from_file()
                    self._loaded = True

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is expired."""
        if self.ttl is None:
            return False
        created_at = entry.get("created_at", 0)
        if isinstance(created_at, str):
            try:
                created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                try:
                    created_at = datetime.fromisoformat(created_at).timestamp()
                except ValueError:
                    created_at = 0
        return (time.time() - created_at) > self.ttl

    def get(self, *args, **kwargs) -> Tuple[Optional[Any], bool]:
        """
        Get a cached value.

        Args:
            *args, **kwargs: Arguments used to generate the cache key

        Returns:
            Tuple of (cached_value, hit). hit is True if cache was found and valid.
        """
        self._ensure_loaded()

        key = self.key_generator.generate_key_for_get(*args, **kwargs)

        if key is None:
            return None, False

        with self._lock:
            entry = self._memory_cache.get(key)
            if entry is None:
                return None, False

            if self._is_expired(entry):
                # Remove expired entry
                del self._memory_cache[key]
                self._dirty = True
                removed_expired = True
            else:
                removed_expired = False
                return entry.get("value"), True

        if removed_expired:
            self._schedule_background_save()
            return None, False

    def set(self, value: Any, *args, **kwargs):
        """
        Set a cached value with background persistence.

        Args:
            value: The value to cache
            *args, **kwargs: Arguments used to generate the cache key
        """
        self._ensure_loaded()

        key = self.key_generator.generate_key_for_set(*args, **kwargs)

        entry = {
            "value": value,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with self._lock:
            self._memory_cache[key] = entry
            self._dirty = True

        # Notify key generator about the new entry
        self.key_generator.on_cache_set(key, *args, **kwargs)

        # Schedule background write
        self._schedule_background_save()

    def _schedule_background_save(self) -> None:
        """Schedule a background save operation."""
        writer = get_background_writer()
        writer.submit(self._background_save)

    def _background_save(self) -> None:
        """Save cache to file in background thread."""
        with self._lock:
            if not self._dirty:
                return
            # Make a copy of the cache data
            data_to_save = dict(self._memory_cache)

        saved = self._save_to_file(data_to_save)
        if saved:
            with self._lock:
                self._dirty = False
        logger.debug(f"Cache saved for {self.tool_name}")

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._memory_cache.clear()
            self._dirty = True

        # Delete file synchronously for clear operation
        if os.path.exists(self.cache_file):
            file_lock = self._get_file_lock()
            try:
                if file_lock:
                    with file_lock:
                        if os.path.exists(self.cache_file):
                            os.remove(self.cache_file)
                else:
                    if os.path.exists(self.cache_file):
                        os.remove(self.cache_file)
            except Exception as e:
                logger.warning(f"Failed to delete cache file {self.cache_file}: {e}")

    def flush(self) -> None:
        """Force save any pending changes to file."""
        data_to_save = None
        with self._lock:
            if self._dirty:
                data_to_save = dict(self._memory_cache)
                self._dirty = False

        if data_to_save:
            saved = self._save_to_file(data_to_save)
            if not saved:
                with self._lock:
                    self._dirty = True

        # Also flush key generator if it has pending updates
        if hasattr(self.key_generator, 'flush'):
            self.key_generator.flush()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._ensure_loaded()

        with self._lock:
            total_entries = len(self._memory_cache)
            expired_entries = sum(1 for e in self._memory_cache.values() if self._is_expired(e))

        return {
            "tool_name": self.tool_name,
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "valid_entries": total_entries - expired_entries,
            "cache_file": self.cache_file,
            "loaded": self._loaded,
        }


class CacheMixin:
    """
    Mixin class to add caching capability to research tools.

    Usage:
        class MyTool(CacheMixin, Tool):
            def __init__(self, ...):
                super().__init__(...)
                self._init_cache("my_tool")

            def _get_cache_key_args(self, **kwargs):
                # Return the args that should be used for cache key
                return (kwargs.get("query"),)

            def __call__(self, query: str):
                # Check cache first
                cached, hit = self._cache_get(query=query)
                if hit:
                    return cached

                # Do actual work
                result = self._do_actual_work(query)

                # Cache the result
                self._cache_set(result, query=query)
                return result
    """

    _tool_cache: Optional[ToolCache] = None

    def _init_cache(
        self,
        tool_name: str,
        cache_dir: Optional[str] = None,
        ttl: Optional[float] = None,
        key_generator: Optional[CacheKeyGenerator] = None,
    ):
        """
        Initialize the cache for this tool.

        Args:
            tool_name: Name of the tool (used for cache file naming)
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds
            key_generator: Custom key generator
        """
        self._tool_cache = ToolCache(
            tool_name=tool_name,
            cache_dir=cache_dir,
            ttl=ttl,
            key_generator=key_generator,
        )

    def _cache_get(self, *args, **kwargs) -> Tuple[Optional[Any], bool]:
        """
        Get a cached value.

        Returns:
            Tuple of (cached_value, hit). hit is True if cache was found.
        """
        if self._tool_cache is None:
            return None, False
        return self._tool_cache.get(*args, **kwargs)

    def _cache_set(self, value: Any, *args, **kwargs):
        """Set a cached value with background persistence."""
        if self._tool_cache is not None:
            self._tool_cache.set(value, *args, **kwargs)

    def _cache_clear(self) -> None:
        """Clear all cached data for this tool."""
        if self._tool_cache is not None:
            self._tool_cache.clear()

    def _cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._tool_cache is None:
            return {"error": "Cache not initialized"}
        return self._tool_cache.stats()


# Specialized cache key generators for research tools

class BibReferenceCacheKeyGenerator(BM25CacheKeyGenerator):
    """Cache key generator for BibReferenceTool using BM25-based fuzzy matching."""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            index_name="bib_reference",
            cache_dir=cache_dir,
            top_k=5,
        )

    def generate_key_for_get(self, title_or_keyword: str) -> Optional[str]:
        """Generate cache key for GET using BM25 fuzzy matching."""
        return super().generate_key_for_get(title_or_keyword)

    def generate_key_for_set(self, title_or_keyword: str) -> str:
        """Generate cache key for SET using normalized title."""
        return super().generate_key_for_set(title_or_keyword)

    def on_cache_set(self, key: str, title_or_keyword: str) -> None:
        """Update BM25 index after cache set."""
        super().on_cache_set(key, title_or_keyword)


class PaperMetadataCacheKeyGenerator(BM25CacheKeyGenerator):
    """Cache key generator for FetchPaperMetaDataTool using BM25-based fuzzy matching."""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            index_name="paper_metadata",
            cache_dir=cache_dir,
            top_k=5,
        )

    def generate_key_for_get(self, paper_title: str) -> Optional[str]:
        """Generate cache key for GET using BM25 fuzzy matching."""
        return super().generate_key_for_get(paper_title)

    def generate_key_for_set(self, paper_title: str) -> str:
        """Generate cache key for SET using normalized title."""
        return super().generate_key_for_set(paper_title)

    def on_cache_set(self, key: str, paper_title: str) -> None:
        """Update BM25 index after cache set."""
        super().on_cache_set(key, paper_title)


class ConferencePageCacheKeyGenerator(DefaultCacheKeyGenerator):
    """Cache key generator for DBLP conference pages using the URL as the key."""

    def generate_key_for_get(self, conf_url: str) -> Optional[str]:
        return conf_url or None

    def generate_key_for_set(self, conf_url: str) -> str:
        return conf_url or ""


def shutdown_background_writer() -> None:
    """
    Shutdown the background writer executor. Call this on application shutdown.

    This function:
    1. Flushes all pending BM25 index updates (saves corpus even if below threshold)
    2. Waits for all background writes to complete
    3. Shuts down the executor
    """
    global _background_writer, _bm25_generators

    # First, flush all registered BM25 generators to save pending items
    with _generators_lock:
        generators_to_flush = list(_bm25_generators)

    for generator in generators_to_flush:
        try:
            generator.flush()
        except Exception as e:
            logger.warning(f"Failed to flush BM25 generator {generator.index_name}: {e}")

    # Wait for background writer to complete all pending tasks and shutdown
    if _background_writer is not None:
        _background_writer.shutdown(wait=True)
        _background_writer = None

    # Clear the generators list
    with _generators_lock:
        _bm25_generators.clear()


# Register shutdown handler to automatically save data on exit
atexit.register(shutdown_background_writer)
