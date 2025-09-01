import feedparser
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import re
from urllib.parse import urljoin

from .tool import Tool, Toolkit
from .request_base import RequestBase
from ..core.logging import logger


class RSSBase(RequestBase):
    """
    Base class for RSS feed operations.
    Provides common functionality for fetching, parsing, and processing RSS feeds.
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3, delay_between_requests: float = 1.0):
        """
        Initialize the RSS base with configuration options.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            delay_between_requests: Delay between requests in seconds
        """
        super().__init__(timeout=timeout, max_retries=max_retries, delay_between_requests=delay_between_requests)
    
    def fetch_rss_feed(self, feed_url: str, max_entries: Optional[int] = 10, fetch_webpage_content: bool = True) -> Dict[str, Any]:
        """
        Fetch and parse an RSS feed from a URL.
        
        Args:
            feed_url: URL of the RSS feed
            max_entries: Maximum number of entries to return (default: 10, None for all)
            fetch_webpage_content: Whether to fetch and extract content from article webpages (default: True)
            
        Returns:
            Dictionary containing parsed feed information
        """
        try:
            # Make the HTTP request
            response = self.request(url=feed_url, method='GET')
            
            # Parse the RSS feed
            feed = feedparser.parse(response.content)
            
            # Check for parsing errors
            if feed.bozo:
                logger.warning(f"RSS feed parsing warnings for {feed_url}: {feed.bozo_exception}")
            
            # Extract feed metadata
            feed_info = {
                'success': True,
                'feed_url': feed_url,
                'title': getattr(feed.feed, 'title', 'Unknown'),
                'description': getattr(feed.feed, 'description', ''),
                'link': getattr(feed.feed, 'link', ''),
                'language': getattr(feed.feed, 'language', ''),
                'updated': getattr(feed.feed, 'updated', ''),
                'generator': getattr(feed.feed, 'generator', ''),
                'total_entries': len(feed.entries),
                'entries': []
            }
            
            # Process entries
            entries = feed.entries[:max_entries] if max_entries is not None else feed.entries
            
            for entry in entries:
                processed_entry = self._process_entry(entry, feed_url, fetch_webpage_content)
                feed_info['entries'].append(processed_entry)
            
            return feed_info
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed from {feed_url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'feed_url': feed_url
            }
    
    def _process_entry(self, entry, base_url: str, fetch_webpage_content: bool = True) -> Dict[str, Any]:
        """
        Process a single RSS entry and extract relevant information.
        
        Args:
            entry: FeedParser entry object
            base_url: Base URL for resolving relative links
            fetch_webpage_content: Whether to fetch and extract content from the article webpage
            
        Returns:
            Dictionary with processed entry information
        """
        # Extract basic information
        processed_entry = {
            'title': getattr(entry, 'title', ''),
            'description': getattr(entry, 'description', ''),
            'link': getattr(entry, 'link', ''),
            'published': getattr(entry, 'published', ''),
            'author': getattr(entry, 'author', ''),
            'id': getattr(entry, 'id', ''),
            'summary': getattr(entry, 'summary', ''),
            'content': getattr(entry, 'content', []),
            'tags': [],
            'categories': [],
            'enclosures': []
        }
        
        
        # Resolve relative links
        if processed_entry['link'] and not processed_entry['link'].startswith(('http://', 'https://')):
            processed_entry['link'] = urljoin(base_url, processed_entry['link'])
        
        # Extract tags/categories
        if hasattr(entry, 'tags'):
            processed_entry['tags'] = [tag.term for tag in entry.tags if hasattr(tag, 'term')]
        
        if hasattr(entry, 'category'):
            processed_entry['categories'] = [entry.category] if isinstance(entry.category, str) else entry.category
        
        # Extract enclosures (media files)
        if hasattr(entry, 'enclosures'):
            for enclosure in entry.enclosures:
                processed_entry['enclosures'].append({
                    'url': getattr(enclosure, 'href', ''),
                    'type': getattr(enclosure, 'type', ''),
                    'length': getattr(enclosure, 'length', ''),
                    'title': getattr(enclosure, 'title', '')
                })
        
        # Parse dates
        processed_entry['published_parsed'] = self._parse_date(entry.published_parsed)
        
        
        # Clean up text content
        processed_entry['title'] = self._clean_text(processed_entry['title'])
        processed_entry['description'] = self._clean_text(processed_entry['description'])
        processed_entry['summary'] = self._clean_text(processed_entry['summary'])
        
        # Fetch webpage content if requested and link is available
        if fetch_webpage_content and processed_entry['link']:
            result = self.request_and_process(url=processed_entry['link'], method='GET')
            if result.get('success') and result.get('content'):
                # Clean up the text and limit length
                text_content = self._clean_text(result['content'])
                if len(text_content) > 10000:  # 10KB limit
                    text_content = text_content[:10000] + "... [Content truncated]"
                processed_entry['webpage_content'] = text_content
                processed_entry['webpage_content_fetched'] = True
            else:
                processed_entry['webpage_content_fetched'] = False
        else:
            processed_entry['webpage_content_fetched'] = False
        
        return processed_entry
    
    def _parse_date(self, date_tuple) -> Optional[str]:
        """
        Parse a date tuple from feedparser into ISO format string.
        
        Args:
            date_tuple: Date tuple from feedparser
            
        Returns:
            ISO format date string or None
        """
        if not date_tuple:
            return None
        
        try:
            # Convert to datetime object
            dt = datetime(*date_tuple[:6])
            # Make timezone-aware if not already
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean HTML tags and normalize whitespace in text.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        if not text:
            return ''
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    

    
    def validate_rss_url(self, url: str) -> Dict[str, Any]:
        """
        Validate if a URL contains a valid RSS feed.
        
        Args:
            url: URL to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            response = self.request(url=url, method='GET')
            content = response.content
            
            # Check if it's XML
            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                return {
                    'success': False,
                    'error': 'Invalid XML content',
                    'url': url
                }
            
            # Check for RSS/Atom elements
            is_rss = root.tag.endswith('rss') or root.tag.endswith('RDF')
            is_atom = root.tag.endswith('feed') or 'atom' in root.tag
            
            if is_rss or is_atom:
                return {
                    'success': True,
                    'is_valid': True,
                    'feed_type': 'RSS' if is_rss else 'Atom',
                    'url': url,
                    'title': self._extract_feed_title(root)
                }
            else:
                return {
                    'success': True,
                    'is_valid': False,
                    'error': 'Not a valid RSS or Atom feed',
                    'url': url
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _extract_feed_title(self, root) -> str:
        """
        Extract feed title from XML root element.
        
        Args:
            root: XML root element
            
        Returns:
            Feed title or empty string
        """
        # Try different possible title elements
        title_selectors = [
            './/title',
            './/channel/title',
            './/feed/title'
        ]
        
        for selector in title_selectors:
            title_elem = root.find(selector)
            if title_elem is not None and title_elem.text:
                return self._clean_text(title_elem.text)
        
        return ''
    



class RSSFetchTool(Tool):
    """Tool for fetching and parsing RSS feeds."""
    
    name: str = "rss_fetch"
    description: str = "Fetch and parse RSS feeds from URLs to get latest articles and updates. Use reasonable limits (10-20 entries) unless you specifically need more for comprehensive analysis."
    inputs: Dict[str, Dict[str, str]] = {
        "feed_url": {
            "type": "string",
            "description": "URL of the RSS feed to fetch"
        },
        "max_entries": {
            "type": "integer",
            "description": "Maximum number of entries to return. Recommended: 10-20 for most use cases, higher only if comprehensive analysis is needed (default: 10)"
        },
        "fetch_webpage_content": {
            "type": "boolean",
            "description": "Whether to fetch and extract content from article webpages. Note: This significantly increases processing time (default: true)"
        }
    }
    required: Optional[List[str]] = ["feed_url"]
    
    def __init__(self, rss_base: RSSBase = None):
        super().__init__()
        self.rss_base = rss_base or RSSBase()
    
    def __call__(self, feed_url: str, max_entries: int = 10, fetch_webpage_content: bool = True) -> Dict[str, Any]:
        """
        Fetch and parse an RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            max_entries: Maximum number of entries to return (default: 10)
            fetch_webpage_content: Whether to fetch and extract content from article webpages
            
        Returns:
            Dictionary with parsed feed information
        """
        return self.rss_base.fetch_rss_feed(feed_url, max_entries, fetch_webpage_content)


class RSSValidateTool(Tool):
    """Tool for validating RSS feed URLs."""
    
    name: str = "rss_validate"
    description: str = "Validate if a URL contains a valid RSS or Atom feed"
    inputs: Dict[str, Dict[str, str]] = {
        "url": {
            "type": "string",
            "description": "URL to validate as an RSS feed"
        }
    }
    required: Optional[List[str]] = ["url"]
    
    def __init__(self, rss_base: RSSBase = None):
        super().__init__()
        self.rss_base = rss_base or RSSBase()
    
    def __call__(self, url: str) -> Dict[str, Any]:
        """
        Validate if a URL contains a valid RSS feed.
        
        Args:
            url: URL to validate
            
        Returns:
            Dictionary with validation results
        """
        return self.rss_base.validate_rss_url(url)

class RSSToolkit(Toolkit):
    """Toolkit for RSS feed operations."""
    
    def __init__(self, name: str = "RSSToolkit"):
        # Create the shared RSS base instance
        rss_base = RSSBase()
        
        # Create tools with the shared RSS base
        tools = [
            RSSFetchTool(rss_base=rss_base),
            RSSValidateTool(rss_base=rss_base)
        ]
        
        super().__init__(name=name, tools=tools)
