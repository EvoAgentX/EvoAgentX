import requests
import html2text
from bs4 import BeautifulSoup
from typing import Tuple, Optional
from ..core.module import BaseModule
from ..core.logging import logger
from pydantic import Field

class SearchBase(BaseModule):
    """
    Base class for search tools that retrieve information from various sources.
    Provides common functionality for search operations.
    """
    
    num_search_pages: Optional[int] = Field(default=5, description="Number of search results to retrieve")
    max_content_words: Optional[int] = Field(default=None, description="Maximum number of words to include in content. Default None means no limit.")
    
    def __init__(
        self, 
        name: str = "SearchBase",
        num_search_pages: Optional[int] = 5, 
        max_content_words: Optional[int] = None, 
        **kwargs
    ):
        """
        Initialize the base search tool.
        
        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, default None means no limit. 
            **kwargs: Additional keyword arguments for parent class initialization
        """ 
        # Pass to parent class initialization
        super().__init__(name=name, num_search_pages=num_search_pages, max_content_words=max_content_words, **kwargs)
        self.content_converter = html2text.HTML2Text()
        # Configure html2text for better content extraction
        self.content_converter.ignore_links = False
        self.content_converter.ignore_images = True
        self.content_converter.body_width = 0  # Don't wrap text
        self.content_converter.unicode_snob = True
        self.content_converter.escape_snob = True
    
    def _truncate_content(self, content: str, max_words: Optional[int] = None) -> str:
        """
        Truncates content to a maximum number of words while preserving original spacing.
        
        Args:
            content (str): The content to truncate
            max_words (Optional[int]): Maximum number of words to include. None means no limit.
            
        Returns:
            str: Truncated content with ellipsis if truncated
        """
        if max_words is None or max_words <= 0:
            return content
            
        words = content.split()
        is_truncated = len(words) > max_words
        word_count = 0
        truncated_content = ""
        
        # Rebuild the content preserving original whitespace
        for i, char in enumerate(content):
            if char.isspace():
                if i > 0 and not content[i-1].isspace():
                    word_count += 1
                if word_count >= max_words:
                    break
            truncated_content += char
            
        # Add ellipsis only if truncated
        return truncated_content + (" ..." if is_truncated else "")
    
    def _scrape_page(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetches the title and main text content from a web page.

        Args:
            url (str): The URL of the web page.

        Returns:
            tuple: (Optional[title], Optional[main textual content])
        """
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None, None

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = soup.title.string if soup.title else "No Title"

        # Try to extract main content for specific sites
        main_content = None
        
        # For Wikipedia, try to get the main content area
        if 'wikipedia.org' in url:
            main_content = soup.find('div', {'id': 'mw-content-text'})
            if main_content:
                # Remove navigation and other non-content elements
                for element in main_content.find_all(['nav', 'script', 'style', 'table']):
                    element.decompose()
                text_content = self.content_converter.handle(str(main_content))
            else:
                text_content = self.content_converter.handle(response.text)
        else:
            # For other sites, try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': 'content'})
            if main_content:
                text_content = self.content_converter.handle(str(main_content))
            else:
                text_content = self.content_converter.handle(response.text)

        return title, text_content