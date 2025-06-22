import requests
from bs4 import BeautifulSoup
from typing import Tuple, Optional
from ..core.module import BaseModule
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
    
    def _truncate_content(self, content: str, max_words: Optional[int]) -> str:
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
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            return None, None

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = soup.title.string if soup.title else "No Title"

        # Extract text content (only from <p> tags)
        paragraphs = soup.find_all("p")
        paragraph_texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        text_content = "\n\n".join(paragraph_texts)

        return title, text_content