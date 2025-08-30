import requests
import json
import os
from typing import Dict, Any, Optional, Union
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import html2text
import time

from ..core.module import BaseModule


class RequestBase(BaseModule):
    """
    Base class for handling HTTP requests, parsing content, and saving data.
    This class provides common functionality for web scraping and HTTP operations.
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3, delay_between_requests: float = 1.0):
        """
        Initialize the RequestBase with configuration options.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            delay_between_requests: Delay between requests in seconds
        """
        super().__init__()
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay_between_requests = delay_between_requests
        self.session = requests.Session()
        
        # Initialize html2text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0  # Don't wrap lines
        
        # Default headers to appear more like a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def request(self, url: str, method: str = 'GET', headers: Optional[Dict[str, str]] = None, 
                params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None,
                json_data: Optional[Dict[str, Any]] = None) -> requests.Response:
        """
        Make an HTTP request with retry logic and error handling.
        
        Args:
            url: The URL to request
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            headers: Additional headers to include
            params: URL parameters
            data: Form data to send
            json_data: JSON data to send
            
        Returns:
            requests.Response object
            
        Raises:
            requests.RequestException: If request fails after all retries
        """
        if headers:
            request_headers = {**self.session.headers, **headers}
        else:
            request_headers = self.session.headers
            
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    headers=request_headers,
                    params=params,
                    data=data,
                    json=json_data,
                    timeout=self.timeout
                )
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # Add delay between requests to be respectful
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_between_requests)
                    
                return response
                
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.delay_between_requests * (attempt + 1))  # Exponential backoff
    
    def parse_html(self, html_content: str) -> BeautifulSoup:
        """
        Parse HTML content using BeautifulSoup.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            BeautifulSoup object for parsing
        """
        return BeautifulSoup(html_content, 'html.parser')
    
    def parse_json(self, json_content: str) -> Dict[str, Any]:
        """
        Parse JSON content.
        
        Args:
            json_content: Raw JSON content
            
        Returns:
            Parsed JSON as dictionary
        """
        return json.loads(json_content)
    
    def extract_text(self, html_content: str, selector: Optional[str] = None) -> str:
        """
        Extract text content from HTML using html2text.
        
        Args:
            html_content: Raw HTML content
            selector: CSS selector to extract specific elements (optional)
            
        Returns:
            Extracted text content
        """
        if selector:
            soup = self.parse_html(html_content)
            elements = soup.select(selector)
            combined_html = '\n'.join([str(elem) for elem in elements])
            return self.html_converter.handle(combined_html)
        else:
            return self.html_converter.handle(html_content)
    
    def extract_links(self, html_content: str, base_url: str = None) -> list:
        """
        Extract all links from HTML content.
        
        Args:
            html_content: Raw HTML content
            base_url: Base URL to resolve relative links
            
        Returns:
            List of extracted URLs
        """
        soup = self.parse_html(html_content)
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if base_url and not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                href = urljoin(base_url, href)
            links.append(href)
        
        return links
    
    def save_content(self, content: Union[str, Dict[str, Any], bytes], file_path: str, 
                    content_type: str = 'text') -> bool:
        """
        Save content to a file.
        
        Args:
            content: Content to save (string, dictionary, or bytes)
            file_path: Path where to save the file
            content_type: Type of content ('text', 'json', 'html', 'pdf', 'binary')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if content_type.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
            elif content_type.lower() in ['pdf', 'binary'] or isinstance(content, bytes):
                with open(file_path, 'wb') as f:
                    if isinstance(content, bytes):
                        f.write(content)
                    else:
                        f.write(str(content).encode('utf-8'))
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(content))
            
            return True
            
        except Exception as e:
            print(f"Error saving content to {file_path}: {e}")
            return False
    
    def get_page_info(self, url: str) -> Dict[str, Any]:
        """
        Get basic information about a webpage.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary containing page information
        """
        try:
            response = self.request(url)
            soup = self.parse_html(response.text)
            
            # Extract basic page information
            info = {
                'url': url,
                'status_code': response.status_code,
                'title': soup.title.string if soup.title else '',
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(response.text),
                'links_count': len(soup.find_all('a', href=True)),
                'images_count': len(soup.find_all('img')),
            }
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                info['description'] = meta_desc.get('content', '')
            
            return info
            
        except Exception as e:
            return {'error': str(e), 'url': url}
    
    def request_and_process(self, url: str, method: str = 'GET', headers: Optional[Dict[str, str]] = None,
                           params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None,
                           json_data: Optional[Dict[str, Any]] = None, return_raw: bool = False,
                           save_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a request and process the response with comprehensive error handling.
        
        Args:
            url: The URL to request
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            headers: Additional headers to include
            params: URL parameters
            data: Form data to send
            json_data: JSON data to send
            return_raw: If True, return raw HTML content, otherwise processed text
            save_file_path: Optional path to save the content
            
        Returns:
            Dictionary containing processed response data
        """
        try:
            response = self.request(
                url=url,
                method=method,
                headers=headers,
                params=params,
                data=data,
                json_data=json_data
            )
            
            # Determine content type
            content_type = response.headers.get('content-type', '').lower()
            
            result = {
                'url': url,
                'method': method.upper(),
                'status_code': response.status_code,
                'success': True,
                'content_type': content_type,
                'content_length': len(response.text),
                'headers': dict(response.headers)
            }
            
            # Process content based on type and return_raw flag
            if return_raw:
                result['content'] = response.text
            else:
                if 'json' in content_type:
                    try:
                        result['content'] = response.json()
                    except json.JSONDecodeError:
                        result['content'] = response.text
                        result['warning'] = 'Content-Type indicates JSON but parsing failed'
                else:
                    result['content'] = self.extract_text(response.text)
            
            # Save to file if requested
            if save_file_path:
                save_success = self._save_response_content(response, save_file_path, content_type)
                result['saved_to_file'] = save_file_path if save_success else None
                if not save_success:
                    result['save_warning'] = f'Failed to save content to {save_file_path}'
            
            return result
            
        except Exception as e:
            return {
                'url': url,
                'method': method.upper(),
                'error': str(e),
                'success': False
            }
    
    def _save_response_content(self, response: requests.Response, file_path: str, content_type: str) -> bool:
        """
        Save response content to file with appropriate format.
        
        Args:
            response: The response object
            file_path: Path to save the file
            content_type: Content type of the response
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if 'json' in content_type:
                try:
                    json_content = response.json()
                    return self.save_content(json_content, file_path, 'json')
                except json.JSONDecodeError:
                    return self.save_content(response.text, file_path, 'text')
            elif 'html' in content_type:
                return self.save_content(response.text, file_path, 'html')
            else:
                return self.save_content(response.text, file_path, 'text')
                
        except Exception as e:
            print(f"Error saving response content: {e}")
            return False
    
    def close(self):
        """Close the session."""
        self.session.close() 