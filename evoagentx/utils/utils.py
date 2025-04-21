import os 
import re 
import time
import regex
import requests
from tqdm import tqdm
from typing import Union, Any, List, Set

from ..core.logging import logger

def make_parent_folder(path: str):
    """Create parent directory for a file path if it doesn't exist.
    
    Ensures that the directory structure needed for a specified file path
    exists, creating any necessary parent directories.
    
    Args:
        path: Full file path for which to create parent directory
        
    Notes:
        - Creates all intermediate directories as needed
        - Uses exist_ok=True to avoid race conditions
        - Logs directory creation with info level
    """
    dir_folder = os.path.dirname(path)
    if not os.path.exists(dir_folder):
        logger.info(f"creating folder {dir_folder} ...")
        os.makedirs(dir_folder, exist_ok=True)

def safe_remove(data: Union[List[Any], Set[Any]], remove_value: Any):
    """Remove an item from a list or set without raising exceptions.
    
    Attempts to remove the specified value from the collection, but
    doesn't raise an exception if the value is not found.
    
    Args:
        data: Collection (list or set) to remove the value from
        remove_value: Value to be removed from the collection
        
    Notes:
        - Silently ignores ValueError exceptions when the value isn't present
        - Useful for cleaning up collections when unsure if values exist
    """
    try:
        data.remove(remove_value)
    except ValueError:
        pass

def generate_dynamic_class_name(base_name: str) -> str:
    """Generate a valid Python class name from a base name string.
    
    Processes an arbitrary string to create a valid CamelCase class name
    by removing special characters, normalizing whitespace, and capitalizing
    each word.
    
    Args:
        base_name: The source string to convert to a class name
        
    Returns:
        A valid CamelCase Python class name derived from the input string
        
    Notes:
        - Removes all non-alphanumeric characters
        - Capitalizes each word component
        - Returns 'DefaultClassName' if processing results in an empty string
    """
    base_name = base_name.strip()
    
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', base_name)
    components = cleaned_name.split()
    class_name = ''.join(x.capitalize() for x in components)

    return class_name if class_name else 'DefaultClassName'

def normalize_text(s: str) -> str:
    """Normalize text by applying a series of transformations.
    
    Applies a sequence of text normalization operations:
    1. Converts text to lowercase
    2. Removes punctuation (except spaces)
    3. Removes articles (a, an, the)
    4. Normalizes whitespace
    
    Args:
        s: Input string to normalize
        
    Returns:
        Normalized version of the input text
        
    Notes:
        - Useful for text comparison and processing
        - Replaces underscores with spaces
        - Preserves most punctuation except underscores
    """
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return text.replace("_", " ")
        # exclude = set(string.punctuation)
        # return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def download_file(url: str, save_file: str, max_retries=3, timeout=10):
    """Download a file from a URL with progress tracking and resume capability.
    
    Downloads a file from the specified URL to the specified location, with
    support for resuming interrupted downloads and tracking progress with a
    progress bar.
    
    Args:
        url: URL of the file to download
        save_file: Local file path where the downloaded file will be saved
        max_retries: Maximum number of retry attempts on connection failure
        timeout: Connection timeout in seconds
        
    Raises:
        ValueError: For unexpected errors during download
        RuntimeError: If maximum retries are exceeded
        
    Notes:
        - Creates parent directories if they don't exist
        - Resumes download if the file exists but is incomplete
        - Shows download progress with tqdm progress bar
        - Validates downloaded file size
        - Retries with exponential backoff on connection failures
    """
    make_parent_folder(save_file)
    for attempt in range(max_retries):
        try:
            resume_byte_pos = 0
            if os.path.exists(save_file):
                resume_byte_pos = os.path.getsize(save_file)
            
            response_head = requests.head(url=url)
            total_size = int(response_head.headers.get("content-length", 0))

            if resume_byte_pos >= total_size:
                logger.info("File already downloaded completely.")
                return

            headers = {'Range': f'bytes={resume_byte_pos}-'} if resume_byte_pos else {}
            response = requests.get(url=url, stream=True, headers=headers, timeout=timeout)
            response.raise_for_status()
            # total_size = int(response.headers.get("content-length", 0))
            mode = 'ab' if resume_byte_pos else 'wb'
            progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, initial=resume_byte_pos)
            
            with open(save_file, mode) as file:
                for chunk_data in response.iter_content(chunk_size=1024):
                    if chunk_data:
                        size = file.write(chunk_data)
                        progress_bar.update(size)
            
            progress_bar.close()

            if os.path.getsize(save_file) >= (total_size + resume_byte_pos):
                logger.info("Download completed successfully.")
                break
            else:
                logger.warning("File size mismatch, retrying...")
                time.sleep(5)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f"Download error: {e}. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(5)
        except Exception as e:
            error_message = f"Unexpected error: {e}"
            logger.error(error_message)
            raise ValueError(error_message)
    else:
        error_message = "Exceeded maximum retries. Download failed."
        logger.error(error_message)
        raise RuntimeError(error_message)
