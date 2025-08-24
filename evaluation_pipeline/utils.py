"""
Utility functions for evaluation pipeline
"""
import json
import os
import asyncio
import time
import traceback
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from functools import wraps

import psutil


def get_optimal_process_count(memory_per_process_mb: int = 512) -> int:
    """
    Calculate optimal number of processes based on system resources
    
    Args:
        memory_per_process_mb: Estimated memory usage per process in MB
        
    Returns:
        Optimal number of processes
    """
    cpu_count = psutil.cpu_count()
    memory_available = psutil.virtual_memory().available
    memory_based_limit = memory_available // (memory_per_process_mb * 1024 * 1024)
    
    # Use conservative estimate: 75% of CPU cores or memory limit, whichever is smaller
    return min(int(cpu_count * 0.75), memory_based_limit, cpu_count)


def retry_with_backoff(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator for retrying function calls with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by for each retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    print(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    print(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            raise last_exception
        return wrapper
    return decorator


def async_retry_with_backoff(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Async decorator for retrying function calls with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by for each retry
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    print(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    print(f"Retrying in {current_delay} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
            
            raise last_exception
        return async_wrapper
    return decorator


def capture_exception_details(exception: Exception) -> Dict[str, Any]:
    """
    Capture detailed exception information for analysis
    
    Args:
        exception: The exception to capture
        
    Returns:
        Dictionary containing exception details
    """
    return {
        'type': type(exception).__name__,
        'module': type(exception).__module__,
        'message': str(exception),
        'traceback': traceback.format_exc(),
        'timestamp': datetime.now().isoformat(),
        'is_expected': _is_expected_exception(exception)
    }


def _is_expected_exception(exception: Exception) -> bool:
    """
    Determine if an exception is expected/known type
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception is expected, False otherwise
    """
    expected_types = (
        ValueError,
        KeyError,
        AttributeError,
        TypeError,
        # Add more expected exception types as needed
    )
    
    return isinstance(exception, expected_types)


def load_test_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all test data files from directory
    
    Args:
        data_dir: Directory containing test data JSON files
        
    Returns:
        List of test data dictionaries
    """
    test_data = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['source_file'] = filename
                    data['test_id'] = generate_test_id(data)
                    test_data.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return test_data


def generate_test_id(test_data: Dict[str, Any]) -> str:
    """
    Generate unique test ID based on test data
    
    Args:
        test_data: Test data dictionary
        
    Returns:
        Unique test ID string
    """
    # Use workflow_id if available, otherwise generate from content
    if 'workflow_id' in test_data:
        return test_data['workflow_id']
    
    # Generate hash from workflow content
    content = json.dumps(test_data, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def save_checkpoint(checkpoint_dir: str, layer: int, data: Dict[str, Any]) -> str:
    """
    Save checkpoint data for a specific layer
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        layer: Layer number (1, 2, or 3)
        data: Data to save
        
    Returns:
        Path to saved checkpoint file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"layer_{layer}_checkpoint_{timestamp}.json"
    filepath = os.path.join(checkpoint_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def load_checkpoint(checkpoint_dir: str, layer: int) -> Optional[Dict[str, Any]]:
    """
    Load latest checkpoint for a specific layer
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        layer: Layer number (1, 2, or 3)
        
    Returns:
        Checkpoint data if found, None otherwise
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find latest checkpoint file for this layer
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) 
        if f.startswith(f"layer_{layer}_checkpoint_") and f.endswith('.json')
    ]
    
    if not checkpoint_files:
        return None
    
    # Sort by filename (which includes timestamp)
    latest_file = sorted(checkpoint_files)[-1]
    filepath = os.path.join(checkpoint_dir, latest_file)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading checkpoint {filepath}: {e}")
        return None


def create_batch_iterator(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split items into batches
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def merge_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple result dictionaries
    
    Args:
        results_list: List of result dictionaries to merge
        
    Returns:
        Merged results dictionary
    """
    merged = {
        'test_results': [],
        'summary': {
            'total_tests': 0,
            'successful': 0,
            'failed': 0,
            'execution_time': 0.0
        },
        'errors': []
    }
    
    for results in results_list:
        if 'test_results' in results:
            merged['test_results'].extend(results['test_results'])
        if 'errors' in results:
            merged['errors'].extend(results['errors'])
        if 'summary' in results:
            summary = results['summary']
            merged['summary']['total_tests'] += summary.get('total_tests', 0)
            merged['summary']['successful'] += summary.get('successful', 0)
            merged['summary']['failed'] += summary.get('failed', 0)
            merged['summary']['execution_time'] += summary.get('execution_time', 0.0)
    
    return merged
