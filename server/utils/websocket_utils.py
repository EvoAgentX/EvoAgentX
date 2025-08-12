"""
Generalized WebSocket utilities for real-time progress tracking and messaging.
Can be reused across different parts of the project.
"""

import asyncio
import json
import os
import queue
import sys
import threading
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Type
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Import loguru for enhanced logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class WebSocketEnhancedSink:
    """
    Enhanced WebSocket sink that captures loguru messages, stdin/stdout, and provides periodic updates.
    Generalized for reuse across different project components.
    """
    
    def __init__(self, websocket_send_func: Callable, context_id: str, context_type: str = "workflow"):
        self.websocket_send_func = websocket_send_func
        self.context_id = context_id
        self.context_type = context_type
        self.message_queue = queue.Queue()
        self.running = True
        self.stdout_buffer = StringIO()
        self.stderr_buffer = StringIO()
        self.last_update_time = time.time()
        self.update_interval = 3.0  # Update every 3 seconds
        
        # Rate limiting for output messages
        self.last_output_time = {"stdout": 0, "stderr": 0}
        self.output_rate_limit = 0.1  # Minimum 100ms between output messages
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Start message processing thread
        self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processing_thread.start()
        
        # Start periodic update thread
        self.update_thread = threading.Thread(target=self._periodic_updates, daemon=True)
        self.update_thread.start()
        
        # Redirect stdout/stderr to capture output
        self._redirect_output()
    
    def _redirect_output(self):
        """Redirect stdout and stderr to capture output."""
        class CapturingStream:
            def __init__(self, original_stream, buffer, sink):
                self.original_stream = original_stream
                self.buffer = buffer
                self.sink = sink
            
            def write(self, text):
                self.original_stream.write(text)
                self.buffer.write(text)
                # Flush to ensure immediate capture
                self.buffer.flush()
                
                # Only send stdout/stderr messages for substantial content to reduce spam
                if text.strip() and len(text.strip()) > 5:
                    self.sink._send_output_message("stdout" if self.original_stream == sys.stdout else "stderr", text)
            
            def flush(self):
                self.original_stream.flush()
                self.buffer.flush()
        
        # Create capturing streams
        self.stdout_capturer = CapturingStream(sys.stdout, self.stdout_buffer, self)
        self.stderr_capturer = CapturingStream(sys.stderr, self.stderr_buffer, self)
        
        # Redirect
        sys.stdout = self.stdout_capturer
        sys.stderr = self.stderr_capturer
    
    def capture_stdin_input(self, input_text: str):
        """Capture stdin input and send via WebSocket."""
        if input_text.strip():
            input_data = {
                "type": "input",
                "content": f"Input received: {input_text.strip()}",
                "result": None
            }
            
            # Send immediately in a thread-safe way
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._send_websocket_message(input_data), 
                        loop
                    )
            except RuntimeError:
                # No event loop available, just print
                print(f"Input message: stdin - {input_text.strip()}")
    
    def _restore_output(self):
        """Restore original stdout and stderr."""
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
        if hasattr(self, 'original_stderr'):
            sys.stderr = self.original_stderr
    
    def _send_output_message(self, output_type: str, content: str):
        """Send stdout/stderr message via WebSocket."""
        if not content.strip():
            return
        
        # Rate limiting - only send if enough time has passed since last message of this type
        current_time = time.time()
        if current_time - self.last_output_time.get(output_type, 0) < self.output_rate_limit:
            return
        
        self.last_output_time[output_type] = current_time
            
        output_data = {
            "type": "output",
            "content": content.strip(),
            "result": None
        }
        
        # Send immediately in a thread-safe way
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._send_websocket_message(output_data), 
                    loop
                )
        except RuntimeError:
            # No event loop available - only log significant stderr messages to avoid spam
            if output_type == "stderr" and len(content.strip()) > 10:
                print(f"Stderr output: {content.strip()}")
            elif output_type == "stdout":
                # Only log stdout if it's substantial content
                if len(content.strip()) > 20:
                    print(f"Stdout output: {content.strip()}")
    
    def write(self, message):
        """Write method called by loguru for each log message."""
        if self.running:
            self.message_queue.put(message)
    
    def _periodic_updates(self):
        """Send periodic status updates every 3 seconds."""
        while self.running:
            try:
                time.sleep(self.update_interval)
                if self.running:
                    # Get current buffer contents
                    stdout_content = self.stdout_buffer.getvalue()
                    stderr_content = self.stderr_buffer.getvalue()
                    
                    # Send periodic update with comprehensive information
                    update_data = {
                        "type": "progress",
                        "content": f"{self.context_type.title()} in progress. Captured {len(stdout_content)} stdout chars, {len(stderr_content)} stderr chars",
                        "result": None
                    }
                    
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self._send_websocket_message(update_data), 
                                loop
                            )
                    except RuntimeError:
                        pass  # No event loop available
                        
            except Exception as e:
                print(f"Error in periodic updates: {e}")
                break
    
    def _process_messages(self):
        """Process queued loguru messages and send via WebSocket."""
        while self.running:
            try:
                # Get message with timeout
                try:
                    message = self.message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if message and self.running:
                    # Parse loguru message
                    parsed_message = self._parse_loguru_message(str(message))
                    if parsed_message:
                        # Send via WebSocket in a thread-safe way
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    self._send_websocket_message(parsed_message), 
                                    loop
                                )
                        except RuntimeError:
                            pass  # No event loop available
                            
            except Exception as e:
                print(f"Error processing messages: {e}")
                break
    
    def _handle_websocket_result(self, future):
        """Handle WebSocket send result."""
        try:
            result = future.result()
            if result:
                print(f"WebSocket message sent successfully: {result}")
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")
    
    def _parse_loguru_message(self, message: str) -> Optional[Dict[str, Any]]:
        """Parse loguru message format and extract relevant information."""
        try:
            # Basic loguru format parsing
            if "|" in message:
                parts = message.split("|")
                if len(parts) >= 3:
                    timestamp = parts[0].strip()
                    level = parts[1].strip()
                    content = "|".join(parts[2:]).strip()
                    
                    # Only send INFO and above to reduce noise
                    if level in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
                        return {
                            "type": "log",
                            "content": content,
                            "result": None
                        }
            return None
        except Exception as e:
            print(f"Error parsing loguru message: {e}")
            return None
    
    async def _send_websocket_message(self, log_data: Dict[str, Any]):
        """Send log data via WebSocket."""
        try:
            if self.websocket_send_func:
                await self.websocket_send_func(json.dumps(log_data))
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")
    
    def get_buffer_contents(self) -> Dict[str, Any]:
        """Get current buffer contents."""
        return {
            "stdout": self.stdout_buffer.getvalue(),
            "stderr": self.stderr_buffer.getvalue(),
            "timestamp": datetime.now().isoformat()
        }
    
    def stop(self):
        """Stop the sink and restore output streams."""
        self.running = False
        self._restore_output()
        
        # Wait for threads to finish
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        if hasattr(self, 'update_thread'):
            self.update_thread.join(timeout=1.0)


class WebSocketProgressTracker:
    """
    Generalized WebSocket progress tracker that can be used for any type of operation.
    Supports the standard message format: {"type": "...", "content": "...", "result": ...}
    """
    
    def __init__(self, websocket_send_func: Callable, context_id: str, context_type: str = "operation"):
        self.websocket_send_func = websocket_send_func
        self.context_id = context_id
        self.context_type = context_type
        self.current_phase = "initializing"
        self.progress = 0.0
        self.execution_id = None
    
    async def send_connection_confirmation(self):
        """Send WebSocket connection confirmation."""
        connection_data = {
            "type": "connection",
            "content": "WebSocket connection established",
            "result": None
        }
        
        try:
            await self.websocket_send_func(json.dumps(connection_data))
        except Exception as e:
            print(f"Error sending connection confirmation: {e}")
    
    async def send_start_notification(self, operation_name: str = None):
        """Send start notification."""
        start_data = {
            "type": "start",
            "content": f"{operation_name or self.context_type.title()} started",
            "result": None
        }
        
        try:
            await self.websocket_send_func(json.dumps(start_data))
        except Exception as e:
            print(f"Error sending start notification: {e}")
    
    async def send_progress_update(self, phase: str, progress: float, message: str = None):
        """Send progress update."""
        self.current_phase = phase
        self.progress = progress
        
        if message:
            content = f"{message} ({progress:.0%} complete)"
        else:
            content = f"{phase.title()} phase: {progress:.0%} complete"
        
        progress_data = {
            "type": "progress",
            "content": content,
            "result": None
        }
        
        try:
            await self.websocket_send_func(json.dumps(progress_data))
        except Exception as e:
            print(f"Error sending progress update: {e}")
    
    async def send_log_message(self, level: str, message: str):
        """Send log message."""
        log_data = {
            "type": "log",
            "content": message,
            "result": None
        }
        
        try:
            await self.websocket_send_func(json.dumps(log_data))
        except Exception as e:
            print(f"Error sending log message: {e}")
    
    async def send_output_message(self, content: str):
        """Send output message."""
        output_data = {
            "type": "output",
            "content": content,
            "result": None
        }
        
        try:
            await self.websocket_send_func(json.dumps(output_data))
        except Exception as e:
            print(f"Error sending output message: {e}")
    
    async def send_completion(self, result: Dict[str, Any], message: str = None):
        """Send completion message."""
        completion_data = {
            "type": "complete",
            "content": message or f"{self.context_type.title()} completed successfully",
            "result": result
        }
        
        try:
            await self.websocket_send_func(json.dumps(completion_data))
        except Exception as e:
            print(f"Error sending completion: {e}")
    
    async def send_error(self, error_message: str):
        """Send error message."""
        error_data = {
            "type": "error",
            "content": error_message,
            "result": None
        }
        
        try:
            await self.websocket_send_func(json.dumps(error_data))
        except Exception as e:
            print(f"Error sending error: {e}")
    
    async def send_phase_update(self, phase: str, progress: float, message: str = None):
        """Send phase-specific update."""
        if message:
            content = f"{phase.title()}: {message} ({progress:.0%} complete)"
        else:
            content = f"{phase.title()} phase: {progress:.0%} complete"
        
        await self.send_progress_update(phase, progress, content)
    
    async def send_status_update(self, status: str, progress: float = None, message: str = None):
        """Send status update."""
        if progress is not None:
            await self.send_progress_update(self.current_phase, progress, message)
        else:
            await self.send_log_message("INFO", f"Status: {status}")


# Convenience functions for common operations
async def send_simple_message(websocket_send_func: Callable, message_type: str, content: str, result: Any = None):
    """Send a simple websocket message."""
    message_data = {
        "type": message_type,
        "content": content,
        "result": result
    }
    
    try:
        await websocket_send_func(json.dumps(message_data))
    except Exception as e:
        print(f"Error sending simple message: {e}")


async def send_progress_message(websocket_send_func: Callable, phase: str, progress: float, message: str = None):
    """Send a progress message."""
    if message:
        content = f"{message} ({progress:.0%} complete)"
    else:
        content = f"{phase.title()}: {progress:.0%} complete"
    
    await send_simple_message(websocket_send_func, "progress", content)


async def send_log_message(websocket_send_func: Callable, level: str, message: str):
    """Send a log message."""
    await send_simple_message(websocket_send_func, "log", message)


async def send_error_message(websocket_send_func: Callable, error_message: str):
    """Send an error message."""
    await send_simple_message(websocket_send_func, "error", error_message)