"""
Workflow-specific logging system for the new socket architecture.
Creates separate loggers for each workflow generation and execution process.
"""

import asyncio
import json
import logging
import sys
import uuid
import time
from datetime import datetime
from typing import Callable, Optional, Dict, Any, List
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr, contextmanager

from loguru import logger as global_logger


class IsolatedWorkflowLogger:
    """
    Creates completely isolated logging for workflow generation or execution.
    Ensures logs from concurrent workflows/executions don't get mixed.
    Captures ALL logs from the workflow engine and marks them with workflow ID.
    """
    
    def __init__(self, workflow_id: str, process_type: str = "generation", process_id: Optional[str] = None):
        self.workflow_id = workflow_id
        self.process_type = process_type  # "generation" or "execution"
        self.process_id = process_id or f"{process_type}_{workflow_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.sink_id = None
        self.global_sink_id = None  # For capturing ALL logs during workflow process
        self.bound_logger = None
        self.websocket_send_func = None
        self.is_active = True
        self.captured_logs = []  # Store all captured logs
    
    def setup_isolated_logging(self, websocket_send_func: Optional[Callable] = None):
        """Setup completely isolated logging for this workflow process."""
        self.websocket_send_func = websocket_send_func
        

        
        # Create bound logger with unique process context
        self.bound_logger = global_logger.bind(
            workflow_id=self.workflow_id,
            process_type=self.process_type,
            process_id=self.process_id
        )
        
        # Create filtered sink that ONLY captures logs from this specific process
        def isolated_sink(message):
            if not self.is_active:
                return
                
            record = message.record
            extra = record.get("extra", {})
            
            # Only process logs from this exact workflow process
            if (extra.get("workflow_id") == self.workflow_id and 
                extra.get("process_type") == self.process_type and
                extra.get("process_id") == self.process_id):
                
                # Extract clean message content
                log_content = str(message).strip()
                if log_content and self.websocket_send_func:
                    # Store the log for later sending - we'll send it when the async context is available
                    self.captured_logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "workflow_id": self.workflow_id,
                        "process_id": self.process_id,
                        "content": log_content,
                        "level": "INFO",
                        "source": "isolated_logger",
                        "pending": True
                    })
        
        # Add isolated sink with strict filtering for bound logs
        try:
            self.sink_id = global_logger.add(
                isolated_sink,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                filter=lambda record: (
                    record.get("extra", {}).get("workflow_id") == self.workflow_id and
                    record.get("extra", {}).get("process_type") == self.process_type and
                    record.get("extra", {}).get("process_id") == self.process_id
                )
            )
            
            # Add global sink to capture ALL logs during workflow process and mark them with workflow ID
            def global_workflow_sink(message):
                if not self.is_active:
                    return
                    
                record = message.record
                extra = record.get("extra", {})
                
                # Skip logs that are already bound to a workflow (to avoid duplicates)
                if extra.get("workflow_id"):
                    return
                
                # Capture ALL unbound logs and attribute them to our workflow ID
                log_content = str(message).strip()
                if log_content and self.websocket_send_func:
                    # Store the captured log with workflow attribution
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "workflow_id": self.workflow_id,  # System knows this log belongs to this workflow
                        "process_id": self.process_id,
                        "content": log_content,  # Original content unchanged
                        "level": record["level"].name,
                        "source": "workflow_engine",
                        "attributed": True  # Flag indicating this was attributed to workflow
                    }
                    self.captured_logs.append(log_entry)
                    
                    print(f"   📤 Sending captured log: {repr(log_content)}")
                    print(f"   📊 Total captured logs: {len(self.captured_logs)}")
                    
                    # Store the captured log for later sending
                    self.captured_logs.append(log_entry)
                else:
                    print(f"   ⏭️  Skipping - no content or websocket function")
            
            self.global_sink_id = global_logger.add(
                global_workflow_sink,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                # No filter - capture ALL logs, we filter in the sink function
            )
            
            # Log the isolation setup
            self.bound_logger.info(f"✅ Isolated logging setup for {self.process_type} process: {self.process_id}")
            self.bound_logger.info(f"🔍 Global log capture enabled - all workflow engine logs will be marked with workflow_id: {self.workflow_id}")
            
        except Exception as e:
            print(f"Warning: Could not setup isolated logging for {self.workflow_id}: {e}")
        
        return self.bound_logger
    
    async def _send_isolated_log(self, log_content: str):
        """Send isolated log message via WebSocket with workflow attribution."""
        try:
            from ..socket_management.protocols import create_message, MessageType
            if self.process_type == "generation":
                log_message = create_message(
                    MessageType.SETUP_LOG,
                    status=None,
                    workflow_id=self.workflow_id,
                    content=log_content,  # Keep original log content unchanged
                    result=None
                )
            else:  # execution
                log_message = create_message(
                    MessageType.RUNTIME_LOG,
                    status=None,
                    workflow_id=self.workflow_id,
                    content=log_content,  # Keep original log content unchanged
                    result=None
                )
            
            if self.websocket_send_func is not None:
                # Check if the WebSocket connection is still healthy before sending
                try:
                    # Send the message object directly (not as JSON string)
                    await self.websocket_send_func(log_message)
                except Exception as send_error:
                    # Mark the websocket function as unavailable to prevent further failures
                    self.websocket_send_func = None
            
        except Exception as e:
            pass
    
    async def _send_captured_log(self, log_content: str):
        """Send captured workflow engine log via WebSocket with workflow attribution but unchanged content."""
        try:
            from ..socket_management.protocols import create_message, MessageType
            if self.process_type == "generation":
                log_message = create_message(
                    MessageType.SETUP_LOG,
                    status=None,
                    workflow_id=self.workflow_id,
                    content=log_content,  # Keep original engine log content unchanged
                    result=None
                )
            else:  # execution
                log_message = create_message(
                    MessageType.RUNTIME_LOG,
                    status=None,
                    workflow_id=self.workflow_id,
                    content=log_content,  # Keep original engine log content unchanged
                    result=None
                )
            
            if self.websocket_send_func is not None:
                # Check if the WebSocket connection is still healthy before sending
                try:
                    # Send the message object directly (not as JSON string)
                    await self.websocket_send_func(log_message)
                except Exception as send_error:
                    # Mark the websocket function as unavailable to prevent further failures
                    self.websocket_send_func = None
            
        except Exception as e:
            pass
    
    def get_captured_logs(self) -> List[Dict[str, Any]]:
        """Get all logs captured from the workflow engine."""
        return self.captured_logs.copy()
    
    async def flush_pending_logs(self):
        """Flush all pending logs to the WebSocket."""
        if not self.websocket_send_func:
            print(f"⚠️  No WebSocket function available for flushing logs")
            return
        
        pending_logs = [log for log in self.captured_logs if log.get("pending", False)]
        if not pending_logs:
            print(f"📭 No pending logs to flush for {self.workflow_id}")
            return
        
        print(f"📤 Flushing {len(pending_logs)} pending logs for {self.workflow_id}")
        
        for log_entry in pending_logs:
            try:
                if log_entry["source"] == "isolated_logger":
                    await self._send_isolated_log(log_entry["content"])
                else:
                    await self._send_captured_log(log_entry["content"])
                
                # Mark as sent
                log_entry["pending"] = False
                print(f"   ✅ Flushed log: {log_entry['content'][:50]}...")
                
            except Exception as e:
                print(f"   ❌ Error flushing log: {e}")
                # Keep it as pending for retry
        
        print(f"📊 Flushed {len([log for log in pending_logs if not log.get('pending', False)])} logs successfully")
    
    def cleanup(self):
        """Cleanup isolated logging resources."""
        if self.bound_logger and self.is_active:
            self.bound_logger.info(f"🧹 Cleaning up isolated logging for {self.process_type} process: {self.process_id}")
        
        self.is_active = False
        
        if self.sink_id:
            try:
                global_logger.remove(self.sink_id)
            except Exception as e:
                print(f"Warning: Could not remove isolated sink {self.sink_id}: {e}")
        
        if self.global_sink_id:
            try:
                global_logger.remove(self.global_sink_id)
            except Exception as e:
                print(f"Warning: Could not remove global workflow sink {self.global_sink_id}: {e}")
        
        self.sink_id = None
        self.global_sink_id = None
        self.bound_logger = None
        self.websocket_send_func = None
        
        # Log final statistics
        if self.captured_logs:
            print(f"📊 Captured {len(self.captured_logs)} workflow engine logs for {self.workflow_id}")
        
        self.captured_logs.clear()


@contextmanager
def isolated_workflow_process(workflow_id: str, process_type: str, websocket_send_func: Optional[Callable] = None):
    """
    Context manager for isolated workflow process logging.
    
    Args:
        workflow_id: The workflow ID
        process_type: "generation" or "execution"  
        websocket_send_func: Optional WebSocket send function
    
    Usage:
        # For workflow generation
        with isolated_workflow_process(workflow_id, "generation", websocket_func) as logger:
            logger.info("This log will only go to this workflow's generation logs")
        
        # For workflow execution  
        with isolated_workflow_process(workflow_id, "execution", websocket_func) as logger:
            logger.info("This log will only go to this workflow's execution logs")
    """
    isolated_logger = IsolatedWorkflowLogger(workflow_id, process_type)
    bound_logger = isolated_logger.setup_isolated_logging(websocket_send_func)
    
    try:
        yield bound_logger, isolated_logger.process_id
    finally:
        # Flush all pending logs before cleanup
        if websocket_send_func:
            try:
                # Create a simple async function to flush logs
                async def flush_logs():
                    await isolated_logger.flush_pending_logs()
                
                # Try to run the flush in the current event loop
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create a task to flush logs
                        asyncio.create_task(flush_logs())
                    else:
                        # Run the coroutine directly
                        loop.run_until_complete(flush_logs())
                except Exception as e:
                    print(f"⚠️  Could not flush pending logs: {e}")
            except Exception as e:
                print(f"⚠️  Error during log flushing: {e}")
        
        isolated_logger.cleanup()


class WorkflowLogCapture:
    """Captures logs for a specific workflow and forwards them via WebSocket."""
    
    def __init__(self, workflow_id: str, websocket_send_func: Optional[Callable] = None, log_type: str = "setup"):
        self.workflow_id = workflow_id
        self.websocket_send_func = websocket_send_func
        self.log_type = log_type  # "setup" or "runtime"
        self.log_buffer: List[Dict[str, Any]] = []
        self.sink_id = None
        self.is_active = True
        
        # Setup loguru sink for this workflow
        self._setup_loguru_sink()
    
    def _setup_loguru_sink(self):
        """Setup a custom loguru sink for this workflow."""
        try:
            def workflow_sink(message):
                if self.is_active:
                    asyncio.create_task(self._process_log_message(message))
            
            self.sink_id = global_logger.add(
                workflow_sink,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                filter=lambda record: record.get("extra", {}).get("workflow_id") == self.workflow_id
            )
        except Exception as e:
            print(f"Warning: Could not setup loguru sink for workflow {self.workflow_id}: {e}")
    
    async def _process_log_message(self, message_record):
        """Process and forward log message via WebSocket."""
        try:
            # Extract message content
            log_content = str(message_record).strip()
            
            if not log_content:
                return
            
            # Store in buffer
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "workflow_id": self.workflow_id,
                "content": log_content,
                "level": "INFO"  # Default level
            }
            self.log_buffer.append(log_entry)
            
            # Send via WebSocket if available
            if self.websocket_send_func:
                message_type = "setup-log" if self.log_type == "setup" else "runtime-log"
                websocket_message = {
                    "type": message_type,
                    "data": {
                        "workflow_id": self.workflow_id,
                        "content": log_content,
                        "result": None
                    }
                }
                try:
                    await self.websocket_send_func(json.dumps(websocket_message))
                except Exception as send_error:
                    print(f"WebSocket send failed for workflow {self.workflow_id}: {send_error}")
                    # Mark the websocket function as unavailable to prevent further failures
                    self.websocket_send_func = None
                
        except Exception as e:
            print(f"Error processing log message for workflow {self.workflow_id}: {e}")
    
    async def log_message(self, level: str, message: str):
        """Manually log a message for this workflow."""
        try:
            # Add workflow_id to logger context
            workflow_logger = global_logger.bind(workflow_id=self.workflow_id)
            
            if level.upper() == "ERROR":
                workflow_logger.error(message)
            elif level.upper() == "WARNING":
                workflow_logger.warning(message)
            elif level.upper() == "DEBUG":
                workflow_logger.debug(message)
            else:
                workflow_logger.info(message)
                
        except Exception as e:
            print(f"Error logging message for workflow {self.workflow_id}: {e}")
    
    async def send_status_update(self, status: str, content: str, result: Any = None):
        """Send a workflow status update via WebSocket."""
        if self.websocket_send_func:
            try:
                message_type = "setup-log" if self.log_type == "setup" else "runtime-log"
                websocket_message = {
                    "type": message_type,
                    "data": {
                        "workflow_id": self.workflow_id,
                        "content": f"{self.workflow_id} updates database status to: {status}",
                        "result": result
                    }
                }
                await self.websocket_send_func(json.dumps(websocket_message))
            except Exception as e:
                print(f"Error sending status update for workflow {self.workflow_id}: {e}")
    
    def get_log_buffer(self) -> List[Dict[str, Any]]:
        """Get all captured logs for this workflow."""
        return self.log_buffer.copy()
    
    def cleanup(self):
        """Cleanup resources and remove loguru sink."""
        self.is_active = False
        if self.sink_id:
            try:
                global_logger.remove(self.sink_id)
            except Exception as e:
                print(f"Warning: Could not remove loguru sink for workflow {self.workflow_id}: {e}")


class WorkflowLoggerManager:
    """Manages workflow-specific loggers across the system."""
    
    def __init__(self):
        self.active_loggers: Dict[str, WorkflowLogCapture] = {}
    
    def create_workflow_logger(
        self, 
        workflow_id: str, 
        websocket_send_func: Optional[Callable] = None,
        log_type: str = "setup"
    ) -> WorkflowLogCapture:
        """Create a new workflow logger."""
        # Cleanup existing logger if exists
        if workflow_id in self.active_loggers:
            self.active_loggers[workflow_id].cleanup()
        
        # Create new logger
        logger = WorkflowLogCapture(workflow_id, websocket_send_func, log_type)
        self.active_loggers[workflow_id] = logger
        
        return logger
    
    def get_workflow_logger(self, workflow_id: str) -> Optional[WorkflowLogCapture]:
        """Get existing workflow logger."""
        return self.active_loggers.get(workflow_id)
    
    def cleanup_workflow_logger(self, workflow_id: str):
        """Cleanup and remove workflow logger."""
        if workflow_id in self.active_loggers:
            self.active_loggers[workflow_id].cleanup()
            del self.active_loggers[workflow_id]
    
    def cleanup_all(self):
        """Cleanup all workflow loggers."""
        for logger in self.active_loggers.values():
            logger.cleanup()
        self.active_loggers.clear()


# Global logger manager instance
workflow_logger_manager = WorkflowLoggerManager()


def create_workflow_logger(
    workflow_id: str, 
    websocket_send_func: Optional[Callable] = None,
    log_type: str = "setup"
) -> WorkflowLogCapture:
    """Convenience function to create a workflow logger."""
    return workflow_logger_manager.create_workflow_logger(workflow_id, websocket_send_func, log_type)


def get_workflow_logger(workflow_id: str) -> Optional[WorkflowLogCapture]:
    """Convenience function to get a workflow logger."""
    return workflow_logger_manager.get_workflow_logger(workflow_id)


def cleanup_workflow_logger(workflow_id: str):
    """Convenience function to cleanup a workflow logger."""
    workflow_logger_manager.cleanup_workflow_logger(workflow_id)
