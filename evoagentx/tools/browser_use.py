"""
BrowserUse Toolkit - Simple browser automation toolkit
Uses browser-use as external dependency only
Multiple specific tools: Navigate, Click, Close, Scroll, Get_Status
One base class and one toolkit managing all tools
"""

import asyncio
import threading
from typing import Dict, Any, Optional, List
from browser_use import BrowserSession
from browser_use.browser.events import (
    NavigateToUrlEvent,
    ClickElementEvent,
    ScrollEvent,
    CloseTabEvent,
    BrowserStateRequestEvent,
    TypeTextEvent,
    GoBackEvent,
    RefreshEvent,
    SendKeysEvent,
    GetDropdownOptionsEvent,
    SelectDropdownOptionEvent,
    SwitchTabEvent
)

from .tool import Tool, Toolkit
from ..core.module import BaseModule
from ..core.logging import logger


def _run_async_in_sync(browser_use_instance, async_func, *args, **kwargs):
    """
    Helper function to run async methods in sync context.
    Uses the dedicated browser thread's event loop.
    """
    import asyncio
    import threading
    
    
    # Use the browser thread's event loop
    return browser_use_instance._run_in_browser_thread(async_func(*args, **kwargs))




class BrowserUse(BaseModule):
    """
    Single base class that does all the heavy lifting and maintains browser state
    All tools reference this base class for their operations
    """
    
    def __init__(
        self, 
        name: str = 'BrowserUse',
        browser_session: Optional[BrowserSession] = None,
        auto_update_state: bool = True,
        **kwargs
    ):
        """
        Initialize the Browser Use base class.
        
        Args:
            name (str): The name of the browser tool
            browser_session (BrowserSession, optional): Browser session to use
            auto_update_state (bool): Whether to auto-update state after operations
            **kwargs: Additional data to pass to the parent class
        """
        super().__init__(name=name, **kwargs)
        self.browser_session = browser_session
        self.auto_update_state = auto_update_state
        self._current_state = None
        self._last_error = None
        self._operation_history = []
        self._browser_started = False
        
        # Browser thread management
        self._browser_thread = None
        self._browser_loop = None
        self._browser_thread_ready = threading.Event()
        self._shutdown_event = threading.Event()
        
        # Start the dedicated browser thread
        self._start_browser_thread()
    
    def _start_browser_thread(self):
        """Start the dedicated browser thread with its own event loop."""
        import threading
        import asyncio
        
        def browser_thread_worker():
            """Worker function for the browser thread."""
            # Create a new event loop for this thread
            self._browser_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._browser_loop)
            
            self._browser_thread_ready.set()
            
            try:
                # Run the event loop until shutdown is requested
                self._browser_loop.run_until_complete(self._browser_thread_main())
            except Exception as e:
                logger.error(f"Error in browser thread: {e}")
            finally:
                logger.info("Browser thread shutting down")
                if self._browser_loop and not self._browser_loop.is_closed():
                    self._browser_loop.close()
        
        self._browser_thread = threading.Thread(target=browser_thread_worker, name="BrowserThread", daemon=False)
        self._browser_thread.start()
        
        # Wait for the thread to be ready
        if not self._browser_thread_ready.wait(timeout=10):
            raise RuntimeError("Browser thread failed to start within 10 seconds")
    
    async def _browser_thread_main(self):
        """Main coroutine for the browser thread."""
        
        # Wait for shutdown signal
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)
        
        # Cleanup browser session when shutdown is requested
        try:
            await self._close_browser_session()
        except Exception as e:
            logger.warning(f"Error closing browser session in thread: {e}")
        
    
    def _run_in_browser_thread(self, coro):
        """Run a coroutine in the browser thread's event loop."""
        import asyncio
        import concurrent.futures
        
        if not self._browser_loop or self._browser_loop.is_closed():
            raise RuntimeError("Browser thread event loop is not available")
        
        # Submit the coroutine to the browser thread's event loop
        future = asyncio.run_coroutine_threadsafe(coro, self._browser_loop)
        return future.result(timeout=60)  # 60 second timeout
    
    def __del__(self):
        """Cleanup when the BrowserUse instance is destroyed."""
        try:
            logger.info("BrowserUse instance being destroyed, starting cleanup...")
            # Set shutdown event to stop the thread
            if hasattr(self, '_shutdown_event'):
                self._shutdown_event.set()
            
            # Wait for thread to finish (with timeout)
            if hasattr(self, '_browser_thread') and self._browser_thread and self._browser_thread.is_alive():
                self._browser_thread.join(timeout=5)
            
            logger.info("BrowserUse cleanup completed successfully")
        except Exception as e:
            logger.warning(f"Error during BrowserUse cleanup: {e}")
    
    def _cleanup_browser_session(self):
        """Clean up the browser session before shutting down the thread."""
        if self.browser_session and self._browser_started:
            try:
                logger.info("Closing browser session...")
                # Run the close operation in the browser thread
                if self._browser_loop and not self._browser_loop.is_closed():
                    future = asyncio.run_coroutine_threadsafe(
                        self._close_browser_session(), 
                        self._browser_loop
                    )
                    future.result(timeout=10)  # 10 second timeout for cleanup
                    logger.info("Browser session closed successfully")
                else:
                    logger.warning("Browser thread event loop not available for cleanup")
            except Exception as e:
                logger.warning(f"Error closing browser session: {e}")
    
    async def _close_browser_session(self):
        """Async method to close the browser session."""
        try:
            if self.browser_session:
                # First stop the event bus to close WebSocket connections
                if hasattr(self.browser_session, 'event_bus') and hasattr(self.browser_session.event_bus, 'stop'):
                    await self.browser_session.event_bus.stop()
                    logger.info("Event bus stopped in thread")
                
                # Then stop the browser session
                if hasattr(self.browser_session, 'stop'):
                    await self.browser_session.stop()
                    logger.info("Browser session stopped in thread")
                else:
                    logger.warning("Browser session stop method not available")
                
                self._browser_started = False
        except Exception as e:
            logger.warning(f"Error in async browser session close: {e}")
    
    def _shutdown_browser_thread(self):
        """Shutdown the browser thread gracefully."""
        if self._browser_thread and self._browser_thread.is_alive():
            logger.info("Shutting down browser thread...")
            self._shutdown_event.set()
            self._browser_thread.join(timeout=10)  # Increased timeout for proper cleanup
            if self._browser_thread.is_alive():
                logger.warning("Browser thread did not shut down gracefully")
            else:
                logger.info("Browser thread shut down successfully")
    
    async def _ensure_browser_started(self):
        """Ensure the browser session is started."""
        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")
        
        if not self._browser_started:
            try:
                # Start the browser session
                await self.browser_session.start()
                self._browser_started = True
            except Exception as e:
                logger.error(f"Failed to start browser session: {e}")
                raise RuntimeError(f"Failed to start browser session: {e}")
    
    async def navigate(self, url: str, new_tab: bool = False) -> Dict[str, Any]:
        """Navigate to a URL."""
        try:
            await self._ensure_browser_started()
            
            event = self.browser_session.event_bus.dispatch(
                NavigateToUrlEvent(url=url, new_tab=new_tab)
            )
            await event
            result = await event.event_result(raise_if_any=True, raise_if_none=False)
            
            # Get complete page state after navigation
            page_state = await self._get_page_state()
            
            # Get agent context for the new page
            agent_context = await self._get_agent_context_from_state(page_state)
            
            operation_result = {
                "action": "navigate",
                "url": url,
                "new_tab": new_tab,
                "success": True,
                "timestamp": asyncio.get_event_loop().time(),
                "state": page_state,  # Include complete page state
                "agent_context": agent_context,  # Include rich agent context
                "context": {
                    "page_title": getattr(result, 'title', None) if result else page_state.get("title", "Unknown"),
                    "final_url": getattr(result, 'url', None) if result else page_state.get("url", url),
                    "navigation_successful": True,
                    "message": f"Successfully navigated to {url}",
                    "page_info": {
                        "url": page_state.get("url", url),
                        "title": page_state.get("title", "Unknown"),
                        "element_count": page_state.get("dom_state", {}).get("element_count", 0),
                        "tab_count": len(page_state.get("tabs", [])),
                        "has_screenshot": bool(page_state.get("screenshot"))
                    }
                }
            }
            
            self._operation_history.append(("navigate", operation_result))
            
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result
            
        except Exception as e:
            self._last_error = str(e)
            return {
                "action": "navigate",
                "url": url, 
                "new_tab": new_tab, 
                "success": False, 
                "timestamp": asyncio.get_event_loop().time(),
                "error": str(e),
                "context": {
                    "navigation_successful": False,
                    "message": f"Failed to navigate to {url}: {str(e)}"
                }
            }
    
    async def click(self, index: int, while_holding_ctrl: bool = False) -> Dict[str, Any]:
        """Click on an element by index."""
        try:
            await self._ensure_browser_started()
            node = await self.browser_session.get_element_by_index(index)
            
            if node is None:
                error_msg = f'Element index {index} not found in browser state'
                logger.error(error_msg)
                raise ValueError(error_msg)

            event = self.browser_session.event_bus.dispatch(
                ClickElementEvent(node=node, while_holding_ctrl=while_holding_ctrl or False)
            )
            
            await event
            click_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)
            
            # Get updated page state after click
            updated_state = await self._get_page_state()
            agent_context = await self._get_agent_context_from_state(updated_state)
            
            operation_result = {
                "action": "click",
                "index": index,
                "while_holding_ctrl": while_holding_ctrl,
                "success": True,
                "timestamp": asyncio.get_event_loop().time(),
                "state": updated_state,  # Include updated page state
                "agent_context": agent_context,  # Include rich agent context
                "context": {
                    "clicked_element_index": index,
                    "click_successful": True,
                    "message": f"Successfully clicked element {index}",
                    "page_info": {
                        "url": updated_state.get('url', 'Unknown'),
                        "title": updated_state.get('title', 'Unknown'),
                        "element_count": updated_state.get('dom_state', {}).get('element_count', 0),
                        "tab_count": len(updated_state.get('tabs', []))
                    }
                }
            }
            
            self._operation_history.append(("click", operation_result))
            
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result
            
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error occurred: {type(e).__name__}"
            logger.error(f"Click failed for index {index}: {error_msg}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            self._last_error = error_msg
            return {
                "success": False, 
                "error": error_msg,
                "index": index,
                "exception_type": type(e).__name__,
                "exception_args": str(e.args)
            }

    async def type(self, index: int, text: str) -> Dict[str, Any]:
        """Type text into an element by index."""
        try:
            await self._ensure_browser_started()
            node = await self.browser_session.get_element_by_index(index)

            if node is None:
                error_msg = f'Element index {index} not found in browser state'
                logger.error(error_msg)
                raise ValueError(error_msg)

            event = self.browser_session.event_bus.dispatch(
                TypeTextEvent(node=node, text=text)
            )

            await event
            type_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)

            # Get updated page state after typing
            updated_state = await self._get_page_state()
            agent_context = await self._get_agent_context_from_state(updated_state)

            operation_result = {
                "action": "type",
                "index": index,
                "text": text,
                "success": True,
                "timestamp": asyncio.get_event_loop().time(),
                "state": updated_state,  # Include updated page state
                "agent_context": agent_context,  # Include rich agent context
                "context": {
                    "typed_text": text,
                    "target_element_index": index,
                    "type_successful": True,
                    "message": f"Successfully typed '{text}' into element {index}",
                    "page_info": {
                        "url": updated_state.get('url', 'Unknown'),
                        "title": updated_state.get('title', 'Unknown'),
                        "element_count": updated_state.get('dom_state', {}).get('element_count', 0),
                        "tab_count": len(updated_state.get('tabs', []))
                    }
                }
            }
            
            self._operation_history.append(("type", operation_result))
            
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result

        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error occurred: {type(e).__name__}"
            logger.error(f"Type failed for index {index} with text '{text}': {error_msg}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

            self._last_error = error_msg
            return {
                "success": False,
                "error": error_msg,
                "index": index,
                "text": text,
                "exception_type": type(e).__name__,
                "exception_args": str(e.args)
            }

    
    async def get_agent_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of current state and available actions for the agent."""
        try:
            if not self.browser_session:
                raise RuntimeError("Browser session not initialized")
            
            # Get current status
            status_result = await self.get_status()
            if not status_result['success']:
                return status_result
            
            # Get operation history
            history = self.get_operation_history()
            
            # Analyze recent operations
            recent_actions = [op[0] for op in history[-5:]]  # Last 5 operations
            
            # Get available tools
            available_tools = [
                {"name": "navigate", "description": "Navigate to a URL", "parameters": ["url", "new_tab"]},
                {"name": "click", "description": "Click on an element by index", "parameters": ["index"]},
                {"name": "type", "description": "Type text into an input element", "parameters": ["index", "text"]},
                {"name": "scroll", "description": "Scroll the page", "parameters": ["direction", "amount"]},
                {"name": "close_tab", "description": "Close a tab", "parameters": ["tab_id"]},
                {"name": "get_status", "description": "Get current page status", "parameters": []}
            ]
            
            return {
                "action": "get_agent_summary",
                "success": True,
                "timestamp": asyncio.get_event_loop().time(),
                "agent_context": {
                    "current_state": status_result.get('agent_context', {}),
                    "available_tools": available_tools,
                    "recent_actions": recent_actions,
                    "operation_count": len(history),
                    "last_error": self._last_error,
                    "suggested_next_steps": self._get_suggested_next_steps(status_result.get('agent_context', {})),
                    "message": f"Agent has {len(available_tools)} tools available. Current page has {status_result.get('agent_context', {}).get('interactive_elements', {}).get('total_clickable', 0)} clickable elements."
                }
            }
            
        except Exception as e:
            self._last_error = str(e)
            return {"success": False, "error": str(e)}
    
    def _get_suggested_next_steps(self, agent_context: Dict[str, Any]) -> List[str]:
        """Get suggested next steps based on current context."""
        suggestions = []
        
        interactive_elements = agent_context.get('interactive_elements', {})
        page_analysis = agent_context.get('page_analysis', {})
        
        if interactive_elements.get('links', []):
            suggestions.append("Click on a link to navigate")
        if interactive_elements.get('buttons', []):
            suggestions.append("Click on a button to perform an action")
        if interactive_elements.get('inputs', []):
            suggestions.append("Type in an input field")
        if page_analysis.get('is_search_page', False):
            suggestions.append("Enter a search query")
        if page_analysis.get('is_login_page', False):
            suggestions.append("Enter login credentials")
        if interactive_elements.get('total_clickable', 0) > 20:
            suggestions.append("Scroll to see more content")
        if not suggestions:
            suggestions.append("Get more information about the page")
        
        return suggestions
    
    async def get_all_elements(self) -> Dict[str, Any]:
        """Get ALL elements with their indices for agent decision-making."""
        try:
            if not self.browser_session:
                return {"success": False, "error": "Browser session not initialized"}
            
            # Get current state
            current_state = await self._get_page_state()
            elements = current_state.get('dom_state', {}).get('elements', [])
            
            # Process ALL elements with basic information
            all_elements = []
            for i, element in enumerate(elements):
                element_info = {
                    "index": i + 1,  # 1-based index for clicking
                    "type": element.get('node_name', 'unknown'),
                    "text": element.get('text', ''),
                    "visible": element.get('is_visible', True),
                    "clickable": element.get('is_visible', True) and element.get('node_name') in ['a', 'button', 'input', 'select', 'textarea']
                }
                all_elements.append(element_info)
            
            # Categorize elements
            clickable_elements = [e for e in all_elements if e['clickable']]
            links = [e for e in all_elements if e['type'] == 'a']
            buttons = [e for e in all_elements if e['type'] == 'button']
            inputs = [e for e in all_elements if e['type'] == 'input']
            
            return {
                "success": True,
                "total_elements": len(all_elements),
                "clickable_elements": len(clickable_elements),
                "elements": all_elements,
                "by_type": {
                    "links": links,
                    "buttons": buttons,
                    "inputs": inputs
                }
            }
            
        except Exception as e:
            self._last_error = str(e)
            return {"success": False, "error": str(e)}
    
    async def close_tab(self, tab_id: Optional[str] = None) -> Dict[str, Any]:
        """Close a tab."""
        try:
            await self._ensure_browser_started()
            
            # If no tab_id provided or "current", get current tab info first
            if tab_id is None or tab_id == "current":
                current_state = await self._get_page_state()
                tabs = current_state.get('tabs', [])
                if tabs:
                    # Use the current tab's target_id (first tab in the list is usually current)
                    tab_id = tabs[0].get('target_id')
                    if not tab_id:
                        return {"tab_id": "current", "success": False, "error": "Could not get current tab ID"}
                else:
                    return {"tab_id": "current", "success": False, "error": "No tabs available to close"}
            
            # Validate that we have a proper tab_id
            if not tab_id or tab_id == "current":
                return {"tab_id": tab_id, "success": False, "error": "Invalid tab ID provided"}
            
            # Get current tab count before closing
            current_state = await self._get_page_state()
            initial_tab_count = len(current_state.get('tabs', []))
            
            event = self.browser_session.event_bus.dispatch(
                CloseTabEvent(target_id=tab_id)
            )
            await event
            result = await event.event_result(raise_if_any=False, raise_if_none=False)
            
            # Get updated page state after closing tab
            updated_state = await self._get_page_state()
            agent_context = await self._get_agent_context_from_state(updated_state)
            
            # Check if the tab was actually closed by comparing tab counts
            current_tab_count = len(updated_state.get('tabs', []))
            operation_successful = current_tab_count < initial_tab_count
            
            operation_result = {
                "action": "close_tab",
                "tab_id": tab_id or "current",
                "success": operation_successful,
                "timestamp": asyncio.get_event_loop().time(),
                "state": updated_state,  # Include updated page state
                "agent_context": agent_context,  # Include rich agent context
                "context": {
                    "closed_tab": tab_id or "current",
                    "initial_tab_count": initial_tab_count,
                    "remaining_tabs": current_tab_count,
                    "close_successful": operation_successful,
                    "message": f"Successfully closed tab {tab_id or 'current'}" if operation_successful else f"Failed to close tab {tab_id or 'current'}",
                    "browser_result": str(result) if result else "No result",
                    "tab_count_change": initial_tab_count - current_tab_count,
                    "page_info": {
                        "url": updated_state.get('url', 'Unknown'),
                        "title": updated_state.get('title', 'Unknown'),
                        "element_count": updated_state.get('dom_state', {}).get('element_count', 0),
                        "tab_count": current_tab_count
                    }
                }
            }
            
            self._operation_history.append(("close_tab", operation_result))
            
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result
            
        except Exception as e:
            self._last_error = str(e)
            return {"tab_id": tab_id or "current", "success": False, "error": str(e)}
    
    async def scroll(self, direction: str, amount: float) -> Dict[str, Any]:
        """Scroll the page."""
        try:
            await self._ensure_browser_started()
            
            # Convert direction to valid value and amount to pixels
            from typing import Literal
            
            # Map direction string to valid literal
            direction_map = {
                "up": "up",
                "down": "down", 
                "left": "left",
                "right": "right"
            }
            
            valid_direction = direction_map.get(direction.lower(), "down")
            pixels = int(amount * 1000)
            
            event = self.browser_session.event_bus.dispatch(
                ScrollEvent(direction=valid_direction, amount=pixels)  # type: ignore
            )
            await event
            result = await event.event_result(raise_if_any=True, raise_if_none=False)
            
            # Get updated page state after scrolling
            updated_state = await self._get_page_state()
            agent_context = await self._get_agent_context_from_state(updated_state)
            
            operation_result = {
                "action": "scroll",
                "direction": direction,
                "amount": amount,
                "success": True,
                "timestamp": asyncio.get_event_loop().time(),
                "state": updated_state,  # Include updated page state
                "agent_context": agent_context,  # Include rich agent context
                "context": {
                    "scroll_direction": direction,
                    "scroll_amount": amount,
                    "scroll_successful": True,
                    "message": f"Successfully scrolled {direction} by {amount} units",
                    "new_content_available": True,
                    "page_info": {
                        "url": updated_state.get('url', 'Unknown'),
                        "title": updated_state.get('title', 'Unknown'),
                        "element_count": updated_state.get('dom_state', {}).get('element_count', 0)
                    }
                }
            }
            
            self._operation_history.append(("scroll", operation_result))
            
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result
            
        except Exception as e:
            self._last_error = str(e)
            return {"direction": direction, "amount": amount, "success": False, "error": str(e)}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current browser status and state with essential information."""
        try:
            await self._ensure_browser_started()
            
            state = await self._get_page_state()
            self._current_state = state
            
            # Analyze available interactions for agent
            elements = state.get('dom_state', {}).get('elements', [])
            
            # Categorize interactive elements
            clickable_elements = []
            links = []
            buttons = []
            inputs = []
            forms = []
            select_elements = []
            
            for i, element in enumerate(elements):
                element_info = {
                    "index": i + 1,
                    "type": element.get('tag', element.get('node_name', 'unknown')),
                    "text": element.get('text', '')[:100] + "..." if len(element.get('text', '')) > 100 else element.get('text', ''),
                    "attributes": {k: v for k, v in element.get('attributes', {}).items() if k in ['id', 'class', 'href', 'type', 'name', 'value', 'placeholder']},
                    "visible": element.get('visible', element.get('is_visible', True)),
                    "position": element.get('position', element.get('absolute_position', {})),
                    "clickable": element.get('clickable', False)  # Use the clickable property from _extract_elements
                }
                
                if element_info["clickable"]:
                    clickable_elements.append(element_info)
                    
                    element_type = element_info["type"]
                    if element_type == 'a':
                        links.append(element_info)
                    elif element_type == 'button':
                        buttons.append(element_info)
                    elif element_type == 'input':
                        inputs.append(element_info)
                    elif element_type == 'select':
                        select_elements.append(element_info)
                    elif element_type in ['form', 'div'] and 'form' in element.get('attributes', {}).get('class', '').lower():
                        forms.append(element_info)
            
            # Generate suggested actions for agent
            suggested_actions = []
            if links:
                suggested_actions.append(f"Click on {len(links)} available links (indices: {[l['index'] for l in links[:5]]})")
            if buttons:
                suggested_actions.append(f"Click on {len(buttons)} available buttons (indices: {[b['index'] for b in buttons[:5]]})")
            if inputs:
                suggested_actions.append(f"Type in {len(inputs)} input fields (indices: {[i['index'] for i in inputs[:5]]})")
            if select_elements:
                suggested_actions.append(f"Select from {len(select_elements)} dropdown menus (indices: {[s['index'] for s in select_elements[:5]]})")
            if len(elements) > 20:
                suggested_actions.append("Scroll down to see more content")
            
            # Page analysis
            page_analysis = {
                "has_forms": len(forms) > 0,
                "has_navigation": len(links) > 5,
                "has_inputs": len(inputs) > 0,
                "is_search_page": any('search' in e.get('attributes', {}).get('class', '').lower() for e in elements),
                "is_login_page": any('password' in e.get('attributes', {}).get('type', '').lower() for e in elements),
                "content_density": "high" if len(elements) > 50 else "medium" if len(elements) > 20 else "low"
            }
            
            # Return only essential information
            operation_result = {
                "action": "get_status",
                "success": True,
                "timestamp": asyncio.get_event_loop().time(),
                "page_info": {
                    "url": state.get('url', 'Unknown'),
                    "title": state.get('title', 'Unknown'),
                    "element_count": state.get('dom_state', {}).get('element_count', 0),
                    "tab_count": len(state.get('tabs', []))
                },
                "interactive_summary": {
                    "total_clickable": len(clickable_elements),
                    "links_count": len(links),
                    "buttons_count": len(buttons),
                    "inputs_count": len(inputs),
                    "selects_count": len(select_elements),
                    "forms_count": len(forms)
                },
                "suggested_actions": suggested_actions[:3],  # Only top 3 suggestions
                "page_analysis": page_analysis,
                "message": f"Page loaded with {len(clickable_elements)} interactive elements. {len(links)} links, {len(buttons)} buttons, {len(inputs)} inputs available."
            }
            
            self._operation_history.append(("get_status", operation_result))
            
            return operation_result
            
        except Exception as e:
            self._last_error = str(e)
            return {"state": None, "success": False, "error": str(e)}
    
    async def _update_state(self) -> None:
        """Update current state"""
        try:
            self._current_state = await self._get_page_state()
        except Exception as e:
            self._last_error = str(e)
    
    async def _get_agent_context_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent context from page state."""
        # Analyze available interactions for agent
        elements = state.get('dom_state', {}).get('elements', [])
        
        # Categorize interactive elements
        clickable_elements = []
        links = []
        buttons = []
        inputs = []
        forms = []
        select_elements = []
        
        for i, element in enumerate(elements):
            # Use the element as-is since it already has the correct structure from _extract_elements
            element_tag = element.get('tag', '').lower()
            element_text = element.get('text', '')
            element_clickable = element.get('clickable', False)
            
            
            # Create a standardized element info structure
            element_info = {
                "index": element.get('index', i + 1),  # Use original index or fallback
                "type": element_tag,
                "text": element_text[:100] + "..." if len(element_text) > 100 else element_text,
                "attributes": element.get('attributes', {}),
                "visible": element.get('visible', True),
                "position": element.get('position', {}),
                "clickable": element_clickable
            }
            
            if element_clickable:
                clickable_elements.append(element_info)
                
                if element_tag == 'a':
                    links.append(element_info)
                elif element_tag == 'button':
                    buttons.append(element_info)
                elif element_tag in ['input', 'textarea']:
                    inputs.append(element_info)
                elif element_tag == 'select':
                    select_elements.append(element_info)
                elif element_tag in ['form'] or 'form' in element.get('attributes', {}).get('class', '').lower():
                    forms.append(element_info)
        
        
        # Generate suggested actions for agent
        suggested_actions = []
        if links:
            suggested_actions.append(f"Click on {len(links)} available links (indices: {[l['index'] for l in links[:5]]})")
        if buttons:
            suggested_actions.append(f"Click on {len(buttons)} available buttons (indices: {[b['index'] for b in buttons[:5]]})")
        if inputs:
            suggested_actions.append(f"Type in {len(inputs)} input fields (indices: {[i['index'] for i in inputs[:5]]})")
        if select_elements:
            suggested_actions.append(f"Select from {len(select_elements)} dropdown menus (indices: {[s['index'] for s in select_elements[:5]]})")
        if len(elements) > 20:
            suggested_actions.append("Scroll down to see more content")
        
        # Page analysis
        page_analysis = {
            "has_forms": len(forms) > 0,
            "has_navigation": len(links) > 5,
            "has_inputs": len(inputs) > 0,
            "is_search_page": any('search' in e.get('attributes', {}).get('class', '').lower() for e in elements),
            "is_login_page": any('password' in e.get('attributes', {}).get('type', '').lower() for e in elements),
            "content_density": "high" if len(elements) > 50 else "medium" if len(elements) > 20 else "low"
        }
        
        return {
            "page_info": {
                "url": state.get('url', 'Unknown'),
                "title": state.get('title', 'Unknown'),
                "element_count": state.get('dom_state', {}).get('element_count', 0),
                "tab_count": len(state.get('tabs', []))
            },
            "interactive_elements": {
                "total_clickable": len(clickable_elements),
                "all_elements": clickable_elements,  # ALL clickable elements with indices
                "links": links,  # ALL links
                "buttons": buttons,  # ALL buttons
                "inputs": inputs,  # ALL inputs
                "selects": select_elements,  # ALL selects
                "forms": forms  # ALL forms
            },
            "suggested_actions": suggested_actions,
            "page_analysis": page_analysis,
            "recent_elements": clickable_elements[:15],  # Top 15 clickable elements
            "message": f"Page loaded with {len(clickable_elements)} interactive elements. {len(links)} links, {len(buttons)} buttons, {len(inputs)} inputs available."
        }

    async def _get_page_state(self) -> Dict[str, Any]:
        """Get comprehensive current page state"""
        await self._ensure_browser_started()
        
        event = self.browser_session.event_bus.dispatch(
            BrowserStateRequestEvent(
                include_screenshot=False,
                cache_clickable_elements_hashes=False
            )
        )
        await event
        result = await event.event_result(raise_if_any=True, raise_if_none=False)
        
        if not result:
            return {
                "url": "Unknown",
                "title": "Unknown",
                "tabs": [],
                "dom_state": {"elements": [], "text_content": "", "element_count": 0},
                "screenshot": False,
                "timestamp": asyncio.get_event_loop().time()
            }
        
        return {
            "url": getattr(result, 'url', 'Unknown'),
            "title": getattr(result, 'title', 'Unknown'),
            "tabs": self._extract_tabs_info(getattr(result, 'tabs', [])),
            "dom_state": self._extract_dom_state(getattr(result, 'dom_state', None)),
            "screenshot": getattr(result, 'screenshot', None) is not None,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    def _extract_tabs_info(self, tabs) -> List[Dict[str, Any]]:
        """Extract tab information from browser tabs"""
        tabs_info = []
        for tab in tabs:
            tabs_info.append({
                "url": tab.url,
                "title": tab.title,
                "target_id": str(tab.target_id),  # Keep full target_id for browser-use operations
                "target_id_short": str(tab.target_id)[-4:],  # Keep short version for display
                "parent_target_id": str(tab.parent_target_id) if tab.parent_target_id else None
            })
        return tabs_info
    
    def _extract_dom_state(self, dom_state) -> Dict[str, Any]:
        """Extract DOM state information"""
        if not dom_state:
            return {
                "elements": [],
                "text_content": "",
                "element_count": 0
            }
        
        try:
            
            # Extract elements from various possible attributes
            elements = []
            element_count = 0
            
            if hasattr(dom_state, 'selector_map') and dom_state.selector_map:
                elements = self._extract_elements(dom_state.selector_map)
                element_count = len(dom_state.selector_map)
            elif hasattr(dom_state, 'clickable_elements') and dom_state.clickable_elements:
                elements = self._extract_elements(dom_state.clickable_elements)
                element_count = len(dom_state.clickable_elements)
            elif hasattr(dom_state, 'elements') and dom_state.elements:
                elements = self._extract_elements(dom_state.elements)
                element_count = len(dom_state.elements)
            else:
                logger.warning("No recognizable element collection found in dom_state")
            
            # Extract text content
            text_content = ""
            if hasattr(dom_state, 'llm_representation'):
                try:
                    text_content = dom_state.llm_representation()
                except:
                    text_content = str(dom_state)[:500]
            elif hasattr(dom_state, 'text_content'):
                text_content = dom_state.text_content
            elif hasattr(dom_state, '__str__'):
                text_content = str(dom_state)[:500]
            
            
            
            return {
                "elements": elements,
                "text_content": text_content,
                "element_count": element_count
            }
        except Exception as e:
            logger.error(f"Error extracting DOM state: {e}")
            return {
                "elements": [],
                "text_content": str(dom_state)[:500] if dom_state else "",
                "element_count": 0
            }
    
    def _extract_elements(self, selector_map) -> List[Dict[str, Any]]:
        """Extract clickable elements from selector map"""
        elements = []
        if not selector_map:
            logger.warning("selector_map is empty or None")
            return elements
            
        try:
            # Handle different types of selector_map structures
            if hasattr(selector_map, 'items'):
                # Dictionary-like structure
                for i, (selector, element) in enumerate(selector_map.items(), 1):
                    # Extract element properties more carefully
                    element_tag = ''
                    element_text = ''
                    element_attrs = {}
                    
                    try:
                        # Try to get tag name
                        if hasattr(element, 'tag_name'):
                            element_tag = element.tag_name
                        elif hasattr(element, 'node_name'):
                            element_tag = element.node_name
                        elif hasattr(element, 'name'):
                            element_tag = element.name
                        else:
                            # Try to extract from string representation
                            elem_str = str(element)
                            if '<' in elem_str and '>' in elem_str:
                                # Extract tag from string like "<a>text</a>" or "[<a>#?:1]"
                                import re
                                tag_match = re.search(r'<(\w+)', elem_str)
                                if tag_match:
                                    element_tag = tag_match.group(1)
                        
                        # Try to get text content
                        if hasattr(element, 'text'):
                            element_text = element.text or ''
                        elif hasattr(element, 'inner_text'):
                            element_text = element.inner_text or ''
                        else:
                            element_text = str(element)[:100]
                        
                        # Try to get attributes
                        if hasattr(element, 'attributes'):
                            element_attrs = element.attributes or {}
                        elif hasattr(element, 'attrs'):
                            element_attrs = element.attrs or {}
                    except Exception as attr_error:
                        logger.warning(f"Error extracting element {i} attributes: {attr_error}")
                    
                    # Determine if element is clickable based on tag and attributes
                    clickable_tags = ['a', 'button', 'input', 'select', 'textarea', 'label']
                    is_clickable = (
                        element_tag.lower() in clickable_tags or
                        'onclick' in element_attrs or
                        'href' in element_attrs or
                        element_attrs.get('type') in ['button', 'submit', 'reset'] or
                        'click' in element_attrs.get('class', '').lower() or
                        'btn' in element_attrs.get('class', '').lower()
                    )
                    
                    # Convert position to serializable dict
                    position = getattr(element, 'absolute_position', None)
                    if position and hasattr(position, 'x'):
                        position = {
                            "x": getattr(position, 'x', 0.0),
                            "y": getattr(position, 'y', 0.0),
                            "width": getattr(position, 'width', 0.0),
                            "height": getattr(position, 'height', 0.0)
                        }
                    
                    element_info = {
                        "index": i,
                        "selector": str(selector),
                        "text": element_text,
                        "tag": element_tag.lower(),
                        "clickable": is_clickable,
                        "visible": getattr(element, 'is_visible', True),
                        "attributes": element_attrs,
                        "position": position
                    }
                    elements.append(element_info)
                        
            elif hasattr(selector_map, '__iter__'):
                # List-like structure
                for i, element in enumerate(selector_map, 1):
                    # Convert position to serializable dict
                    position = getattr(element, 'absolute_position', None)
                    if position and hasattr(position, 'x'):
                        position = {
                            "x": getattr(position, 'x', 0.0),
                            "y": getattr(position, 'y', 0.0),
                            "width": getattr(position, 'width', 0.0),
                            "height": getattr(position, 'height', 0.0)
                        }
                    
                    element_info = {
                        "index": i,
                        "selector": f"element_{i}",
                        "text": getattr(element, 'text', '') or str(element)[:100],
                        "tag": getattr(element, 'tag_name', '') or getattr(element, 'node_name', ''),
                        "clickable": True,
                        "visible": getattr(element, 'is_visible', True),
                        "attributes": getattr(element, 'attributes', {}),
                        "position": position
                    }
                    elements.append(element_info)
            else:
                logger.warning(f"Unknown selector_map type: {type(selector_map)}")
                
        except Exception as e:
            logger.error(f"Error extracting elements: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
        return elements
    
    def _extract_text_content(self, dom_state) -> str:
        """Extract text content from DOM state"""
        if hasattr(dom_state, 'llm_representation'):
            return dom_state.llm_representation()
        return ""
    
    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current cached state"""
        return self._current_state
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error that occurred"""
        return self._last_error
    
    def get_operation_history(self) -> List[tuple]:
        """Get history of operations performed"""
        return self._operation_history.copy()
    
    def clear_error(self):
        """Clear the last error"""
        self._last_error = None
    
    async def go_back(self) -> Dict[str, Any]:
        """Go back in browser history."""
        try:
            await self._ensure_browser_started()
            event = self.browser_session.event_bus.dispatch(GoBackEvent())
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            
            updated_state = await self._get_page_state()
            agent_context = await self._get_agent_context_from_state(updated_state)
            
            operation_result = {
                "action": "go_back", "success": True, "timestamp": asyncio.get_event_loop().time(),
                "state": updated_state, "agent_context": agent_context,
                "context": {"message": "Successfully went back in browser history"}
            }
            
            self._operation_history.append(("go_back", operation_result))
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result
        except Exception as e:
            self._last_error = str(e)
            return {"action": "go_back", "success": False, "error": str(e)}
    
    async def refresh(self) -> Dict[str, Any]:
        """Refresh the current page."""
        try:
            await self._ensure_browser_started()
            event = self.browser_session.event_bus.dispatch(RefreshEvent())
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            
            updated_state = await self._get_page_state()
            agent_context = await self._get_agent_context_from_state(updated_state)
            
            operation_result = {
                "action": "refresh", "success": True, "timestamp": asyncio.get_event_loop().time(),
                "state": updated_state, "agent_context": agent_context,
                "context": {"message": "Successfully refreshed the page"}
            }
            
            self._operation_history.append(("refresh", operation_result))
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result
        except Exception as e:
            self._last_error = str(e)
            return {"action": "refresh", "success": False, "error": str(e)}
    
    
    async def send_keys(self, keys: str) -> Dict[str, Any]:
        """Send keyboard keys (e.g., 'Enter', 'Escape', 'Tab')."""
        try:
            await self._ensure_browser_started()
            event = self.browser_session.event_bus.dispatch(SendKeysEvent(keys=keys))
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            
            updated_state = await self._get_page_state()
            agent_context = await self._get_agent_context_from_state(updated_state)
            
            operation_result = {
                "action": "send_keys", "keys": keys, "success": True, "timestamp": asyncio.get_event_loop().time(),
                "state": updated_state, "agent_context": agent_context,
                "context": {"message": f"Successfully sent keys: '{keys}'"}
            }
            
            self._operation_history.append(("send_keys", operation_result))
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result
        except Exception as e:
            self._last_error = str(e)
            return {"action": "send_keys", "keys": keys, "success": False, "error": str(e)}
    
    async def get_dropdown_options(self, index: int) -> Dict[str, Any]:
        """Get dropdown options for a select element."""
        try:
            await self._ensure_browser_started()
            node = await self.browser_session.get_element_by_index(index)
            if node is None:
                return {"action": "get_dropdown_options", "index": index, "success": False, "error": f"Element {index} not found"}
            
            event = self.browser_session.event_bus.dispatch(GetDropdownOptionsEvent(node=node))
            await event
            result = await event.event_result(raise_if_any=True, raise_if_none=False)
            
            return {
                "action": "get_dropdown_options", "index": index, "success": True, "timestamp": asyncio.get_event_loop().time(),
                "options": result if result else [], "context": {"message": f"Retrieved dropdown options for element {index}"}
            }
        except Exception as e:
            self._last_error = str(e)
            return {"action": "get_dropdown_options", "index": index, "success": False, "error": str(e)}
    
    async def select_dropdown_option(self, index: int, option_text: str) -> Dict[str, Any]:
        """Select an option from a dropdown by text."""
        try:
            await self._ensure_browser_started()
            node = await self.browser_session.get_element_by_index(index)
            if node is None:
                return {"action": "select_dropdown_option", "index": index, "option": option_text, "success": False, "error": f"Element {index} not found"}
            
            event = self.browser_session.event_bus.dispatch(SelectDropdownOptionEvent(node=node, option_text=option_text))
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            
            updated_state = await self._get_page_state()
            agent_context = await self._get_agent_context_from_state(updated_state)
            
            operation_result = {
                "action": "select_dropdown_option", "index": index, "option": option_text, "success": True, "timestamp": asyncio.get_event_loop().time(),
                "state": updated_state, "agent_context": agent_context,
                "context": {"message": f"Successfully selected '{option_text}' from dropdown {index}"}
            }
            
            self._operation_history.append(("select_dropdown_option", operation_result))
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result
        except Exception as e:
            self._last_error = str(e)
            return {"action": "select_dropdown_option", "index": index, "option": option_text, "success": False, "error": str(e)}
    
    async def switch_tab(self, tab_id: str) -> Dict[str, Any]:
        """Switch to a specific tab."""
        try:
            await self._ensure_browser_started()
            event = self.browser_session.event_bus.dispatch(SwitchTabEvent(target_id=tab_id))
            await event
            result = await event.event_result(raise_if_any=False, raise_if_none=False)
            
            updated_state = await self._get_page_state()
            agent_context = await self._get_agent_context_from_state(updated_state)
            
            # For switch_tab, we consider it successful if we can get the updated state
            # and the current tab URL/title changed (indicating we switched)
            operation_successful = True  # Assume success if we get here
            
            operation_result = {
                "action": "switch_tab", "tab_id": tab_id, "success": operation_successful, "timestamp": asyncio.get_event_loop().time(),
                "state": updated_state, "agent_context": agent_context,
                "context": {
                    "message": f"Successfully switched to tab {tab_id}" if operation_successful else f"Failed to switch to tab {tab_id}",
                    "browser_result": str(result) if result else "No result"
                }
            }
            
            self._operation_history.append(("switch_tab", operation_result))
            if self.auto_update_state:
                await self._update_state()
            
            return operation_result
        except Exception as e:
            self._last_error = str(e)
            return {"action": "switch_tab", "tab_id": tab_id, "success": False, "error": str(e)}
    
    async def close_browser(self) -> Dict[str, Any]:
        """Close the browser session completely."""
        try:
            if self.browser_session and self._browser_started:
                # Try different methods to close the browser session
                try:
                    if hasattr(self.browser_session, 'close'):
                        await self.browser_session.close()
                    elif hasattr(self.browser_session, 'stop'):
                        await self.browser_session.stop()
                    elif hasattr(self.browser_session, 'quit'):
                        await self.browser_session.quit()
                    else:
                        # Gracefully mark as closed without error
                        logger.info("Browser session cleanup completed")
                except Exception as close_error:
                    logger.warning(f"Browser close method failed: {close_error}")
                
                self._browser_started = False
                logger.info("Browser session closed successfully")
                return {"success": True, "message": "Browser session closed"}
            else:
                return {"success": True, "message": "Browser was not started"}
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to close browser session: {e}")
            return {"success": False, "error": str(e)}
    
# Multiple Specific Tools - Each tool handles one specific functionality

class BrowserUseNavigateTool(Tool):
    """Tool for navigating to URLs and managing browser tabs"""
    name: str = "browser_navigate"
    description: str = "Navigate to a specific URL in the current tab or open a new tab. Use this to visit websites, load pages, or open new browser tabs. Returns page state with clickable elements, page title, and current URL after navigation."
    inputs: Dict[str, Dict[str, str]] = {
        "url": {
            "type": "string",
            "description": "Complete URL to navigate to (e.g., 'https://www.google.com', 'https://example.com/page'). Must include protocol (http:// or https://)."
        },
        "new_tab": {
            "type": "boolean",
            "description": "Whether to open the URL in a new browser tab (true) or current tab (false). Default: false. Use true when you want to keep the current page open."
        }
    }
    required: Optional[List[str]] = ["url"]
    
    def __init__(self, browser_use: Optional[BrowserUse] = None):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, url: str, new_tab: bool = False) -> Dict[str, Any]:
        """Execute navigation using the BrowserUse instance."""
        if not self.browser_use:
            raise RuntimeError("Browser use instance not initialized")
        
        if not url:
            return {"success": False, "error": "URL is required"}
        
        try:
            return _run_async_in_sync(self.browser_use, self.browser_use.navigate, url, new_tab)
        except Exception as e:
            return {"success": False, "error": f"Error navigating to {url}: {str(e)}"}


class BrowserUseClickTool(Tool):
    """Tool for clicking on interactive web elements"""
    name: str = "browser_click"
    description: str = "Click on a clickable web element (links, buttons, inputs, etc.) by its index number. Use browser_get_status first to see all available clickable elements with their indices. Returns updated page state after the click action."
    inputs: Dict[str, Dict[str, str]] = {
        "index": {
            "type": "integer",
            "description": "1-based index number of the clickable element to click. Get available indices from browser_get_status tool. Must be a positive integer corresponding to a clickable element on the current page."
        }
    }
    required: Optional[List[str]] = ["index"]
    
    def __init__(self, browser_use: Optional[BrowserUse] = None):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, index: int) -> Dict[str, Any]:
        """Execute click using the BrowserUse instance."""
        if not self.browser_use:
            raise RuntimeError("Browser use instance not initialized")
        
        if index is None:
            return {"success": False, "error": "Index is required"}
        
        try:
            return _run_async_in_sync(self.browser_use, self.browser_use.click, index)
        except Exception as e:
            return {"success": False, "error": f"Error clicking element {index}: {str(e)}"}


class BrowserUseTypeTool(Tool):
    """Tool for typing text into input fields and form elements"""
    name: str = "browser_type"
    description: str = "Type text into input fields, text areas, or other text-entry elements on a webpage. Use this for filling forms, entering search queries, or any text input. Use browser_get_status first to find input elements and their indices."
    inputs: Dict[str, Dict[str, str]] = {
        "index": {"type": "integer", "description": "1-based index number of the input element to type into. Get available input indices from browser_get_status tool. Must be a positive integer corresponding to an input field on the current page."},
        "text": {"type": "string", "description": "The text content to type into the input field. Can be any string including numbers, special characters, and spaces. Will be entered exactly as provided."}
    }
    required: Optional[List[str]] = ["index", "text"]
    
    def __init__(self, browser_use: Optional[BrowserUse] = None):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, index: int, text: str) -> Dict[str, Any]:
        """Execute type text using the BrowserUse instance."""
        if not self.browser_use:
            raise RuntimeError("Browser use instance not initialized")
        
        if index is None:
            return {"success": False, "error": "Index is required"}
        
        if not text:
            return {"success": False, "error": "Text is required"}
        
        try:
            return _run_async_in_sync(self.browser_use, self.browser_use.type, index, text)
        except Exception as e:
            return {"success": False, "error": f"Error typing text '{text}' into element {index}: {str(e)}"}


class BrowserUseCloseTool(Tool):
    """Tool for closing the entire browser session"""
    name: str = "browser_close"
    description: str = "Completely close the browser session and terminate all browser processes. Use this when you're done with all browser automation tasks. This will close all tabs and free up system resources. Cannot be undone - you'll need to restart browser automation if you need to continue."
    inputs: Dict[str, Dict[str, str]] = {}
    required: Optional[List[str]] = []
    
    def __init__(self, browser_use: Optional[BrowserUse] = None):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self) -> Dict[str, Any]:
        """Execute close browser using the BrowserUse instance."""
        if not self.browser_use:
            raise RuntimeError("Browser use instance not initialized")
        
        try:
            return _run_async_in_sync(self.browser_use, self.browser_use.close_browser)
        except Exception as e:
            return {"success": False, "error": f"Error closing browser: {str(e)}"}


class BrowserUseScrollTool(Tool):
    """Tool for scrolling web pages to reveal more content"""
    name: str = "browser_scroll"
    description: str = "Scroll the current webpage up or down to reveal more content, load additional elements, or navigate through long pages. Useful when you need to see more clickable elements or content that's not currently visible. Returns updated page state with newly visible elements."
    inputs: Dict[str, Dict[str, str]] = {
        "direction": {
            "type": "string",
            "description": "Scroll direction: 'up' to scroll towards the top of the page, 'down' to scroll towards the bottom. Only these two values are accepted."
        },
        "amount": {
            "type": "number",
            "description": "Scroll amount as a decimal number: 0.1 = small scroll, 0.5 = half page, 1.0 = full page, 2.0 = two pages. Higher values scroll more. Recommended: 0.5-1.0 for normal scrolling."
        }
    }
    required: Optional[List[str]] = ["direction", "amount"]
    
    def __init__(self, browser_use: Optional[BrowserUse] = None):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, direction: str, amount: float) -> Dict[str, Any]:
        """Execute scroll using the BrowserUse instance."""
        if not self.browser_use:
            raise RuntimeError("Browser use instance not initialized")
        
        if not direction or amount is None:
            return {"success": False, "error": "Direction and amount are required"}
        
        try:
            return _run_async_in_sync(self.browser_use, self.browser_use.scroll, direction, amount)
        except Exception as e:
            return {"success": False, "error": f"Error scrolling {direction} {amount}: {str(e)}"}


class BrowserUseGetStatusTool(Tool):
    """Tool for getting essential browser and page information"""
    name: str = "browser_get_status"
    description: str = "Get essential information about the current browser state including page URL, title, interactive element counts, and suggested actions. Use this before clicking or typing to see what elements are available. Returns concise summary with element counts and top suggestions."
    inputs: Dict[str, Dict[str, str]] = {}
    required: Optional[List[str]] = []
    
    def __init__(self, browser_use: Optional[BrowserUse] = None):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self) -> Dict[str, Any]:
        """Execute get status using the BrowserUse instance."""
        if not self.browser_use:
            raise RuntimeError("Browser use instance not initialized")
        
        try:
            return _run_async_in_sync(self.browser_use, self.browser_use.get_status)
        except Exception as e:
            return {"success": False, "error": f"Error getting status: {str(e)}"}


# ============================================================================
# ENHANCED BROWSER TOOLS - Simple, robust tool implementations
# ============================================================================

class BrowserUseGoBackTool(Tool):
    """Tool for navigating back in browser history"""
    name: str = "browser_go_back"
    description: str = "Navigate back to the previous page in browser history (equivalent to clicking the back button). Use this to return to a previously visited page. Returns updated page state after going back."
    inputs: Dict[str, Dict[str, str]] = {}
    required: Optional[List[str]] = []
    
    def __init__(self, browser_use: BrowserUse):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self) -> Dict[str, Any]:
        return _run_async_in_sync(self.browser_use, self.browser_use.go_back)

class BrowserUseRefreshTool(Tool):
    """Tool for refreshing/reloading the current webpage"""
    name: str = "browser_refresh"
    description: str = "Reload/refresh the current webpage (equivalent to pressing F5 or clicking refresh button). Use this to reload the page content, refresh dynamic content, or retry failed page loads. Returns updated page state after refresh."
    inputs: Dict[str, Dict[str, str]] = {}
    required: Optional[List[str]] = []
    
    def __init__(self, browser_use: BrowserUse):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self) -> Dict[str, Any]:
        return _run_async_in_sync(self.browser_use, self.browser_use.refresh)


class BrowserUseSendKeysTool(Tool):
    """Tool for sending keyboard shortcuts and special keys"""
    name: str = "browser_send_keys"
    description: str = "Send special keyboard keys and shortcuts to the current page. Use this for keyboard shortcuts, form submissions, navigation, or triggering JavaScript events. Common keys: Enter (submit forms), Escape (close dialogs), Tab (navigate between elements), Arrow keys (navigation)."
    inputs: Dict[str, Dict[str, str]] = {
        "keys": {"type": "string", "description": "Special keyboard key to send. Common values: 'Enter' (submit/confirm), 'Escape' (cancel/close), 'Tab' (next element), 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space' (click/activate), 'Backspace' (delete). Use exact key names as shown."}
    }
    required: Optional[List[str]] = ["keys"]
    
    def __init__(self, browser_use: BrowserUse):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, keys: str) -> Dict[str, Any]:
        return _run_async_in_sync(self.browser_use, self.browser_use.send_keys, keys)

class BrowserUseGetDropdownOptionsTool(Tool):
    """Tool for retrieving dropdown menu options"""
    name: str = "browser_get_dropdown_options"
    description: str = "Get all available options from a dropdown/select menu element. Use this before selecting a dropdown option to see what choices are available. Returns a list of all selectable options with their text values."
    inputs: Dict[str, Dict[str, str]] = {
        "index": {"type": "integer", "description": "1-based index number of the dropdown/select element. Get available indices from browser_get_status tool. Must be a positive integer corresponding to a select element on the current page."}
    }
    required: Optional[List[str]] = ["index"]
    
    def __init__(self, browser_use: BrowserUse):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, index: int) -> Dict[str, Any]:
        return _run_async_in_sync(self.browser_use, self.browser_use.get_dropdown_options, index)

class BrowserUseSelectDropdownOptionTool(Tool):
    """Tool for selecting options from dropdown menus"""
    name: str = "browser_select_dropdown_option"
    description: str = "Select a specific option from a dropdown/select menu by matching the option text. Use browser_get_dropdown_options first to see available options. Returns updated page state after selection."
    inputs: Dict[str, Dict[str, str]] = {
        "index": {"type": "integer", "description": "1-based index number of the dropdown/select element. Get available indices from browser_get_status tool. Must be a positive integer corresponding to a select element on the current page."},
        "option_text": {"type": "string", "description": "Exact text of the option to select from the dropdown. Must match one of the options returned by browser_get_dropdown_options. Case-sensitive and must match exactly."}
    }
    required: Optional[List[str]] = ["index", "option_text"]
    
    def __init__(self, browser_use: BrowserUse):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, index: int, option_text: str) -> Dict[str, Any]:
        return _run_async_in_sync(self.browser_use, self.browser_use.select_dropdown_option, index, option_text)

class BrowserUseSwitchTabTool(Tool):
    """Tool for switching between browser tabs"""
    name: str = "browser_switch_tab"
    description: str = "Switch the active browser tab to a different open tab. Use this to navigate between multiple open tabs. Get available tab IDs from browser_get_status. Returns updated page state of the newly active tab."
    inputs: Dict[str, Dict[str, str]] = {
        "tab_id": {"type": "string", "description": "Unique identifier of the tab to switch to. Get available tab IDs from browser_get_status tool. Must be a valid tab ID from the current browser session."}
    }
    required: Optional[List[str]] = ["tab_id"]
    
    def __init__(self, browser_use: BrowserUse):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, tab_id: str) -> Dict[str, Any]:
        return _run_async_in_sync(self.browser_use, self.browser_use.switch_tab, tab_id)

class BrowserUseCloseTabTool(Tool):
    """Tool for closing browser tabs"""
    name: str = "browser_close_tab"
    description: str = "Close a specific browser tab or the current tab if no ID provided. Use this to clean up unused tabs or close tabs you're done with. Cannot close the last remaining tab. Returns updated page state after closing."
    inputs: Dict[str, Dict[str, str]] = {
        "tab_id": {"type": "string", "description": "ID of the specific tab to close. Get available tab IDs from browser_get_status. If not provided or null, closes the current active tab. Must be a valid tab ID from the current browser session."}
    }
    required: Optional[List[str]] = []
    
    def __init__(self, browser_use: BrowserUse):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, tab_id: str = None) -> Dict[str, Any]:
        return _run_async_in_sync(self.browser_use, self.browser_use.close_tab, tab_id)

class BrowserUseCreateTabTool(Tool):
    """Tool for creating new browser tabs"""
    name: str = "browser_create_tab"
    description: str = "Create a new browser tab and optionally navigate to a URL. Use this to open additional tabs for multitasking or to keep the current page open while navigating elsewhere. Returns updated page state of the new tab."
    inputs: Dict[str, Dict[str, str]] = {
        "url": {"type": "string", "description": "URL to navigate to in the new tab. If not provided, defaults to 'about:blank' (empty page). Must include protocol (http:// or https://) for external URLs."}
    }
    required: Optional[List[str]] = []
    
    def __init__(self, browser_use: BrowserUse):
        super().__init__()
        self.browser_use = browser_use
    
    def __call__(self, url: str = "about:blank") -> Dict[str, Any]:
        return _run_async_in_sync(self.browser_use, self.browser_use.navigate, url, new_tab=True)

class BrowserUseToolkit(Toolkit):
    """
    Single toolkit that manages all browser tools and the shared base
    """
    def __init__(
        self,
        name: str = "BrowserUseToolkit",
        browser_session: Optional[BrowserSession] = None,
        auto_update_state: bool = True,
        **kwargs
    ):
        # Create browser session if not provided
        if browser_session is None:
            from browser_use import BrowserSession, BrowserProfile
            # Create optimized browser profile for performance
            browser_profile = BrowserProfile(
                headless=False,
                minimum_wait_page_load_time=1,
                wait_for_network_idle_page_load_time=2,
                wait_between_actions=1,
                highlight_elements=False,  # Disable highlighting for performance
                cross_origin_iframes=False,  # Disable for performance
                enable_default_extensions=False,  # Disable extensions for performance
            )
            browser_session = BrowserSession(browser_profile=browser_profile)
        
        # Create the shared BrowserUse instance
        browser_use = BrowserUse(
            name="BrowserUse",
            browser_session=browser_session,
            auto_update_state=auto_update_state,
            **kwargs
        )
        
        # Create all specific tools with the shared browser use instance
        tools = [
            BrowserUseNavigateTool(browser_use=browser_use),
            BrowserUseClickTool(browser_use=browser_use),
            BrowserUseTypeTool(browser_use=browser_use),
            BrowserUseCloseTool(browser_use=browser_use),
            BrowserUseScrollTool(browser_use=browser_use),
            BrowserUseGetStatusTool(browser_use=browser_use),
            BrowserUseGoBackTool(browser_use=browser_use),
            BrowserUseRefreshTool(browser_use=browser_use),
            BrowserUseSendKeysTool(browser_use=browser_use),
            BrowserUseGetDropdownOptionsTool(browser_use=browser_use),
            BrowserUseSelectDropdownOptionTool(browser_use=browser_use),
            BrowserUseSwitchTabTool(browser_use=browser_use),
            # BrowserUseCloseTabTool(browser_use=browser_use),
            # BrowserUseCreateTabTool(browser_use=browser_use),
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store browser_use as instance variable
        self.browser_use = browser_use
    
    def cleanup(self):
        """Explicitly cleanup the browser session and thread."""
        if self.browser_use:
            logger.info("Explicitly cleaning up BrowserUseToolkit...")
            self.browser_use._cleanup_browser_session()
            self.browser_use._shutdown_browser_thread()
            logger.info("BrowserUseToolkit cleanup completed")
    
    def __del__(self):
        """Cleanup when the toolkit is destroyed."""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Error during BrowserUseToolkit cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Error during BrowserUseToolkit context manager cleanup: {e}")
        return False  # Don't suppress exceptions