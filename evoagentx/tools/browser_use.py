"""
BrowserUse Toolkit - Simple browser automation toolkit
Uses browser-use as external dependency only
Multiple specific tools: Navigate, Click, Close, Scroll, Get_Status
One base class and one toolkit managing all tools
"""

import asyncio
from typing import Dict, Any, Optional, List
from browser_use import BrowserSession
from browser_use.browser.events import (
    NavigateToUrlEvent,
    ClickElementEvent,
    ScrollEvent,
    CloseTabEvent,
    BrowserStateRequestEvent
)

from .tool import Tool, Toolkit
from ..core.module import BaseModule
from ..core.logging import logger


def _run_async_in_sync(async_func, *args, **kwargs):
    """
    Helper function to run async methods in sync context.
    Handles event loop management and threading when needed.
    """
    import asyncio
    import concurrent.futures
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new task if loop is already running
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                return future.result()
        else:
            return loop.run_until_complete(async_func(*args, **kwargs))
    except RuntimeError:
        return asyncio.run(async_func(*args, **kwargs))




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
    
    async def _ensure_browser_started(self):
        """Ensure the browser session is started."""
        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")
        
        if not self._browser_started:
            try:
                # Start the browser session
                await self.browser_session.start()
                self._browser_started = True
                logger.info("Browser session started successfully")
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
                raise ValueError(f'Element index {index} not found in browser state')

            event = self.browser_session.event_bus.dispatch(
                ClickElementEvent(node=node, while_holding_ctrl=while_holding_ctrl or False)
            )
            
            await event
            # Wait for handler to complete and get any exception or metadata
            click_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)
            
            return {
                "success": True,
                "index": index,
                "message": f"Successfully clicked element {index}"
            }
            
        except Exception as e:
            self._last_error = str(e)
            return {"success": False, "error": str(e)}

    
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
                {"name": "scroll", "description": "Scroll the page", "parameters": ["direction", "amount"]},
                {"name": "close_tab", "description": "Close a tab", "parameters": ["tab_id"]},
                {"name": "get_status", "description": "Get current page status", "parameters": []},
                {"name": "search_google", "description": "Search Google for a query", "parameters": ["query"]}
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
            
            event = self.browser_session.event_bus.dispatch(
                CloseTabEvent(target_id=tab_id or "current")
            )
            await event
            result = await event.event_result(raise_if_any=True, raise_if_none=False)
            
            operation_result = {
                "action": "close_tab",
                "tab_id": tab_id or "current",
                "success": True,
                "timestamp": asyncio.get_event_loop().time(),
                "context": {
                    "closed_tab": tab_id or "current",
                    "remaining_tabs": len(self._current_state.get('tabs', [])) - 1 if self._current_state else 0,
                    "close_successful": True,
                    "message": f"Successfully closed tab {tab_id or 'current'}"
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
        """Get current browser status and state with rich agent context."""
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
                    "type": element.get('node_name', 'unknown'),
                    "text": element.get('text', '')[:100] + "..." if len(element.get('text', '')) > 100 else element.get('text', ''),
                    "attributes": {k: v for k, v in element.get('attributes', {}).items() if k in ['id', 'class', 'href', 'type', 'name', 'value', 'placeholder']},
                    "visible": element.get('is_visible', True),
                    "position": element.get('absolute_position', {}),
                    "clickable": element.get('is_visible', True) and element.get('node_name') in ['a', 'button', 'input', 'select', 'textarea']
                }
                
                if element_info["clickable"]:
                    clickable_elements.append(element_info)
                    
                    if element.get('node_name') == 'a':
                        links.append(element_info)
                    elif element.get('node_name') == 'button':
                        buttons.append(element_info)
                    elif element.get('node_name') == 'input':
                        inputs.append(element_info)
                    elif element.get('node_name') == 'select':
                        select_elements.append(element_info)
                    elif element.get('node_name') in ['form', 'div'] and 'form' in element.get('class', '').lower():
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
            
            operation_result = {
                "action": "get_status",
                "success": True,
                "state": state,  # Complete DOM state
                "timestamp": asyncio.get_event_loop().time(),
                "agent_context": {
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
        logger.info(f"_get_agent_context_from_state received {len(elements)} elements")
        
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
            
            # Debug log for first few elements
            if i < 5:
                logger.info(f"Processing element {i+1}: tag='{element_tag}', clickable={element_clickable}, text='{element_text[:30]}'")
            
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
        
        logger.info(f"Categorized: {len(clickable_elements)} clickable, {len(links)} links, {len(buttons)} buttons, {len(inputs)} inputs")
        
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
                "target_id": str(tab.target_id)[-4:],
                "parent_target_id": str(tab.parent_target_id)[-4:] if tab.parent_target_id else None
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
            # Debug: Log the dom_state structure
            logger.info(f"DOM state type: {type(dom_state)}")
            logger.info(f"DOM state attributes: {dir(dom_state)}")
            
            # Extract elements from various possible attributes
            elements = []
            element_count = 0
            
            if hasattr(dom_state, 'selector_map') and dom_state.selector_map:
                logger.info(f"Found selector_map with {len(dom_state.selector_map)} items")
                elements = self._extract_elements(dom_state.selector_map)
                element_count = len(dom_state.selector_map)
            elif hasattr(dom_state, 'clickable_elements') and dom_state.clickable_elements:
                logger.info(f"Found clickable_elements with {len(dom_state.clickable_elements)} items")
                elements = self._extract_elements(dom_state.clickable_elements)
                element_count = len(dom_state.clickable_elements)
            elif hasattr(dom_state, 'elements') and dom_state.elements:
                logger.info(f"Found elements with {len(dom_state.elements)} items")
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
            
            logger.info(f"Extracted {len(elements)} clickable elements from {element_count} total elements")
            
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
            logger.info(f"Processing selector_map of type: {type(selector_map)}")
            
            # Handle different types of selector_map structures
            if hasattr(selector_map, 'items'):
                # Dictionary-like structure
                logger.info(f"Processing {len(selector_map)} items from dict-like selector_map")
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
                    
                    if i <= 5:  # Log first 5 elements for debugging
                        logger.info(f"Element {i}: {element_info['tag']} - {element_info['text'][:50]} - clickable: {is_clickable}")
                        
            elif hasattr(selector_map, '__iter__'):
                # List-like structure
                logger.info(f"Processing list-like selector_map with {len(list(selector_map))} items")
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
                    if i <= 3:  # Log first 3 elements for debugging
                        logger.info(f"Element {i}: {element_info['tag']} - {element_info['text'][:50]}")
            else:
                logger.warning(f"Unknown selector_map type: {type(selector_map)}")
                logger.info(f"Selector_map content preview: {str(selector_map)[:200]}")
                
        except Exception as e:
            logger.error(f"Error extracting elements: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
        logger.info(f"Successfully extracted {len(elements)} elements, {len([e for e in elements if e['clickable']])} clickable")
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
    
    async def search_google(self, query: str) -> Dict[str, Any]:
        """
        Client-side search function - searches Google for a query
        This is implemented in the base class, not as a separate tool
        """
        try:
            if not self.browser_session:
                raise RuntimeError("Browser session not initialized")
            
            # Navigate to Google search
            search_url = f"https://www.google.com/search?q={query}"
            nav_result = await self.navigate(search_url)
            
            if nav_result['success']:
                # Get the search results page status
                status_result = await self.get_status()
                
                return {
                    "success": True,
                    "query": query,
                    "search_url": search_url,
                    "page_status": status_result,
                    "message": f"Successfully searched for '{query}' on Google"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to navigate to Google search: {nav_result.get('error', 'Unknown error')}"
                }
        except Exception as e:
            return {"success": False, "error": f"Error searching for '{query}': {str(e)}"}

# Multiple Specific Tools - Each tool handles one specific functionality

class BrowserUseNavigateTool(Tool):
    """Tool for navigating to URLs"""
    name: str = "browser_navigate"
    description: str = "Navigate to a URL or open new tab"
    inputs: Dict[str, Dict[str, str]] = {
        "url": {
            "type": "string",
            "description": "URL to navigate to"
        },
        "new_tab": {
            "type": "boolean",
            "description": "Open in new tab (default: false)"
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
            return _run_async_in_sync(self.browser_use.navigate, url, new_tab)
        except Exception as e:
            return {"success": False, "error": f"Error navigating to {url}: {str(e)}"}


class BrowserUseClickTool(Tool):
    """Tool for clicking on elements"""
    name: str = "browser_click"
    description: str = "Click on an element by index"
    inputs: Dict[str, Dict[str, str]] = {
        "index": {
            "type": "integer",
            "description": "Index of element to click (1-based)"
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
            return _run_async_in_sync(self.browser_use.click, index)
        except Exception as e:
            return {"success": False, "error": f"Error clicking element {index}: {str(e)}"}


class BrowserUseCloseTool(Tool):
    """Tool for closing browser session"""
    name: str = "browser_close"
    description: str = "Close the browser session completely"
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
            return _run_async_in_sync(self.browser_use.close_browser)
        except Exception as e:
            return {"success": False, "error": f"Error closing browser: {str(e)}"}


class BrowserUseScrollTool(Tool):
    """Tool for scrolling pages"""
    name: str = "browser_scroll"
    description: str = "Scroll the page up or down"
    inputs: Dict[str, Dict[str, str]] = {
        "direction": {
            "type": "string",
            "description": "Direction to scroll: up or down"
        },
        "amount": {
            "type": "number",
            "description": "Amount to scroll (0.5 = half page, 1.0 = full page)"
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
            return _run_async_in_sync(self.browser_use.scroll, direction, amount)
        except Exception as e:
            return {"success": False, "error": f"Error scrolling {direction} {amount}: {str(e)}"}


class BrowserUseGetStatusTool(Tool):
    """Tool for getting browser status"""
    name: str = "browser_get_status"
    description: str = "Get current browser status and state"
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
            return _run_async_in_sync(self.browser_use.get_status)
        except Exception as e:
            return {"success": False, "error": f"Error getting status: {str(e)}"}


class BrowserUseToolkit(Toolkit):
    """
    Single toolkit that manages all browser tools and the shared base
    Similar to WikipediaSearchToolkit - orchestrates the base class and all tools
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
                headless=False,  # Use headless for better performance
                minimum_wait_page_load_time=0.1,  # Reduce wait times
                wait_for_network_idle_page_load_time=0.2,
                wait_between_actions=0.1,
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
            BrowserUseCloseTool(browser_use=browser_use),
            BrowserUseScrollTool(browser_use=browser_use),
            BrowserUseGetStatusTool(browser_use=browser_use),
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store browser_use as instance variable
        self.browser_use = browser_use