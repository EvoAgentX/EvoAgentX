from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from typing import Dict, Any, List, Callable, Optional, Tuple, Union
from pydantic import Field
from .tool import Tool
from evoagentx.core.logging import logger
import html2text
import time

# Define selector map as a constant to avoid repetition
SELECTOR_MAP = {
    "css": By.CSS_SELECTOR,
    "xpath": By.XPATH,
    "id": By.ID,
    "class": By.CLASS_NAME,
    "name": By.NAME,
    "tag": By.TAG_NAME,
}

class BrowserTool(Tool):
    """
    A tool for interacting with web browsers using Selenium.
    Allows agents to navigate to URLs, interact with elements, extract information,
    and more from web pages.
    """
    
    timeout: int = Field(default=10, description="Default timeout in seconds for browser operations")
    browser_type: str = Field(default="chrome", description="Type of browser to use ('chrome', 'firefox', 'safari', 'edge')")
    headless: bool = Field(default=False, description="Whether to run the browser in headless mode")
    timeout: int = Field(default=10, description="Default timeout in seconds for browser operations")
    
    def __init__(
        self,
        name: str = "Browser Tool",
        browser_type: str = "chrome",
        headless: bool = False,
        timeout: int = 10,
        **kwargs
    ):
        """
        Initialize the browser tool with Selenium WebDriver.
        
        Args:
            name (str): Name of the tool
            browser_type (str): Type of browser to use ('chrome', 'firefox', 'safari', 'edge')
            headless (bool): Whether to run the browser in headless mode
            timeout (int): Default timeout in seconds for browser operations
            **kwargs: Additional keyword arguments for parent class initialization
        """
        # Pass to parent class initialization
        super().__init__(name=name, timeout=timeout, browser_type=browser_type, headless=headless, **kwargs)
        self.driver = None
        
        # Storage for element references from snapshots
        self.element_references = {}
    
    # Helper methods to reduce duplication
    
    def _check_driver_initialized(self) -> Union[None, Dict[str, Any]]:
        """
        Check if the browser driver is initialized.
        
        Returns:
            Union[None, Dict[str, Any]]: None if driver is initialized, error response otherwise
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
        return None
    
    def _get_selector_by_type(self, selector_type: str) -> Union[str, Dict[str, Any]]:
        """
        Get the Selenium By selector for the given selector type.
        
        Args:
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            
        Returns:
            Union[str, Dict[str, Any]]: The By selector or error response
        """
        by_type = SELECTOR_MAP.get(selector_type.lower())
        if not by_type:
            return {"status": "error", "message": f"Invalid selector type: {selector_type}"}
        return by_type
    
    def _wait_for_page_load(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for the page to load completely.
        
        Args:
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            bool: True if page loaded, False if timed out
        """
        timeout = timeout or self.timeout
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            return True
        except TimeoutException:
            return False
    
    def _parse_element_reference(self, ref: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse an element reference into selector type and selector.
        
        Args:
            ref (str): Element reference ID from the page snapshot
            
        Returns:
            Tuple[Optional[str], Optional[str], Optional[str]]: 
                (selector_type, selector, error_message) - error_message is None if successful
        """
        if not self.element_references:
            return None, None, "No page snapshot available. Use browser_snapshot or navigate_to_url first."
            
        stored_ref = self.element_references.get(ref)
        if not stored_ref:
            return None, None, f"Element reference '{ref}' not found. Use browser_snapshot or navigate_to_url first."
        
        # Parse the stored reference to get selector and type
        if ":" in stored_ref:
            ref_parts = stored_ref.split(":", 1)
            if len(ref_parts) != 2:
                return None, None, f"Invalid stored reference format: {stored_ref}"
            
            selector_type, selector = ref_parts
            return selector_type, selector, None
        
        return None, None, f"Invalid stored reference format: {stored_ref}"
    
    def _find_element_with_wait(self, by_type: str, selector: str, 
                               timeout: Optional[int] = None, 
                               wait_condition=EC.presence_of_element_located) -> Tuple[Optional[Any], Optional[str]]:
        """
        Find an element on the page with wait condition.
        
        Args:
            by_type (str): Selenium By selector type
            selector (str): The selector string
            timeout (int, optional): Custom timeout for this operation
            wait_condition: The EC condition to wait for
            
        Returns:
            Tuple[Optional[Any], Optional[str]]: (element, error_message) - error_message is None if successful
        """
        timeout = timeout or self.timeout
        try:
            element = WebDriverWait(self.driver, timeout).until(
                wait_condition((by_type, selector))
            )
            return element, None
        except TimeoutException:
            return None, f"Element not found or condition not met with selector: {selector}"
        except Exception as e:
            logger.error(f"Error finding element {selector}: {str(e)}")
            return None, str(e)
    
    def _handle_function_params(self, function_params: Optional[list], 
                              function_name: str, 
                              param_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract parameters from nested function_params format.
        
        Args:
            function_params (list, optional): Nested function parameters
            function_name (str): The function name to look for
            param_mapping (Dict[str, str]): Mapping of parameter names
            
        Returns:
            Dict[str, Any]: Extracted parameters
        """
        result = {}
        if not function_params:
            return result
            
        for param in function_params:
            fn_name = param.get("function_name", "")
            if fn_name == function_name or fn_name in param_mapping.get("alt_names", []):
                args = param.get("function_args", {})
                for param_name, result_name in param_mapping.items():
                    if param_name == "alt_names":
                        continue
                    if param_name in args:
                        result[result_name] = args[param_name]
                break
        
        return result
    
    # Original methods with improved implementation using the helper methods
    
    def initialize_browser(self, function_params: list = None, continue_after_tool_call: bool = None) -> Dict[str, Any]:
        """
        Start or restart a browser session. Must be called before any other browser operations.
        
        This function supports multiple parameter styles:
        1. Standard style: no parameters
        2. Nested function_params style:
           function_params=[{"function_name": "initialize_browser", "function_args": {}}]
           
        Args:
            function_params (list, optional): Nested function parameters
            continue_after_tool_call (bool, optional): Whether to continue after the tool call
        
        Returns:
            Dict[str, Any]: Status information about the browser initialization
        """
        try:
            if self.driver:
                # Close any existing session
                try:
                    self.driver.quit()
                except Exception as e:
                    logger.warning(f"Error closing existing browser session: {str(e)}")
                    
            options = None
            if self.browser_type == "chrome":
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                options = Options()
                if self.headless:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                
                # Create service with Chrome executable path
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            elif self.browser_type == "firefox":
                from selenium.webdriver.firefox.options import Options
                options = Options()
                if self.headless:
                    options.add_argument("--headless")
                self.driver = webdriver.Firefox(options=options)
            elif self.browser_type == "safari":
                self.driver = webdriver.Safari()
            elif self.browser_type == "edge":
                from selenium.webdriver.edge.options import Options
                options = Options()
                if self.headless:
                    options.add_argument("--headless")
                self.driver = webdriver.Edge(options=options)
            else:
                return {"status": "error", "message": f"Unsupported browser type: {self.browser_type}"}
            
            self.driver.set_page_load_timeout(self.timeout)
            return {"status": "success", "message": f"Browser {self.browser_type} initialized successfully"}
        except Exception as e:
            logger.error(f"Error initializing browser: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def navigate_to_url(self, url: str = None, timeout: int = None, 
                       function_params: list = None, continue_after_tool_call: bool = None) -> Dict[str, Any]:
        """
        Navigate to a URL and capture a snapshot of the page. This provides element references used for interaction.
        
        This function supports multiple parameter styles:
        1. Standard style: url parameter
        2. Nested function_params style:
           function_params=[{"function_name": "navigate_to_url", "function_args": {"url": "..."}}]
        
        Args:
            url (str, optional): The complete URL (with https://) to navigate to
            timeout (int, optional): Custom timeout in seconds (default: 10)
            function_params (list, optional): Nested function parameters
            continue_after_tool_call (bool, optional): Whether to continue after the tool call
            
        Returns:
            Dict[str, Any]: Information about the navigation result and page snapshot
        """
        if not self.driver:
            init_result = self.initialize_browser()
            if init_result["status"] == "error":
                return init_result
        
        # Handle nested function_params format
        if function_params and not url:
            params = self._handle_function_params(
                function_params, 
                "navigate_to_url", 
                {"url": "url", "timeout": "timeout", "alt_names": ["browser_navigate"]}
            )
            url = params.get("url")
            timeout = params.get("timeout", timeout)
                    
        if not url:
            return {"status": "error", "message": "URL parameter is required"}
                
        timeout = timeout or self.timeout
        try:
            self.driver.get(url)
            
            # Wait for page to load
            page_loaded = self._wait_for_page_load(timeout)
            if not page_loaded:
                logger.warning(f"Page load timeout for URL: {url}, but continuing with snapshot")
            
            # Automatically take a snapshot of the page
            snapshot_result = self.browser_snapshot()
            
            if snapshot_result["status"] == "success":
                return {
                    "status": "success", 
                    "url": url,
                    "title": self.driver.title,
                    "current_url": self.driver.current_url,
                    "snapshot": {
                        "interactive_elements": snapshot_result.get("interactive_elements", [])
                    }
                }
            else:
                # Return navigation success but note snapshot failure
                return {
                    "status": "partial_success", 
                    "url": url,
                    "title": self.driver.title,
                    "current_url": self.driver.current_url,
                    "snapshot_error": snapshot_result.get("message", "Unknown error capturing snapshot")
                }
                
        except TimeoutException:
            return {"status": "timeout", "message": f"Timed out loading URL: {url}"}
        except Exception as e:
            logger.error(f"Error navigating to URL {url}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def find_element(self, selector: str, selector_type: str = "css", timeout: int = None) -> Dict[str, Any]:
        """
        Find an element on the current page and return information about it.
        
        Args:
            selector (str): The selector to find the element
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Information about the found element
        """
        # Check if browser is initialized
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
            
        timeout = timeout or self.timeout
        
        # Get the selector type
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):  # Error response
            return by_type
            
        try:
            # Find the element
            element, error = self._find_element_with_wait(
                by_type, selector, timeout, EC.presence_of_element_located
            )
            if error:
                return {"status": "not_found", "message": f"Element not found with {selector_type}: {selector}"}
            
            # Extract element properties
            element_properties = self._extract_element_properties(element, selector)
            
            return {
                "status": "success",
                "element": element_properties
            }
        except Exception as e:
            logger.error(f"Error finding element {selector}: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _extract_element_properties(self, element, selector: str) -> Dict[str, Any]:
        """
        Extract common properties from a WebElement.
        
        Args:
            element: The Selenium WebElement
            selector (str): The selector used to find the element (for error messages)
            
        Returns:
            Dict[str, Any]: Element properties
        """
        element_properties = {
            "text": element.text,
            "tag_name": element.tag_name,
            "is_displayed": element.is_displayed(),
            "is_enabled": element.is_enabled(),
        }
        
        # Get attributes safely
        for attr in ["href", "id", "class"]:
            try:
                value = element.get_attribute(attr)
                if value:
                    element_properties[attr] = value
            except StaleElementReferenceException:
                logger.warning(f"Element became stale when trying to get {attr} attribute for {selector}")
            except Exception as e:
                logger.warning(f"Could not get {attr} attribute for {selector}: {str(e)}")
                
        return element_properties
    
    def find_multiple_elements(self, selector: str, selector_type: str = "css", timeout: int = None) -> Dict[str, Any]:
        """
        Find multiple elements on the current page and return information about them.
        
        Args:
            selector (str): The selector to find the elements
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Information about the found elements
        """
        # Check if browser is initialized
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
            
        timeout = timeout or self.timeout
        
        # Get the selector type
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):  # Error response
            return by_type
            
        try:
            # First check if at least one element exists
            element, error = self._find_element_with_wait(
                by_type, selector, timeout, EC.presence_of_element_located
            )
            if error:
                return {"status": "not_found", "message": f"No elements found with {selector_type}: {selector}"}
            
            # Then get all matching elements
            elements = self.driver.find_elements(by_type, selector)
            
            # Extract element properties
            elements_properties = []
            for idx, element in enumerate(elements):
                try:
                    element_properties = self._extract_element_properties(element, f"{selector}[{idx}]")
                    element_properties["index"] = idx
                    elements_properties.append(element_properties)
                except StaleElementReferenceException:
                    logger.warning(f"Element {idx} became stale while extracting properties")
                except Exception as e:
                    logger.warning(f"Error extracting properties for element {idx}: {str(e)}")
                
            return {
                "status": "success",
                "count": len(elements_properties),
                "elements": elements_properties
            }
        except Exception as e:
            logger.error(f"Error finding elements {selector}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def click_element(self, selector: str, selector_type: str = "css", timeout: int = None) -> Dict[str, Any]:
        """
        Click on an element on the current page.
        
        Args:
            selector (str): The selector to find the element
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Result of the click operation
        """
        # Check if browser is initialized
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
            
        timeout = timeout or self.timeout
        
        # Get the selector type
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):  # Error response
            return by_type
                
        try:
            # Find and click the element
            element, error = self._find_element_with_wait(
                by_type, selector, timeout, EC.element_to_be_clickable
            )
            if error:
                return {"status": "not_found", "message": f"Element not clickable with {selector_type}: {selector}"}
                
            element.click()
            
            # Wait for page to load after click
            page_loaded = self._wait_for_page_load(timeout)
            if not page_loaded:
                return {
                    "status": "partial_success",
                    "message": "Element clicked, but page load timed out",
                    "selector": selector,
                    "current_url": self.driver.current_url
                }
            
            return {
                "status": "success",
                "message": f"Clicked element with {selector_type}: {selector}",
                "current_url": self.driver.current_url,
                "title": self.driver.title
            }
        except Exception as e:
            logger.error(f"Error clicking element {selector}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def input_text(self, element: str = None, ref: str = None, text: str = None, 
                   submit: bool = False, slowly: bool = True,
                   function_params: list = None, continue_after_tool_call: bool = None) -> Dict[str, Any]:
        """
        Type text into a form field, search box, or other input element using a reference ID from a snapshot.
        
        This function only works with element references from a snapshot. Use browser_snapshot
        or navigate_to_url first to capture the page elements.
        
        This function supports multiple parameter styles:
        1. Standard style: element (description), ref (element ID), text
        2. Nested function_params style:
           function_params=[{"function_name": "browser_type", "function_args": {...}}]
        
        Args:
            element (str, optional): Human-readable description of the element (e.g., 'Search field', 'Username input')
            ref (str, optional): Element ID from the page snapshot (e.g., 'e0', 'e1', 'e2') - NOT a CSS selector
            text (str, optional): Text to input into the element
            submit (bool): Press Enter after typing to submit forms (default: false)
            slowly (bool): Type one character at a time to trigger JS events (default: true)
            function_params (list, optional): Nested function parameters
            continue_after_tool_call (bool, optional): Whether to continue after the tool call
            
        Returns:
            Dict[str, Any]: Result of the text input operation
        """
        # Check if browser is initialized
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
            
        # Handle nested function_params format
        if function_params:
            params = self._handle_function_params(
                function_params, 
                "input_text", 
                {"element": "element", "ref": "ref", "text": "text", 
                 "submit": "submit", "slowly": "slowly", "alt_names": ["browser_type"]}
            )
            element = params.get("element", element)
            ref = params.get("ref", ref)
            text = params.get("text", text)
            if "submit" in params:
                submit = params["submit"]
            if "slowly" in params:
                slowly = params["slowly"]
        
        if not ref or not text:
            return {"status": "error", "message": "Both ref and text parameters are required"}
           
        # Parse the reference
        selector_type, selector, error = self._parse_element_reference(ref)
        if error:
            return {"status": "error", "message": error}
                
        # Use a human-readable description or the ref ID if not provided
        element_desc = element or ref
            
        # Get the selector type
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):  # Error response
            return by_type
        
        try:
            # Find the element
            web_element, error = self._find_element_with_wait(
                by_type, selector, self.timeout, EC.element_to_be_clickable
            )
            if error:
                return {"status": "not_found", "message": f"Element not found: {element_desc}"}
            
            # Clear existing content
            web_element.clear()
            
            # Type text
            if slowly:
                # Type character by character
                for char in text:
                    web_element.send_keys(char)
                    # Small delay between keypresses
                    time.sleep(0.05)
            else:
                # Type all at once
                web_element.send_keys(text)
            
            # Submit if requested
            if submit:
                from selenium.webdriver.common.keys import Keys
                web_element.send_keys(Keys.ENTER)
                
                # Wait for page to load after submission
                page_loaded = self._wait_for_page_load(self.timeout)
                if not page_loaded:
                    # Take a new snapshot after submitting
                    self.browser_snapshot()
                    return {
                        "status": "partial_success",
                        "message": "Text entered and submitted, but page load timed out",
                        "element": element_desc,
                        "text": text
                    }
                
                # Take a new snapshot after submitting
                snapshot_result = self.browser_snapshot()
                if snapshot_result["status"] != "success":
                    logger.warning(f"Failed to capture snapshot after form submission: {snapshot_result.get('message')}")
            
            return {
                "status": "success",
                "message": f"Successfully input text into {element_desc}" + 
                           (" and submitted" if submit else ""),
                "element": element_desc,
                "text": text
            }
            
        except TimeoutException:
            return {"status": "not_found", "message": f"Element not found: {element_desc}"}
        except Exception as e:
            logger.error(f"Error inputting text to element {element_desc}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_page_content(self) -> Dict[str, Any]:
        """
        Get the current page title, URL and body content.
        
        Returns:
            Dict[str, Any]: Information about the current page
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        try:
            # Get title and URL
            title = self.driver.title
            current_url = self.driver.current_url
            
            # Extract only the body content using JavaScript
            body_content = self.driver.execute_script("""
                var body = document.body;
                return body ? body.outerHTML : "";
            """)
            
            # Get a summary of key elements for easier navigation
            element_summary = self.driver.execute_script("""
                // Get common interactive elements
                var summary = {
                    links: [],
                    buttons: [],
                    inputs: [],
                    forms: []
                };
                
                // Get links
                var links = document.querySelectorAll('a');
                for (var i = 0; i < Math.min(links.length, 20); i++) {
                    var link = links[i];
                    summary.links.push({
                        text: link.textContent.trim().substring(0, 50),
                        href: link.getAttribute('href'),
                        id: link.id,
                        class: link.className
                    });
                }
                
                // Get buttons
                var buttons = document.querySelectorAll('button, input[type="button"], input[type="submit"]');
                for (var i = 0; i < Math.min(buttons.length, 20); i++) {
                    var button = buttons[i];
                    summary.buttons.push({
                        text: button.textContent ? button.textContent.trim().substring(0, 50) : button.value,
                        id: button.id,
                        class: button.className,
                        type: button.type
                    });
                }
                
                // Get inputs
                var inputs = document.querySelectorAll('input:not([type="button"]):not([type="submit"]), textarea, select');
                for (var i = 0; i < Math.min(inputs.length, 20); i++) {
                    var input = inputs[i];
                    summary.inputs.push({
                        type: input.type,
                        name: input.name,
                        id: input.id,
                        placeholder: input.placeholder
                    });
                }
                
                // Get forms
                var forms = document.querySelectorAll('form');
                for (var i = 0; i < Math.min(forms.length, 10); i++) {
                    var form = forms[i];
                    summary.forms.push({
                        id: form.id,
                        action: form.action,
                        method: form.method
                    });
                }
                
                return summary;
            """)
            
            return {
                "status": "success",
                "title": title,
                "url": current_url,
                "body_content": body_content,
                "element_summary": element_summary
            }
        except Exception as e:
            logger.error(f"Error getting page content: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def switch_to_frame(self, frame_reference: str, reference_type: str = "index") -> Dict[str, Any]:
        """
        Switch to a frame on the page.
        
        Args:
            frame_reference (str): Reference to the frame (index, name, or ID)
            reference_type (str): Type of reference ('index', 'name', 'id', 'element')
            
        Returns:
            Dict[str, Any]: Result of the frame switch operation
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        try:
            if reference_type == "index":
                try:
                    index = int(frame_reference)
                    self.driver.switch_to.frame(index)
                except ValueError:
                    return {"status": "error", "message": f"Invalid frame index: {frame_reference}"}
            elif reference_type == "name" or reference_type == "id":
                self.driver.switch_to.frame(frame_reference)
            elif reference_type == "element":
                # First find the element
                selector_parts = frame_reference.split(":", 1)
                if len(selector_parts) != 2:
                    return {"status": "error", "message": "Element reference must be in format 'selector_type:selector'"}
                
                selector_type, selector = selector_parts
                element_result = self.find_element(selector, selector_type)
                
                if element_result["status"] != "success":
                    return {"status": "error", "message": f"Could not find frame element: {element_result['message']}"}
                
                # Get the actual WebElement (not just the properties)
                selector_map = {
                    "css": By.CSS_SELECTOR,
                    "xpath": By.XPATH,
                    "id": By.ID,
                    "class": By.CLASS_NAME,
                    "name": By.NAME,
                    "tag": By.TAG_NAME,
                }
                by_type = selector_map.get(selector_type.lower())
                element = self.driver.find_element(by_type, selector)
                self.driver.switch_to.frame(element)
            else:
                return {"status": "error", "message": f"Invalid reference type: {reference_type}"}
                
            return {
                "status": "success",
                "message": f"Switched to frame using {reference_type}: {frame_reference}"
            }
        except Exception as e:
            logger.error(f"Error switching to frame {frame_reference}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def switch_to_window(self, window_reference: str, reference_type: str = "index") -> Dict[str, Any]:
        """
        Switch to a window or tab.
        
        Args:
            window_reference (str): Reference to the window (index, handle, or title)
            reference_type (str): Type of reference ('index', 'handle', 'title')
            
        Returns:
            Dict[str, Any]: Result of the window switch operation
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        try:
            window_handles = self.driver.window_handles
            
            if not window_handles:
                return {"status": "error", "message": "No window handles available"}
                
            if reference_type == "index":
                try:
                    index = int(window_reference)
                    if index < 0 or index >= len(window_handles):
                        return {"status": "error", "message": f"Window index out of range: {index}"}
                    
                    self.driver.switch_to.window(window_handles[index])
                except ValueError:
                    return {"status": "error", "message": f"Invalid window index: {window_reference}"}
            elif reference_type == "handle":
                if window_reference not in window_handles:
                    return {"status": "error", "message": f"Window handle not found: {window_reference}"}
                
                self.driver.switch_to.window(window_reference)
            elif reference_type == "title":
                current_handle = self.driver.current_window_handle
                window_found = False
                
                for handle in window_handles:
                    try:
                        self.driver.switch_to.window(handle)
                        if self.driver.title == window_reference:
                            window_found = True
                            break
                    except Exception:
                        pass
                
                if not window_found:
                    # Switch back to the original window
                    self.driver.switch_to.window(current_handle)
                    return {"status": "error", "message": f"No window with title '{window_reference}' found"}
            else:
                return {"status": "error", "message": f"Invalid reference type: {reference_type}"}
                
            return {
                "status": "success",
                "message": f"Switched to window using {reference_type}: {window_reference}",
                "title": self.driver.title,
                "url": self.driver.current_url
            }
        except Exception as e:
            logger.error(f"Error switching to window {window_reference}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_tool_descriptions(self) -> List[str]:
        """
        Returns a brief description of the browser tool.
        
        Returns:
            List[str]: Tool descriptions
        """
        return [
            "Interact with web browsers to navigate, click, input text, and extract information from web pages."
        ]

    def select_dropdown_option(self, select_selector: str, 
                              option_value: str,
                              select_by: str = "value",
                              selector_type: str = "css") -> Dict[str, Any]:
        """
        Select an option from a dropdown
        select_by can be 'value', 'text', or 'index'
        
        Args:
            select_selector (str): The selector to find the dropdown element
            option_value (str): The value to select (depends on select_by)
            select_by (str): Method to select by ('value', 'text', 'index')
            selector_type (str): Type of selector for the dropdown
            
        Returns:
            Dict[str, Any]: Result of the selection operation
        """
        # Check if browser is initialized
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
        
        try:
            from selenium.webdriver.support.ui import Select
            
            # Get the selector type
            by_type = self._get_selector_by_type(selector_type)
            if isinstance(by_type, dict):  # Error response
                return by_type
            
            # Find the dropdown element
            element, error = self._find_element_with_wait(
                by_type, select_selector, self.timeout, EC.presence_of_element_located
            )
            if error:
                return {"status": "not_found", "message": f"Dropdown element not found with {selector_type}: {select_selector}"}
            
            # Create select object
            select = Select(element)
            
            # Select based on method
            if select_by.lower() == "value":
                select.select_by_value(option_value)
            elif select_by.lower() == "text":
                select.select_by_visible_text(option_value)
            elif select_by.lower() == "index":
                try:
                    select.select_by_index(int(option_value))
                except ValueError:
                    return {"status": "error", "message": f"Invalid index value: {option_value}. Must be an integer."}
            else:
                return {"status": "error", "message": f"Invalid select_by option: {select_by}"}
            
            return {"status": "success", "message": f"Selected option with {select_by}: {option_value}"}
        except Exception as e:
            logger.error(f"Error selecting dropdown option: {str(e)}")
            return {"status": "error", "message": str(e)}

    def close_browser(self) -> Dict[str, Any]:
        """
        Close the browser and end the session. Call this when you're done to free resources.
        
        Returns:
            Dict[str, Any]: Status of the browser closure
        """
        if not self.driver:
            return {"status": "success", "message": "Browser already closed"}
            
        try:
            self.driver.quit()
            self.driver = None
            return {"status": "success", "message": "Browser closed successfully"}
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def browser_click(self, element: str = None, ref: str = None, 
                     function_params: list = None, continue_after_tool_call: bool = None) -> Dict[str, Any]:
        """
        Click on a button, link, or other clickable element using a reference ID from a snapshot.
        
        This function only works with element references from a snapshot. You MUST call browser_snapshot
        or navigate_to_url first to capture the page elements.
        
        Common usage pattern:
        1. First get a snapshot: browser_snapshot() or navigate_to_url()
        2. Find the element reference (e.g. 'e0', 'e1') from the snapshot's interactive_elements
        3. Use that reference to click: browser_click(element='Login button', ref='e0')
        
        This function supports multiple parameter styles:
        1. Standard style: element (description), ref (element ID)
        2. Nested function_params style:
           function_params=[{"function_name": "browser_click", "function_args": {...}}]
        
        Args:
            element (str, optional): Human-readable description of what you're clicking (e.g., 'Login button', 'Next page link')
            ref (str, optional): Element ID from the page snapshot (e.g., 'e0', 'e1', 'e2') - NOT a CSS selector
            function_params (list, optional): Nested function parameters
            continue_after_tool_call (bool, optional): Whether to continue after the tool call
            
        Returns:
            Dict[str, Any]: Result of the click operation with detailed feedback
        """
        # Check if browser is initialized
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
            
        # Handle nested function_params format
        if function_params and not ref:
            params = self._handle_function_params(
                function_params, 
                "browser_click", 
                {"element": "element", "ref": "ref"}
            )
            element = params.get("element", element)
            ref = params.get("ref", ref)
        
        # Validate required parameters
        if not ref:
            return {
                "status": "error", 
                "message": "Element reference (ref) parameter is required. You must first call browser_snapshot() or navigate_to_url() to get element references.",
                "required_steps": [
                    "1. Call browser_snapshot() or navigate_to_url() to get page elements",
                    "2. Find the element reference (e.g. 'e0') in the response's interactive_elements",
                    "3. Use that reference to click: browser_click(element='Button name', ref='e0')"
                ]
            }
            
        # Check if we have any stored references
        if not self.element_references:
            return {
                "status": "error",
                "message": "No element references found. You must first capture a page snapshot.",
                "required_steps": [
                    "1. Call browser_snapshot() or navigate_to_url() to capture the page state",
                    "2. Use the element references returned in the snapshot"
                ]
            }
            
        # Parse the reference
        selector_type, selector, error = self._parse_element_reference(ref)
        if error:
            return {
                "status": "error", 
                "message": error,
                "help": "Make sure you're using a valid element reference from a recent snapshot"
            }
                
        # Use a human-readable description or the ref ID if not provided
        element_desc = element or ref
        
        # Get the selector type
        by_type = self._get_selector_by_type(selector_type)
        if isinstance(by_type, dict):  # Error response
            return by_type
                
        try:
            # First check if element exists at all
            try:
                element_exists = self.driver.find_element(by_type, selector)
            except Exception:
                return {
                    "status": "not_found",
                    "message": f"Element not found: {element_desc}",
                    "suggestion": "The page may have changed. Try getting a new snapshot with browser_snapshot()"
                }
            
            # Then check if it's clickable
            web_element, error = self._find_element_with_wait(
                by_type, selector, self.timeout, EC.element_to_be_clickable
            )
            if error:
                # Element exists but isn't clickable - get more details
                try:
                    is_visible = element_exists.is_displayed()
                    is_enabled = element_exists.is_enabled()
                    element_tag = element_exists.tag_name
                    element_classes = element_exists.get_attribute("class")
                    
                    return {
                        "status": "not_clickable",
                        "message": f"Element found but not clickable: {element_desc}",
                        "element_state": {
                            "visible": is_visible,
                            "enabled": is_enabled,
                            "tag": element_tag,
                            "classes": element_classes
                        },
                        "suggestion": "The element might be disabled, hidden, or covered by another element"
                    }
                except Exception:
                    return {
                        "status": "not_clickable",
                        "message": f"Element found but not clickable: {element_desc}",
                        "suggestion": "The element might be disabled, hidden, or covered by another element"
                    }
                
            # Try to click the element
            web_element.click()
            
            # Wait for page to load after click
            page_loaded = self._wait_for_page_load(self.timeout)
            if not page_loaded:
                # Take a snapshot anyway even if page load times out
                snapshot_result = self.browser_snapshot()
                return {
                    "status": "partial_success",
                    "message": "Element clicked, but page load timed out",
                    "element": element_desc,
                    "current_url": self.driver.current_url,
                    "snapshot": snapshot_result if snapshot_result["status"] == "success" else None,
                    "suggestion": "The page might still be loading. You may want to wait and take another snapshot."
                }
            
            # Take a new snapshot after the click
            snapshot_result = self.browser_snapshot()
            
            if snapshot_result["status"] == "success":
                return {
                    "status": "success",
                    "message": f"Successfully clicked on {element_desc}",
                    "element": element_desc,
                    "current_url": self.driver.current_url,
                    "title": self.driver.title,
                    "snapshot": {
                        "interactive_elements": snapshot_result.get("interactive_elements", [])
                    }
                }
            else:
                # Return click success but note snapshot failure
                return {
                    "status": "success",
                    "message": f"Successfully clicked on {element_desc} but snapshot failed",
                    "element": element_desc,
                    "current_url": self.driver.current_url,
                    "title": self.driver.title,
                    "snapshot_error": snapshot_result.get("message", "Unknown error capturing snapshot"),
                    "suggestion": "You may want to take another snapshot with browser_snapshot()"
                }
                
        except TimeoutException:
            return {
                "status": "timeout",
                "message": f"Timed out waiting for element to be clickable: {element_desc}",
                "suggestion": "The element might be taking too long to load or become clickable"
            }
        except Exception as e:
            logger.error(f"Error clicking element: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "element": element_desc,
                "suggestion": "Try getting a new snapshot of the page with browser_snapshot()"
            }
    
    def _classify_element_interactivity(self, element_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an element's interactivity based on its properties.
        This method contains all rules for determining if an element is interactive or editable.
        
        Args:
            element_data (Dict[str, Any]): Element data including properties, attributes, etc.
            
        Returns:
            Dict[str, Any]: Element data with interactivity classifications added
        """
        # Start with basic properties
        element_data["interactable"] = False
        element_data["editable"] = False
        
        # Get commonly used properties
        tag_name = element_data.get("properties", {}).get("tag", "").upper()
        role = element_data.get("attributes", {}).get("role", "").lower()
        
        # Check if element is disabled
        is_disabled = (
            element_data.get("attributes", {}).get("disabled") is not None or
            element_data.get("attributes", {}).get("aria-disabled") == "true" or
            element_data.get("attributes", {}).get("aria-hidden") == "true"
        )
        
        # Check visibility
        is_visible = element_data.get("visible", True)
        
        if not is_disabled and is_visible:
            # Interactive Tags Check
            interactive_tags = {
                'A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA',
                'DETAILS', 'AUDIO', 'VIDEO', 'IFRAME', 'EMBED',
                'OBJECT', 'SUMMARY', 'MENU'
            }
            
            # Interactive Roles Check
            interactive_roles = {
                'button', 'link', 'checkbox', 'menuitem',
                'menuitemcheckbox', 'menuitemradio', 'option',
                'radio', 'searchbox', 'slider', 'spinbutton',
                'switch', 'tab', 'textbox', 'combobox',
                'listbox', 'menu', 'menubar', 'radiogroup',
                'tablist', 'toolbar', 'tree', 'treegrid'
            }
            
            # Check for interactive attributes
            has_interactive_attrs = any([
                element_data.get("attributes", {}).get(attr) is not None
                for attr in ['onclick', 'onkeydown', 'onkeyup', 'onmousedown', 
                           'onmouseup', 'tabindex']
            ])
            
            # Determine interactability
            element_data["interactable"] = (
                tag_name in interactive_tags or
                role in interactive_roles or
                has_interactive_attrs
            )
            
            # Determine editability
            editable_input_types = {'text', 'search', 'email', 'number', 'tel', 
                                  'url', 'password'}
            editable_roles = {'textbox', 'searchbox', 'spinbutton'}
            
            element_data["editable"] = (
                # Standard input fields
                (tag_name == 'INPUT' and 
                 element_data.get("attributes", {}).get("type", "text").lower() in editable_input_types) or
                tag_name == 'TEXTAREA' or
                # Rich text editing
                element_data.get("attributes", {}).get("contenteditable") == "true" or
                # Explicit input roles
                role in editable_roles
            )
        
        return element_data

    def _process_accessibility_tree(self, accessibility_tree):
        """
        Process the accessibility tree to extract all elements and store their references.
        
        This method processes all elements in the page structure, assigns unique IDs,
        and stores their selectors for later interaction.
        
        Args:
            accessibility_tree (dict): The accessibility tree from JavaScript
            
        Returns:
            list: A list of all elements with their IDs and properties
        """
        all_elements = []
        
        # Function to extract all elements and store references
        def extract_elements(node, path="", index=0):
            if not node:
                return index
                
            current_path = path + "/" + (node.get("name") or node.get("role") or "element")
            
            # Generate a unique element ID for all elements
            element_id = f"e{index}"
            
            element_info = {
                "id": element_id,
                "description": current_path.strip("/"),
                "purpose": node.get("semantic_info", {}).get("purpose", ""),
                "label": node.get("semantic_info", {}).get("label", ""),
                "category": node.get("semantic_info", {}).get("category", ""),
                "isPrimary": node.get("semantic_info", {}).get("isPrimary", False),
                "visible": node.get("visible", True),
                "properties": node.get("properties", {}),
                "attributes": node.get("attributes", {})
            }
            
            # Store the first selector as the actual reference for this element
            if "all_refs" in node:
                self.element_references[element_id] = node["all_refs"][0]
                
            # Classify element interactivity
            element_info = self._classify_element_interactivity(element_info)
            
            all_elements.append(element_info)
            index += 1
            
            # Process children
            for child in node.get("children", []):
                index = extract_elements(child, current_path, index)
            
            return index
        
        # Extract all elements
        extract_elements(accessibility_tree)
        
        return all_elements

    def browser_snapshot(self, function_params: list = None, continue_after_tool_call: bool = None) -> Dict[str, Any]:
        """
        Capture a fresh snapshot of the current page with all interactive elements. 
        Use after page state changes not caused by navigation or clicking.
        
        This function supports multiple parameter styles:
        1. Standard style: no parameters
        2. Nested function_params style:
           function_params=[{"function_name": "browser_snapshot", "function_args": {}}]
        
        Args:
            function_params (list, optional): Nested function parameters
            continue_after_tool_call (bool, optional): Whether to continue after the tool call
            
        Returns:
            Dict[str, Any]: The accessibility snapshot of the page with interactive elements
        """
        # Check if browser is initialized
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
            
        try:
            # Get basic page info
            title = self.driver.title
            current_url = self.driver.current_url
            
            # Get accessibility tree using JavaScript
            accessibility_tree = self.driver.execute_script("""
                function getAccessibilityTree(node, depth = 0, maxDepth = 10) {
                    if (!node || depth > maxDepth) return null;
                    
                    let result = {
                        role: node.role || node.tagName,
                        name: node.name || '',
                        type: node.type || '',
                        value: node.value || '',
                        description: node.description || '',
                        properties: {},
                        visible: isElementVisible(node)
                    };
                    
                    // Helper function for element visibility
                    function isElementVisible(element) {
                        if (!element.getBoundingClientRect) return true;
                        const style = window.getComputedStyle(element);
                        const rect = element.getBoundingClientRect();
                        
                        // Check basic visibility
                        const isVisible = style.display !== 'none' && 
                                        style.visibility !== 'hidden' && 
                                        style.opacity !== '0' &&
                                        rect.width > 0 && 
                                        rect.height > 0;
                                        
                        // Check if element is in viewport
                        const isInViewport = rect.top >= 0 &&
                                           rect.left >= 0 &&
                                           rect.bottom <= window.innerHeight &&
                                           rect.right <= window.innerWidth;
                                           
                        return isVisible && isInViewport;
                    }
                    
                    // Add text content
                    if (node.textContent) {
                        result.text_content = node.textContent.trim();
                    }

                    // Add identifier properties for references
                    if (node.id) result.properties.id = node.id;
                    if (node.className) result.properties.class = node.className;
                    if (node.tagName) result.properties.tag = node.tagName.toLowerCase();
                    
                    // Add attributes
                    if (node.attributes) {
                        result.attributes = {};
                        for (let attr of node.attributes) {
                            result.attributes[attr.name] = attr.value;
                        }
                    }

                    // Add custom ref property that combines selector types
                    let refs = [];
                    // Store all possible selectors, but don't use them as primary ref
                    if (node.id) refs.push(`id:${node.id}`);
                    if (node.className && typeof node.className === 'string') 
                        refs.push(`class:${node.className}`);
                    if (node.tagName) refs.push(`tag:${node.tagName.toLowerCase()}`);
                    
                    // For inputs, add name attribute
                    if (node.getAttribute && node.getAttribute('name')) {
                        result.properties.name = node.getAttribute('name');
                        refs.push(`name:${node.getAttribute('name')}`);
                    }
                    
                    // Create XPath and CSS selectors
                    try {
                        // CSS selector
                        let cssPath = getCssPath(node);
                        if (cssPath) refs.push(`css:${cssPath}`);
                        
                        // XPath
                        let xpath = getXPath(node);
                        if (xpath) refs.push(`xpath:${xpath}`);
                    } catch (e) {}
                    
                    // Store all refs but don't set primary ref here
                    if (refs.length > 0) {
                        result.all_refs = refs;
                    }

                    // Add semantic information about the element
                    result.semantic_info = {
                        // What the element represents
                        purpose: (function() {
                            if (node.tagName === 'INPUT') {
                                if (node.type === 'submit') return 'submit button';
                                if (node.type === 'search') return 'search box';
                                if (node.type === 'text') return 'text input';
                                return `${node.type || 'text'} input`;
                            }
                            if (node.tagName === 'BUTTON') return 'button';
                            if (node.tagName === 'A') return 'link';
                            if (node.tagName === 'SELECT') return 'dropdown';
                            if (node.tagName === 'TEXTAREA') return 'text area';
                            if (node.getAttribute('role')) return node.getAttribute('role');
                            return 'interactive element';
                        })(),
                        
                        // The visible or accessible text
                        label: (function() {
                            return node.getAttribute('aria-label') ||
                                   node.getAttribute('title') ||
                                   node.getAttribute('placeholder') ||
                                   node.getAttribute('alt') ||
                                   (node.tagName === 'INPUT' ? node.value : node.textContent.trim());
                        })(),
                        
                        // Is this a primary action?
                        isPrimary: !!(
                            node.classList.contains('primary') ||
                            node.getAttribute('aria-label')?.toLowerCase().includes('search') ||
                            node.getAttribute('title')?.toLowerCase().includes('search') ||
                            node.type === 'search' ||
                            node.getAttribute('role') === 'main' ||
                            node.id?.toLowerCase().includes('main') ||
                            node.classList.contains('main')
                        ),
                        
                        // Basic category
                        category: (function() {
                            if (node.type === 'search' || 
                                node.getAttribute('role') === 'searchbox') return 'search';
                            if (node.type === 'submit' || 
                                node.tagName === 'BUTTON' ||
                                node.getAttribute('role') === 'button') return 'action';
                            if (node.tagName === 'A' ||
                                node.getAttribute('role') === 'link') return 'navigation';
                            if (node.tagName === 'INPUT' || 
                                node.tagName === 'TEXTAREA' ||
                                node.getAttribute('role') === 'textbox') return 'input';
                            if (node.tagName === 'SELECT' ||
                                ['listbox', 'combobox'].includes(node.getAttribute('role'))) return 'selection';
                            return 'interactive';
                        })()
                    };
                    
                    // Process children
                    result.children = [];
                    if (node.children) {
                        for (let i = 0; i < node.children.length; i++) {
                            const childTree = getAccessibilityTree(node.children[i], depth + 1, maxDepth);
                            if (childTree) {
                                result.children.push(childTree);
                            }
                        }
                    }
                    
                    return result;
                }
                
                return getAccessibilityTree(document.body);
            """)
            
            # Process the accessibility tree and extract all elements
            all_elements = self._process_accessibility_tree(accessibility_tree)
            page_content = html2text.html2text(self.driver.page_source)
            
            return {
                "status": "success",
                "title": title,
                "url": current_url,
                "accessibility_tree": accessibility_tree,
                "page_content": page_content,
                "interactive_elements": [e for e in all_elements if e.get("interactable") or e.get("editable")]
            }
            
        except Exception as e:
            logger.error(f"Error generating accessibility snapshot: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def browser_console_messages(self, function_params: list = None, continue_after_tool_call: bool = None) -> Dict[str, Any]:
        """
        Retrieve JavaScript console messages (logs, warnings, errors) from the browser for debugging.
        
        This function supports multiple parameter styles:
        1. Standard style: no parameters
        2. Nested function_params style:
           function_params=[{"function_name": "browser_console_messages", "function_args": {}}]
        
        Args:
            function_params (list, optional): Nested function parameters
            continue_after_tool_call (bool, optional): Whether to continue after the tool call
            
        Returns:
            Dict[str, Any]: The console messages including logs, warnings and errors
        """
        # Check if browser is initialized
        driver_check = self._check_driver_initialized()
        if driver_check:
            return driver_check
            
        try:
            logs = self._collect_browser_logs()
            
            return {
                "status": "success",
                "console_messages": logs
            }
            
        except Exception as e:
            logger.error(f"Error retrieving console messages: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _collect_browser_logs(self) -> List[Dict[str, Any]]:
        """
        Collect logs from both the browser driver and JavaScript console.
        
        Returns:
            List[Dict[str, Any]]: Combined logs from both sources
        """
        logs = []
        
        # Try to get browser logs if available
        try:
            browser_logs = self.driver.get_log('browser')
            for log in browser_logs:
                # Map browser log levels to standard levels
                level = log.get("level", "").upper()
                if level == "SEVERE":
                    level = "ERROR"
                elif level == "INFO":
                    level = "LOG"
                
                logs.append({
                    "level": level,
                    "message": log.get("message", ""),
                    "timestamp": log.get("timestamp", "")
                })
        except Exception as log_error:
            # Browser logs might not be available in all drivers/configurations
            logs.append({
                "level": "WARNING",
                "message": f"Could not retrieve browser logs: {str(log_error)}",
                "timestamp": ""
            })
        
        # Try to execute JavaScript to get console message history
        try:
            # Add script to capture console messages if not already added
            self.driver.execute_script("""
                if (!window._consoleLogs) {
                    window._consoleLogs = [];
                    
                    // Store original console methods
                    const originalConsole = {
                        log: console.log,
                        info: console.info,
                        warn: console.warn,
                        error: console.error,
                        debug: console.debug
                    };
                    
                    // Helper function to add message with proper level
                    function addMessage(level, args) {
                        window._consoleLogs.push({
                            level: level.toUpperCase(),
                            message: Array.from(args).join(' '),
                            timestamp: new Date().toISOString()
                        });
                    }
                    
                    // Override console methods to capture logs
                    console.log = function() {
                        addMessage('LOG', arguments);
                        originalConsole.log.apply(console, arguments);
                    };
                    
                    console.info = function() {
                        addMessage('INFO', arguments);
                        originalConsole.info.apply(console, arguments);
                    };
                    
                    console.warn = function() {
                        addMessage('WARN', arguments);
                        originalConsole.warn.apply(console, arguments);
                    };
                    
                    console.error = function() {
                        addMessage('ERROR', arguments);
                        originalConsole.error.apply(console, arguments);
                    };
                    
                    console.debug = function() {
                        addMessage('DEBUG', arguments);
                        originalConsole.debug.apply(console, arguments);
                    };
                }
            """)
            
            # Wait a bit to ensure delayed messages are captured
            time.sleep(2)
            
            # Get the captured console logs
            js_logs = self.driver.execute_script("return window._consoleLogs || [];")
            
            # Add JS logs to our logs collection
            for log in js_logs:
                if log not in logs:  # Avoid duplicates
                    logs.append(log)
                    
        except Exception as js_error:
            logs.append({
                "level": "WARNING",
                "message": f"Could not retrieve JavaScript console logs: {str(js_error)}",
                "timestamp": ""
            })
            
        return logs

    def get_tools(self) -> List[Callable]:
        """
        Returns a list of callable functions for all tools
        
        Returns:
            List[Callable]: A list of callable functions
        """
        # IMPORTANT: This order MUST match the exact order of schemas in get_tool_schemas()
        return [
            self.initialize_browser,
            self.navigate_to_url,
            self.input_text,
            self.browser_click,
            self.browser_snapshot,
            self.browser_console_messages,
            self.close_browser
        ]

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for browser tools.
        
        Returns:
            List[Dict[str, Any]]: Function schemas in OpenAI format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "initialize_browser",
                    "description": "Start or restart a browser session. Must be called before any other browser operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "navigate_to_url",
                    "description": "Navigate to a URL and capture a snapshot of all page elements. Returns both interactive and non-interactive elements with their properties.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The complete URL (with https://) to navigate to"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Custom timeout in seconds (default: 10)"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "input_text",
                    "description": "Type text into a form field, search box, or other input element using a reference ID from a snapshot. The element must be editable.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element": {
                                "type": "string",
                                "description": "Human-readable description of the element (e.g., 'Search field', 'Username input')"
                            },
                            "ref": {
                                "type": "string",
                                "description": "Element ID from the page snapshot (e.g., 'e0', 'e1', 'e2'). Must refer to an editable element."
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to input into the element"
                            },
                            "submit": {
                                "type": "boolean",
                                "description": "Press Enter after typing to submit forms (default: false)"
                            },
                            "slowly": {
                                "type": "boolean",
                                "description": "Type one character at a time to trigger JS events (default: true)"
                            }
                        },
                        "required": ["element", "ref", "text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_click",
                    "description": "Click on a button, link, or other clickable element using a reference ID from a snapshot. IMPORTANT: You must first call browser_snapshot() or navigate_to_url() to get element references, then use those references to click.\n\nExample workflow:\n1. Get snapshot: browser_snapshot() or navigate_to_url()\n2. Find element ref (e.g. 'e0') in response's interactive_elements\n3. Click using that ref: browser_click(element='Login', ref='e0')",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element": {
                                "type": "string",
                                "description": "Human-readable description of what you're clicking (e.g., 'Login button', 'Next page link', 'Submit button')"
                            },
                            "ref": {
                                "type": "string",
                                "description": "Element ID from the page snapshot (e.g., 'e0', 'e1', 'e2'). You MUST get this ID from a previous snapshot's interactive_elements."
                            }
                        },
                        "required": ["element", "ref"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_snapshot",
                    "description": "Capture a fresh snapshot of the current page, including all elements (both interactive and non-interactive). Use this to get up-to-date element references before clicking or typing.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_console_messages",
                    "description": "Retrieve JavaScript console messages (logs, warnings, errors) from the browser for debugging.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "close_browser",
                    "description": "Close the browser and end the session. Call this when you're done to free resources.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
        