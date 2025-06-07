import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate 
from evoagentx.tools.browser_tool import BrowserTool
import tempfile
import http.server
import socketserver
import threading

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)

# Create a simple HTML page with console messages
TEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Console Message Test</title>
    <link rel="icon" href="data:,">  <!-- Empty favicon to prevent 404 -->
</head>
<body>
    <h1>Testing Console Messages</h1>
    <script>
        // Helper function to ensure messages are captured
        function logMessage(type, msg) {
            const timestamp = new Date().toISOString();
            console[type](`[${timestamp}] ${msg}`);
        }
        
        // Generate different types of console messages
        logMessage('log', 'This is a log message');
        logMessage('info', 'This is an info message');
        logMessage('warn', 'This is a warning message');
        logMessage('error', 'This is an error message');
        logMessage('debug', 'This is a debug message');
        
        // Generate some dynamic messages
        setTimeout(() => {
            logMessage('log', 'Delayed message after 1 second');
            logMessage('error', 'Delayed error after 1 second');
        }, 1000);
        
        // Force a sync of console messages
        console.log('---End of immediate messages---');
    </script>
</body>
</html>
"""

class TestServer:
    def __init__(self, port=8000):
        self.port = port
        self.httpd = None
        self.server_thread = None
        
        # Create a temporary file with our test HTML
        self.temp_dir = tempfile.mkdtemp()
        with open(os.path.join(self.temp_dir, 'index.html'), 'w') as f:
            f.write(TEST_HTML)
            
        # Set up the server
        handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer(("", self.port), handler)
        
        # Change to temp directory
        os.chdir(self.temp_dir)
    
    def start(self):
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

async def test_browser_console_messages():
    """
    Test browser console messages by interacting with a button that triggers console output.
    """
    from evoagentx.tools.browser_tool import BrowserTool
    
    # Create a temporary HTML file with a button that logs to console
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Console Test</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            button { 
                padding: 10px 20px; 
                font-size: 16px; 
                cursor: pointer;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>Console Message Test</h1>
        <button id="testButton" onclick="triggerConsoleMessages()">Test Console Messages</button>
        
        <script>
            function triggerConsoleMessages() {
                console.log('Button clicked - INFO message');
                console.info('This is an informational message');
                console.warn('This is a warning message');
                console.error('This is an error message');
                console.debug('This is a debug message');
                
                // Test different data types
                console.log('Array test:', [1, 2, 3]);
                console.log('Object test:', {name: 'test', value: 42});
                
                // Test message with calculation
                let result = 10 * 5;
                console.log('Calculation result:', result);
                
                // Simulate an error
                try {
                    throw new Error('Test error in try-catch');
                } catch (e) {
                    console.error('Caught error:', e.message);
                }
            }
            
            // Log initial page load
            console.log('Page loaded successfully');
        </script>
    </body>
    </html>
    """
    
    import os
    import tempfile
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(test_html)
        test_file_path = f.name
    
    try:
        # Initialize browser tool
        browser_tool = BrowserTool()
        print("\nStep 1: Initializing browser...")
        init_result = browser_tool.initialize_browser()
        print("Initialization Result:")
        print("-" * 30)
        print(init_result)
        print("-" * 30)
        
        if init_result["status"] == "success":
            # Navigate to the test file
            file_url = f"file://{test_file_path}"
            print("\nStep 2: Navigating to test page...")
            nav_result = browser_tool.navigate_to_url(url=file_url)
            print("Navigation Result:")
            print("-" * 30)
            print(nav_result)
            print("-" * 30)
            
            # Get initial console messages
            print("\nStep 3: Getting initial console messages...")
            initial_messages = browser_tool.browser_console_messages()
            print("Initial Console Messages:")
            print("-" * 30)
            print(initial_messages)
            print("-" * 30)
            
            # Find and click the test button
            print("\nStep 4: Finding and clicking test button...")
            interactive_elements = nav_result.get("snapshot", {}).get("interactive_elements", [])
            button_ref = None
            for element in interactive_elements:
                if "button" in element.get("description", "").lower():
                    button_ref = element["id"]
                    break
            
            if button_ref:
                click_result = browser_tool.browser_click(
                    element="Test Console Messages button",
                    ref=button_ref
                )
                print("Click Result:")
                print("-" * 30)
                print(click_result)
                print("-" * 30)
                
                # Get console messages after click
                print("\nStep 5: Getting console messages after click...")
                messages_after_click = browser_tool.browser_console_messages()
                print("Console Messages After Click:")
                print("-" * 30)
                print(messages_after_click)
                print("-" * 30)
            else:
                print("Error: Button not found in interactive elements")
        
        print("\nStep 6: Closing browser...")
        close_result = browser_tool.close_browser()
        print("Browser Close Result:")
        print("-" * 30)
        print(close_result)
        print("-" * 30)
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        # Ensure browser is closed
        try:
            if 'browser_tool' in locals():
                browser_tool.close_browser()
        except Exception as close_error:
            print(f"Error closing browser: {str(close_error)}")
    finally:
        # Clean up the temporary file
        try:
            os.unlink(test_file_path)
        except Exception as e:
            print(f"Error removing temporary file: {str(e)}")

def build_customize_agent_with_tools():

    code_writer = CustomizeAgent(
        name="CodeWriter",
        description="Writes Python code based on requirements",
        prompt_template= StringTemplate(
            instruction="Visit init_page and search for 'python' in CS field, then visit the first page and return the content"
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "init_page", "type": "string", "description": "The url to visit"}
        ],
        outputs=[
            {"name": "result", "type": "string", "description": "The result of the search"}
        ],
        tool_names=["browser_tool"],
        tool_dict={"browser_tool": BrowserTool(headless=False, browser_type="chrome")}
    )

    message = code_writer(
        inputs={"init_page": "https://www.wikipedia.org/"}
    )
    print(f"Response from {code_writer.name}:")
    print(message.content.result)


if __name__ == "__main__":
    # import asyncio
    # asyncio.run(test_browser_console_messages())
    build_customize_agent_with_tools()