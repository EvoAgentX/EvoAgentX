import os 
import asyncio
import time
from typing import Dict, Any
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import ActionAgent
from evoagentx.core.registry import register_action_function, ACTION_FUNCTION_REGISTRY

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)


# ============================================================================
# REGISTERED FUNCTIONS - Key Cases Only
# ============================================================================

@register_action_function
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@register_action_function
def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


@register_action_function
def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@register_action_function
def validate_email(email: str) -> dict:
    """Validate email format and return validation result."""
    import re
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    is_valid = bool(re.match(email_pattern, email))
    
    return {
        "email": email,
        "is_valid": is_valid,
        "domain": email.split('@')[-1] if '@' in email else None,
        "validation_message": "Valid email format" if is_valid else "Invalid email format"
    }


async def fetch_data_async(url: str) -> str:
    """Simulate fetching data from URL asynchronously."""
    await asyncio.sleep(0.1)  # Simulate async operation
    return f"Data from {url}"


@register_action_function
def error_prone_function(input_data: Any) -> Dict[str, Any]:
    """Function that can fail in various ways for testing error handling."""
    if input_data is None:
        raise ValueError("Input data cannot be None")
    
    if isinstance(input_data, str):
        if input_data.lower() == "crash":
            raise RuntimeError("Simulated crash")
        elif input_data.lower() == "timeout":
            time.sleep(10)  # Simulate timeout
    
    if isinstance(input_data, int):
        if input_data < 0:
            raise ValueError("Negative numbers not allowed")
        if input_data > 1000:
            raise OverflowError("Number too large")
    
    return {
        "input_type": type(input_data).__name__,
        "input_value": input_data,
        "processed": True,
        "timestamp": time.time()
    }


# ============================================================================
# KEY EXAMPLE FUNCTIONS
# ============================================================================

def demo_basic_functionality():
    """Demonstrate basic ActionAgent functionality."""
    print("1. Basic Functionality:")
    
    # Simple math agent
    math_agent = ActionAgent(
        name="MathAgent",
        description="Performs mathematical operations",
        inputs=[
            {"name": "a", "type": "int", "description": "First number", "required": True},
            {"name": "b", "type": "int", "description": "Second number", "required": True}
        ],
        outputs=[
            {"name": "result", "type": "int", "description": "Sum of the numbers", "required": True}
        ],
        execute_func=add_numbers
    )
    
    result = math_agent(inputs={"a": 5, "b": 3})
    print(f"   Math Agent Result: {result.content.result}")


def demo_async_functionality():
    """Demonstrate async ActionAgent functionality."""
    print("\n2. Async Functionality:")
    
    async def run_async_demo():
        data_agent = ActionAgent(
            name="DataAgent",
            description="Fetches data from URLs",
            inputs=[
                {"name": "url", "type": "str", "description": "URL to fetch", "required": True}
            ],
            outputs=[
                {"name": "result", "type": "str", "description": "Fetched data", "required": True}
            ],
            execute_func=fetch_data_async,
            async_execute_func=fetch_data_async
        )
        
        result = await data_agent(inputs={"url": "https://api.example.com"})
        print(f"   Data Agent Result: {result.content.result}")
    
    asyncio.run(run_async_demo())


def demo_error_handling():
    """Demonstrate ActionAgent error handling."""
    print("\n3. Error Handling:")
    
    divide_agent = ActionAgent(
        name="DivideAgent",
        description="Divides numbers with error handling",
        inputs=[
            {"name": "a", "type": "int", "description": "Numerator", "required": True},
            {"name": "b", "type": "int", "description": "Denominator", "required": True}
        ],
        outputs=[
            {"name": "result", "type": "float", "description": "Quotient", "required": False},
            {"name": "error", "type": "str", "description": "Error message if any", "required": False}
        ],
        execute_func=divide_numbers
    )
    
    # Normal execution
    result = divide_agent(inputs={"a": 10, "b": 2})
    print(f"   Normal division: {result.content.result}")
    
    # Error case
    error_result = divide_agent(inputs={"a": 10, "b": 0})
    print(f"   Division by zero: {error_result.content.error}")


def demo_validation():
    """Demonstrate ActionAgent for data validation."""
    print("\n4. Data Validation:")
    
    email_agent = ActionAgent(
        name="EmailValidator",
        description="Validates email addresses",
        inputs=[
            {"name": "email", "type": "str", "description": "Email address to validate", "required": True}
        ],
        outputs=[
            {"name": "email", "type": "str", "description": "Input email address", "required": True},
            {"name": "is_valid", "type": "bool", "description": "Whether email is valid", "required": True},
            {"name": "domain", "type": "str", "description": "Email domain", "required": False},
            {"name": "validation_message", "type": "str", "description": "Validation result message", "required": True}
        ],
        execute_func=validate_email
    )
    
    test_emails = ["user@example.com", "invalid-email", "test@domain.co.uk"]
    
    for email in test_emails:
        result = email_agent(inputs={"email": email})
        status = "✅" if result.content.is_valid else "❌"
        print(f"   {status} {result.content.email} → {result.content.validation_message}")


def demo_auto_async_wrapper():
    """Demonstrate ActionAgent with auto-generated async wrapper."""
    print("\n5. Auto Async Wrapper:")
    
    def multiply_numbers(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y
    
    multiply_agent = ActionAgent(
        name="MultiplyAgent",
        description="Multiplies numbers",
        inputs=[
            {"name": "x", "type": "int", "description": "First number", "required": True},
            {"name": "y", "type": "int", "description": "Second number", "required": True}
        ],
        outputs=[
            {"name": "result", "type": "int", "description": "Product of the numbers", "required": True}
        ],
        execute_func=multiply_numbers
        # async_execute_func will be auto-generated
    )
    
    async def run_async():
        result = await multiply_agent(inputs={"x": 6, "y": 8})
        print(f"   Async multiply result: {result.content.result}")
    
    asyncio.run(run_async())


def demo_save_load_functionality():
    """Demonstrate ActionAgent save/load functionality."""
    print("\n6. Save/Load Functionality:")
    
    # Create and save agent
    math_agent = ActionAgent(
        name="MathAgent",
        description="Performs mathematical operations",
        inputs=[
            {"name": "a", "type": "int", "description": "First number", "required": True},
            {"name": "b", "type": "int", "description": "Second number", "required": True}
        ],
        outputs=[
            {"name": "result", "type": "int", "description": "Result", "required": True}
        ],
        execute_func=add_numbers
    )
    
    save_path = "./examples/output/test_math_agent.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    math_agent.save_module(save_path)
    print(f"   ✅ Agent saved to {save_path}")
    
    # Load and test agent
    loaded_agent = ActionAgent.load_module(save_path)
    result = loaded_agent(inputs={"a": 5, "b": 3})
    print(f"   ✅ Loaded agent result: {result.content.result}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"   ✅ Cleaned up {save_path}")


def demo_input_validation():
    """Demonstrate ActionAgent input validation."""
    print("\n7. Input Validation:")
    
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    agent = ActionAgent(
        name="ValidAgent",
        description="Valid agent",
        inputs=[
            {"name": "a", "type": "int", "description": "First number", "required": True},
            {"name": "b", "type": "int", "description": "Second number", "required": True}
        ],
        outputs=[
            {"name": "result", "type": "int", "description": "Sum", "required": True}
        ],
        execute_func=add_numbers
    )
    
    # Test valid inputs
    try:
        result = agent(inputs={"a": 5, "b": 3})
        print(f"   ✅ Valid inputs: {result.content.result}")
    except Exception as e:
        print(f"   ❌ Valid inputs failed: {e}")
    
    # Test missing required input
    try:
        result = agent(inputs={"a": 5})  # Missing 'b'
        print(f"   ❌ Should have failed for missing input, but got: {result}")
    except ValueError as e:
        print(f"   ✅ Correctly caught missing input error: {e}")


def demo_function_registry():
    """Demonstrate ActionAgent function registry functionality."""
    print("\n8. Function Registry:")
    
    # Show registered functions
    print("   Registered action functions:")
    for func_name in ACTION_FUNCTION_REGISTRY.functions.keys():
        print(f"     - {func_name}")
    
    # Test getting a registered function
    try:
        add_func = ACTION_FUNCTION_REGISTRY.get_function("add_numbers")
        result = add_func(5, 3)
        print(f"   ✅ Retrieved registered function result: {result}")
    except Exception as e:
        print(f"   ❌ Failed to retrieve registered function: {e}")


def demo_advanced_error_handling():
    """Demonstrate ActionAgent with advanced error handling scenarios."""
    print("\n9. Advanced Error Handling:")
    
    error_agent = ActionAgent(
        name="ErrorProneAgent",
        description="Tests various error scenarios",
        inputs=[
            {"name": "input_data", "type": "any", "description": "Input data that may cause errors", "required": True}
        ],
        outputs=[
            {"name": "input_type", "type": "str", "description": "Type of input data", "required": False},
            {"name": "input_value", "type": "any", "description": "Input value", "required": False},
            {"name": "processed", "type": "bool", "description": "Whether processing succeeded", "required": False},
            {"name": "timestamp", "type": "float", "description": "Processing timestamp", "required": False},
            {"name": "error", "type": "str", "description": "Error message if any", "required": False}
        ],
        execute_func=error_prone_function
    )
    
    test_cases = [
        {"input_data": "normal"},
        {"input_data": "crash"},
        {"input_data": None},
        {"input_data": -5},
        {"input_data": 2000}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = error_agent(inputs=test_case)
            if hasattr(result.content, 'error'):
                print(f"   Test {i} error: {result.content.error}")
            else:
                print(f"   Test {i} success: {result.content.input_type} = {result.content.input_value}")
        except Exception as e:
            print(f"   Test {i} exception: {e}")


def demo_edge_cases():
    """Demonstrate ActionAgent edge cases."""
    print("\n10. Edge Cases:")
    
    # Test with no inputs
    no_input_agent = ActionAgent(
        name="NoInputAgent",
        description="Agent with no inputs",
        inputs=[],
        outputs=[
            {"name": "message", "type": "str", "description": "Simple message", "required": True}
        ],
        execute_func=lambda: "Hello from no-input agent"
    )
    
    result = no_input_agent(inputs={})
    print(f"   No-input agent: {result.content.message}")
    
    # Test with unexpected inputs
    unexpected_agent = ActionAgent(
        name="UnexpectedInputAgent",
        description="Agent that handles unexpected inputs",
        inputs=[
            {"name": "a", "type": "int", "description": "First number", "required": True},
            {"name": "b", "type": "int", "description": "Second number", "required": True}
        ],
        outputs=[
            {"name": "result", "type": "int", "description": "Sum", "required": True}
        ],
        execute_func=add_numbers
    )
    
    result = unexpected_agent(inputs={"a": 5, "b": 3, "c": 10, "d": "extra"})
    print(f"   Unexpected inputs agent: {result.content.result}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("ActionAgent Examples - Key Cases Only")
    print("=" * 50)
    
    # Run key demonstrations
    demo_basic_functionality()
    demo_async_functionality()
    demo_error_handling()
    demo_validation()
    demo_auto_async_wrapper()
    demo_save_load_functionality()
    demo_input_validation()
    demo_function_registry()
    demo_advanced_error_handling()
    demo_edge_cases()
    
    print("\n" + "=" * 50)
    print("All ActionAgent examples completed!")
    print(f"Total registered functions: {len(ACTION_FUNCTION_REGISTRY.functions)}") 