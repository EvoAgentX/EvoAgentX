import os
from openai import OpenAI

# Get API key from environment or use a default (replace with your valid key)
api_key = os.environ.get("OPENAI_API_KEY", "your_valid_api_key_here")

# Initialize the client
client = OpenAI(api_key=api_key)

try:
    # Make a simple API call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    
    # Print the response
    print("API call successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"Error: {e}") 