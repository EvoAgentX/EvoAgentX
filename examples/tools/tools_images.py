#!/usr/bin/env python3

"""
Example demonstrating how to use image handling toolkits from EvoAgentX.
This script provides comprehensive examples for:
- ImageAnalysisToolkit for analyzing images using AI
- OpenAI Image Generation for creating images from text prompts
- Flux Image Generation for creating images using Flux Kontext Max
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import (
    ImageAnalysisToolkit,
    OpenAIImageGenerationToolkit,
    FluxImageGenerationToolkit
)


def run_image_analysis_example():
    """Simple example using ImageAnalysisToolkit to analyze images."""
    print("\n===== IMAGE ANALYSIS TOOL EXAMPLE =====\n")
    
    # Check for OpenRouter API key
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("To test image analysis, set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your-openrouter-api-key-here'")
        print("Get your key from: https://openrouter.ai/")
        return
    
    try:
        # Initialize the image analysis toolkit
        toolkit = ImageAnalysisToolkit(
            name="DemoImageAnalysisToolkit",
            api_key=openrouter_api_key,
            model="openai/gpt-4o-mini"
        )
        
        print("✓ ImageAnalysisToolkit initialized")
        print(f"✓ Using OpenRouter API key: {openrouter_api_key[:8]}...")
        
        # Get the analysis tool - the actual tool name is "image_analysis"
        analyze_tool = toolkit.get_tool("image_analysis")
        
        # Test image analysis with a sample image
        test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        
        print(f"Analyzing image: {test_image_url}")
        print("Prompt: Describe this image in detail.")
        
        result = analyze_tool(
            prompt="Describe this image in detail.",
            image_url=test_image_url
        )
        
        # The tool returns content and usage directly, not in a success wrapper
        if 'error' not in result:
            print("✓ Image analysis successful")
            print(f"Analysis: {result.get('content', 'No content')}")
            
            # Display usage information if available
            if 'usage' in result:
                usage = result['usage']
                print(f"Token usage: {usage}")
        else:
            print(f"❌ Image analysis failed: {result.get('error', 'Unknown error')}")
        
        print("\n✓ ImageAnalysisToolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def run_openai_image_generation_example():
    """Simple example using OpenAI Image Generation Toolkit."""
    print("\n===== OPENAI IMAGE GENERATION TOOL EXAMPLE =====\n")
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org_id = os.getenv("OPENAI_ORGANIZATION_ID")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("To test OpenAI image generation, set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-openai-api-key-here'")
        print("Get your key from: https://platform.openai.com/")
        return
    
    try:
        # Initialize the OpenAI image generation toolkit
        toolkit = OpenAIImageGenerationToolkit(
            name="DemoOpenAIImageToolkit",
            api_key=openai_api_key,
            organization_id=openai_org_id,
            model="gpt-4o",
            save_path="./generated_images"
        )
        
        print("✓ OpenAI Image Generation Toolkit initialized")
        print(f"✓ Using OpenAI API key: {openai_api_key[:8]}...")
        if openai_org_id:
            print(f"✓ Using OpenAI Organization ID: {openai_org_id[:8]}...")
        
        # Get the generation tool - the actual tool name is "image_generation"
        generate_tool = toolkit.get_tool("image_generation")
        
        # Test image generation
        test_prompt = "A futuristic cyberpunk city with neon lights and flying cars, digital art style"
        print(f"Generating image with prompt: '{test_prompt}'")
        
        result = generate_tool(
            prompt=test_prompt,
            size="1024x1024",
            quality="high"
        )
        
        # The tool returns file_path directly, not in a success wrapper
        if 'error' not in result:
            print("✓ Image generation successful")
            print(f"Generated image path: {result.get('file_path', 'No path')}")
            print(f"Storage handler: {result.get('storage_handler', 'Unknown')}")
            
            # Check if file exists
            file_path = result.get('file_path', '')
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"File size: {file_size} bytes")
                print("✓ Generated image file saved successfully")
            else:
                print("⚠ Generated image file not found")
        else:
            print(f"❌ Image generation failed: {result.get('error', 'Unknown error')}")
        
        print("\n✓ OpenAI Image Generation Toolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def run_flux_image_generation_example():
    """Simple example using Flux Image Generation Toolkit."""
    print("\n===== FLUX IMAGE GENERATION TOOL EXAMPLE =====\n")
    
    # Check for BFL API key
    bfl_api_key = os.getenv("BFL_API_KEY")
    if not bfl_api_key:
        print("❌ BFL_API_KEY not found in environment variables")
        print("To test Flux image generation, set your BFL API key:")
        print("export BFL_API_KEY='your-bfl-api-key-here'")
        print("Get your key from: https://flux.ai/")
        return
    
    try:
        # Initialize the Flux image generation toolkit
        toolkit = FluxImageGenerationToolkit(
            name="DemoFluxImageToolkit",
            api_key=bfl_api_key,
            save_path="./flux_generated_images"
        )
        
        print("✓ Flux Image Generation Toolkit initialized")
        print(f"✓ Using BFL API key: {bfl_api_key[:8]}...")
        
        # Get the generation tool - the actual tool name is "flux_image_generation"
        generate_tool = toolkit.get_tool("flux_image_generation")
        
        # Test image generation
        test_prompt = "A futuristic cyberpunk city with neon lights and flying cars, digital art style"
        print(f"Generating image with prompt: '{test_prompt}'")
        
        result = generate_tool(
            prompt=test_prompt,
            seed=42,
            output_format="jpeg",
            prompt_upsampling=False,
            safety_tolerance=2
        )
        
        # The tool returns file_path directly, not in a success wrapper
        if 'error' not in result:
            print("✓ Image generation successful")
            print(f"Generated image path: {result.get('file_path', 'No path')}")
            print(f"Storage handler: {result.get('storage_handler', 'Unknown')}")
            
            # Check if file exists
            file_path = result.get('file_path', '')
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"File size: {file_size} bytes")
                print("✓ Generated image file saved successfully")
            else:
                print("⚠ Generated image file not found")
        else:
            print(f"❌ Image generation failed: {result.get('error', 'Unknown error')}")
        
        print("\n✓ Flux Image Generation Toolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Main function to run all image tool examples"""
    print("===== IMAGE TOOL EXAMPLES =====")
    
    # Run image analysis example
    run_image_analysis_example()
    
    # # Run OpenAI image generation example
    run_openai_image_generation_example()
    
    # Run Flux image generation example
    run_flux_image_generation_example()
    
    print("\n===== ALL IMAGE TOOL EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main()
