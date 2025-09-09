#!/usr/bin/env python3

"""
Example demonstrating how to use image handling toolkits from EvoAgentX.
This script provides comprehensive examples for:
- ImageAnalysisToolkit for analyzing images using AI
- OpenAI Image Generation for creating images from text prompts
- OpenAI Image Editing for editing existing images
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
    OpenAIImageToolkitV2,
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


## (Removed) standalone OpenAI image generation example


## (Removed) standalone OpenAI image editing example


def run_openai_image_toolkit_pipeline():
    """Pipeline: generate → edit → analyze using OpenAIImageToolkitV2."""
    print("\n===== OPENAI IMAGE TOOLKIT PIPELINE (GEN → EDIT → ANALYZE) =====\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org_id = os.getenv("OPENAI_ORGANIZATION_ID")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return

    toolkit = OpenAIImageToolkitV2(
        name="DemoOpenAIImageToolkitV2",
        api_key=openai_api_key,
        organization_id=openai_org_id,
        generation_model="gpt-image-1",
        save_path="./generated_images"
    )

    gen = toolkit.get_tool("openai_image_generation_v2")
    edit = toolkit.get_tool("openai_gpt_image1_edit_v2")
    analyze = toolkit.get_tool("openai_image_analysis_v2")

    # 1) Generate
    gen_prompt = "A cute baby owl sitting on a tree branch at sunset, digital art"
    print(f"Generating: {gen_prompt}")
    gen_result = gen(prompt=gen_prompt, model="gpt-image-1", size="1024x1024")
    if 'error' in gen_result:
        print(f"❌ Generation failed: {gen_result['error']}")
        return
    gen_paths = gen_result.get('results', [])
    if not gen_paths:
        print("❌ No generated images returned")
        return
    src_path = gen_paths[0]
    print(f"Generated image: {src_path}")

    # 2) Edit
    print("Editing the generated image...")
    edit_result = edit(
        prompt="Add a red scarf around the owl's neck",
        images=src_path,
        size="1024x1024",
        background="opaque",
        quality="high",
        n=1,
        image_name="edited_minimal"
    )
    if 'error' in edit_result:
        print(f"❌ Edit failed: {edit_result['error']}")
        return
    edited_paths = edit_result.get('results', [])
    if not edited_paths:
        print("❌ No edited images returned")
        return
    edited_path = edited_paths[0]
    print(f"Edited image: {edited_path}")

    # 3) Analyze (convert local file → data URL)
    print("Analyzing the edited image...")
    try:
        import base64, mimetypes
        with open(edited_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        mime, _ = mimetypes.guess_type(edited_path)
        mime = mime or 'image/png'
        data_url = f"data:{mime};base64,{b64}"
        analysis = analyze(
            prompt="Summarize what's in this image in one sentence.",
            image_url=data_url,
            model="gpt-4o-mini"
        )
        if 'error' in analysis:
            print(f"❌ Analyze failed: {analysis['error']}")
        else:
            print("✓ Analysis:")
            print(analysis.get('content', ''))
    except Exception as e:
        print(f"❌ Failed to analyze edited image: {e}")

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
    # run_image_analysis_example()
    
    run_openai_image_toolkit_pipeline()
    
    # Run Flux image generation example
    # run_flux_image_generation_example()
    
    print("\n===== ALL IMAGE TOOL EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main()
