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
    OpenAIImageToolkit,
    OpenRouterImageToolkit,
    FluxImageToolkit
)
from evoagentx.tools.storage_handler import LocalStorageHandler

# Import the new unified image collections
from evoagentx.tools.collection_image import (
    ImageGenerationCollection,
    ImageEditingCollection, 
    ImageAnalysisCollection,
    ImageCollectionToolkit
)


def run_image_analysis_example():
    """Simple example using OpenRouter image analysis to analyze images."""
    print("\n===== IMAGE ANALYSIS TOOL EXAMPLE =====\n")

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return

    try:
        ortk = OpenRouterImageToolkit(name="DemoORImageToolkit", api_key=openrouter_api_key)
        # Use the correct tool name from OpenRouterImageAnalysisTool
        analyze_tool = ortk.get_tool("openrouter_image_analysis")
        test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        print(f"Analyzing image: {test_image_url}")
        result = analyze_tool(prompt="Describe this image in detail.", image_url=test_image_url)
        print("Raw analysis response:")
        print(result)
        if 'error' in result:
            print(f"❌ Image analysis failed: {result['error']}")
        else:
            print("✓ Analysis:")
            print(result.get('content', ''))
    except Exception as e:
        print(f"Error: {str(e)}")


## (Removed) standalone OpenAI image generation example


## (Removed) standalone OpenAI image editing example


def run_openai_image_toolkit_pipeline():
    """Single-call demo using OpenAIImageToolkit: generation only."""
    print("\n===== OPENAI IMAGE TOOLKIT (SINGLE CALL: GENERATION) =====\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org_id = os.getenv("OPENAI_ORGANIZATION_ID")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return

    # Handle optional organization_id parameter
    if openai_org_id:
        toolkit = OpenAIImageToolkit(
            name="DemoOpenAIImageToolkit",
            api_key=openai_api_key,
            organization_id=openai_org_id,
            generation_model="gpt-image-1",
            save_path="./generated_images"
        )
    else:
        toolkit = OpenAIImageToolkit(
            name="DemoOpenAIImageToolkit",
            api_key=openai_api_key,
            generation_model="gpt-image-1",
            save_path="./generated_images"
        )

    # Single tool call: generation only
    gen = toolkit.get_tool("openai_image_generation")

    gen_prompt = "A cute baby owl sitting on a tree branch at sunset, digital art"
    print(f"Generating: {gen_prompt}")
    gen_result = gen(prompt=gen_prompt, model="gpt-image-1", size="1024x1024")
    print("Raw generation response:")
    print(gen_result)
    if 'error' in gen_result:
        print(f"❌ Generation failed: {gen_result['error']}")
        return
    gen_paths = gen_result.get('results', [])
    if not gen_paths:
        print("❌ No generated images returned")
        return
    src_path = gen_paths[0]
    print(f"Generated image: {src_path}")


def run_flux_image_toolkit_pipeline():
    """Single-call demo using Flux backend: generation only."""
    print("\n===== FLUX IMAGE TOOLKIT (SINGLE CALL: GENERATION) =====\n")

    flux_api_key = os.getenv("FLUX_API_KEY")
    if not flux_api_key:
        print("❌ FLUX_API_KEY not found in environment variables")
        return

    # Initialize toolkit
    flux = FluxImageToolkit(
        name="DemoFluxImageToolkitPipeline",
        api_key=flux_api_key,
        save_path="./flux_generated_images"
    )
    gen_tool = flux.get_tool("flux_image_generation")

    # Single tool call: generation only
    gen_prompt = "A neon-lit cyberpunk alley with rain reflections, cinematic"
    print(f"Generating: {gen_prompt}")
    gen_res = gen_tool(
        prompt=gen_prompt,
        seed=42,
        output_format="jpeg",
        prompt_upsampling=False,
        safety_tolerance=2
    )
    print("Raw Flux generation response:")
    print(gen_res)
    if not gen_res.get('success'):
        print(f"❌ Generation failed: {gen_res.get('error')}")
        return

    base_path = gen_res.get('file_path')
    if not base_path or not os.path.exists(base_path):
        print("❌ Generation did not return a valid file path")
        return
    print(f"Generated: {base_path}")


def run_openrouter_edit_pipeline():
    """Single-call demo using OpenRouter: edit only (uses existing base image)."""
    print("\n===== OPENROUTER EDIT (SINGLE CALL: EDIT ONLY) =====\n")

    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        print("❌ OPENROUTER_API_KEY not found")
        return

    # Use a single storage handler across OR tools so that saved_paths (filenames)
    # are resolved relative to the same base directory when read back.
    or_base_dir = "./openrouter_images"
    or_storage = LocalStorageHandler(base_path=or_base_dir)
    ortk = OpenRouterImageToolkit(name="DemoORImageToolkit", api_key=or_key, storage_handler=or_storage)

    # Single tool call: edit only, using an existing base image in repo
    edit_tool = ortk.get_tool("openrouter_image_edit")

    base_image_path = os.path.join(or_base_dir, "base.png")
    if not os.path.exists(base_image_path):
        print(f"❌ Base image not found at {base_image_path}. Please add one to proceed.")
        return

    edit_prompt = "Add a bold 'GEMINI' text at the top"
    print(f"Editing image: {base_image_path}")
    edit_res = edit_tool(
        prompt=edit_prompt,
        image_paths=[base_image_path],
        model="google/gemini-2.5-flash-image-preview",
        save_path=or_base_dir,
        output_basename="edited"
    )
    print("Raw OpenRouter edit response:")
    print(edit_res)
    edited = edit_res.get('saved_paths', [])
    if not edited:
        print("❌ No edited image saved")
        return
    edited_path = edited[0]
    print(f"Edited image: {edited_path}")



def main():
    """Main function to run all image tool examples"""
    print("===== IMAGE TOOL EXAMPLES =====")
    
    # Uncomment the examples you want to run:
    
    # Individual toolkit examples
    run_image_analysis_example() 
    run_openai_image_toolkit_pipeline()
    run_flux_image_toolkit_pipeline()
    run_openrouter_edit_pipeline()
    
    print("\n===== ALL IMAGE TOOL EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main()
