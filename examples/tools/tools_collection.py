#!/usr/bin/env python3

"""
Tool Collections Simple Demo for EvoAgentX

Simple demonstration of Tool Collections:
- SearchCollection: Multi-engine search with fallback
- ImageGenerationCollection: Multi-provider image generation  
- ImageEditingCollection: Multi-provider image editing
- ImageAnalysisCollection: Multi-provider image analysis
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Ensure project root is on sys.path to import evoagentx
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evoagentx.tools import (
    ImageGenerationCollectionToolkit,
    ImageEditingCollectionToolkit,
    ImageAnalysisCollectionToolkit,
    SearchCollectionToolkit
)


def demo_search():
    """Simple search demo"""
    print("\n=== SEARCH DEMO ===")
    
    try:
        search_collection = SearchCollectionToolkit(max_content_words=100, num_results=2).tools[0]
        
        query = "artificial intelligence"
        print(f"Searching: {query}")
        
        result = search_collection(query=query)
        print(f"\nRaw Search Result:")
        print(result)
        
        # Showcase immediate-return unified schema
        if isinstance(result, dict) and result.get("error") is None:
            items = result.get("results", [])
            print(f"Returned {len(items)} items from '{result.get('metadata', {}).get('source')}'.")
            if items:
                print(f"Top result title: {items[0].get('title')}")
    
    except Exception as e:
        print(f"Search error: {str(e)}")





def demo_image_generation():
    """Simple image generation demo"""
    print("\n=== IMAGE GENERATION DEMO ===")
    
    # Check for API keys
    # New OpenAI Image Org keys for collection toolkit
    openai_org_key = os.getenv("OPENAI_IMAGE_ORG_API_KEY")
    openai_org_id = os.getenv("OPENAI_IMAGE_ORG_ID")
    openrouter_key = os.getenv("OPENROUTER_API_KEY") 
    flux_key = os.getenv("FLUX_API_KEY")
    
    # Require OpenAI org key + org id together, or OpenRouter, or Flux
    if not ((openai_org_key and openai_org_id) or openrouter_key or flux_key):
        print("No image generation API keys found")
        return None
    
    try:
        gen_toolkit = ImageGenerationCollectionToolkit(
            openai_org_api_key=openai_org_key,
            openai_org_id=openai_org_id,
            openrouter_api_key=openrouter_key,
            flux_api_key=flux_key,
            save_path="./demo_images",
            auto_postprocess=True
        )
        gen_tool = gen_toolkit.get_tool("image_generation")

        input_params = {
            "prompt": "A cute robot in a garden",
            "size": "1024x512",
            "image_name": "demo_robot",
            "output_format": "webp",
        }
        print("Generation input:")
        print(input_params)

        result = gen_tool(**input_params)
        print("\nRaw generation result:")
        print(result)
        
        # With immediate-return, result is a single unified dict
        first_image = None
        first_url = None
        if isinstance(result, dict) and result.get("success") and result.get("images"):
            first_image = result["images"][0]
            print(f"Provider '{result.get('provider')}' produced image: {first_image}")
            urls = result.get("urls")
            if isinstance(urls, list) and urls:
                first_url = urls[0]
                print(f"Image URL available: {first_url}")
        
        if first_url:
            return first_url
        else:
            print("No image URL returned; skipping edit chaining")
            return None
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return None


def demo_image_editing(src_image_url: str = None):
    """Simple image editing demo. Returns edited image URL if success, else None"""
    print("\n=== IMAGE EDITING DEMO ===")
    
    # Check for API keys
    # New OpenAI Image Org keys for collection toolkit
    openai_org_key = os.getenv("OPENAI_IMAGE_ORG_API_KEY")
    openai_org_id = os.getenv("OPENAI_IMAGE_ORG_ID")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    flux_key = os.getenv("FLUX_API_KEY")
    
    # Require OpenAI org key + org id together, or OpenRouter, or Flux
    if not ((openai_org_key and openai_org_id) or openrouter_key or flux_key):
        print("No image editing API keys found")
        return None
    
    if not src_image_url:
        # Fallback to a reliable public demo image URL when generation doesn't return a usable URL
        src_image_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
        print("No source image URL from generation; using demo URL:", src_image_url)
    
    try:
        edit_toolkit = ImageEditingCollectionToolkit(
            openai_org_api_key=openai_org_key,
            openai_org_id=openai_org_id,
            openrouter_api_key=openrouter_key,
            flux_api_key=flux_key,
            save_path="./demo_images",
            auto_postprocess=True
        )
        edit_tool = edit_toolkit.get_tool("image_editing")

        input_params = {
            "prompt": "Add a red hat to the subject",
            "image_urls": [src_image_url],
            "size": "500x500",
            "image_name": "demo_robot_edited",
            "output_format": "webp",
        }
        print("Editing input:")
        print(input_params)

        edit_result = edit_tool(**input_params)
        
        print("\nRaw editing result:")
        print(edit_result)
        
        # Immediate-return unified output
        edited_image_url = None
        if isinstance(edit_result, dict) and edit_result.get("success"):
            urls = edit_result.get("urls")
            if isinstance(urls, list) and urls:
                edited_image_url = urls[0]
                print(f"Provider '{edit_result.get('provider')}' produced edited image URL: {edited_image_url}")
        if not edited_image_url:
            print("No edited image returned by any provider.")
            return None
        return edited_image_url
        
    except Exception as e:
        print(f"Editing error: {str(e)}")
        return None


def demo_image_analysis(src_image_url: str = None):
    """Simple image analysis demo. If src_image_url provided, analyze it; otherwise, fall back to a demo URL."""
    print("\n=== IMAGE ANALYSIS DEMO ===")
    
    # Check for API keys
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if not openrouter_key:
        print("No image analysis API keys found")
        return
    
    try:
        analysis_toolkit = ImageAnalysisCollectionToolkit(
            openrouter_api_key=openrouter_key,
            save_path="./demo_images"
        )
        analysis_tool = analysis_toolkit.get_tool("image_analysis")

        prompt = "What do you see in this image?"

        if not src_image_url:
            src_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        input_params = {
            "prompt": prompt,
            "image_urls": [src_image_url],
        }
        print("Analysis input:")
        print(input_params)

        result = analysis_tool(**input_params)
        
        print("\nRaw analysis result:")
        print(result)
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")




## Removed unified toolkit demo to keep the collections showcase minimal


def main():
    """Run simple Tool Collection demos."""
    print("=== TOOL COLLECTIONS SIMPLE DEMO ===")
    print("Demonstrating search, image generation, editing, analysis, and unified toolkit\n")
    
    # Core demos
    # demo_search()
    # generated_image_url = demo_image_generation()
    edited_image_url = demo_image_editing()
    # demo_image_analysis()

    print("\n=== DEMO COMPLETED ===")


if __name__ == "__main__":
    main()