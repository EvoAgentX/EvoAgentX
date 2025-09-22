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

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools.collection_search import SearchCollection, SearchCollectionToolkit
from evoagentx.tools.collection_image import (
    ImageGenerationCollection, 
    ImageEditingCollection, 
    ImageAnalysisCollection,
    ImageCollectionToolkit
)


def demo_search():
    """Simple search demo"""
    print("\n=== SEARCH DEMO ===")
    
    try:
        search_collection = SearchCollection(max_content_words=100, num_results=2)
        
        query = "artificial intelligence"
        print(f"Searching: {query}")
        
        results = search_collection(query=query)
        print(f"\nRaw Search Results:")
        print(results)
        
    except Exception as e:
        print(f"Search error: {str(e)}")





def demo_image_generation():
    """Simple image generation demo"""
    print("\n=== IMAGE GENERATION DEMO ===")
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY") 
    flux_key = os.getenv("FLUX_API_KEY")
    
    if not any([openai_key, openrouter_key, flux_key]):
        print("No image generation API keys found")
        return None
    
    try:
        image_gen = ImageGenerationCollection(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            flux_api_key=flux_key,
            base_path="./demo_images"
        )
        
        prompt = "A cute robot in a garden"
        print(f"Generating: {prompt}")
        
        result = image_gen(prompt=prompt, image_name="demo_robot")
        print(f"\nRaw Generation Results:")
        print(result)
        
        # Extract the first successful image path from the collection outputs
        first_image = None
        if isinstance(result, dict):
            for tool_name, res in result.items():
                if isinstance(res, dict) and res.get("success") and res.get("images"):
                    first_image = res["images"][0]
                    print(f"Provider '{tool_name}' produced image: {first_image}")
                    break
        
        if first_image:
            return first_image
        else:
            print("No image generated to edit")
            return None
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return None


def demo_image_editing(src_image_path: str = None):
    """Simple image editing demo. Returns edited image path if success, else None"""
    print("\n=== IMAGE EDITING DEMO ===")
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    flux_key = os.getenv("FLUX_API_KEY")
    
    if not any([openai_key, openrouter_key, flux_key]):
        print("No image editing API keys found")
        return None
    
    if not src_image_path:
        print("No source image path provided from generation step; skipping real edit.")
        return None
    
    try:
        image_edit = ImageEditingCollection(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            flux_api_key=flux_key,
            base_path="./demo_images"
        )
        
        prompt = "Add a red hat to the subject"
        print(f"Editing with prompt: {prompt}")
        print(f"Editing source image: {src_image_path}")
        
        # Perform real editing using the generated image
        edit_result = image_edit(
            prompt=prompt,
            image_path=src_image_path,
            size="1024x1024",
            quality="high",
            n=1,
            image_name="demo_robot_edited"
        )
        
        print("\nRaw Editing Results:")
        print(edit_result)
        
        # Extract first successful edited image path
        edited_image = None
        if isinstance(edit_result, dict):
            for tool_name, res in edit_result.items():
                if isinstance(res, dict) and res.get("success") and res.get("images"):
                    edited_image = res["images"][0]
                    print(f"Provider '{tool_name}' produced edited image: {edited_image}")
                    break
        if not edited_image:
            print("No edited image returned by any provider.")
            return None
        return edited_image
        
    except Exception as e:
        print(f"Editing error: {str(e)}")
        return None


def demo_image_analysis(local_image_path: str = None):
    """Simple image analysis demo. If local_image_path provided and exists, analyze it; otherwise, fall back to a demo URL."""
    print("\n=== IMAGE ANALYSIS DEMO ===")
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if not any([openai_key, openrouter_key]):
        print("No image analysis API keys found")
        return
    
    try:
        image_analysis = ImageAnalysisCollection(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            base_path="./demo_images"
        )
        
        prompt = "What do you see in this image?"
        
        if local_image_path:
            print(f"Analyzing local edited image with prompt: {prompt}")
            result = image_analysis(prompt=prompt, image_path=local_image_path)
        else:
            test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            print(f"Analyzing image with prompt: {prompt}")
            result = image_analysis(prompt=prompt, image_url=test_image_url)
        
        print(f"\nRaw Analysis Results:")
        print(result)
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")




def demo_image_toolkit():
    """Demo unified ImageCollectionToolkit"""
    print("\n=== UNIFIED IMAGE TOOLKIT DEMO ===")
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    flux_key = os.getenv("FLUX_API_KEY")
    
    if not any([openai_key, openrouter_key, flux_key]):
        print("No image API keys found")
        return
    
    try:
        # Initialize unified toolkit
        image_toolkit = ImageCollectionToolkit(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            flux_api_key=flux_key,
            base_path="./demo_images"
        )
        
        # Check available providers
        providers = image_toolkit.get_available_providers()
        print(f"Available providers: {providers}")
        
        # Demo generation using the toolkit
        prompt = "A serene mountain landscape at sunset"
        print(f"\nGenerating with unified toolkit: {prompt}")
        
        # Use the generation collection through the toolkit
        gen_result = image_toolkit.generate_image(prompt=prompt, image_name="mountain_sunset")
        print(f"Generation result: {gen_result}")
        
        # Demo analysis with a web image
        test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        analysis_prompt = "Describe this natural landscape"
        
        print(f"\nAnalyzing with unified toolkit: {analysis_prompt}")
        analysis_result = image_toolkit.analyze_image(prompt=analysis_prompt, image_url=test_image_url)
        print(f"Analysis result: {analysis_result}")
        
    except Exception as e:
        print(f"Unified toolkit error: {str(e)}")


def main():
    """Run simple Tool Collection demos."""
    print("=== TOOL COLLECTIONS SIMPLE DEMO ===")
    print("Demonstrating search, image generation, editing, analysis, and unified toolkit\n")
    
    # Core demos
    # demo_search()
    generated_image_path = demo_image_generation()
    edited_image_path = demo_image_editing(generated_image_path)
    demo_image_analysis(edited_image_path)
    demo_image_toolkit()
    print("\n=== DEMO COMPLETED ===")


if __name__ == "__main__":
    main()