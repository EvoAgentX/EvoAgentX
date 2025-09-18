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
    flux_key = os.getenv("BFL_API_KEY")
    
    if not any([openai_key, openrouter_key, flux_key]):
        print("No image generation API keys found")
        return
    
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
        
    except Exception as e:
        print(f"Generation error: {str(e)}")


def demo_image_editing():
    """Simple image editing demo"""
    print("\n=== IMAGE EDITING DEMO ===")
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    flux_key = os.getenv("BFL_API_KEY")
    
    if not any([openai_key, openrouter_key, flux_key]):
        print("No image editing API keys found")
        return
    
    try:
        image_edit = ImageEditingCollection(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            flux_api_key=flux_key,
            base_path="./demo_images"
        )
        
        # Use a sample image path (would need existing image)
        prompt = "Add a red hat to the subject"
        print(f"Editing with prompt: {prompt}")
        print("Note: Would need existing image file for actual editing")
        
        # Simulate with dummy data for demo
        print("\nRaw Editing Results: (simulated - need existing image)")
        print("{'error': 'No images provided for editing'}")
        
    except Exception as e:
        print(f"Editing error: {str(e)}")


def demo_image_analysis():
    """Simple image analysis demo"""
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
        
        test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        prompt = "What do you see in this image?"
        
        print(f"Analyzing image with prompt: {prompt}")
        
        result = image_analysis(prompt=prompt, image_url=test_image_url)
        print(f"\nRaw Analysis Results:")
        print(result)
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")




def main():
    """Run simple Tool Collection demos."""
    print("=== TOOL COLLECTIONS SIMPLE DEMO ===")
    print("Demonstrating search, image generation, editing, and analysis\n")
    
    # Core demos
    demo_search()
    demo_image_generation()
    demo_image_editing()
    demo_image_analysis()
    print("\n=== DEMO COMPLETED ===")


if __name__ == "__main__":
    main()