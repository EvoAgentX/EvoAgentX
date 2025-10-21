#!/usr/bin/env python3

"""
Example demonstrating how to use image handling toolkits from EvoAgentX.
This script provides comprehensive examples for:
- ImageAnalysisToolkit for analyzing images using AI
- OpenAI Image Generation and Editing (with auto postprocessing support)
- Flux Image Generation and Editing (with auto postprocessing support)
- OpenRouter Image Generation, Editing, and Analysis (with auto postprocessing support)
"""

import os
from pyexpat import model
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import (
    OpenAIImageToolkit,
    FluxImageToolkit,
    OpenRouterImageToolkit
)


def run_openai_image_toolkit_pipeline():
    """Pipeline: generate → edit → analyze using OpenAIImageToolkit."""
    print("\n===== OPENAI IMAGE TOOLKIT PIPELINE (GEN → EDIT → ANALYZE) =====\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org_id = os.getenv("OPENAI_ORGANIZATION_ID")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return

    toolkit = OpenAIImageToolkit(
        name="DemoOpenAIImageToolkit",
        api_key=openai_api_key,
        organization_id=openai_org_id,
        model="gpt-image-1",
        save_path="./openai_images"
    )

    gen = toolkit.get_tool("openai_image_generation")
    edit = toolkit.get_tool("openai_image_edit")
    analyze = toolkit.get_tool("openai_image_analysis")

    # 1) Generate with custom parameters
    gen_prompt = "A cute baby owl sitting on a tree branch at sunset, digital art"
    print(f"Generating: {gen_prompt}")
    print(f"Parameters: model=gpt-image-1, size=1024x1024, quality=high, output_format=png")
    gen_result = gen(
        prompt=gen_prompt, 
        model="gpt-image-1", 
        size="1024x1024",
        quality="high",
        output_format="png",
        background="opaque",
        n=1
    )
    if 'error' in gen_result:
        print(f"❌ Generation failed: {gen_result['error']}")
        return
    gen_paths = gen_result.get('results', [])
    if not gen_paths:
        print("❌ No generated images returned")
        return
    src_path = gen_paths[0]
    print(f"✓ Generated image: {src_path}")

    # 2) Edit with custom parameters
    print("\nEditing the generated image...")
    print(f"Parameters: size=1024x1024, quality=high, output_format=jpeg, output_compression=90")
    edit_result = edit(
        prompt="Add a red scarf around the owl's neck",
        image_paths=[src_path],
        size="1024x1024",
        background="opaque",
        quality="high",
        output_format="jpeg",
        output_compression=90,
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
    print(f"✓ Edited image: {edited_path}")

    # 3) Analyze (convert local file → data URL)
    print("\nAnalyzing the edited image...")
    try:
        analysis = analyze(
            prompt="Summarize what's in this image in one sentence.",
            image_paths=[edited_path],
            model="gpt-4o-mini"
        )
        if 'error' in analysis:
            print(f"❌ Analyze failed: {analysis['error']}")
        else:
            print("✓ Analysis:")
            print(analysis.get('content', ''))
    except Exception as e:
        print(f"❌ Failed to analyze edited image: {e}")


def run_openai_postprocessing_test():
    """Test OpenAI image generation with auto postprocessing for unsupported sizes/formats."""
    print("\n===== OPENAI IMAGE POSTPROCESSING TEST (GEN → EDIT) =====\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org_id = os.getenv("OPENAI_ORGANIZATION_ID")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return

    toolkit = OpenAIImageToolkit(
        name="DemoOpenAIPostprocessToolkit",
        api_key=openai_api_key,
        organization_id=openai_org_id,
        model="dall-e-3",
        save_path="./openai_images",
        auto_postprocess=True  # Enable auto postprocessing
    )

    gen = toolkit.get_tool("openai_image_generation")
    edit = toolkit.get_tool("openai_image_edit")

    # 1) Generate with unsupported size and format (auto postprocess)
    gen_prompt = "A cat playing with a ball in the living room"
    print(f"Generating: {gen_prompt}")
    print(f"Parameters: model=dall-e-3, size=800x600 (unsupported), format=webp (unsupported), quality=hd")
    print(f"Note: dall-e-3 only supports sizes [1024x1024, 1792x1024, 1024x1792] and format [PNG]")
    gen_result = gen(
        prompt=gen_prompt,
        model="dall-e-3",
        size="800x600",  # Unsupported → will auto postprocess
        output_format="webp",
        quality="hd",
        style="vivid"
    )
    
    if 'error' in gen_result:
        print(f"❌ Generation failed: {gen_result['error']}")
        return
    gen_paths = gen_result.get('results', [])
    if not gen_paths:
        print("❌ No generated images returned")
        return
    src_path = gen_paths[0]
    print(f"✓ Generated image: {src_path}")
    
    # Verify result
    if os.path.exists(src_path):
        try:
            from PIL import Image
            with Image.open(src_path) as img:
                print(f"✓ Image dimensions: {img.size[0]}x{img.size[1]} (target was 800x600)")
                print(f"✓ Image format: {img.format} (target was JPEG)")
        except Exception as e:
            print(f"⚠ Could not verify image: {e}")

    # 2) Edit with unsupported size and format (auto postprocess)
    print("\nEditing the generated image...")
    print(f"Parameters: size=600x600 (unsupported), quality=high")
    print(f"Note: gpt-image-1 supports sizes [1024x1024, 1536x1024, 1024x1536, auto]")
    edit_result = edit(
        prompt="Add a red scarf around the cat's neck",
        image_paths=[src_path],
        size="600x600",  # Unsupported → will auto postprocess
        quality="high",
        output_format="webp",
        image_name="edited_cat"
    )
    if 'error' in edit_result:
        print(f"❌ Edit failed: {edit_result['error']}")
        return
    edited_paths = edit_result.get('results', [])
    if not edited_paths:
        print("❌ No edited images returned")
        return
    edited_path = edited_paths[0]
    print(f"✓ Edited image: {edited_path}")
    # Verify result
    if os.path.exists(edited_path):
        try:
            from PIL import Image
            with Image.open(edited_path) as img:
                print(f"✓ Image dimensions: {img.size[0]}x{img.size[1]} (target was 600x600)")
                print(f"✓ Image format: {img.format} (target was WEBP)")
        except Exception as e:
            print(f"⚠ Could not verify image: {e}")


def run_flux_image_toolkit_pipeline():
    """Pipeline: generate → edit → analyze using Flux backend."""
    print("\n===== FLUX IMAGE TOOLKIT PIPELINE (GEN → EDIT → ANALYZE) =====\n")

    flux_api_key = os.getenv("FLUX_API_KEY")
    if not flux_api_key:
        print("❌ FLUX_API_KEY not found in environment variables")
        return

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return
        
    # Initialize toolkit
    flux = FluxImageToolkit(
        name="DemoFluxImageToolkitPipeline",
        flux_api_key=flux_api_key,
        openrouter_api_key=openrouter_api_key,
        save_path="./flux_images"
    )
    gen = flux.get_tool("flux_image_generation")
    edit = flux.get_tool("flux_image_edit")
    analyze = flux.get_tool("openrouter_image_analysis") if flux.get_tool("openrouter_image_analysis") else None

    # 1) Generate base image
    gen_prompt = "A cute baby owl sitting on a tree branch at sunset, digital art"
    print(f"Generating: {gen_prompt}")
    gen_result = gen(
        prompt=gen_prompt,
        seed=42,
        aspect_ratio="16:9",
        output_format="jpeg",
        prompt_upsampling=False,
        safety_tolerance=2
    )
    if 'error' in gen_result:
        print(f"❌ Generation failed: {gen_result['error']}")
        return
    gen_paths = gen_result.get('results', [])
    if not gen_paths:
        print("❌ No generated images returned")
        return
    src_path = gen_paths[0]
    print(f"✓ Generated image: {src_path}")

    # 2) Edit the generated image
    print("\nEditing the generated image...")
    edit_result = edit(
        prompt="Add a red scarf around the owl's neck",
        image_paths=[src_path],
        seed=43,
        output_format="jpeg",
        prompt_upsampling=False,
        safety_tolerance=2
    )
    if 'error' in edit_result:
        print(f"❌ Edit failed: {edit_result['error']}")
        return
    edited_paths = edit_result.get('results', [])
    if not edited_paths:
        print("❌ No edited images returned")
        return
    edited_path = edited_paths[0]
    print(f"✓ Edited image: {edited_path}")

    # 3) Analyze (using OpenRouter analysis tool)
    print("\nAnalyzing the edited image...")
    try:
        analysis = analyze(
            prompt="Summarize what's in this image in one sentence.",
            image_paths=[edited_path],
            model="gpt-4o-mini"
        )
        if 'error' in analysis:
            print(f"❌ Analyze failed: {analysis['error']}")
        else:
            print("✓ Analysis:")
            print(analysis.get('content', ''))
    except Exception as e:
        print(f"❌ Failed to analyze edited image: {e}")
    
    print("\n✓ Flux Image Toolkit Pipeline test completed")


def run_flux_postprocessing_test():
    """Test Flux image generation with auto postprocessing for unsupported sizes/formats."""
    print("\n===== FLUX IMAGE POSTPROCESSING TEST (GEN → EDIT) =====\n")
    
    flux_api_key = os.getenv("FLUX_API_KEY")
    if not flux_api_key:
        print("❌ FLUX_API_KEY not found in environment variables")
        return
    
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize toolkit with auto_postprocess enabled
        toolkit = FluxImageToolkit(
            name="DemoFluxPostprocessToolkit",
            flux_api_key=flux_api_key,
            openrouter_api_key=openrouter_api_key,
            save_path="./flux_images",
            auto_postprocess=True  # Enable auto postprocessing
        )
        
        gen = toolkit.get_tool("flux_image_generation")
        edit = toolkit.get_tool("flux_image_edit")
        
        # 1) Generate with unsupported size and format (auto postprocess)
        gen_prompt = "A cat playing with a ball in the living room"
        print(f"Generating: {gen_prompt}")
        print(f"Parameters: output_size=800x600 (needs postprocessing), format=webp (needs postprocessing)")
        print(f"Note: Flux natively supports aspect ratios (e.g., 16:9) and formats [jpeg, png]")
        gen_result = gen(
            prompt=gen_prompt,
            seed=100,
            output_size="800x600",  # Exact size → needs postprocessing
            output_format="webp",   # WEBP → needs postprocessing
            output_quality=90
        )
        
        if 'error' in gen_result:
            print(f"❌ Generation failed: {gen_result['error']}")
            return
        gen_paths = gen_result.get('results', [])
        if not gen_paths:
            print("❌ No generated images returned")
            return
        src_path = gen_paths[0]
        print(f"✓ Generated image: {src_path}")
        
        # Verify result
        if os.path.exists(src_path):
            try:
                from PIL import Image
                with Image.open(src_path) as img:
                    print(f"✓ Image dimensions: {img.size[0]}x{img.size[1]} (target was 800x600)")
                    print(f"✓ Image format: {img.format} (target was WEBP)")
            except Exception as e:
                print(f"⚠ Could not verify image: {e}")
        
        # 2) Edit with unsupported size
        print("\nEditing the generated image...")
        print(f"Parameters: output_size=600x600 (needs postprocessing)")
        edit_result = edit(
            prompt="Add a red scarf around the cat's neck",
            image_paths=[src_path],
            seed=101,
            output_size="600x600",  # Exact size → needs postprocessing
            output_format="webp",
            image_name="edited_cat"
        )
        if 'error' in edit_result:
            print(f"❌ Edit failed: {edit_result['error']}")
            return
        edited_paths = edit_result.get('results', [])
        if not edited_paths:
            print("❌ No edited images returned")
            return
        edited_path = edited_paths[0]
        print(f"✓ Edited image: {edited_path}")
        
        # Verify result
        if os.path.exists(edited_path):
            try:
                from PIL import Image
                with Image.open(edited_path) as img:
                    print(f"✓ Image dimensions: {img.size[0]}x{img.size[1]} (target was 600x600)")
                    print(f"✓ Image format: {img.format} (target was WEBP)")
            except Exception as e:
                print(f"⚠ Could not verify image: {e}")
        
        print("\n✓ Flux postprocessing test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def run_openrouter_image_toolkit_pipeline():
    """Pipeline: generate → edit → analyze using OpenRouterImageToolkit."""
    print("\n===== OPENROUTER IMAGE TOOLKIT PIPELINE (GEN → EDIT → ANALYZE) =====\n")

    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        print("❌ OPENROUTER_API_KEY not found")
        return

    toolkit = OpenRouterImageToolkit(
        name="DemoORImageToolkit", 
        api_key=or_key,
        save_path="./openrouter_images"
    )
    
    gen = toolkit.get_tool("openrouter_image_generation")
    edit = toolkit.get_tool("openrouter_image_edit")
    analyze = toolkit.get_tool("openrouter_image_analysis")

    # 1) Generate with custom parameters
    gen_prompt = "A cute baby owl sitting on a tree branch at sunset, digital art"
    print(f"Generating: {gen_prompt}")
    print(f"Parameters: model=google/gemini-2.5-flash-image, output_format=png")
    gen_result = gen(
        prompt=gen_prompt,
        model="google/gemini-2.5-flash-image",
        output_format="png",
        image_name="or_gen"
    )
    
    if 'error' in gen_result:
        print(f"❌ Generation failed: {gen_result['error']}")
        return
    
    gen_paths = gen_result.get('results', [])
    if not gen_paths:
        print("❌ No generated images returned")
        return
    src_path = gen_paths[0]
    print(f"✓ Generated image: {src_path}")

    # 2) Edit the generated image
    print("\nEditing the generated image...")
    print(f"Parameters: model=google/gemini-2.5-flash-image, output_format=jpeg")
    edit_result = edit(
        prompt="Add a red scarf around the owl's neck",
        image_paths=[src_path],
        model="google/gemini-2.5-flash-image",
        output_format="jpeg",
        output_quality=90,
        image_name="or_edited"
    )
    
    if 'error' in edit_result:
        print(f"❌ Edit failed: {edit_result['error']}")
        return
    
    edited_paths = edit_result.get('results', [])
    if not edited_paths:
        print("❌ No edited images returned")
        return
    edited_path = edited_paths[0]
    print(f"✓ Edited image: {edited_path}")
    
    # 3) Analyze the edited image
    print("\nAnalyzing the edited image...")
    try:
        analysis = analyze(
            prompt="Summarize what's in this image in one sentence.",
            image_paths=[edited_path],
            model="openai/gpt-4o-mini"
        )
        if 'error' in analysis:
            print(f"❌ Analyze failed: {analysis['error']}")
        else:
            print("✓ Analysis:")
            print(analysis.get('content', ''))
    except Exception as e:
        print(f"❌ Failed to analyze edited image: {e}")



def run_openrouter_postprocessing_test():
    """Test OpenRouter image generation/editing with auto postprocessing for custom sizes/formats."""
    print("\n===== OPENROUTER IMAGE POSTPROCESSING TEST (GEN → EDIT) =====\n")
    
    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        print("❌ OPENROUTER_API_KEY not found")
        return
    
    try:
        # Initialize toolkit with auto_postprocess enabled
        ortk = OpenRouterImageToolkit(
            name="DemoORPostprocessToolkit", 
            api_key=or_key, 
            auto_postprocess=True  # Enable auto postprocessing
        )
        
        gen = ortk.get_tool("openrouter_image_generation")
        edit = ortk.get_tool("openrouter_image_edit")
        
        # 1) Generate with custom size and format (requires postprocessing)
        gen_prompt = "A cat playing with a ball in the living room"
        print(f"Generating: {gen_prompt}")
        print(f"Parameters: output_size=800x600, output_format=webp, quality=90")
        print(f"Note: OpenRouter will generate at default size and postprocess to target size/format")
        gen_result = gen(
            prompt=gen_prompt,
            model="google/gemini-2.5-flash-image",
            output_size="800x600",  # Custom size → needs postprocessing
            output_format="webp",   # Custom format
            output_quality=90,
            image_name="or_gen_pp"
        )
        
        if 'error' in gen_result:
            print(f"❌ Generation failed: {gen_result['error']}")
            return
        gen_paths = gen_result.get('results', [])
        if not gen_paths:
            print("❌ No generated images returned")
            return
        src_path = gen_paths[0]
        print(f"✓ Generated image: {src_path}")
        
        # Verify result
        if os.path.exists(src_path):
            try:
                from PIL import Image
                with Image.open(src_path) as img:
                    print(f"✓ Image dimensions: {img.size[0]}x{img.size[1]} (target was 800x600)")
                    print(f"✓ Image format: {img.format} (target was WEBP)")
            except Exception as e:
                print(f"⚠ Could not verify image: {e}")
        
        # 2) Edit with custom size and format (requires postprocessing)
        print("\nEditing the generated image...")
        print(f"Parameters: output_size=600x600, output_format=webp, quality=85")
        edit_result = edit(
            prompt="Add a red scarf around the cat's neck",
            image_paths=[src_path],
            model="google/gemini-2.5-flash-image",
            output_size="600x600",  # Custom size → needs postprocessing
            output_format="webp",
            output_quality=85,
            image_name="or_edit_pp"
        )
        if 'error' in edit_result:
            print(f"❌ Edit failed: {edit_result['error']}")
            return
        edited_paths = edit_result.get('results', [])
        if not edited_paths:
            print("❌ No edited images returned")
            return
        edited_path = edited_paths[0]
        print(f"✓ Edited image: {edited_path}")
        
        # Verify result
        if os.path.exists(edited_path):
            try:
                from PIL import Image
                with Image.open(edited_path) as img:
                    print(f"✓ Image dimensions: {img.size[0]}x{img.size[1]} (target was 600x600)")
                    print(f"✓ Image format: {img.format} (target was WEBP)")
            except Exception as e:
                print(f"⚠ Could not verify image: {e}")
        
        print("\n✓ OpenRouter postprocessing test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Main function to run all image tool examples"""
    print("===== IMAGE TOOL EXAMPLES =====")
    
    # 1. Full pipeline: generate → edit → analyze (OpenAI)
    run_openai_image_toolkit_pipeline()
    
    # 2. OpenAI with AUTO POSTPROCESSING enabled (unsupported sizes/formats)
    # run_openai_postprocessing_test()
    
    # 3. Flux full pipeline: generate → edit → analyze (Flux, where analysis is done using OpenRouter)
    # run_flux_image_toolkit_pipeline()

    # 4. Flux with AUTO POSTPROCESSING enabled (unsupported sizes/formats)
    # run_flux_postprocessing_test()
    
    # 5. OpenRouter full pipeline: generate → edit → analyze (OpenRouter)
    # run_openrouter_image_toolkit_pipeline()

    # 6. OpenRouter with AUTO POSTPROCESSING enabled (unsupported sizes/formats)
    # run_openrouter_postprocessing_test()

    print("\n===== ALL IMAGE TOOL EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main()
