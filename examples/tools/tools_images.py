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
    FluxImageGenerationToolkit,
    OpenRouterImageToolkit
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
        analyze_tool = ortk.get_tool("image_analysis")
        test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        print(f"Analyzing image: {test_image_url}")
        result = analyze_tool(prompt="Describe this image in detail.", image_url=test_image_url)
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
        generation_model="gpt-image-1",
        save_path="./generated_images"
    )

    gen = toolkit.get_tool("openai_image_generation")
    edit = toolkit.get_tool("openai_image_edit")
    analyze = toolkit.get_tool("openai_image_analysis")

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
        analysis = analyze(
            prompt="Summarize what's in this image in one sentence.",
            image_path=edited_path,
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
    print("\n===== IMAGE GENERATION TOOL EXAMPLE =====\n")
    
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
        
        print("✓ Image Generation Toolkit initialized")
        print(f"✓ Using BFL API key: {bfl_api_key[:8]}...")
        
        # Get the generation tool - the actual tool name is "flux_image_generation_edit"
        generate_tool = toolkit.get_tool("flux_image_generation_edit")
        
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
        
        print("\n✓ Image Generation Toolkit test completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def run_flux_image_toolkit_pipeline():
    """Pipeline: generate → edit → analyze using Flux backend (input_image editing)."""
    print("\n===== IMAGE TOOLKIT PIPELINE (GEN → EDIT → ANALYZE) =====\n")

    bfl_api_key = os.getenv("BFL_API_KEY")
    if not bfl_api_key:
        print("❌ BFL_API_KEY not found in environment variables")
        return

    # Initialize toolkit
    flux = FluxImageGenerationToolkit(
        name="DemoFluxImageToolkitPipeline",
        api_key=bfl_api_key,
        save_path="./flux_generated_images"
    )
    gen = flux.get_tool("flux_image_generation_edit")
    analyze = flux.get_tool("image_analysis") if flux.get_tool("image_analysis") else None

    # 1) Generate base image
    gen_prompt = "A neon-lit cyberpunk alley with rain reflections, cinematic"
    print(f"Generating: {gen_prompt}")
    gen_res = gen(
        prompt=gen_prompt,
        seed=42,
        output_format="jpeg",
        prompt_upsampling=False,
        safety_tolerance=2
    )
    if 'error' in gen_res:
        print(f"❌ Generation failed: {gen_res['error']}")
        return
    base_path = gen_res.get('file_path')
    if not base_path or not os.path.exists(base_path):
        print("❌ Generation did not return a valid file path")
        return
    print(f"Generated: {base_path}")

    # 2) Edit by sending input_image (base64)
    try:
        import base64
        with open(base_path, 'rb') as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')
        edit_prompt = "Add a glowing red umbrella held by a person in the foreground"
        print("Editing the generated image...")
        edit_res = gen(
            prompt=edit_prompt,
            input_image=b64_img,
            seed=43,
            output_format="jpeg",
            prompt_upsampling=False,
            safety_tolerance=2
        )
        if 'error' in edit_res:
            print(f"❌ Edit failed: {edit_res['error']}")
            return
        edited_path = edit_res.get('file_path')
        if not edited_path or not os.path.exists(edited_path):
            print("❌ Edit did not return a valid file path")
            return
        print(f"Edited: {edited_path}")
    except Exception as e:
        print(f"❌ Failed to edit: {e}")

    # 3) Analyze
    if analyze and edited_path and os.path.exists(edited_path):
        try:
            import base64, mimetypes
            with open(edited_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            mime, _ = mimetypes.guess_type(edited_path)
            mime = mime or 'image/jpeg'
            data_url = f"data:{mime};base64,{b64}"
            analysis = analyze(
                prompt="Summarize what's in this image in one sentence.",
                image_url=data_url,
            )
            if 'error' in analysis:
                print(f"❌ Analyze failed: {analysis['error']}")
            else:
                print("✓ Analysis:")
                print(analysis.get('content', ''))
        except Exception as e:
            print(f"❌ Failed to analyze: {e}")


def run_openrouter_edit_pipeline():
    """OpenRouter: generate → edit (with generated image as input) → save."""
    print("\n===== OPENROUTER EDIT PIPELINE (GEN → EDIT) =====\n")

    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        print("❌ OPENROUTER_API_KEY not found")
        return

    ortk = OpenRouterImageToolkit(name="DemoORImageToolkit", api_key=or_key)
    gen = ortk.get_tool("openrouter_image_generation_edit")

    # 1) generate
    res = gen(
        prompt="A minimalist poster of a mountain at sunrise",
        model="google/gemini-2.5-flash-image-preview",
        save_path="./openrouter_images",
        output_basename="base"
    )
    bases = res.get('saved_paths', [])
    if not bases:
        print("❌ No base image saved; cannot proceed to edit")
        return
    base_path = bases[0]
    print(f"Base image: {base_path}")

    # 2) edit
    edit_prompt = "Add a bold 'GEMINI' text at the top"
    edit_res = gen(
        prompt=edit_prompt,
        image_paths=[base_path],
        model="google/gemini-2.5-flash-image-preview",
        save_path="./openrouter_images",
        output_basename="edited"
    )
    edited = edit_res.get('saved_paths', [])
    if not edited:
        print("❌ No edited image saved")
        return
    print(f"Edited image: {edited[0]}")


def main():
    """Main function to run all image tool examples"""
    print("===== IMAGE TOOL EXAMPLES =====")
    
    # run_image_analysis_example() 
    
    # run_openai_image_toolkit_pipeline()
   
    # run_flux_image_toolkit_pipeline()
   
    # run_openrouter_edit_pipeline()
    
    print("\n===== ALL IMAGE TOOL EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main()
