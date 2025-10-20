#!/usr/bin/env python3

"""
Minimal, raw-input/output demos for individual image toolkits.
Focus: OpenAIImageToolkit, OpenRouterImageToolkit, FluxImageToolkit.
Each demo prints full input dicts and raw outputs.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Ensure project root is on sys.path to import evoagentx
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evoagentx.tools import (
    OpenAIImageToolkit,
    OpenRouterImageToolkit,
    FluxImageToolkit,
)


def run_openai_toolkit_examples():
    """Demonstrate OpenAIImageToolkit: generation, edit, analysis (raw IO)."""
    print("\n===== OPENAI IMAGE TOOLKIT (GEN/EDIT/ANALYSIS) =====\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org_id = os.getenv("OPENAI_ORGANIZATION_ID")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found; skipping OpenAI toolkit demos")
        return

    try:
        tk = OpenAIImageToolkit(api_key=openai_api_key, organization_id=openai_org_id, save_path="./generated_images")

        # Generation
        gen = tk.get_tool("openai_image_generation")
        gen_input = {
            "prompt": "A watercolor painting of a mountain lake at dawn",
            "image_name": "openai_gen_demo",
            "size": "1024x1024",
        }
        print("OpenAI generation input:")
        print(gen_input)
        gen_out = gen(**gen_input)
        print("Raw generation output:")
        print(gen_out)

        # Editing
        edit = tk.get_tool("openai_image_edit")
        demo_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
        edit_input = {
            "prompt": "Add a small red hat to the central subject",
            "image_urls": [demo_url],
            "image_name": "openai_edit_demo",
        }
        print("OpenAI edit input:")
        print(edit_input)
        edit_out = edit(**edit_input)
        print("Raw edit output:")
        print(edit_out)

        # Analysis
        analysis = tk.get_tool("openai_image_analysis")
        analysis_input = {
            "prompt": "Describe notable visual elements in this image.",
            "image_urls": [demo_url],
        }
        print("OpenAI analysis input:")
        print(analysis_input)
        analysis_out = analysis(**analysis_input)
        print("Raw analysis output:")
        print(analysis_out)
    except Exception as e:
        print(f"Error running OpenAI toolkit demos: {str(e)}")


def run_openrouter_toolkit_examples():
    """Demonstrate OpenRouterImageToolkit: generation, edit, analysis (raw IO)."""
    print("\n===== OPENROUTER IMAGE TOOLKIT (GEN/EDIT/ANALYSIS) =====\n")

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("❌ OPENROUTER_API_KEY not found; skipping OpenRouter toolkit demos")
        return

    try:
        tk = OpenRouterImageToolkit(api_key=openrouter_api_key)

        demo_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"

        # Generation
        gen = tk.get_tool("openrouter_image_generation")
        gen_input = {
            "prompt": "An isometric city at night with glowing windows",
            "output_basename": "openrouter_gen_demo",
        }
        print("OpenRouter generation input:")
        print(gen_input)
        gen_out = gen(**gen_input)
        print("Raw generation output:")
        print(gen_out)

        # Editing
        edit = tk.get_tool("openrouter_image_edit")
        edit_input = {
            "prompt": "Overlay a translucent geometric pattern",
            "image_urls": [demo_url],
            "output_basename": "openrouter_edit_demo",
        }
        print("OpenRouter edit input:")
        print(edit_input)
        edit_out = edit(**edit_input)
        print("Raw edit output:")
        print(edit_out)

        # Analysis
        analysis = tk.get_tool("openrouter_image_analysis")
        analysis_input = {
            "prompt": "Summarize the visual composition",
            "image_urls": [demo_url],
        }
        print("OpenRouter analysis input:")
        print(analysis_input)
        analysis_out = analysis(**analysis_input)
        print("Raw analysis output:")
        print(analysis_out)
    except Exception as e:
        print(f"Error running OpenRouter toolkit demos: {str(e)}")


def run_flux_toolkit_examples():
    """Demonstrate FluxImageToolkit: generation and edit (raw IO)."""
    print("\n===== FLUX IMAGE TOOLKIT (GEN/EDIT) =====\n")

    flux_api_key = os.getenv("FLUX_API_KEY")
    if not flux_api_key:
        print("❌ FLUX_API_KEY not found; skipping Flux toolkit demos")
        return

    try:
        tk = FluxImageToolkit(api_key=flux_api_key, save_path="./flux_generated_images")

        # Generation
        gen = tk.get_tool("flux_image_generation")
        gen_input = {
            "prompt": "A neon-lit cyberpunk alley with rain reflections, cinematic",
            "image_name": "flux_gen_demo",
        }
        print("Flux generation input:")
        print(gen_input)
        gen_out = gen(**gen_input)
        print("Raw Flux generation output:")
        print(gen_out)

        # Editing
        edit = tk.get_tool("flux_image_edit")
        demo_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
        edit_input = {
            "prompt": "Apply a subtle vignette and cool tone",
            "image_urls": [demo_url],
            "image_name": "flux_edit_demo",
        }
        print("Flux edit input:")
        print(edit_input)
        edit_out = edit(**edit_input)
        print("Raw Flux edit output:")
        print(edit_out)
    except Exception as e:
        print(f"Error running Flux toolkit demos: {str(e)}")


def main():
    """Run individual image toolkit demos (OpenAI, OpenRouter, Flux)."""
    print("===== INDIVIDUAL IMAGE TOOLKIT DEMOS =====")
    run_openai_toolkit_examples()
    # run_openrouter_toolkit_examples()
    run_flux_toolkit_examples()
    print("\n===== ALL INDIVIDUAL TOOLKIT DEMOS COMPLETED =====")



if __name__ == "__main__":
    main()
