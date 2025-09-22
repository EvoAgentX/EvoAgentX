from typing import Dict, List, Optional, Any, Union
import os
from dotenv import load_dotenv
from .tool import Tool, Toolkit
from .collection_base import ToolCollection
from .storage_handler import FileStorageHandler, LocalStorageHandler

# Import toolkit classes
from .image_openai import OpenAIImageToolkit
from .image_openrouter import OpenRouterImageToolkit
from .image_flux import FluxImageToolkit
load_dotenv()


class ImageGenerationCollection(ToolCollection):
    """
    Image generation collection with automatic provider fallback.
    Supports OpenAI DALL-E, OpenRouter, and Flux providers.
    """
    name: str = "image_generation"
    description: str = "Generate images from text prompts with automatic provider fallback"

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Text prompt for image generation"},
        "model": {"type": "string", "description": "Model to use"},
        "size": {"type": "string", "description": "Image size/aspect ratio"},
        "quality": {"type": "string", "description": "Image quality"},
        "n": {"type": "integer", "description": "Number of images (1-10)"},
        "seed": {"type": "integer", "description": "Random seed"},
        "style": {"type": "string", "description": "Art style"},
        "output_format": {"type": "string", "description": "Output format"},
        "image_name": {"type": "string", "description": "Base name for saved images"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self,
        name: str = "image_generation",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/generated",
        execution_order: Optional[List[str]] = None,
    ):
        # Load environment variables if no keys provided
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not flux_api_key:
            flux_api_key = os.getenv("FLUX_API_KEY")
            
        # Initialize storage handler
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)
            
        # Initialize toolkits
        toolkits = []
        
        if openai_api_key:
            toolkits.append(OpenAIImageToolkit(
                api_key=openai_api_key,
                storage_handler=storage_handler,
                save_path=base_path
            ))
            
        if openrouter_api_key:
            toolkits.append(OpenRouterImageToolkit(
                api_key=openrouter_api_key,
                storage_handler=storage_handler
            ))
            
        if flux_api_key:
            toolkits.append(FluxImageToolkit(
                api_key=flux_api_key,
                storage_handler=storage_handler,
                save_path=base_path
            ))
            
        # Set default execution order using actual tool names
        if execution_order is None:
            execution_order = ["openai_image_generation", "openrouter_image_generation", "flux_image_generation"]
        
        super().__init__(
            name=name,
            description="Generate images from text prompts with automatic provider fallback",
            kits=toolkits,
            execution_order=execution_order,
            argument_mapping_function={
                "openai_image_generation": self._map_args_for_openai,
                "openrouter_image_generation": self._map_args_for_openrouter,
                "flux_image_generation": self._map_args_for_flux
            },
            output_mapping_function={
                "openai_image_generation": self._convert_output_from_openai,
                "openrouter_image_generation": self._convert_output_from_openrouter,
                "flux_image_generation": self._convert_output_from_flux
            }
        )
        
        self.storage_handler = storage_handler

    def _map_args_for_openai(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for OpenAI image generation.
        
        OpenAI Input Format:
        - prompt: str (required)
        - model: str (dall-e-2 | dall-e-3 | gpt-image-1)
        - size: str (model-specific sizes)
        - quality: str (standard | hd for dall-e-3)
        - n: int (1-10, 1 for dall-e-3)
        - style: str (vivid | natural for dall-e-3)
        - response_format: str (url | b64_json)
        - image_name: str (optional save name)
        """
        mapped = {
            "prompt": inputs.get("prompt"),
            "model": inputs.get("model", "dall-e-3"),
            "image_name": inputs.get("image_name"),
        }
        
        # Map optional parameters
        if "size" in inputs:
            mapped["size"] = inputs["size"]
        if "quality" in inputs:
            mapped["quality"] = inputs["quality"]
        if "n" in inputs:
            mapped["n"] = inputs["n"]
        if "style" in inputs:
            mapped["style"] = inputs["style"]
        if "output_format" in inputs:
            mapped["response_format"] = "b64_json"  # Always use b64_json for consistency
            
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_openrouter(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for OpenRouter image generation.
        
        OpenRouter Input Format:
        - prompt: str (required)
        - model: str (default: google/gemini-2.5-flash-image-preview)
        - image_urls: list (optional, for editing)
        - image_paths: list (optional, for editing)
        - output_basename: str (base filename)
        """
        mapped = {
            "prompt": inputs.get("prompt"),
            "model": inputs.get("model", "google/gemini-2.5-flash-image-preview"),
            "output_basename": inputs.get("image_name", "generated"),
        }
        
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_flux(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for Flux image generation.
        
        Flux Input Format:
        - prompt: str (required)
        - seed: int (default: 42)
        - aspect_ratio: str (e.g., "1:1")
        - output_format: str (jpeg | png)
        - prompt_upsampling: bool (default: false)
        - safety_tolerance: int (default: 2)
        """
        mapped = {
            "prompt": inputs.get("prompt"),
            "seed": inputs.get("seed", 42),
            "output_format": inputs.get("output_format", "jpeg"),
        }
        
        # Map size to aspect_ratio if provided
        if "size" in inputs:
            size = inputs["size"]
            if isinstance(size, str) and "x" in size:
                # Convert "1024x1024" to "1:1"
                w, h = size.split("x")
                try:
                    ratio_w, ratio_h = int(w), int(h)
                    gcd = self._gcd(ratio_w, ratio_h)
                    mapped["aspect_ratio"] = f"{ratio_w//gcd}:{ratio_h//gcd}"
                except ValueError:
                    pass  # Invalid size format, skip
                    
        return {k: v for k, v in mapped.items() if v is not None}

    def _convert_output_from_openai(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI output to unified format.
        
        OpenAI Output Format:
        - {"results": [filename1, filename2, ...], "count": int}
        - {"error": "error message"}
        
        Unified Output Format:
        - {"success": bool, "images": [filename1, ...], "count": int, "provider": "openai"}
        - {"success": false, "error": "error message", "provider": "openai"}
        """
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "provider": "openai"
            }
        
        return {
            "success": True,
            "images": result.get("results", []),
            "count": result.get("count", 0),
            "provider": "openai"
        }

    def _convert_output_from_openrouter(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenRouter output to unified format.
        
        OpenRouter Output Format:
        - {"saved_paths": [filename1, ...]}
        - {"warning": "message", "raw": {...}}
        - {"error": "error message"}
        
        Unified Output Format:
        - {"success": bool, "images": [filename1, ...], "count": int, "provider": "openrouter"}
        - {"success": false, "error": "error message", "provider": "openrouter"}
        """
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "provider": "openrouter"
            }
        
        if "saved_paths" in result:
            return {
                "success": True,
                "images": result["saved_paths"],
                "count": len(result["saved_paths"]),
                "provider": "openrouter"
            }
        
        return {
            "success": False,
            "error": result.get("warning", "No images generated"),
            "provider": "openrouter"
        }

    def _convert_output_from_flux(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Flux output to unified format.
        
        Flux Output Format:
        - {"file_path": "/path/to/file.png"}
        - {"error": "error message"}
        
        Unified Output Format:
        - {"success": bool, "images": ["/path/to/file.png"], "count": 1, "provider": "flux"}
        - {"success": false, "error": "error message", "provider": "flux"}
        """
        if "error" in result:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "provider": "flux"
            }
        
        return {
            "success": True,
            "images": [result["file_path"]],
            "count": 1,
            "provider": "flux"
        }

    def _gcd(self, a: int, b: int) -> int:
        """Calculate greatest common divisor."""
        while b:
            a, b = b, a % b
        return a

    def _get_next_execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[str]:
        """Determine next tool to execute; stop when a provider succeeded with images."""
        for tool_name in self.execution_order:
            if tool_name not in outputs:
                return tool_name
            tool_output = outputs.get(tool_name)
            if (
                isinstance(tool_output, dict)
                and tool_output.get("success") is True
                and len(tool_output.get("images", [])) > 0
            ):
                return None
        return None

    def __call__(
        self,
        prompt: str,
        model: str = None,
        size: str = None,
        quality: str = None,
        n: int = None,
        seed: int = None,
        style: str = None,
        output_format: str = None,
        image_name: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not prompt:
            return {"success": False, "error": "Missing required parameter 'prompt'"}
        inputs = {
            "prompt": prompt,
            "model": model,
            "size": size,
            "quality": quality,
            "n": n,
            "seed": seed,
            "style": style,
            "output_format": output_format,
            "image_name": image_name,
            **kwargs,
        }
        # Remove None values
        inputs = {k: v for k, v in inputs.items() if v is not None}
        return self._run_pipeline(inputs)


class ImageEditingCollection(ToolCollection):
    """
    Image editing collection with automatic provider fallback.
    Supports OpenAI, OpenRouter, and Flux providers.
    """

    name: str = "image_editing"
    description: str = "Edit and modify images with automatic provider fallback"

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Edit instruction"},
        "images": {"type": "array", "description": "Image paths or URLs"},
        "image_path": {"type": "string", "description": "Single image path"},
        "image_url": {"type": "string", "description": "Single image URL"},
        "mask_path": {"type": "string", "description": "Mask image path"},
        "model": {"type": "string", "description": "Model to use"},
        "size": {"type": "string", "description": "Output size"},
        "quality": {"type": "string", "description": "Image quality"},
        "n": {"type": "integer", "description": "Number of variations"},
        "seed": {"type": "integer", "description": "Random seed"},
        "output_format": {"type": "string", "description": "Output format"},
        "image_name": {"type": "string", "description": "Base name for saved images"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self,
        name: str = "image_editing",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/edited",
        execution_order: Optional[List[str]] = None,
    ):
        # Get API keys from environment if not provided
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not flux_api_key:
            flux_api_key = os.getenv("FLUX_API_KEY")
            
        # Initialize storage handler
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)
            
        # Initialize toolkits
        toolkits = []
        
        if openai_api_key:
            toolkits.append(OpenAIImageToolkit(
                api_key=openai_api_key,
                storage_handler=storage_handler,
                save_path=base_path
            ))
            
        if openrouter_api_key:
            toolkits.append(OpenRouterImageToolkit(
                api_key=openrouter_api_key,
                storage_handler=storage_handler
            ))
            
        if flux_api_key:
            toolkits.append(FluxImageToolkit(
                api_key=flux_api_key,
                storage_handler=storage_handler,
                save_path=base_path
            ))
            
        # Set default execution order using actual tool names
        if execution_order is None:
            execution_order = ["openai_image_edit", "openrouter_image_edit", "flux_image_edit"]
            
        super().__init__(
            name=name,
            description="Edit and modify images with automatic provider fallback",
            kits=toolkits,
            execution_order=execution_order,
            argument_mapping_function={
                "openai_image_edit": self._map_args_for_openai,
                "openrouter_image_edit": self._map_args_for_openrouter,
                "flux_image_edit": self._map_args_for_flux
            },
            output_mapping_function={
                "openai_image_edit": self._convert_output_from_openai,
                "openrouter_image_edit": self._convert_output_from_openrouter,
                "flux_image_edit": self._convert_output_from_flux
            }
        )
        
        self.storage_handler = storage_handler

    def _map_args_for_openai(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for OpenAI image editing.
        
        OpenAI Input Format:
        - prompt: str (required)
        - images: list[str] (required, image paths)
        - mask_path: str (optional PNG mask)
        - size: str (1024x1024 | 1536x1024 | 1024x1536 | auto)
        - n: int (1-10)
        - quality: str (auto | high | medium | low)
        - output_format: str (png | jpeg | webp)
        - image_name: str (optional output base name)
        """
        # Handle different input formats
        images = inputs.get("images")
        if not images:
            if "image_path" in inputs:
                images = [inputs["image_path"]]
            elif "image_url" in inputs:
                # OpenAI edit doesn't support URLs directly, would need to download
                raise ValueError("OpenAI image editing requires local image files, not URLs")
        
        if not images:
            raise ValueError("No images provided for editing")
            
        mapped = {
            "prompt": inputs.get("prompt"),
            "images": images,
            "image_name": inputs.get("image_name"),
        }
        
        # Map optional parameters
        if "mask_path" in inputs:
            mapped["mask_path"] = inputs["mask_path"]
        if "size" in inputs:
            mapped["size"] = inputs["size"]
        if "n" in inputs:
            mapped["n"] = inputs["n"]
        if "quality" in inputs:
            mapped["quality"] = inputs["quality"]
        if "output_format" in inputs:
            mapped["output_format"] = inputs["output_format"]
            
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_openrouter(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for OpenRouter image editing.
        
        OpenRouter Input Format:
        - prompt: str (required)
        - image_urls: list (optional, remote URLs)
        - image_paths: list (optional, local paths)
        - model: str (default: google/gemini-2.5-flash-image-preview)
        - output_basename: str (base filename)
        """
        mapped = {
            "prompt": inputs.get("prompt"),
            "model": inputs.get("model", "google/gemini-2.5-flash-image-preview"),
            "output_basename": inputs.get("image_name", "edited"),
        }
        
        # Handle different input formats
        if "images" in inputs:
            # Assume local paths for now
            mapped["image_paths"] = inputs["images"]
        elif "image_path" in inputs:
            mapped["image_paths"] = [inputs["image_path"]]
        elif "image_url" in inputs:
            mapped["image_urls"] = [inputs["image_url"]]
            
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_flux(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for Flux image editing.
        
        Flux Input Format:
        - prompt: str (required)
        - image_path: str (local path to image for editing)
        - seed: int (default: 42)
        - aspect_ratio: str (e.g., "1:1")
        - output_format: str (jpeg | png)
        """
        mapped = {
            "prompt": inputs.get("prompt"),
            "seed": inputs.get("seed", 42),
            "output_format": inputs.get("output_format", "jpeg"),
        }
        
        # Handle input image - pass local path; provider will read and convert to base64
        images = inputs.get("images")
        image_path = inputs.get("image_path")
        
        if images:
            image_path = images[0]  # Use first image
        
        if image_path:
            mapped["image_path"] = image_path
            
        return {k: v for k, v in mapped.items() if v is not None}

    def _convert_output_from_openai(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI output to unified format.
        
        OpenAI Output Format:
        - {"results": [filename1, filename2, ...], "count": int}
        - {"error": "error message"}
        
        Unified Output Format:
        - {"success": bool, "images": [filename1, ...], "count": int, "provider": "openai"}
        - {"success": false, "error": "error message", "provider": "openai"}
        """
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "provider": "openai"
            }
        
        return {
            "success": True,
            "images": result.get("results", []),
            "count": result.get("count", 0),
            "provider": "openai"
        }

    def _convert_output_from_openrouter(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenRouter output to unified format.
        
        OpenRouter Output Format:
        - {"saved_paths": [filename1, ...]}
        - {"warning": "message", "raw": {...}}
        - {"error": "error message"}
        
        Unified Output Format:
        - {"success": bool, "images": [filename1, ...], "count": int, "provider": "openrouter"}
        - {"success": false, "error": "error message", "provider": "openrouter"}
        """
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "provider": "openrouter"
            }
        
        if "saved_paths" in result:
            return {
                "success": True,
                "images": result["saved_paths"],
                "count": len(result["saved_paths"]),
                "provider": "openrouter"
            }
        
        return {
            "success": False,
            "error": result.get("warning", "No images generated"),
            "provider": "openrouter"
        }

    def _convert_output_from_flux(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Flux output to unified format.
        
        Flux Output Format:
        - {"success": true, "file_path": "filename", "full_path": "...", "message": "..."}
        - {"success": false, "error": "error message"}
        
        Unified Output Format:
        - {"success": bool, "images": [filename1, ...], "count": int, "provider": "flux"}
        - {"success": false, "error": "error message", "provider": "flux"}
        """
        if not result.get("success", False):
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "provider": "flux"
            }
        
        return {
            "success": True,
            "images": [result["file_path"]],
            "count": 1,
            "provider": "flux"
        }

    def _get_next_execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[str]:
        """Determine next tool to execute; stop when a provider succeeded with images."""
        for tool_name in self.execution_order:
            if tool_name not in outputs:
                return tool_name
            tool_output = outputs.get(tool_name)
            if (
                isinstance(tool_output, dict)
                and tool_output.get("success") is True
                and len(tool_output.get("images", [])) > 0
            ):
                return None
        return None

    def __call__(
        self,
        prompt: str,
        images: list = None,
        image_path: str = None,
        image_url: str = None,
        mask_path: str = None,
        model: str = None,
        size: str = None,
        quality: str = None,
        n: int = None,
        seed: int = None,
        output_format: str = None,
        image_name: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute image editing with automatic provider fallback.
        Requires a prompt and at least one image input (images, image_path, or image_url).
        """
        # Validate required prompt
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            return {"error": "prompt is required for image editing"}

        # Consolidate image inputs
        consolidated_images: List[str] = []
        if images:
            consolidated_images.extend(images)
        if image_path:
            consolidated_images.append(image_path)
        # image_url is handled separately by providers that support URLs

        if not consolidated_images and not image_url:
            return {"error": "At least one of images, image_path, or image_url must be provided"}

        inputs: Dict[str, Any] = {
            "prompt": prompt,
            "images": consolidated_images if consolidated_images else None,
            "image_path": image_path,
            "image_url": image_url,
            "mask_path": mask_path,
            "model": model,
            "size": size,
            "quality": quality,
            "n": n,
            "seed": seed,
            "output_format": output_format,
            "image_name": image_name,
        }

        # Remove None values
        inputs = {k: v for k, v in inputs.items() if v is not None}

        return self._run_pipeline(inputs)


class ImageAnalysisCollection(ToolCollection):
    """
    Image analysis collection with automatic provider fallback.
    Supports OpenAI and OpenRouter providers.
    """

    name: str = "image_analysis"
    description: str = "Analyze and understand images with automatic provider fallback"

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Analysis question or instruction"},
        "image_url": {"type": "string", "description": "HTTP(S) image URL"},
        "image_path": {"type": "string", "description": "Local image file path"},
        "pdf_path": {"type": "string", "description": "Local PDF file path"},
        "model": {"type": "string", "description": "Model to use"},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(
        self,
        name: str = "image_analysis",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images/analysis_cache",
        execution_order: Optional[List[str]] = None,
    ):
        # Get API keys from environment if not provided
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            
        # Initialize storage handler
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)
            
        # Initialize toolkits
        toolkits = []
        
        if openai_api_key:
            toolkits.append(OpenAIImageToolkit(
                api_key=openai_api_key,
                storage_handler=storage_handler,
                save_path=base_path
            ))
            
        if openrouter_api_key:
            toolkits.append(OpenRouterImageToolkit(
                api_key=openrouter_api_key,
                storage_handler=storage_handler
            ))
            
        # Set default execution order using actual tool names
        if execution_order is None:
            execution_order = ["openai_image_analysis", "openrouter_image_analysis"]
            
        super().__init__(
            name=name,
            description="Analyze and understand images with automatic provider fallback",
            kits=toolkits,
            execution_order=execution_order,
            argument_mapping_function={
                "openai_image_analysis": self._map_args_for_openai,
                "openrouter_image_analysis": self._map_args_for_openrouter
            },
            output_mapping_function={
                "openai_image_analysis": self._convert_output_from_openai,
                "openrouter_image_analysis": self._convert_output_from_openrouter
            }
        )
        
        self.storage_handler = storage_handler

    def _map_args_for_openai(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for OpenAI image analysis.
        
        OpenAI Input Format:
        - prompt: str (required)
        - image_url: str (HTTP(S) URL, optional)
        - image_path: str (local path, optional)
        - model: str (e.g., gpt-4o-mini, gpt-4.1, gpt-5)
        """
        mapped = {
            "prompt": inputs.get("prompt"),
            "model": inputs.get("model", "gpt-4o-mini"),
        }
        
        # Handle image input
        if "image_url" in inputs:
            mapped["image_url"] = inputs["image_url"]
        elif "image_path" in inputs:
            mapped["image_path"] = inputs["image_path"]
            
        return {k: v for k, v in mapped.items() if v is not None}

    def _map_args_for_openrouter(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments for OpenRouter image analysis.
        
        OpenRouter Input Format:
        - prompt: str (required)
        - image_url: str (HTTP(S) URL, optional)
        - image_path: str (local image path, optional)
        - pdf_path: str (local PDF path, optional)
        """
        mapped = {
            "prompt": inputs.get("prompt"),
        }
        
        # Handle different input types
        if "image_url" in inputs:
            mapped["image_url"] = inputs["image_url"]
        elif "image_path" in inputs:
            mapped["image_path"] = inputs["image_path"]
        elif "pdf_path" in inputs:
            mapped["pdf_path"] = inputs["pdf_path"]
            
        return {k: v for k, v in mapped.items() if v is not None}

    def _convert_output_from_openai(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI output to unified format.
        
        OpenAI Output Format:
        - {"content": "analysis text"}
        - {"error": "error message"}
        
        Unified Output Format:
        - {"success": bool, "content": "analysis text", "provider": "openai"}
        - {"success": false, "error": "error message", "provider": "openai"}
        """
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "provider": "openai"
            }
        
        return {
            "success": True,
            "content": result.get("content", ""),
            "provider": "openai"
        }

    def _convert_output_from_openrouter(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenRouter output to unified format.
        
        OpenRouter Output Format:
        - {"content": "analysis text", "usage": {...}}
        - {"error": "error message", "raw": "..."}
        
        Unified Output Format:
        - {"success": bool, "content": "analysis text", "usage": {...}, "provider": "openrouter"}
        - {"success": false, "error": "error message", "provider": "openrouter"}
        """
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "provider": "openrouter"
            }
        
        return {
            "success": True,
            "content": result.get("content", ""),
            "usage": result.get("usage", {}),
            "provider": "openrouter"
        }

    def _get_next_execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[str]:
        """Determine next tool to execute; stop when analysis succeeded with content."""
        for tool_name in self.execution_order:
            if tool_name not in outputs:
                return tool_name
            tool_output = outputs.get(tool_name)
            if isinstance(tool_output, dict):
                content = tool_output.get("content")
                # Stop if we have explicit success with non-empty content,
                # or if no error is present and content exists
                if (tool_output.get("success") is True and content) or (tool_output.get("error") is None and content):
                    return None
        return None

    def __call__(self, prompt: str, image_url: str = None, image_path: str = None, pdf_path: str = None, model: str = None, **kwargs) -> Dict[str, Any]:
        """
        Execute image analysis with provider fallback.
        Validates that at least one image input is provided.
        """
        # Validate required prompt parameter
        if not prompt:
            return {
                "success": False,
                "error": "Missing required parameter 'prompt'"
            }
        
        # Validate input - need at least one image source
        has_image = any([
            image_url,
            image_path,
            pdf_path  # PDF is also valid for OpenRouter
        ])
        
        if not has_image:
            return {
                "success": False,
                "error": "No image input provided. Please specify image_url, image_path, or pdf_path."
            }
        
        # Build unified inputs
        inputs = {
            "prompt": prompt,
            "image_url": image_url,
            "image_path": image_path,
            "pdf_path": pdf_path,
            "model": model,
            **kwargs,
        }
        inputs = {k: v for k, v in inputs.items() if v is not None}
        
        # Check content type and provider compatibility
        if pdf_path and self.execution_order:
            # OpenAI doesn't support PDF, remove it from execution order for this call
            original_order = self.execution_order.copy()
            # Remove OpenAI analysis tool by its name
            filtered_order = [name for name in self.execution_order if name != "openai_image_analysis"]
            self.execution_order = filtered_order
            try:
                result = self._run_pipeline(inputs)
            finally:
                self.execution_order = original_order
            return result
        
        return self._run_pipeline(inputs)


class ImageCollectionToolkit(Toolkit):
    """
    Unified image toolkit combining generation, editing, and analysis capabilities
    with automatic provider fallback across OpenAI, OpenRouter, and Flux.
    """

    def __init__(
        self,
        name: str = "ImageCollectionToolkit",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        storage_handler: Optional[FileStorageHandler] = None,
        base_path: str = "./images",
    ):
        # Get API keys from environment if not provided
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openrouter_api_key:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not flux_api_key:
            flux_api_key = os.getenv("FLUX_API_KEY")
            
        # Initialize storage handler
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)
            
        # Initialize ToolCollection instances
        tools = []
        
        # Image Generation Collection
        generation_collection = ImageGenerationCollection(
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            flux_api_key=flux_api_key,
            storage_handler=storage_handler,
            base_path=f"{base_path}/generated"
        )
        tools.append(generation_collection)
        
        # Image Editing Collection
        editing_collection = ImageEditingCollection(
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            flux_api_key=flux_api_key,
            storage_handler=storage_handler,
            base_path=f"{base_path}/edited"
        )
        tools.append(editing_collection)
        
        # Image Analysis Collection
        analysis_collection = ImageAnalysisCollection(
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            storage_handler=storage_handler,
            base_path=f"{base_path}/analysis_cache"
        )
        tools.append(analysis_collection)
        
        # Initialize parent Toolkit
        super().__init__(name=name, tools=tools)
        
        # Store configuration
        self.openai_api_key = openai_api_key
        self.openrouter_api_key = openrouter_api_key
        self.flux_api_key = flux_api_key
        self.storage_handler = storage_handler
        self.base_path = base_path
        
        # Store references to collections for easy access
        self.generation = generation_collection
        self.editing = editing_collection
        self.analysis = analysis_collection
    
    def get_available_providers(self) -> dict:
        """Get information about available providers for each capability."""
        return {
            "generation": {
                "available": [tool.name for tool in self.generation.tools],
                "total_providers": len(self.generation.tools)
            },
            "editing": {
                "available": [tool.name for tool in self.editing.tools],
                "total_providers": len(self.editing.tools)
            },
            "analysis": {
                "available": [tool.name for tool in self.analysis.tools],
                "total_providers": len(self.analysis.tools)
            }
        }
    
    def generate_image(self, **kwargs):
        """Generate images using the ImageGenerationCollection."""
        return self.generation(**kwargs)
    
    def edit_image(self, **kwargs):
        """Edit images using the ImageEditingCollection."""
        return self.editing(**kwargs)
    
    def analyze_image(self, **kwargs):
        """Analyze images using the ImageAnalysisCollection."""
        return self.analysis(**kwargs)