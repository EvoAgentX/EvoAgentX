import os
import time
import requests
from typing import Dict, List, Optional
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .image_postprocessor import FluxImagePostProcessor
from .utils import file_to_base64


class FluxImageEditTool(Tool):
    name: str = "flux_image_edit"
    description: str = "Flux image editing supporting model flux-kontext-max. It supports automatic postprocessing for unsupported sizes/formats."

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "Edit instruction. Required."},
        "image_urls": {"type": "array", "description": "Remote image URLs (optional)."},
        "image_paths": {"type": "array", "description": "Local image paths (optional). Single string will be normalized to array."},
        "seed": {"type": "integer", "description": "Random seed, default is 42.", "default": 42},
        "aspect_ratio": {"type": "string", "description": "Aspect ratio, e.g. '1:1', '16:9' etc.", "default": None},
        "output_format": {"type": "string", "description": "Image format, default is jpeg.", "default": "jpeg"},
        "output_size": {"type": "string", "description": "Output image size, e.g. '1024x768' (needs postprocessing) or '16:9' (native support)."},
        "output_quality": {"type": "integer", "description": "Output quality (0-100), for JPEG/WEBP compression, default is 95.", "default": 95},
        "prompt_upsampling": {"type": "boolean", "description": "Enable prompt upsampling, default is false.", "default": False},
        "safety_tolerance": {"type": "integer", "description": "Safety tolerance level, default is 2.", "default": 2},
        "image_name": {"type": "string", "description": "Optional save name."},
    }
    required: List[str] = ["prompt"]

    def __init__(self, name: str = None, api_key: str = None, model: str = "flux-kontext-max",
                 save_path: str = "./flux_edited_images",
                 storage_handler: Optional[FileStorageHandler] = None, auto_postprocess: bool = False):
        super().__init__()
        self.name = name or self.name
        self.api_key = api_key or os.getenv("FLUX_API_KEY")
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)
        
        self.auto_postprocess = auto_postprocess
        self.postprocessor = FluxImagePostProcessor()

    def __call__(
        self,
        prompt: str,
        image_urls: list = None,
        image_paths: list = None,
        seed: int = 42,
        model: str = None,
        aspect_ratio: str = None,
        output_format: str = "jpeg",
        output_size: str = None,
        output_quality: int = 95,
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        image_name: str = None,
    ):
        temp_downloaded_files = []  # Track temporary downloaded files for cleanup
        try:
            # Get actual parameters
            actual_model = model if model else self.model
            
            # Validate that at least one image source is provided
            if not image_urls and not image_paths:
                return {"error": "At least one of image_urls or image_paths must be provided for editing."}
            
            # Collect all image paths
            all_image_paths = []
            
            # Download remote images if image_urls provided
            if image_urls:
                import tempfile
                if isinstance(image_urls, str):
                    image_urls = [image_urls]
                
                for idx, url in enumerate(image_urls):
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        # Save to temporary file
                        suffix = url.split('.')[-1] if '.' in url.split('/')[-1] else 'png'
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{suffix}') as tmp:
                            tmp.write(response.content)
                            all_image_paths.append(tmp.name)
                            temp_downloaded_files.append(tmp.name)
                    except Exception as e:
                        # Clean up any downloaded files
                        self._cleanup_temp_files(temp_downloaded_files)
                        return {"error": f"Failed to download image from {url}: {e}"}
            
            # Add local image paths
            if image_paths:
                if isinstance(image_paths, str):
                    all_image_paths.append(image_paths)
                else:
                    all_image_paths.extend(image_paths)
            
            if not all_image_paths:
                self._cleanup_temp_files(temp_downloaded_files)
                return {"error": "No valid images provided"}
            
            # Use first image for editing
            first_image_path = all_image_paths[0]
            
            # Convert to base64
            input_image = file_to_base64(first_image_path, self.storage_handler)
            
            # Check if postprocessing is needed
            needs_pp = self.postprocessor.needs_postprocessing(
                output_size or aspect_ratio, 
                output_format
            )
            
            target_size = output_size
            target_format = output_format
            target_quality = output_quality if output_quality is not None else 95
            
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔄 Detected incompatible parameters, automatic postprocessing will be enabled:")
                for reason in needs_pp["reason"]:
                    print(f"   • {reason}")
                
                # Get compatible API parameters
                compat_params = self.postprocessor.get_compatible_params(
                    output_size or aspect_ratio, 
                    output_format
                )
                api_aspect_ratio = compat_params["api_params"].get("aspect_ratio")
                api_format = compat_params["api_params"].get("output_format")
                
                print(f"📝 API will use: aspect_ratio={api_aspect_ratio}, format={api_format}")
                print(f"🎯 Postprocessing target: size={target_size}, format={target_format}")
            else:
                api_aspect_ratio = aspect_ratio or output_size
                api_format = output_format
            
            # Build request payload
            payload = {
                "prompt": prompt,
                "input_image": input_image,
                "seed": seed,
                "output_format": api_format if self.auto_postprocess and needs_pp["need_format_conversion"] else output_format,
                "prompt_upsampling": prompt_upsampling,
                "safety_tolerance": safety_tolerance,
            }
            
            # Handle aspect ratio
            if self.auto_postprocess and needs_pp["need_resize"]:
                payload["aspect_ratio"] = api_aspect_ratio
            elif aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
            elif output_size and ":" in output_size:
                payload["aspect_ratio"] = output_size
            
            headers = {
                "accept": "application/json",
                "x-key": self.api_key,
                "Content-Type": "application/json",
            }

            # Call API
            response = requests.post(f"https://api.bfl.ai/v1/{actual_model}", json=payload, headers=headers)
            response.raise_for_status()
            request_data = response.json()

            request_id = request_data["id"]
            polling_url = request_data["polling_url"]

            # Poll for result
            while True:
                time.sleep(2)
                poll_result = requests.get(
                    polling_url,
                    headers={
                        "accept": "application/json",
                        "x-key": self.api_key,
                    },
                    params={"id": request_id},
                ).json()

                if poll_result["status"] == "Ready":
                    image_url = poll_result["result"]["sample"]
                    break
                elif poll_result["status"] in ["Error", "Failed"]:
                    # Clean up temporary files
                    self._cleanup_temp_files(temp_downloaded_files)
                    return {"error": f"Editing failed: {poll_result}"}

            # Download image
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image_bytes = image_response.content
            
            # Process and save images
            results = []
            try:
                # Apply postprocessing if needed
                if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                    print("🔧 Postprocessing image...")
                    image_bytes, ext = self.postprocessor.process_image(
                        image_bytes,
                        target_size=target_size,
                        target_format=target_format,
                        compression_quality=target_quality
                    )
                else:
                    # Determine file extension
                    ext = output_format.lower()
                    if ext == "jpeg":
                        ext = "jpg"

                # Generate unique filename
                filename = self._get_unique_filename(image_name, 0, ext)
                
                # Save using storage handler
                save_result = self.storage_handler.save(filename, image_bytes)
                
                if save_result["success"]:
                    results.append(filename)
                else:
                    results.append(f"Error saving image: {save_result.get('error', 'Unknown error')}")
            except Exception as e:
                results.append(f"Error saving image: {e}")

            # Clean up temporary downloaded files
            self._cleanup_temp_files(temp_downloaded_files)
            
            return {"results": results, "count": len(results)}
        except Exception as e:
            # Clean up temporary downloaded files even on error
            self._cleanup_temp_files(temp_downloaded_files)
            return {"error": f"Image editing failed: {e}"}
    
    def _get_unique_filename(self, image_name: str, index: int, ext: str = "jpg") -> str:
        """Generate a unique filename for the image"""
        import time
        
        if image_name:
            base = image_name.rsplit(".", 1)[0]
            # Try without index suffix first (for single image case)
            if index == 0:
                filename = f"{base}.{ext}"
                if not self.storage_handler.exists(filename):
                    return filename
            # If first attempt failed or not first image, use index suffix
            filename = f"{base}_{index+1}.{ext}"
        else:
            ts = int(time.time())
            filename = f"generated_{ts}_{index+1}.{ext}"
        
        # Check if file exists and generate unique name
        counter = 1
        while self.storage_handler.exists(filename):
            if image_name:
                base = image_name.rsplit(".", 1)[0]
                filename = f"{base}_{index+1}_{counter}.{ext}"
            else:
                filename = f"generated_{ts}_{index+1}_{counter}.{ext}"
            counter += 1
            
        return filename
    
    def _cleanup_temp_files(self, temp_files: list):
        """Clean up temporary files"""
        import os
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass

