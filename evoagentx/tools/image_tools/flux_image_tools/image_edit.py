from typing import Dict, List, Optional
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .flux_image_postprocessor import FluxImagePostProcessor
from .utils import file_to_base64
import requests
import time


class FluxImageEditTool(Tool):
    name: str = "flux_image_edit"
    description: str = (
        "Edit/convert images using bfl.ai flux-kontext-max API."
        "Provide input image (path or base64) and edit prompt."
    )

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "Description of the edit operation."},
        "input_image": {"type": "string", "description": "Input image, can be a file path or base64 encoded image."},
        "images": {"type": "array", "description": "Input image path list (single string will be normalized to array)."},
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

    def __init__(self, api_key: str, storage_handler: Optional[FileStorageHandler] = None, 
                 base_path: str = "./imgs", save_path: str = None, auto_postprocess: bool = False):
        super().__init__()
        self.api_key = api_key
        
        # handle backward compatibility: if save_path is provided, use it as base_path
        if save_path is not None:
            base_path = save_path
            
        # initialize storage handler
        if storage_handler is None:
            self.storage_handler = LocalStorageHandler(base_path=base_path)
        else:
            self.storage_handler = storage_handler
        
        self.auto_postprocess = auto_postprocess
        self.postprocessor = FluxImagePostProcessor()

    def __call__(
        self,
        prompt: str,
        input_image: str = None,
        images: list = None,
        seed: int = 42,
        aspect_ratio: str = None,
        output_format: str = "jpeg",
        output_size: str = None,
        output_quality: int = 95,
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        image_name: str = None,
    ):
        try:
            # handle input image
            if input_image is None and images is None:
                return {"error": "must provide input_image or images parameter"}
            
            # if images parameter is provided, convert first to base64
            if images is not None:
                if isinstance(images, str):
                    image_paths = [images]
                else:
                    image_paths = list(images)
                
                if len(image_paths) == 0:
                    return {"error": "images list cannot be empty"}
                
                # use first image
                first_image = image_paths[0]
                
                # check if already base64
                if first_image.startswith("data:") or (len(first_image) > 100 and "/" not in first_image):
                    # maybe already base64
                    if first_image.startswith("data:"):
                        # extract base64 part
                        input_image = first_image.split(",", 1)[1] if "," in first_image else first_image
                    else:
                        input_image = first_image
                else:
                    # convert from file path
                    input_image = file_to_base64(first_image, self.storage_handler)
            elif input_image is not None:
                # check if input_image is a file path
                if not input_image.startswith("data:") and "/" in input_image:
                    # maybe a file path
                    try:
                        input_image = file_to_base64(input_image, self.storage_handler)
                    except Exception:
                        # if conversion fails, assume it is already base64
                        pass
                elif input_image.startswith("data:"):
                    # extract base64 part
                    input_image = input_image.split(",", 1)[1] if "," in input_image else input_image
            
            # check if postprocessing is needed
            needs_pp = self.postprocessor.needs_postprocessing(
                output_size or aspect_ratio, 
                output_format
            )
            
            target_size = output_size
            target_format = output_format
            target_quality = output_quality
            
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔄 detected incompatible parameters, auto postprocessing enabled:")
                for reason in needs_pp["reason"]:
                    print(f"   • {reason}")
                
                # get compatible API parameters
                compat_params = self.postprocessor.get_compatible_params(
                    output_size or aspect_ratio, 
                    output_format
                )
                api_aspect_ratio = compat_params["api_params"].get("aspect_ratio")
                api_format = compat_params["api_params"].get("output_format")
                
                print(f"📝 API will use: aspect_ratio={api_aspect_ratio}, format={api_format}")
                print(f"🎯 postprocessing target: size={target_size}, format={target_format}")
            else:
                api_aspect_ratio = aspect_ratio or output_size
                api_format = output_format
            
            # build request payload
            payload = {
                "prompt": prompt,
                "input_image": input_image,
                "seed": seed,
                "output_format": api_format if self.auto_postprocess and needs_pp["need_format_conversion"] else output_format,
                "prompt_upsampling": prompt_upsampling,
                "safety_tolerance": safety_tolerance,
            }
            
            # handle aspect ratio
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

            # send edit request
            response = requests.post("https://api.bfl.ai/v1/flux-kontext-max", json=payload, headers=headers)
            response.raise_for_status()
            request_data = response.json()

            request_id = request_data["id"]
            polling_url = request_data["polling_url"]

            # poll result
            while True:
                time.sleep(2)
                result = requests.get(
                    polling_url,
                    headers={
                        "accept": "application/json",
                        "x-key": self.api_key,
                    },
                    params={"id": request_id},
                ).json()

                if result["status"] == "Ready":
                    image_url = result["result"]["sample"]
                    break
                elif result["status"] in ["Error", "Failed"]:
                    return {"error": f"edit failed: {result}"}

            # download image
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image_content = image_response.content
            
            # apply postprocessing (if needed)
            if self.auto_postprocess and (needs_pp["need_resize"] or needs_pp["need_format_conversion"]):
                print("🔧 postprocessing image...")
                image_content, ext = self.postprocessor.process_image(
                    image_content,
                    target_size=target_size,
                    target_format=target_format,
                    compression_quality=target_quality
                )
            else:
                ext = output_format if output_format != "jpeg" else "jpg"

            # generate unique filename
            filename = self._get_unique_filename(image_name, seed, ext)
            
            # save image using storage handler
            result = self.storage_handler.save(filename, image_content)
            
            if result["success"]:
                # return translated path
                translated_path = self.storage_handler.translate_in(filename)
                return {
                    "results": [translated_path],
                    "count": 1,
                    "file_path": translated_path,  # backward compatibility
                    "full_path": result.get("full_path", filename),
                }
            else:
                return {
                    "error": f"save image failed: {result.get('error', 'unknown error')}"
                }
        except Exception as e:
            return {"error": f"image edit failed: {e}"}
    
    def _get_unique_filename(self, image_name: str, seed: int, ext: str = "jpg") -> str:
        """generate unique filename for the image"""
        import time
        
        if image_name:
            base = image_name.rsplit(".", 1)[0]
            filename = f"{base}_edited.{ext}"
        else:
            ts = int(time.time())
            filename = f"flux_edit_{seed}_{ts}.{ext}"
        
        counter = 1
        
        # exists, generate unique filename
        while self.storage_handler.exists(filename):
            if image_name:
                base = image_name.rsplit(".", 1)[0]
                filename = f"{base}_edited_{counter}.{ext}"
            else:
                filename = f"flux_edit_{seed}_{ts}_{counter}.{ext}"
            counter += 1
            
        return filename

