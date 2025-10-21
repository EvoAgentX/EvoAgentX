from typing import Dict, Tuple, Optional
from PIL import Image
from io import BytesIO


class OpenRouterImagePostProcessor:
    """
    Image postprocessor for OpenRouter image generation/editing APIs.
    Handles format conversion and size adjustment for unsupported parameters.
    """
    
    # OpenRouter models typically support specific formats and sizes
    # google/gemini-2.5-flash-image supports most common formats but may have size constraints
    MODEL_SUPPORTED_FORMATS = {
        "google/gemini-2.5-flash-image": ["png", "jpeg", "jpg", "webp"],
        # Add other OpenRouter models as needed
        "default": ["png", "jpeg", "jpg", "webp"]
    }
    
    # Most OpenRouter models don't have strict size limitations like OpenAI
    # but we'll track commonly supported sizes
    MODEL_SUPPORTED_SIZES = {
        "google/gemini-2.5-flash-image": None,  # No strict size constraints, generates flexible sizes
        "default": None  # Most models support flexible sizes
    }
    
    def __init__(self):
        pass
    
    @staticmethod
    def needs_postprocessing(
        model: str, 
        requested_size: Optional[str] = None,
        requested_format: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Check if postprocessing is needed for the requested parameters.
        
        Args:
            model: Model name (e.g., "google/gemini-2.5-flash-image")
            requested_size: Requested output size (e.g., "512x512")
            requested_format: Requested output format (e.g., "png", "jpeg", "webp")
            
        Returns:
            Dict containing need_resize, need_format_conversion, and reason list
        """
        result = {
            "need_resize": False,
            "need_format_conversion": False,
            "reason": []
        }
        
        # OpenRouter models typically don't have strict size constraints
        # Size postprocessing is needed when user explicitly requests a specific size
        if requested_size:
            result["need_resize"] = True
            result["reason"].append(f"Explicit size {requested_size} requested for postprocessing")
        
        # Format conversion might be needed for less common formats
        supported_formats = OpenRouterImagePostProcessor.MODEL_SUPPORTED_FORMATS.get(model, OpenRouterImagePostProcessor.MODEL_SUPPORTED_FORMATS["default"])
        if requested_format and requested_format.lower() not in supported_formats:
            result["need_format_conversion"] = True
            result["reason"].append(f"Format {requested_format} may need conversion (supported: {supported_formats})")
        
        return result
    
    @staticmethod
    def get_compatible_params(
        model: str,
        size: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> Dict:
        """
        Get compatible API parameters and target parameters for postprocessing.
        
        Args:
            model: Model name
            size: Requested output size
            output_format: Requested output format
            
        Returns:
            Dict containing api_params (for API call) and target_params (for postprocessing)
        """
        api_params = {}
        target_params = {
            "size": size,
            "format": output_format
        }
        
        # OpenRouter doesn't typically use size parameter in API call
        # We'll let the API generate at its default size and postprocess
        
        # Use PNG as default format for API (most widely supported)
        supported_formats = OpenRouterImagePostProcessor.MODEL_SUPPORTED_FORMATS.get(model, OpenRouterImagePostProcessor.MODEL_SUPPORTED_FORMATS["default"])
        if output_format and output_format.lower() in supported_formats:
            api_params["output_format"] = output_format.lower()
        else:
            # Use PNG as default
            api_params["output_format"] = "png"
        
        return {
            "api_params": api_params,
            "target_params": target_params
        }
    
    @staticmethod
    def process_image(
        image_bytes: bytes,
        target_size: Optional[str] = None,
        target_format: Optional[str] = None,
        compression_quality: int = 95
    ) -> Tuple[bytes, str]:
        """
        Process image: resize and/or convert format.
        
        Args:
            image_bytes: Original image bytes
            target_size: Target size (e.g., "512x512", "1024x768")
            target_format: Target format (png/jpeg/webp)
            compression_quality: Compression quality (1-100)
            
        Returns:
            Tuple[processed image bytes, file extension]
        """
        try:
            # Open image from bytes
            img = Image.open(BytesIO(image_bytes))
            
            # Resize if target size is specified
            if target_size:
                try:
                    target_w, target_h = map(int, target_size.split('x'))
                    # Use high quality resampling algorithm
                    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                except Exception as e:
                    print(f"⚠️ Size adjustment failed: {e}, using original size")
            
            # Determine output format
            output_format = (target_format or "png").lower()
            
            # Handle transparency based on output format
            if output_format in ["jpeg", "jpg"] and img.mode in ("RGBA", "LA", "P"):
                # JPEG doesn't support transparency, convert to RGB with white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif output_format in ["png", "webp"] and img.mode not in ("RGBA", "RGB"):
                img = img.convert("RGBA" if output_format != "jpeg" else "RGB")
            
            # Save to BytesIO
            output = BytesIO()
            save_kwargs = {}
            
            if output_format in ["jpeg", "jpg"]:
                save_kwargs["quality"] = compression_quality
                save_kwargs["optimize"] = True
                actual_format = "JPEG"
                ext = "jpg"
            elif output_format == "webp":
                save_kwargs["quality"] = compression_quality
                save_kwargs["method"] = 6  # Best compression
                actual_format = "WEBP"
                ext = "webp"
            elif output_format == "heic":
                # HEIC support requires pillow-heif
                try:
                    import pillow_heif
                    pillow_heif.register_heif_opener()
                    actual_format = "HEIF"
                    ext = "heic"
                except ImportError:
                    print("⚠️ HEIC format requires pillow-heif, falling back to PNG")
                    actual_format = "PNG"
                    ext = "png"
                    save_kwargs["optimize"] = True
            else:  # png or other
                save_kwargs["optimize"] = True
                actual_format = "PNG"
                ext = "png"
            
            img.save(output, format=actual_format, **save_kwargs)
            return output.getvalue(), ext
            
        except Exception as e:
            print(f"⚠️ Image postprocessing failed: {e}, returning original image")
            # Return original image bytes
            return image_bytes, "png"
    
    @staticmethod
    def process_multiple_images(
        images_data: list,
        target_size: Optional[str] = None,
        target_format: Optional[str] = None,
        compression_quality: int = 95
    ) -> list:
        """
        Batch process multiple images.
        
        Args:
            images_data: List of image data (bytes)
            target_size: Target size for all images
            target_format: Target format for all images
            compression_quality: Compression quality (1-100)
            
        Returns:
            List of processed image data [(bytes, ext), ...]
        """
        results = []
        for img_bytes in images_data:
            processed_bytes, ext = OpenRouterImagePostProcessor.process_image(
                img_bytes, target_size, target_format, compression_quality
            )
            results.append((processed_bytes, ext))
        return results

