from typing import Dict, Tuple, Optional
from PIL import Image
from io import BytesIO


class OpenAIImagePostProcessor:
    """
    Image postprocessor, for processing images returned by OpenAI API
    support format conversion and size adjustment
    """
    
    # supported sizes for each model
    MODEL_SUPPORTED_SIZES = {
        "dall-e-2": ["256x256", "512x512", "1024x1024"],
        "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"],
        "gpt-image-1": ["1024x1024", "1536x1024", "1024x1536", "auto"],
    }
    
    # supported output formats for each model
    MODEL_SUPPORTED_FORMATS = {
        "dall-e-2": ["png"],  # dall-e-2 only supports PNG
        "dall-e-3": ["png"],  # dall-e-3 only supports PNG
        "gpt-image-1": ["png", "jpeg", "webp"],  # gpt-image-1 supports multiple formats
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
        check if need postprocessing
        
        Args:
            model: model name
            requested_size: requested size
            requested_format: requested output format
            
        Returns:
            Dict containing need_resize and need_format_conversion
        """
        result = {
            "need_resize": False,
            "need_format_conversion": False,
            "reason": []
        }
        
        supported_sizes = OpenAIImagePostProcessor.MODEL_SUPPORTED_SIZES.get(model, [])
        supported_formats = OpenAIImagePostProcessor.MODEL_SUPPORTED_FORMATS.get(model, ["png"])
        
        # check size
        if requested_size and requested_size != "auto" and requested_size not in supported_sizes:
            result["need_resize"] = True
            result["reason"].append(f"size {requested_size} is not supported by {model}")
        
        # check format
        if requested_format and requested_format.lower() not in supported_formats:
            result["need_format_conversion"] = True
            result["reason"].append(f"format {requested_format} is not supported by {model}")
        
        return result
    
    @staticmethod
    def get_compatible_params(
        model: str,
        size: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> Dict:
        """
        get compatible API parameters (for API call) and target parameters (for postprocessing)
        
        Args:
            model: model name
            size: requested size
            output_format: requested output format
            
        Returns:
            Dict containing api_params (for API call) and target_params (for postprocessing)
        """
        supported_sizes = OpenAIImagePostProcessor.MODEL_SUPPORTED_SIZES.get(model, [])
        supported_formats = OpenAIImagePostProcessor.MODEL_SUPPORTED_FORMATS.get(model, ["png"])
        
        api_params = {}
        target_params = {
            "size": size,
            "format": output_format
        }
        
        # process size
        if size and size != "auto":
            if size in supported_sizes:
                api_params["size"] = size
            else:
                # use the closest supported size
                api_params["size"] = OpenAIImagePostProcessor._get_closest_size(size, supported_sizes)
        elif model == "gpt-image-1":
            api_params["size"] = "auto"
        
        # process format
        if output_format:
            if output_format.lower() in supported_formats:
                api_params["output_format"] = output_format.lower()
            else:
                # use default format (PNG)
                api_params["output_format"] = "png"
        
        return {
            "api_params": api_params,
            "target_params": target_params
        }
    
    @staticmethod
    def _get_closest_size(target_size: str, supported_sizes: list) -> str:
        """
        get the closest supported size to the target size
        
        Args:
            target_size: target size (e.g. "800x600")
            supported_sizes: supported sizes list
            
        Returns:
            the closest supported size
        """
        try:
            target_w, target_h = map(int, target_size.split('x'))
            target_area = target_w * target_h
            
            best_size = supported_sizes[0]
            min_diff = float('inf')
            
            for size in supported_sizes:
                if size == "auto":
                    continue
                w, h = map(int, size.split('x'))
                area = w * h
                diff = abs(area - target_area)
                if diff < min_diff:
                    min_diff = diff
                    best_size = size
            
            return best_size
        except Exception:
            return supported_sizes[0] if supported_sizes else "1024x1024"
    
    @staticmethod
    def process_image(
        image_bytes: bytes,
        target_size: Optional[str] = None,
        target_format: Optional[str] = None,
        compression_quality: int = 95
    ) -> Tuple[bytes, str]:
        """
        process image (adjust size and convert format)
        
        Args:
            image_bytes: original image bytes
            target_size: target size (e.g. "800x600")
            target_format: target format (png/jpeg/webp)
            compression_quality: compression quality (0-100)
            
        Returns:
            Tuple[processed image bytes, actual used format extension]
        """
        try:
            # open image
            img = Image.open(BytesIO(image_bytes))
            
            # adjust size
            if target_size and target_size != "auto":
                try:
                    target_w, target_h = map(int, target_size.split('x'))
                    # use high quality resampling algorithm
                    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                except Exception as e:
                    print(f"⚠️ size adjustment failed: {e}, using original size")
            
            # convert format
            output_format = (target_format or "png").lower()
            
            # process transparency
            if output_format == "jpeg" and img.mode in ("RGBA", "LA", "P"):
                # JPEG does not support transparency, convert to RGB
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif output_format in ["png", "webp"] and img.mode not in ("RGBA", "RGB"):
                img = img.convert("RGBA" if output_format != "jpeg" else "RGB")
            
            # save to BytesIO
            output = BytesIO()
            save_kwargs = {}
            
            if output_format == "jpeg":
                save_kwargs["quality"] = compression_quality
                save_kwargs["optimize"] = True
                actual_format = "JPEG"
                ext = "jpg"
            elif output_format == "webp":
                save_kwargs["quality"] = compression_quality
                save_kwargs["method"] = 6  # best compression
                actual_format = "WEBP"
                ext = "webp"
            else:  # png
                save_kwargs["optimize"] = True
                actual_format = "PNG"
                ext = "png"
            
            img.save(output, format=actual_format, **save_kwargs)
            return output.getvalue(), ext
            
        except Exception as e:
            print(f"⚠️ image postprocessing failed: {e}, returning original image")
            # return original image
            return image_bytes, "png"
    
    @staticmethod
    def process_multiple_images(
        images_data: list,
        target_size: Optional[str] = None,
        target_format: Optional[str] = None,
        compression_quality: int = 95
    ) -> list:
        """
        batch process multiple images
        
        Args:
            images_data: image data list (bytes)
            target_size: target size
            target_format: target format
            compression_quality: compression quality
            
        Returns:
            list of processed image data [(bytes, ext), ...]
        """
        results = []
        for img_bytes in images_data:
            processed_bytes, ext = OpenAIImagePostProcessor.process_image(
                img_bytes, target_size, target_format, compression_quality
            )
            results.append((processed_bytes, ext))
        return results

