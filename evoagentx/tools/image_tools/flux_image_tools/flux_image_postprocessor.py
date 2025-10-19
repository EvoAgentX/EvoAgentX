from typing import Dict, Tuple, Optional
from PIL import Image
from io import BytesIO


class FluxImagePostProcessor:
    """
    Flux image postprocessor, for processing images returned by Flux API
    支持格式转换和尺寸调整
    """
    
    # Flux supported aspect ratios
    # according to Flux documentation, Flux-kontext-max supports multiple aspect ratios
    FLUX_SUPPORTED_ASPECT_RATIOS = [
        "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21", 
        "3:2", "2:3", "5:4", "4:5"
    ]
    
    # Flux supported output formats
    FLUX_SUPPORTED_FORMATS = ["jpeg", "png"]
    
    def __init__(self):
        pass
    
    @staticmethod
    def needs_postprocessing(
        requested_size: Optional[str] = None,
        requested_format: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        check if postprocessing is needed
        
        Args:
            requested_size: requested size (e.g. "800x600")
            requested_format: requested output format
            
        Returns:
            dictionary containing need_resize and need_format_conversion
        """
        result = {
            "need_resize": False,
            "need_format_conversion": False,
            "reason": []
        }
        
        # check size - if provided exact size rather than aspect ratio, need postprocessing
        if requested_size:
            # if aspect ratio format (e.g. "16:9"), check if supported
            if ":" in requested_size:
                if requested_size not in FluxImagePostProcessor.FLUX_SUPPORTED_ASPECT_RATIOS:
                    result["need_resize"] = True
                    result["reason"].append(f"aspect ratio {requested_size} is not supported by Flux")
            else:
                # 如果是精确尺寸 (例如 "800x600")，需要后处理
                result["need_resize"] = True
                result["reason"].append(f"exact size {requested_size} needs postprocessing (Flux uses aspect ratio)")
        
        # check format
        if requested_format:
            format_lower = requested_format.lower()
            # webp is not natively supported by Flux, needs conversion
            if format_lower not in FluxImagePostProcessor.FLUX_SUPPORTED_FORMATS:
                result["need_format_conversion"] = True
                result["reason"].append(f"format {requested_format} is not supported by Flux")
        
        return result
    
    @staticmethod
    def get_compatible_params(
        size: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> Dict:
        """
        get compatible API parameters (for API call) and target parameters (for postprocessing)
        
        Args:
            size: requested size
            output_format: requested output format
            
        Returns:
            dictionary containing api_params (for API call) and target_params (for postprocessing)
        """
        api_params = {}
        target_params = {
            "size": size,
            "format": output_format
        }
        
        # process size
        if size:
            # if aspect ratio format and supported, use directly
            if ":" in size and size in FluxImagePostProcessor.FLUX_SUPPORTED_ASPECT_RATIOS:
                api_params["aspect_ratio"] = size
            else:
                # if exact size or unsupported aspect ratio, select the closest aspect ratio
                api_params["aspect_ratio"] = FluxImagePostProcessor._get_closest_aspect_ratio(size)
        else:
            # use default 1:1
            api_params["aspect_ratio"] = "1:1"
        
        # process format
        if output_format:
            format_lower = output_format.lower()
            if format_lower in FluxImagePostProcessor.FLUX_SUPPORTED_FORMATS:
                api_params["output_format"] = format_lower
            else:
                # use default format (JPEG)
                api_params["output_format"] = "jpeg"
        else:
            api_params["output_format"] = "jpeg"
        
        return {
            "api_params": api_params,
            "target_params": target_params
        }
    
    @staticmethod
    def _get_closest_aspect_ratio(target_size: str) -> str:
        """
        get the closest aspect ratio to the target size
        
        Args:
            target_size: target size (e.g. "800x600" or "16:9")
            
        Returns:
            the closest supported aspect ratio
        """
        try:
            # if already aspect ratio format
            if ":" in target_size:
                parts = target_size.split(":")
                if len(parts) == 2:
                    target_w, target_h = float(parts[0]), float(parts[1])
                else:
                    return "1:1"
            else:
                # if pixel size format
                parts = target_size.lower().split("x")
                if len(parts) == 2:
                    target_w, target_h = float(parts[0]), float(parts[1])
                else:
                    return "1:1"
            
            target_ratio = target_w / target_h
            
            best_aspect_ratio = "1:1"
            min_diff = float('inf')
            
            for aspect_ratio in FluxImagePostProcessor.FLUX_SUPPORTED_ASPECT_RATIOS:
                w, h = map(float, aspect_ratio.split(':'))
                ratio = w / h
                diff = abs(ratio - target_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_aspect_ratio = aspect_ratio
            
            return best_aspect_ratio
        except Exception:
            return "1:1"
    
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
            if target_size:
                try:
                    # if aspect ratio format, skip (API already processed)
                    if ":" not in target_size:
                        # if exact size
                        target_w, target_h = map(int, target_size.split('x'))
                        # use high quality resampling algorithm
                        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                except Exception as e:
                    print(f"⚠️ size adjustment failed: {e}, using original size")
            
            # convert format
            output_format = (target_format or "jpeg").lower()
            
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
            return image_bytes, "jpg"

