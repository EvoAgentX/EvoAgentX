from typing import Dict, Optional, List
from ...tool import Tool
from .openai_utils import create_openai_client, ensure_image_edit_compatible


class OpenAIGPTImage1EditV2(Tool):
    name: str = "openai_gpt_image1_edit_v2"
    description: str = "Edit images using OpenAI gpt-image-1 (direct, minimal validation)."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Edit instruction. Required."},
        "images": {"type": "array", "description": "Image path(s) png/webp/jpg <50MB. Required. Single string accepted and normalized to array."},
        "mask_path": {"type": "string", "description": "Optional PNG mask path (same size as first image)."},
        "size": {"type": "string", "description": "1024x1024 | 1536x1024 | 1024x1536 | auto"},
        "n": {"type": "integer", "description": "1-10"},
        "background": {"type": "string", "description": "transparent | opaque | auto"},
        "input_fidelity": {"type": "string", "description": "high | low"},
        "output_compression": {"type": "integer", "description": "0-100 for jpeg/webp"},
        "output_format": {"type": "string", "description": "png | jpeg | webp (default png)"},
        "partial_images": {"type": "integer", "description": "0-3 partial streaming"},
        "quality": {"type": "string", "description": "auto | high | medium | low"},
        "stream": {"type": "boolean", "description": "streaming mode"},
        "image_name": {"type": "string", "description": "Optional output base name"},
    }
    required: Optional[List[str]] = ["prompt", "images"]

    def __init__(self, api_key: str, organization_id: str = None, save_path: str = "./edited_images"):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.save_path = save_path

    def __call__(
        self,
        prompt: str,
        images: list,
        mask_path: str = None,
        size: str = None,
        n: int = None,
        background: str = None,
        input_fidelity: str = None,
        output_compression: int = None,
        output_format: str = None,
        partial_images: int = None,
        quality: str = None,
        stream: bool = None,
        image_name: str = None,
    ):
        try:
            client = create_openai_client(self.api_key, self.organization_id)

            # Accept either list[str] or a single string at runtime
            if isinstance(images, str):
                image_paths = [images]
            else:
                image_paths = list(images)

            opened_images = []
            temp_paths = []
            mask_fh = None
            try:
                # ensure compatibility and open files
                for p in image_paths:
                    use_path, tmp = ensure_image_edit_compatible(p)
                    if tmp:
                        temp_paths.append(tmp)
                    opened_images.append(open(use_path, "rb"))

                api_kwargs = {
                    "model": "gpt-image-1",
                    "prompt": prompt,
                    "image": opened_images if len(opened_images) > 1 else opened_images[0],
                }
                if size is not None:
                    api_kwargs["size"] = size
                if n is not None:
                    api_kwargs["n"] = n
                if background is not None:
                    api_kwargs["background"] = background
                if input_fidelity is not None:
                    api_kwargs["input_fidelity"] = input_fidelity
                if output_compression is not None:
                    api_kwargs["output_compression"] = output_compression
                if output_format is not None:
                    api_kwargs["output_format"] = output_format
                if partial_images is not None:
                    api_kwargs["partial_images"] = partial_images
                if quality is not None:
                    api_kwargs["quality"] = quality
                if stream is not None:
                    api_kwargs["stream"] = stream

                if mask_path:
                    mask_fh = open(mask_path, "rb")
                    api_kwargs["mask"] = mask_fh

                response = client.images.edit(**api_kwargs)
            finally:
                for fh in opened_images:
                    try:
                        fh.close()
                    except Exception:
                        pass
                if mask_fh:
                    try:
                        mask_fh.close()
                    except Exception:
                        pass
                # cleanup temps
                import os
                for tp in temp_paths:
                    try:
                        if tp and os.path.exists(tp):
                            os.remove(tp)
                    except Exception:
                        pass

            # Save base64 images
            import os, base64, time
            os.makedirs(self.save_path, exist_ok=True)
            results = []
            for i, img in enumerate(response.data):
                try:
                    img_bytes = base64.b64decode(img.b64_json)
                    ts = int(time.time())
                    if image_name:
                        filename = f"{image_name.rsplit('.', 1)[0]}_{i+1}.png"
                    else:
                        filename = f"gpt_image1_edit_v2_{ts}_{i+1}.png"
                    out_path = os.path.join(self.save_path, filename)
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)
                    results.append(out_path)
                except Exception as e:
                    results.append(f"Error saving image {i+1}: {e}")

            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"error": f"gpt-image-1 editing failed: {e}"}


