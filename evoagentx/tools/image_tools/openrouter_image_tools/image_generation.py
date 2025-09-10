from typing import Dict, List
import os
import requests
from ...tool import Tool


class OpenRouterImageGenerationEditTool(Tool):
    name: str = "openrouter_image_generation_edit"
    description: str = (
        "Text-to-image and image-editing via OpenRouter models (e.g., google/gemini-2.5-flash-image-preview). "
        "Without images: generate. With images (URLs only for now): edit/compose."
    )

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "Text prompt."},
        "image_urls": {"type": "array", "description": "Remote image URLs (optional)."},
        "model": {"type": "string", "description": "OpenRouter model id.", "default": "google/gemini-2.5-flash-image-preview"},
        "api_key": {"type": "string", "description": "OpenRouter API key (fallback to env OPENROUTER_API_KEY)."},
        "save_path": {"type": "string", "description": "Directory to save images (when data URLs).", "default": "./openrouter_images"},
        "output_basename": {"type": "string", "description": "Base filename for outputs.", "default": "or_gen"}
    }
    required: List[str] = ["prompt"]

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    def __call__(
        self,
        prompt: str,
        image_urls: list = None,
        model: str = "google/gemini-2.5-flash-image-preview",
        api_key: str = None,
        save_path: str = "./openrouter_images",
        output_basename: str = "or_gen",
    ):
        key = api_key or self.api_key
        if not key:
            return {"error": "OPENROUTER_API_KEY not provided."}

        messages = [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": messages, "modalities": ["image", "text"]}

        if image_urls:
            payload["messages"][0] = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
                ]
            }

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        url = "https://openrouter.ai/api/v1/chat/completions"
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        saved_paths: List[str] = []
        if data.get("choices"):
            msg = data["choices"][0]["message"]
            images = msg.get("images") or []
            for im in images:
                image_url = im.get("image_url", {}).get("url")
                if not image_url:
                    continue
                # Save data URL locally and collect file path
                if image_url.startswith("data:") and "," in image_url:
                    import base64
                    os.makedirs(save_path or "./openrouter_images", exist_ok=True)
                    header, b64data = image_url.split(",", 1)
                    # mime → extension
                    mime = "image/png"
                    if ";" in header:
                        mime = header.split(":", 1)[1].split(";", 1)[0] or mime
                    ext = ".png"
                    if mime == "image/jpeg":
                        ext = ".jpg"
                    elif mime == "image/webp":
                        ext = ".webp"
                    elif mime == "image/heic":
                        ext = ".heic"
                    elif mime == "image/heif":
                        ext = ".heif"
                    base = output_basename or "or_gen"
                    idx = 0
                    while True:
                        name = f"{base}{'' if idx==0 else '_'+str(idx)}{ext}"
                        fullpath = os.path.join(save_path or "./openrouter_images", name)
                        if not os.path.exists(fullpath):
                            break
                        idx += 1
                    with open(fullpath, "wb") as f:
                        f.write(base64.b64decode(b64data))
                    saved_paths.append(fullpath)

        if saved_paths:
            return {"saved_paths": saved_paths}
        return {"warning": "No image returned or saved.", "raw": data}


