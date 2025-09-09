from typing import Dict, List
import os
import io
from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler


class GeminiImageGenerationTool(Tool):
    name: str = "gemini_image_generation"
    description: str = (
        "Generate or edit images using Google Gemini. "
        "If only prompt is provided, runs text-to-image. If prompt + image_path is provided, runs image editing."
    )

    inputs: Dict[str, Dict] = {
        "prompt": {"type": "string", "description": "Text prompt describing the desired image or edit."},
        "image_path": {"type": "string", "description": "Optional local path to an input image for editing."},
        "model": {"type": "string", "description": "Gemini model id.", "default": "gemini-2.5-flash-image-preview"},
        "output_format": {"type": "string", "description": "Output image format (png|jpeg).", "default": "png"},
        "output_filename": {"type": "string", "description": "Optional filename to save to (basename or path)."},
    }
    required: List[str] = ["prompt"]

    def __init__(self, api_key: str = None, save_path: str = "./imgs", storage_handler: FileStorageHandler = None):
        super().__init__()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.save_path = save_path
        self.storage_handler = storage_handler

    def __call__(
        self,
        prompt: str,
        image_path: str = None,
        model: str = "gemini-2.5-flash-image-preview",
        output_format: str = "png",
        output_filename: str = None,
    ):
        try:
            from google import genai
            from PIL import Image
        except Exception as e:
            raise RuntimeError(
                f"Failed to import dependencies: {e}. Install 'google-genai' and 'pillow'."
            )

        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY not provided. Pass api_key= or set env GOOGLE_API_KEY.")

        client = genai.Client(api_key=self.api_key)

        # Build contents: text-only => generation; text+image => editing
        contents = [prompt]
        pil_image = None
        if image_path:
            # Open image via PIL and pass the PIL.Image to SDK as in official examples
            pil_image = Image.open(image_path)
            contents.append(pil_image)

        response = client.models.generate_content(model=model, contents=contents)

        # Collect first inline image; also collect any textual output
        text_chunks: List[str] = []
        image_bytes: bytes = None

        first_candidate = None
        if getattr(response, "candidates", None):
            first_candidate = response.candidates[0]

        if not first_candidate:
            return {"error": "No candidates returned by Gemini."}

        parts = getattr(first_candidate.content, "parts", []) or []
        for part in parts:
            if getattr(part, "text", None):
                text_chunks.append(part.text)
            elif getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                # inline_data.data is raw bytes of an image
                image_bytes = part.inline_data.data
                break

        result: Dict[str, str] = {}
        if text_chunks:
            result["text"] = "\n".join(t.strip() for t in text_chunks if t)

        if image_bytes:
            # Decide filename
            fmt = (output_format or "png").lower()
            if fmt not in ("png", "jpeg", "jpg"):
                fmt = "png"
            base = output_filename or ("gemini_gen.png" if fmt == "png" else "gemini_gen.jpg")

            # De-duplicate filename
            filename = base
            i = 1
            while self.storage_handler.exists(filename):
                name, ext = os.path.splitext(base)
                filename = f"{name}_{i}{ext or ('.png' if fmt=='png' else '.jpg')}"
                i += 1

            # Prepend save_path if provided
            full_filename = f"{self.save_path}/{filename}" if self.save_path else filename

            # If output_format differs from encoded, we can re-encode via PIL
            # Load bytes into PIL and save via storage handler
            try:
                from PIL import Image  # type: ignore
                pil = Image.open(io.BytesIO(image_bytes))
                with io.BytesIO() as buf:
                    pil.save(buf, format="PNG" if fmt == "png" else "JPEG")
                    saved = self.storage_handler.save(full_filename, buf.getvalue())
            except Exception:
                # Fallback: save raw bytes
                saved = self.storage_handler.save(full_filename, image_bytes)

            if saved.get("success"):
                result["file_path"] = full_filename
                result["storage_handler"] = type(self.storage_handler).__name__
            else:
                result["error"] = f"Failed to save image: {saved.get('error', 'Unknown error')}"

        return result or {"warning": "No image or text returned."}


class GeminiImageGenerationToolkit(Toolkit):
    """
    Toolkit for Google Gemini image generation/editing with storage handler integration.
    """

    def __init__(self, name: str = "GeminiImageGenerationToolkit", api_key: str = None, save_path: str = "./imgs", storage_handler: FileStorageHandler = None):
        if storage_handler is None:
            from .storage_file import LocalStorageHandler
            storage_handler = LocalStorageHandler(base_path=save_path)

        tool = GeminiImageGenerationTool(
            api_key=api_key,
            save_path=save_path,
            storage_handler=storage_handler,
        )

        tools = [tool]
        super().__init__(name=name, tools=tools)
        self.api_key = api_key
        self.save_path = save_path
        self.storage_handler = storage_handler


