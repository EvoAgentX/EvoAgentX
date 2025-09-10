from typing import Dict
import base64
import mimetypes


def guess_mime_from_name(name: str, default: str = "image/png") -> str:
    guess, _ = mimetypes.guess_type(name)
    return guess or default


def path_to_data_url(path: str) -> str:
    mime = guess_mime_from_name(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def url_to_image_part(url: str) -> Dict:
    return {"type": "image_url", "image_url": {"url": url}}


def path_to_image_part(path: str) -> Dict:
    return url_to_image_part(path_to_data_url(path))


