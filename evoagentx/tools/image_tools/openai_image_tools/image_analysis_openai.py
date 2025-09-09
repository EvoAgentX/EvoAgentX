from typing import Dict, Optional, List
from ...tool import Tool
from .openai_utils import create_openai_client


class OpenAIImageAnalysisV2(Tool):
    name: str = "openai_image_analysis_v2"
    description: str = "Simple image analysis via OpenAI Responses API (input_text + input_image)."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "User question/instruction. Required."},
        "image_url": {"type": "string", "description": "HTTP(S) image URL. Required."},
        "model": {"type": "string", "description": "OpenAI model for responses.create (e.g., gpt-4o-mini, gpt-4.1, gpt-5). Optional."},
    }
    required: Optional[List[str]] = ["prompt", "image_url"]

    def __init__(self, api_key: str, organization_id: str = None, model: str = "gpt-4o-mini"):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model

    def __call__(
        self,
        prompt: str,
        image_url: str,
        model: str = None,
    ):
        try:
            client = create_openai_client(self.api_key, self.organization_id)
            actual_model = model if model else self.model

            response = client.responses.create(
                model=actual_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ],
            )

            # Prefer unified output_text when present
            text = getattr(response, "output_text", None)
            if text is None:
                # Fallback: try to assemble from content if SDK shape differs
                try:
                    choices = getattr(response, "output", None) or getattr(response, "choices", None)
                    if choices and isinstance(choices, list):
                        first = choices[0]
                        text = getattr(first, "message", {}).get("content", "") if isinstance(first, dict) else ""
                except Exception:
                    text = ""

            return {"content": text or ""}
        except Exception as e:
            return {"error": f"OpenAI image analysis failed: {e}"}


