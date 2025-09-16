"""
OpenAI image utilities (refactored)
- Generation: supports dall-e-2, dall-e-3, gpt-image-1
- Editing: supports gpt-image-1 only
"""

from typing import Dict, Tuple

# Model config (generation)
OPENAI_MODEL_CONFIG: Dict[str, Dict] = {
    "dall-e-2": {
        "supported_params": ["prompt", "model", "size", "n", "response_format"],
        "size_options": ["256x256", "512x512", "1024x1024"],
        "quality_options": ["standard"],
        "n_max": 10,
        "supports_generation": True,
        "supports_editing": True,
    },
    "dall-e-3": {
        "supported_params": ["prompt", "model", "size", "quality", "n", "style", "response_format"],
        "size_options": ["1024x1024", "1792x1024", "1024x1792"],
        "quality_options": ["standard", "hd"],
        "style_options": ["vivid", "natural"],
        "n_max": 1,
        "supports_generation": True,
        "supports_editing": False,
    },
    "gpt-image-1": {
        "supported_params": [
            "prompt",
            "model",
            "size",
            "quality",
            "n",
            "background",
            "moderation",
            "output_compression",
            "output_format",
            "partial_images",
            "stream",
        ],
        "size_options": ["1024x1024", "1536x1024", "1024x1536", "auto"],
        "quality_options": ["auto", "high", "medium", "low"],
        "background_options": ["auto", "transparent", "opaque"],
        "moderation_options": ["auto", "low"],
        "output_format_options": ["png", "jpeg", "webp"],
        "n_max": 10,
        "supports_generation": True,
        "supports_editing": True,
    },
}

# Model config (editing, gpt-image-1 only)
OPENAI_EDITING_MODEL_CONFIG: Dict[str, Dict] = {
    "gpt-image-1": {
        "supported_params": [
            "prompt",
            "model",
            "size",
            "n",
            "background",
            "input_fidelity",
            "output_compression",
            "output_format",
            "partial_images",
            "quality",
            "stream",
        ],
        "size_options": ["1024x1024", "1536x1024", "1024x1536", "auto"],
        "quality_options": ["auto", "high", "medium", "low"],
        "background_options": ["auto", "transparent", "opaque"],
        "input_fidelity_options": ["high", "low"],
        "output_format_options": ["png", "jpeg", "webp"],
        "n_max": 10,
    }
}


def get_model_config(model: str, operation: str = "generation") -> Dict:
    if operation == "editing":
        return OPENAI_EDITING_MODEL_CONFIG.get(model, {})
    return OPENAI_MODEL_CONFIG.get(model, {})


def validate_parameter(model: str, param: str, value: any, operation: str = "generation") -> Tuple[bool, str]:
    config = get_model_config(model, operation)
    if not config:
        return False, f"Unsupported model: {model}"
    if param not in config.get("supported_params", []):
        return False, f"Parameter '{param}' is not supported by {model}"

    if param == "size" and value not in config.get("size_options", []):
        return False, f"Invalid size '{value}' for {model}. Supported: {config['size_options']}"
    if param == "quality" and value not in config.get("quality_options", []):
        return False, f"Invalid quality '{value}' for {model}. Supported: {config['quality_options']}"
    if param == "style" and value not in config.get("style_options", []):
        return False, f"Invalid style '{value}' for {model}. Supported: {config['style_options']}"
    if param == "background" and value not in config.get("background_options", []):
        return False, f"Invalid background '{value}' for {model}. Supported: {config['background_options']}"
    if param == "moderation" and value not in config.get("moderation_options", []):
        return False, f"Invalid moderation '{value}' for {model}. Supported: {config['moderation_options']}"
    if param == "input_fidelity" and value not in config.get("input_fidelity_options", []):
        return False, f"Invalid input_fidelity '{value}' for {model}. Supported: {config['input_fidelity_options']}"
    if param == "output_format" and value not in config.get("output_format_options", []):
        return False, f"Invalid output_format '{value}'. Supported: {config['output_format_options']}"
    if param == "n" and value > config.get("n_max", 10):
        return False, f"Invalid n {value}. Max: {config['n_max']}"
    if param == "output_compression" and (value < 0 or value > 100):
        return False, "output_compression must be between 0 and 100"
    if param == "partial_images" and (value < 0 or value > 3):
        return False, "partial_images must be between 0 and 3"
    return True, ""


def validate_parameters(model: str, params: Dict, operation: str = "generation") -> Dict:
    validated_params = {}
    warnings = []
    errors = []
    for param, value in params.items():
        if value is None:
            continue
        ok, msg = validate_parameter(model, param, value, operation)
        if ok:
            validated_params[param] = value
        else:
            if "not supported" in msg:
                warnings.append(msg)
            else:
                errors.append(msg)
    return {"validated_params": validated_params, "warnings": warnings, "errors": errors}


def build_validation_params(**kwargs) -> Dict:
    return {k: v for k, v in kwargs.items() if v is not None}


def handle_validation_result(validation_result: Dict) -> Dict | None:
    if validation_result["errors"]:
        return {"error": f"Parameter validation failed: {'; '.join(validation_result['errors'])}"}
    if validation_result["warnings"]:
        print(f"⚠️ Parameter warnings: {'; '.join(validation_result['warnings'])}")
        print("📝 Note: Continue with supported parameters only")
    return None


def create_openai_client(api_key: str, organization_id: str | None = None):
    from openai import OpenAI
    return OpenAI(api_key=api_key, organization=organization_id)

