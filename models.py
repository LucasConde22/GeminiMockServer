"""
Gemini Supported Models Definition.
Lists the 29 models currently supported by the Gemini API Key
(https://aistudio.google.com/docs/api-key) and provides model metadata.
"""

from typing import Any, Dict, List, Optional

SUPPORTED_MODELS: List[str] = [
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.8-live",
    "gemini-3.8-live-extended-thinking",
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-nano-banana-2.1",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image",
    "gemini-3.5-transcribe",
    "gemini-3.5-transcribe-live",
    "gemini-omni-1.2-flash",
    "gemini-omni-1.1-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.5-live-translate-preview",
    "gemini-3.1-flash-live-preview",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-tts",
]


def normalize_model_name(name: str) -> str:
    """Strips 'models/' prefix and trailing whitespace."""
    clean = name.strip()
    if clean.startswith("models/"):
        clean = clean[len("models/"):]
    return clean


def is_model_supported(model_name: str) -> bool:
    clean = normalize_model_name(model_name)
    return clean in SUPPORTED_MODELS


def get_model_info(model_name: str) -> Dict[str, Any]:
    clean = normalize_model_name(model_name)
    return {
        "name": f"models/{clean}",
        "version": "1.0",
        "displayName": clean,
        "description": f"Official Google Gemini model ({clean}) locally supported via Antigravity",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 8192,
        "supportedGenerationMethods": [
            "generateContent",
            "streamGenerateContent",
            "countTokens",
        ],
    }


def get_all_models_response() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "models": [get_model_info(m) for m in SUPPORTED_MODELS]
    }
