"""
Gemini Mock Server powered by Antigravity.
Emulates the Google Gemini REST API locally, delegating prompt execution
to Google Antigravity CLI using Gemini 3.8 Flash with low reasoning effort,
and returning 100% Gemini-compatible responses with automatic cleanup.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request, Response

from antigravity_client import (
    AntigravityClient,
    find_agy_binary,
    purge_conversations_by_preview,
)
from models import (
    SUPPORTED_MODELS,
    get_all_models_response,
    get_model_info,
    is_model_supported,
    normalize_model_name,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gemini_mock")

app = Flask(__name__)

# Global client holder
antigravity_client: AntigravityClient = None


def make_gemini_response(text: str, usage: Dict[str, int], model_name: str) -> Dict[str, Any]:
    """Builds a response matching Google Gemini API v1beta / v1 format."""
    clean_model = normalize_model_name(model_name)
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                    },
                ],
            }
        ],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("candidates_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
        "modelVersion": clean_model or "gemini-3.8-flash",
    }


def make_gemini_error(code: int, message: str, status_str: str = "INTERNAL") -> Tuple[Response, int]:
    payload = {
        "error": {
            "code": code,
            "message": message,
            "status": status_str,
        }
    }
    return jsonify(payload), code


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
@app.route("/v1beta/health", methods=["GET"])
@app.route("/v1/health", methods=["GET"])
def health_check():
    """Health check and server info."""
    return jsonify({
        "status": "ok",
        "service": "gemini-mock-antigravity",
        "antigravity_cli": antigravity_client.agy_bin if antigravity_client else "not configured",
        "execution_model": antigravity_client.model if antigravity_client else "gemini-3.8-flash",
        "execution_effort": antigravity_client.effort if antigravity_client else "low",
        "supported_models_count": len(SUPPORTED_MODELS),
        "description": "Universal Gemini API Mock powered by Antigravity (Gemini 3.8 Flash low effort)",
    })


@app.route("/models", methods=["GET"])
@app.route("/v1beta/models", methods=["GET"])
@app.route("/v1/models", methods=["GET"])
def list_models():
    """Lists all 29 official models supported by Gemini API Key."""
    return jsonify(get_all_models_response())


@app.route("/models/<path:model_name>", methods=["GET"])
@app.route("/v1beta/models/<path:model_name>", methods=["GET"])
@app.route("/v1/models/<path:model_name>", methods=["GET"])
def get_model(model_name: str):
    """Returns details for a specific model."""
    clean_name = normalize_model_name(model_name)
    if not is_model_supported(clean_name):
        return make_gemini_error(404, f"Model '{clean_name}' not found", "NOT_FOUND")
    return jsonify(get_model_info(clean_name))


def generate_dummy_embedding(text: str, dim: int = 768) -> List[float]:
    """Generates a deterministic float vector based on sha256 hash for embedContent."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
    values = []
    for _ in range(dim):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        val = (seed / 0x7FFFFFFF) * 2.0 - 1.0
        values.append(round(val, 6))
    return values


@app.route("/models/<path:model_action>", methods=["POST"])
@app.route("/v1beta/models/<path:model_action>", methods=["POST"])
@app.route("/v1/models/<path:model_action>", methods=["POST"])
def handle_model_action(model_action: str):
    """
    Handles official Gemini API model actions:
      - models/<model>:generateContent
      - models/<model>:streamGenerateContent (supports ?alt=sse and array response)
      - models/<model>:countTokens
      - models/<model>:embedContent
      - models/<model>:batchEmbedContents
    """
    if ":" in model_action:
        model_name, action = model_action.split(":", 1)
    else:
        model_name, action = model_action, "generateContent"

    clean_model = normalize_model_name(model_name)
    logger.info(f"==> Request received for model '{clean_model}', action '{action}'")

    req_data = request.get_json(silent=True) or {}

    # Action: embedContent
    if action == "embedContent":
        content = req_data.get("content", {})
        if isinstance(content, dict):
            parts = content.get("parts", [])
            text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
        else:
            text = str(content)
        dim = req_data.get("config", {}).get("outputDimensionality", 768)
        return jsonify({"embedding": {"values": generate_dummy_embedding(text, dim)}})

    # Action: batchEmbedContents
    if action == "batchEmbedContents":
        requests_list = req_data.get("requests", [])
        embeddings = []
        for r in requests_list:
            content = r.get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
            embeddings.append({"values": generate_dummy_embedding(text)})
        return jsonify({"embeddings": embeddings})

    # Normalized extraction of contents
    raw_contents = req_data.get("contents")
    if raw_contents is None:
        logger.error("No 'contents' provided in request")
        return make_gemini_error(400, "'contents' is required", "INVALID_ARGUMENT")

    if isinstance(raw_contents, str):
        contents = [{"role": "user", "parts": [{"text": raw_contents}]}]
    elif isinstance(raw_contents, dict):
        contents = [raw_contents]
    elif isinstance(raw_contents, list):
        contents = raw_contents
    else:
        contents = [{"role": "user", "parts": [{"text": str(raw_contents)}]}]

    # Normalize system instructions and generationConfig (supports camelCase and snake_case)
    system_instruction = req_data.get("systemInstruction") or req_data.get("system_instruction")
    generation_config = req_data.get("generationConfig") or req_data.get("generation_config", {})

    # Action: countTokens
    if action == "countTokens":
        total_chars = 0
        for c in contents:
            if isinstance(c, dict):
                for p in c.get("parts", []):
                    if isinstance(p, dict):
                        total_chars += len(p.get("text", ""))
                    elif isinstance(p, str):
                        total_chars += len(p)
            elif isinstance(c, str):
                total_chars += len(c)
        estimated_tokens = max(1, total_chars // 4)
        return jsonify({"totalTokens": estimated_tokens})

    # Action: generateContent & streamGenerateContent
    if action not in ("generateContent", "streamGenerateContent"):
        return make_gemini_error(
            400,
            f"Action '{action}' is not supported. Supported actions: generateContent, streamGenerateContent, countTokens, embedContent",
            "INVALID_ARGUMENT",
        )

    try:
        result = antigravity_client.generate_content(
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            model_name=clean_model,
        )
        response_text = result["text"]
        usage = result["usage"]
        logger.info(f"<== Response successfully generated ({len(response_text)} characters)")

        gemini_response = make_gemini_response(response_text, usage, clean_model)

        if action == "streamGenerateContent":
            # Per official Google API: if ?alt=sse or Accept: text/event-stream, return SSE
            alt_param = request.args.get("alt", "")
            accept_header = request.headers.get("Accept", "")
            if alt_param == "sse" or "text/event-stream" in accept_header:
                def generate_sse():
                    yield f"data: {json.dumps(gemini_response, ensure_ascii=False)}\n\n"
                return Response(generate_sse(), mimetype="text/event-stream")
            else:
                return jsonify([gemini_response]), 200

        return jsonify(gemini_response), 200

    except TimeoutError as te:
        logger.error(f"Timeout in Antigravity: {te}")
        return make_gemini_error(504, str(te), "DEADLINE_EXCEEDED")
    except Exception as e:
        logger.exception(f"Error processing request with Antigravity: {e}")
        return make_gemini_error(500, f"Internal error in mock server: {str(e)}", "INTERNAL")


def create_app(
    agy_bin: str = None,
    model: str = "gemini-3.8-flash",
    effort: str = "low",
    timeout: int = 60,
) -> Flask:
    global antigravity_client
    antigravity_client = AntigravityClient(
        agy_bin=agy_bin,
        model=model,
        effort=effort,
        timeout=timeout,
    )
    return app


def main():
    parser = argparse.ArgumentParser(
        description="Universal Gemini API Mock Server powered by Antigravity (Gemini 3.8 Flash low effort)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "5050")),
        help="Server port (default: 5050 or $PORT)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Server host (default: 0.0.0.0 or $HOST)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("ANTIGRAVITY_MODEL", "gemini-3.8-flash"),
        help="Model to use in Antigravity (default: gemini-3.8-flash)",
    )
    parser.add_argument(
        "--effort",
        type=str,
        default=os.getenv("ANTIGRAVITY_EFFORT", "low"),
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort level (default: low)",
    )
    parser.add_argument(
        "--agy-bin",
        type=str,
        default=os.getenv("ANTIGRAVITY_BIN"),
        help="Path to Antigravity CLI binary (agy)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("TIMEOUT", "60")),
        help="Timeout in seconds for Antigravity invocations (default: 60)",
    )
    parser.add_argument(
        "--purge-preview",
        type=str,
        default=None,
        help="Purges existing conversations in Antigravity matching this preview and exits",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    args = parser.parse_args()

    if args.purge_preview:
        count = purge_conversations_by_preview(args.purge_preview)
        print(f"Purge completed: {count} conversations deleted.")
        sys.exit(0)

    try:
        resolved_bin = find_agy_binary(args.agy_bin)
        print(f"[*] Antigravity binary detected: {resolved_bin}")
    except FileNotFoundError as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)

    create_app(
        agy_bin=args.agy_bin,
        model=args.model,
        effort=args.effort,
        timeout=args.timeout,
    )

    print("=" * 65)
    print("🚀 GEMINI MOCK SERVER (Universal Gemini API Bridge) STARTED")
    print(f"   Base URL:         http://localhost:{args.port}/v1beta")
    print(f"   Host:             {args.host}:{args.port}")
    print(f"   AI Engine:        Antigravity using {args.model} (effort: {args.effort})")
    print(f"   Gemini Models:    {len(SUPPORTED_MODELS)} supported models")
    print("   Cleanup:          Temporary conversations are purged automatically")
    print("=" * 65)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
