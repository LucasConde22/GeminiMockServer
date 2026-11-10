"""
Zero-dependency Gemini Mock Server using Python's built-in http.server.
Fully compatible with official Gemini API clients without requiring Flask.
"""

import argparse
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
import os
import sys
from urllib.parse import parse_qs, urlparse

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gemini_mock.standalone")

antigravity_client: AntigravityClient = None


def make_gemini_response(text: str, usage: dict, model_name: str) -> dict:
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
                "safetyRatings": [],
            }
        ],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("candidates_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
        "modelVersion": clean_model or "gemini-3.8-flash",
    }


def make_gemini_error(code: int, message: str, status_str: str = "INTERNAL") -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "status": status_str,
        }
    }


class GeminiMockHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, data: dict):
        response_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("", "/health", "/v1beta/health", "/v1/health"):
            self._send_json(200, {
                "status": "ok",
                "service": "gemini-mock-antigravity-standalone",
                "antigravity_cli": antigravity_client.agy_bin if antigravity_client else "none",
                "execution_model": antigravity_client.model if antigravity_client else "gemini-3.8-flash",
                "execution_effort": antigravity_client.effort if antigravity_client else "low",
                "supported_models_count": len(SUPPORTED_MODELS),
            })
            return

        if path in ("/models", "/v1beta/models", "/v1/models"):
            self._send_json(200, get_all_models_response())
            return

        for prefix in ("/v1beta/models/", "/v1/models/", "/models/"):
            if path.startswith(prefix):
                mname = path[len(prefix):]
                clean = normalize_model_name(mname)
                if is_model_supported(clean):
                    self._send_json(200, get_model_info(clean))
                else:
                    self._send_json(404, make_gemini_error(404, f"Model '{clean}' not found", "NOT_FOUND"))
                return

        self._send_json(404, make_gemini_error(404, f"Route not found: {self.path}", "NOT_FOUND"))

    def do_POST(self):
        parsed = urlparse(self.path)
        raw_path = parsed.path
        query_params = parse_qs(parsed.query)

        normalized_path = raw_path
        for prefix in ("/v1beta/", "/v1/", "/"):
            if normalized_path.startswith(prefix):
                normalized_path = normalized_path[len(prefix):]
                break

        if not normalized_path.startswith("models/"):
            self._send_json(404, make_gemini_error(404, f"Route not found: {raw_path}", "NOT_FOUND"))
            return

        model_action = normalized_path[len("models/"):]
        if ":" in model_action:
            model_name, action = model_action.split(":", 1)
        else:
            model_name, action = model_action, "generateContent"

        clean_model = normalize_model_name(model_name)

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)

        try:
            req_data = json.loads(post_body.decode("utf-8")) if post_body else {}
        except Exception:
            self._send_json(400, make_gemini_error(400, "Request body is not valid JSON", "INVALID_ARGUMENT"))
            return

        # Action: embedContent
        if action == "embedContent":
            content = req_data.get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
            values = []
            for _ in range(768):
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                values.append(round((seed / 0x7FFFFFFF) * 2.0 - 1.0, 6))
            self._send_json(200, {"embedding": {"values": values}})
            return

        raw_contents = req_data.get("contents")
        if raw_contents is None:
            self._send_json(400, make_gemini_error(400, "'contents' is required", "INVALID_ARGUMENT"))
            return

        if isinstance(raw_contents, str):
            contents = [{"role": "user", "parts": [{"text": raw_contents}]}]
        elif isinstance(raw_contents, dict):
            contents = [raw_contents]
        elif isinstance(raw_contents, list):
            contents = raw_contents
        else:
            contents = [{"role": "user", "parts": [{"text": str(raw_contents)}]}]

        if action == "countTokens":
            total_chars = 0
            for c in contents:
                if isinstance(c, dict):
                    for p in c.get("parts", []):
                        if isinstance(p, dict):
                            total_chars += len(p.get("text", ""))
                elif isinstance(c, str):
                    total_chars += len(c)
            self._send_json(200, {"totalTokens": max(1, total_chars // 4)})
            return

        if action not in ("generateContent", "streamGenerateContent"):
            self._send_json(400, make_gemini_error(400, f"Action '{action}' is not supported", "INVALID_ARGUMENT"))
            return

        system_instruction = req_data.get("systemInstruction") or req_data.get("system_instruction")
        generation_config = req_data.get("generationConfig") or req_data.get("generation_config", {})

        try:
            result = antigravity_client.generate_content(
                contents=contents,
                system_instruction=system_instruction,
                generation_config=generation_config,
                model_name=clean_model,
            )
            response_data = make_gemini_response(result["text"], result["usage"], clean_model)
            alt = query_params.get("alt", [""])[0]
            if action == "streamGenerateContent" and (alt == "sse" or "text/event-stream" in self.headers.get("Accept", "")):
                sse_payload = f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(sse_payload)))
                self.end_headers()
                self.wfile.write(sse_payload)
            elif action == "streamGenerateContent":
                self._send_json(200, [response_data])
            else:
                self._send_json(200, response_data)
        except TimeoutError as te:
            self._send_json(504, make_gemini_error(504, str(te), "DEADLINE_EXCEEDED"))
        except Exception as e:
            self._send_json(500, make_gemini_error(500, str(e), "INTERNAL"))


def run_standalone(
    host: str = "0.0.0.0",
    port: int = 5050,
    model: str = "gemini-3.8-flash",
    effort: str = "low",
    agy_bin: str = None,
    timeout: int = 60,
):
    global antigravity_client
    antigravity_client = AntigravityClient(
        agy_bin=agy_bin,
        model=model,
        effort=effort,
        timeout=timeout,
    )
    server_address = (host, port)
    httpd = HTTPServer(server_address, GeminiMockHandler)
    logger.info(
        f"Starting standalone HTTP server at http://{host}:{port}/v1beta "
        f"using model {model} (effort={effort})"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="gemini-3.8-flash")
    parser.add_argument("--effort", type=str, default="low")
    parser.add_argument("--agy-bin", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    run_standalone(
        host=args.host,
        port=args.port,
        model=args.model,
        effort=args.effort,
        agy_bin=args.agy_bin,
        timeout=args.timeout,
    )
