"""
Unit Tests for Gemini Mock Server (Universal Gemini API Bridge).
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from antigravity_client import (
    AntigravityClient,
    extract_json_content,
    extract_text_from_parts,
    normalize_gemini_schema,
)
from models import (
    SUPPORTED_MODELS,
    get_all_models_response,
    get_model_info,
    is_model_supported,
)
from server import app, create_app


class TestSupportedModels(unittest.TestCase):
    def test_supported_models_count(self):
        # Exactly 29 official models supported by Gemini API Key
        self.assertEqual(len(SUPPORTED_MODELS), 29)

    def test_key_models_present(self):
        expected_models = [
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
        for m in expected_models:
            self.assertTrue(is_model_supported(m), f"Model {m} should be supported")

    def test_model_info_structure(self):
        info = get_model_info("gemini-3.8-flash")
        self.assertEqual(info["name"], "models/gemini-3.8-flash")
        self.assertEqual(info["displayName"], "gemini-3.8-flash")
        self.assertIn("generateContent", info["supportedGenerationMethods"])


class TestAntigravityUtilities(unittest.TestCase):
    def test_normalize_gemini_schema(self):
        gemini_schema = {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "age": {"type": "INTEGER"},
                "items": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
            },
            "required": ["name"],
        }
        normalized = normalize_gemini_schema(gemini_schema)
        self.assertEqual(normalized["type"], "object")
        self.assertEqual(normalized["properties"]["name"]["type"], "string")
        self.assertEqual(normalized["properties"]["age"]["type"], "integer")
        self.assertEqual(normalized["properties"]["items"]["type"], "array")
        self.assertEqual(normalized["properties"]["items"]["items"]["type"], "string")

    def test_extract_text_from_parts(self):
        parts = [
            {"text": "Hello "},
            {"text": "World!"},
            "Direct text",
        ]
        result = extract_text_from_parts(parts)
        self.assertEqual(result, "Hello \nWorld!\nDirect text")

    def test_extract_json_content_plain(self):
        raw = '{"name": "Alice", "city": "London"}'
        result = extract_json_content(raw)
        data = json.loads(result)
        self.assertEqual(data["name"], "Alice")
        self.assertEqual(data["city"], "London")

    def test_extract_json_content_markdown_block(self):
        raw = """```json
{
  "name": "Bob",
  "email": "bob@example.com"
}
```"""
        result = extract_json_content(raw)
        data = json.loads(result)
        self.assertEqual(data["name"], "Bob")
        self.assertEqual(data["email"], "bob@example.com")

    def test_extract_json_content_with_surrounding_text(self):
        raw = 'Sure! Here is the JSON:\n{"name": "Charlie"}\nHope this helps.'
        result = extract_json_content(raw)
        data = json.loads(result)
        self.assertEqual(data["name"], "Charlie")


class TestGeminiMockServer(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock(spec=AntigravityClient)
        self.mock_client.agy_bin = "/mock/bin/agy"
        self.mock_client.model = "gemini-3.8-flash"
        self.mock_client.effort = "low"
        self.app = app
        self.client = self.app.test_client()

        import server
        server.antigravity_client = self.mock_client

    def test_health_check(self):
        for path in ("/health", "/v1beta/health", "/v1/health"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["service"], "gemini-mock-antigravity")
            self.assertEqual(data["execution_model"], "gemini-3.8-flash")
            self.assertEqual(data["execution_effort"], "low")
            self.assertEqual(data["supported_models_count"], 29)

    def test_list_all_29_models(self):
        response = self.client.get("/v1beta/models")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("models", data)
        self.assertEqual(len(data["models"]), 29)
        model_names = [m["name"] for m in data["models"]]
        self.assertIn("models/gemini-3.8-flash", model_names)
        self.assertIn("models/gemini-4-argon", model_names)
        self.assertIn("models/gemini-2.5-pro", model_names)

    def test_get_single_model(self):
        response = self.client.get("/v1beta/models/gemini-3.8-flash")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["name"], "models/gemini-3.8-flash")

    def test_get_invalid_model_returns_404(self):
        response = self.client.get("/v1beta/models/unknown-model")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn("error", data)

    def test_count_tokens(self):
        payload = {
            "contents": [
                {"parts": [{"text": "Hello world from Antigravity Gemini Mock Server!"}]}
            ]
        }
        response = self.client.post("/v1beta/models/gemini-3.8-flash:countTokens", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("totalTokens", data)
        self.assertGreater(data["totalTokens"], 0)

    def test_generate_content_success(self):
        expected_json = json.dumps({"first_name": "John", "last_name": "Doe", "email": "john@test.com"})
        self.mock_client.generate_content.return_value = {
            "text": expected_json,
            "usage": {"prompt_tokens": 15, "candidates_tokens": 20, "total_tokens": 35},
        }

        payload = {
            "contents": [
                {"parts": [{"text": "My name is John Doe, email john@test.com"}]}
            ],
            "system_instruction": {
                "parts": [{"text": "You are a data extractor"}]
            },
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "first_name": {"type": "STRING"},
                        "last_name": {"type": "STRING"},
                        "email": {"type": "STRING"},
                    },
                },
            },
        }

        response = self.client.post(
            "/v1beta/models/gemini-3.8-flash:generateContent",
            json=payload,
            headers={"x-goog-api-key": "fake-key"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("candidates", data)
        self.assertGreater(len(data["candidates"]), 0)
        part_text = data["candidates"][0]["content"]["parts"][0]["text"]
        self.assertEqual(part_text, expected_json)

        parsed_inner = json.loads(part_text)
        self.assertEqual(parsed_inner["first_name"], "John")

        self.assertEqual(data["usageMetadata"]["promptTokenCount"], 15)
        self.assertEqual(data["usageMetadata"]["candidatesTokenCount"], 20)

    def test_stream_generate_content(self):
        self.mock_client.generate_content.return_value = {
            "text": "Streaming response text",
            "usage": {"prompt_tokens": 5, "candidates_tokens": 5, "total_tokens": 10},
        }

        payload = {"contents": [{"parts": [{"text": "Hello streaming"}]}]}
        response = self.client.post(
            "/v1beta/models/gemini-3.8-flash:streamGenerateContent",
            json=payload,
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["candidates"][0]["content"]["parts"][0]["text"], "Streaming response text")

    def test_unsupported_action(self):
        response = self.client.post(
            "/v1beta/models/gemini-3.8-flash:invalidAction",
            json={"contents": [{"parts": [{"text": "hola"}]}]},
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["status"], "INVALID_ARGUMENT")

    def test_embed_content(self):
        payload = {
            "content": {
                "parts": [{"text": "What is the meaning of life?"}]
            },
            "config": {
                "outputDimensionality": 768
            }
        }
        response = self.client.post("/v1beta/models/gemini-3.8-flash:embedContent", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("embedding", data)
        self.assertIn("values", data["embedding"])
        self.assertEqual(len(data["embedding"]["values"]), 768)

    def test_stream_generate_content_sse(self):
        self.mock_client.generate_content.return_value = {
            "text": "Streaming response via SSE",
            "usage": {"prompt_tokens": 5, "candidates_tokens": 5, "total_tokens": 10},
        }

        payload = {"contents": "Hello streaming"}
        response = self.client.post(
            "/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse",
            json=payload,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.content_type)
        self.assertTrue(response.data.startswith(b"data: "))

    def test_official_camel_case_system_instruction_and_config(self):
        self.mock_client.generate_content.return_value = {
            "text": '{"result": "ok"}',
            "usage": {"prompt_tokens": 10, "candidates_tokens": 5, "total_tokens": 15},
        }

        payload = {
            "contents": "My text",
            "systemInstruction": {
                "parts": [{"text": "System rule"}]
            },
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {"result": {"type": "STRING"}}
                }
            }
        }

        response = self.client.post("/v1beta/models/gemini-3.8-flash:generateContent", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["candidates"][0]["content"]["parts"][0]["text"], '{"result": "ok"}')

    def test_empty_contents(self):
        response = self.client.post(
            "/v1beta/models/gemini-3.8-flash:generateContent",
            json={},
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
