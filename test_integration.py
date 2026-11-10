"""
End-to-end integration test with live Antigravity CLI and live Flask test client.
"""

import json
import unittest

from antigravity_client import AntigravityClient
from server import app, create_app


class TestAntigravityLiveIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize with live system agy binary
        create_app()
        cls.client = app.test_client()

    def test_live_registration_parsing(self):
        """
        Sends the exact payload that ProveedorIA in the Telegram Bot uses,
        and verifies that Antigravity produces valid JSON parsable by ProveedorIA.
        """
        user_message = "Mi nombre completo es Juan Pérez, mi correo es juan.perez@example.com"
        system_instruction = (
            "Eres un extractor de datos de registro para la plataforma FiuFlix.\n"
            "Tu tarea es analizar el mensaje del usuario y extraer:\n"
            "- nombre\n"
            "- apellido\n"
            "- email\n"
        )
        schema = {
            "type": "OBJECT",
            "properties": {
                "nombre": {"type": "STRING"},
                "apellido": {"type": "STRING"},
                "email": {"type": "STRING"},
            },
        }

        payload = {
            "contents": [
                {"parts": [{"text": user_message}]}
            ],
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        }

        response = self.client.post(
            "/v1beta/models/gemini-3.5-flash-lite:generateContent",
            json=payload,
            headers={"Content-Type": "application/json", "x-goog-api-key": "test-key"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Check candidates
        self.assertIn("candidates", data)
        self.assertTrue(len(data["candidates"]) > 0)

        # Check parts text
        part_text = data["candidates"][0]["content"]["parts"][0]["text"]
        print("\n[LIVE TEST OUTPUT FROM ANTIGRAVITY]:", part_text)

        # Must parse as valid JSON
        extracted = json.loads(part_text)
        self.assertIn("nombre", extracted)
        self.assertIn("apellido", extracted)
        self.assertIn("email", extracted)
        self.assertEqual(extracted["email"], "juan.perez@example.com")


if __name__ == "__main__":
    unittest.main()
