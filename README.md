# Gemini Mock Server (Universal Gemini API Bridge with Antigravity)

A local Python server that **universally** emulates the **Google Gemini REST API** (v1beta and v1), delegating prompt execution to **Google Antigravity** (`agy`) using **Gemini 3.8 Flash with low reasoning effort (`--effort low`)** for maximum speed.

It allows developers to test AI agents, bots, and LLM applications locally **without spending tokens or requiring a funded Google Gemini API Key**, providing the exact same endpoints, structured outputs, and behaviors as production.

---

## 📋 Installation Requirements

1. **Python**:
   - Python 3.10 or higher (tested on Python 3.12).
2. **Google Antigravity CLI (`agy`)**:
   - Must be installed and authenticated on your machine (usually at `~/.local/bin/agy` or available in your `PATH`).
   - To verify:
     ```bash
     agy --help
     ```
3. **Python Dependencies**:
   - For the standard Flask server:
     ```bash
     pip install -r requirements.txt
     ```
   - **Zero-Dependencies Mode**: If you prefer not to install external pip libraries or run inside a restricted container, you can execute `standalone_server.py`, which relies exclusively on the Python standard library (`http.server`, `json`, `sqlite3`).

---

## ⚡ Speed Optimization: Gemini 3.8 Flash (Low Effort)

Every request forwarded to Antigravity is executed using:
- **Model:** `gemini-3.8-flash`
- **Reasoning Effort:** `--effort low`

This combination minimizes latency and suppresses excessive thinking tokens, achieving full response roundtrips in 2 to 6 seconds for complex extractions and tasks.

You can customize the model or effort level via environment variables or CLI flags:
```bash
./start.sh --model gemini-3.8-flash --effort low
# or using environment variables:
ANTIGRAVITY_MODEL=gemini-3.8-flash ANTIGRAVITY_EFFORT=low ./start.sh
```

---

## 🧹 Automatic Conversation Cleanup

Antigravity logs CLI executions in its local database (`conversation_summaries.db`, SQLite tables, and `brain/` folders).

To prevent automated test requests from cluttering your Antigravity UI and history:
- **Real-Time Auto-Cleanup:** After processing each request, the server retrieves the `conversation_id` created by `agy` and **immediately purges it** from SQLite databases and disk.
- **Manual Purge Utility:** If you need to clean up residual conversations matching a specific preview or title:
  ```bash
  python3 server.py --purge-preview "[SYSTEM INSTRUCTION]"
  ```

---

## 🤖 Supported Gemini Models

The server exposes and validates the **29 official models** currently supported by the Gemini API Key ([Google AI Studio](https://aistudio.google.com/docs/api-key)):

| Supported Models | | |
|---|---|---|
| `gemini-4-argon` | `gemini-3.8-flash` | `gemini-3.8-live` |
| `gemini-3.8-live-extended-thinking` | `gemini-3.8-flash-tts` | `gemini-3.8-flash-lite-tts` |
| `gemini-3.6-flash` | `gemini-3.5-flash-lite` | `gemini-3.1-flash-lite` |
| `gemini-nano-banana-2.1` | `gemini-3.1-flash-image` | `gemini-3.1-flash-lite-image` |
| `gemini-3-pro-image` | `gemini-3.5-transcribe` | `gemini-3.5-transcribe-live` |
| `gemini-omni-1.2-flash` | `gemini-omni-1.1-flash` | `gemini-3.1-pro-preview` |
| `gemini-3-flash-preview` | `gemini-3.5-live-translate-preview` | `gemini-3.1-flash-live-preview` |
| `gemini-3.1-flash-tts-preview` | `gemini-2.5-flash` | `gemini-2.5-flash-image` |
| `gemini-2.5-flash-native-audio-preview-12-2025` | `gemini-2.5-flash-preview-tts` | `gemini-2.5-flash-lite` |
| `gemini-2.5-pro` | `gemini-2.5-pro-preview-tts` | |

Query the model catalog directly:
```bash
curl http://localhost:5050/v1beta/models
```
Or inspect a specific model:
```bash
curl http://localhost:5050/v1beta/models/gemini-3.8-flash
```

---

## 🌐 Universal Gemini API Compatibility

This server is designed as a drop-in replacement for the official Google API:

1. **Input Flexibility:**
   - Multi-turn conversation turns (`contents` with roles `user`, `model`, `system`).
   - System instructions (`systemInstruction` and `system_instruction`).
   - Generation parameters (`temperature`, `maxOutputTokens`, `stopSequences`).
   - Structured JSON outputs via **JSON Schema** (`generationConfig.responseMimeType: "application/json"` and `responseSchema`).
   - Multimodal part references (`inlineData`, `fileData`).
2. **Emulated Endpoints:**
   - `POST /v1beta/models/<model>:generateContent` (also `/v1/...` and `/models/...`)
   - `POST /v1beta/models/<model>:streamGenerateContent` (with SSE `?alt=sse` and array modes)
   - `POST /v1beta/models/<model>:countTokens`
   - `POST /v1beta/models/<model>:embedContent` and `:batchEmbedContents`
   - `GET /v1beta/models` and `GET /v1beta/models/<model>`
   - `GET /health` and `GET /v1beta/health`
3. **Flexible Authentication:**
   - Accepts API keys via `x-goog-api-key` header, `Authorization: Bearer <key>` header, or `?key=<key>` query parameter.

---

## 🚀 Running the Server

### Using the Startup Script (Recommended)

```bash
cd gemini-mock
./start.sh
```

By default it listens on `http://0.0.0.0:5050`. To override the port:

```bash
PORT=8000 ./start.sh
```

### Directly with Python

```bash
# Standard Flask server
python3 server.py --port 5050

# Standalone server (zero pip dependencies)
python3 standalone_server.py --port 5050
```

---

## 💡 Usage Examples

### 1. With the Telegram Bot (`bot/`)

In your bot's `.env` file (`bot/.env`):

```dotenv
# If running inside Docker on Linux:
URL_BASE_IA=http://172.18.0.1:5050/v1beta

# If running on the host machine directly:
# URL_BASE_IA=http://localhost:5050/v1beta

MODELO_IA=gemini-3.8-flash
API_KEY_IA=local-dev-mock-key
```

### 2. Using `curl` (Natural Language Prompt)

```bash
curl -X POST http://localhost:5050/v1beta/models/gemini-3.8-flash:generateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: test-key" \
  -d '{
    "contents": [
      { "parts": [{ "text": "Explain briefly what Docker is in two sentences." }] }
    ]
  }'
```

### 3. Using `curl` (Structured JSON Extraction with Schema)

```bash
curl -X POST http://localhost:5050/v1beta/models/gemini-3.8-flash:generateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: test-key" \
  -d '{
    "contents": [
      { "parts": [{ "text": "My name is Alan Turing and my email is alan@turing.ac.uk" }] }
    ],
    "systemInstruction": {
      "parts": [{ "text": "Extract user contact details." }]
    },
    "generationConfig": {
      "responseMimeType": "application/json",
      "responseSchema": {
        "type": "OBJECT",
        "properties": {
          "name": { "type": "STRING" },
          "email": { "type": "STRING" }
        },
        "required": ["name", "email"]
      }
    }
  }'
```

### 4. Using the Official Python SDK (`google-genai`)

```python
from google import genai
from google.genai import types

# Configure the client to point to your local mock server
client = genai.Client(
    api_key="dummy-key",
    http_options=types.HttpOptions(base_url="http://localhost:5050")
)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Hello world!",
)
print(response.text)
```

---

## 🧪 Tests

```bash
cd gemini-mock

# Unit tests (20 tests)
python3 -m unittest test_server.py

# End-to-end live integration test with Antigravity
python3 -m unittest test_integration.py
```
