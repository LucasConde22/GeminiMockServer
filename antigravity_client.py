"""
Antigravity Bridge for Gemini Mock Server.
Translates Google Gemini API requests into Antigravity CLI (agy) commands,
executes them non-interactively using Gemini 3.8 Flash with low reasoning effort,
sanitizes output, and automatically purges ephemeral conversations from Antigravity.
"""

import glob
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gemini_mock.antigravity")

DEFAULT_CLI_PATHS = [
    os.path.expanduser("~/.local/bin/agy"),
    "/usr/local/bin/agy",
    "/usr/bin/agy",
]

DEFAULT_MODEL = os.getenv("ANTIGRAVITY_MODEL", "gemini-3.8-flash")
DEFAULT_EFFORT = os.getenv("ANTIGRAVITY_EFFORT", "low")


def find_agy_binary(custom_path: Optional[str] = None) -> str:
    """Locate the agy CLI binary path."""
    if custom_path and os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
        return custom_path

    env_path = os.getenv("ANTIGRAVITY_BIN")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    which_path = shutil.which("agy")
    if which_path:
        return which_path

    for candidate in DEFAULT_CLI_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "Antigravity 'agy' binary not found on the system. "
        "Ensure it is in PATH or set the ANTIGRAVITY_BIN environment variable."
    )


def normalize_gemini_schema(schema: Any) -> Any:
    """
    Recursively converts Gemini schema types (e.g. UPPERCASE 'OBJECT', 'STRING')
    into standard JSON schema types (lowercase 'object', 'string').
    """
    if isinstance(schema, dict):
        normalized = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                normalized[k] = v.lower()
            else:
                normalized[k] = normalize_gemini_schema(v)
        return normalized
    elif isinstance(schema, list):
        return [normalize_gemini_schema(item) for item in schema]
    return schema


def extract_text_from_parts(parts: List[Any]) -> str:
    """Extract plain text and multimodal hints from Gemini content parts."""
    text_chunks = []
    for part in parts:
        if isinstance(part, dict):
            if "text" in part:
                text_chunks.append(part["text"])
            elif "inlineData" in part or "inline_data" in part:
                blob = part.get("inlineData") or part.get("inline_data") or {}
                mime = blob.get("mimeType") or blob.get("mime_type", "multimedia")
                text_chunks.append(f"[Attached media content: {mime}]")
            elif "fileData" in part or "file_data" in part:
                f_data = part.get("fileData") or part.get("file_data") or {}
                uri = f_data.get("fileUri") or f_data.get("file_uri", "file")
                text_chunks.append(f"[Attached file: {uri}]")
            elif "functionCall" in part or "function_call" in part:
                fn = part.get("functionCall") or part.get("function_call") or {}
                text_chunks.append(f"[Previous function call: {fn.get('name')}({json.dumps(fn.get('args', {}))})]")
            elif "functionResponse" in part or "function_response" in part:
                resp = part.get("functionResponse") or part.get("function_response") or {}
                text_chunks.append(f"[Previous function response for {resp.get('name')}: {json.dumps(resp.get('response', {}))}]")
        elif isinstance(part, str):
            text_chunks.append(part)
    return "\n".join(text_chunks)


def extract_json_content(raw_text: str) -> str:
    """
    Extracts and sanitizes a valid JSON string from raw model output,
    stripping markdown fences (```json ... ```) or conversational fluff.
    """
    cleaned = raw_text.strip()

    # Check for Markdown code fences
    fence_pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(fence_pattern, cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    # Try parsing directly
    try:
        parsed = json.loads(cleaned)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # Find the outermost JSON object or array delimiters
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidate = cleaned[first_bracket : last_bracket + 1]
        try:
            parsed = json.loads(candidate)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    return cleaned


def cleanup_antigravity_conversation(conversation_id: str) -> None:
    """
    Deletes the ephemeral conversation created by Antigravity CLI from
    SQLite databases, conversation history, brain folders, and caches.
    This guarantees that automated API requests do not pollute the user's Antigravity history.
    """
    if not conversation_id or not isinstance(conversation_id, str):
        return

    # Safety check: do not delete main system conversations
    if len(conversation_id) < 10:
        return

    base_dirs = [
        os.path.expanduser("~/.gemini/antigravity-cli"),
        os.path.expanduser("~/.gemini/antigravity"),
    ]

    for base in base_dirs:
        if not os.path.isdir(base):
            continue

        # 1. Clean from SQLite databases
        db_path = os.path.join(base, "conversation_summaries.db")
        if os.path.isfile(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("DELETE FROM conversation_summaries WHERE conversation_id = ?", (conversation_id,))
                try:
                    cur.execute("DELETE FROM conversations_fts WHERE conversation_id = ?", (conversation_id,))
                except Exception:
                    pass
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug(f"Failed to clean up {conversation_id} from {db_path}: {e}")

        # 2. Remove conversation files (.db, .db-wal, .db-shm, etc.)
        for f in glob.glob(os.path.join(base, "conversations", f"{conversation_id}*")):
            try:
                os.remove(f)
            except Exception as e:
                logger.debug(f"Failed to remove conversation file {f}: {e}")

        # 3. Remove brain folder if any
        brain_dir = os.path.join(base, "brain", conversation_id)
        if os.path.isdir(brain_dir):
            try:
                shutil.rmtree(brain_dir, ignore_errors=True)
            except Exception as e:
                logger.debug(f"Failed to remove brain dir {brain_dir}: {e}")

        # 4. Remove reference from last_conversations.json cache
        cache_file = os.path.join(base, "cache", "last_conversations.json")
        if os.path.isfile(cache_file):
            try:
                with open(cache_file, "r") as cf:
                    cache_data = json.load(cf)
                modified = False
                if isinstance(cache_data, dict):
                    for k, v in list(cache_data.items()):
                        if v == conversation_id:
                            del cache_data[k]
                            modified = True
                if modified:
                    with open(cache_file, "w") as cf:
                        json.dump(cache_data, cf)
            except Exception:
                pass


def purge_conversations_by_preview(pattern: str = "[SYSTEM INSTRUCTION]") -> int:
    """
    Purges any leftover conversations in Antigravity matching a given preview or title substring.
    """
    base_dirs = [
        os.path.expanduser("~/.gemini/antigravity-cli"),
        os.path.expanduser("~/.gemini/antigravity"),
    ]
    purged_cids = set()

    for base in base_dirs:
        db_path = os.path.join(base, "conversation_summaries.db")
        if not os.path.isfile(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT conversation_id, title, preview FROM conversation_summaries")
            for cid, title, preview in cur.fetchall():
                if (
                    pattern in title
                    or pattern in preview
                    or "[INSTRUCCIÓN" in title
                    or "[INSTRUCCIÓN" in preview
                    or "[INSTRUCCIONES" in title
                    or "[INSTRUCCIONES" in preview
                ):
                    purged_cids.add(cid)
            conn.close()
        except Exception:
            pass

    for cid in purged_cids:
        cleanup_antigravity_conversation(cid)

    return len(purged_cids)


class AntigravityClient:
    def __init__(
        self,
        agy_bin: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        timeout: int = 60,
    ):
        self.agy_bin = find_agy_binary(agy_bin)
        self.model = model
        self.effort = effort
        self.timeout = timeout
        logger.info(
            f"Antigravity client initialized: bin={self.agy_bin}, "
            f"model={self.model}, effort={self.effort}"
        )

    def build_prompt_and_instructions(
        self,
        contents: Any,
        system_instruction: Optional[Any] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
        """
        Builds a general-purpose prompt compatible with any Gemini REST API request:
        - Normalizes contents (string, dict, or list of Content turns)
        - Incorporates system instructions (systemInstruction or system_instruction)
        - Supports camelCase and snake_case generationConfig parameters
        - Respects JSON Schema / application/json requests
        - Appends generation constraints (stop sequences, temperature, etc.)
        """
        generation_config = generation_config or {}
        response_mime = generation_config.get("responseMimeType") or generation_config.get("response_mime_type", "")
        raw_schema = generation_config.get("responseSchema") or generation_config.get("response_schema")
        is_json = response_mime == "application/json" or raw_schema is not None

        normalized_schema = normalize_gemini_schema(raw_schema) if raw_schema else None

        prompt_sections = []

        # 1. System instructions (supports string, dict with parts, or list)
        if system_instruction:
            if isinstance(system_instruction, str):
                sys_text = system_instruction
            elif isinstance(system_instruction, dict):
                sys_parts = system_instruction.get("parts", [])
                sys_text = extract_text_from_parts(sys_parts)
            elif isinstance(system_instruction, list):
                sys_text = extract_text_from_parts(system_instruction)
            else:
                sys_text = str(system_instruction)

            if sys_text.strip():
                prompt_sections.append(f"[SYSTEM INSTRUCTION]\n{sys_text.strip()}")

        # 2. Normalize contents format (list, single dict, or raw text)
        if isinstance(contents, str):
            contents_list = [{"role": "user", "parts": [{"text": contents}]}]
        elif isinstance(contents, dict):
            contents_list = [contents]
        elif isinstance(contents, list):
            contents_list = contents
        else:
            contents_list = [{"role": "user", "parts": [{"text": str(contents)}]}]

        # 3. Multi-turn conversation history / contents
        convo_lines = []
        for item in contents_list:
            if isinstance(item, str):
                convo_lines.append(f"User: {item.strip()}")
                continue
            elif not isinstance(item, dict):
                continue

            role = item.get("role", "user")
            parts = item.get("parts", [])
            text = extract_text_from_parts(parts)
            if not text.strip():
                continue

            if role in ("model", "assistant"):
                convo_lines.append(f"Model: {text.strip()}")
            elif role == "system":
                convo_lines.append(f"System: {text.strip()}")
            else:
                convo_lines.append(f"User: {text.strip()}")

        if convo_lines:
            if len(convo_lines) == 1 and not system_instruction and not is_json:
                prompt_sections.append(convo_lines[0].replace("User: ", "", 1))
            else:
                prompt_sections.append("[CONVERSATION / USER MESSAGE]\n" + "\n".join(convo_lines))

        # 4. Optional stop sequences and generation constraints
        stop_seqs = generation_config.get("stopSequences") or generation_config.get("stop_sequences")
        if stop_seqs and isinstance(stop_seqs, list):
            prompt_sections.append(
                f"[STOP SEQUENCE RULE]\nStop generation if any of these sequences are encountered: {stop_seqs}"
            )

        # 5. JSON format and schema enforcement (if requested)
        if is_json:
            json_instruction = (
                "[REQUIRED RESPONSE FORMAT]\n"
                "You MUST respond strictly with a valid, well-formed JSON object or array.\n"
                "Do NOT use markdown code fences (```json or ```). Do NOT include any leading or trailing text.\n"
                "Your entire response must be raw JSON parseable directly by a standard JSON parser."
            )
            if normalized_schema:
                json_instruction += (
                    "\nThe JSON must strictly conform to the following schema:\n"
                    f"{json.dumps(normalized_schema, ensure_ascii=False, indent=2)}"
                )
            prompt_sections.append(json_instruction)

        full_prompt = "\n\n".join(prompt_sections)
        return full_prompt, is_json, normalized_schema

    def generate_content(
        self,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes Antigravity CLI using Gemini 3.8 Flash with low reasoning effort,
        extracts the response, and automatically deletes the ephemeral conversation.
        """
        prompt, is_json, schema = self.build_prompt_and_instructions(
            contents=contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )

        safe_workdir = tempfile.gettempdir()
        created_conversation_id: Optional[str] = None

        # Build CLI command: using gemini-3.8-flash with low effort for fastest execution
        cmd = [
            self.agy_bin,
            "--dangerously-skip-permissions",
            "--disable-slash-commands",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "-p",
            prompt,
        ]

        logger.info(
            f"Running Antigravity with model '{self.model}' (effort={self.effort}) "
            f"for requested model '{model_name or self.model}'..."
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=safe_workdir,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("Timeout executing Antigravity CLI")
            raise TimeoutError(f"Antigravity CLI exceeded the timeout limit of {self.timeout}s") from exc

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or f"Exit code: {result.returncode}"
            logger.error(f"Error executing Antigravity CLI: {error_msg}")
            raise RuntimeError(f"Antigravity CLI error: {error_msg}")

        # Parse agy's JSON output
        try:
            agy_data = json.loads(result.stdout)
            created_conversation_id = agy_data.get("conversation_id")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse agy stdout as JSON: {result.stdout[:200]}")
            raw_text = result.stdout.strip()
            if is_json:
                raw_text = extract_json_content(raw_text)
            return {
                "text": raw_text,
                "usage": {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0},
            }
        finally:
            # Automatic cleanup of ephemeral conversation
            if created_conversation_id:
                cleanup_antigravity_conversation(created_conversation_id)
                logger.debug(f"Ephemeral conversation {created_conversation_id} deleted from Antigravity.")

        # Process response text
        if is_json and "structured_output" in agy_data and agy_data["structured_output"] is not None:
            final_text = json.dumps(agy_data["structured_output"], ensure_ascii=False)
        else:
            raw_response = agy_data.get("response", "")
            if is_json:
                final_text = extract_json_content(raw_response)
            else:
                final_text = raw_response

        # Extract token usage
        usage_info = agy_data.get("usage", {})
        usage = {
            "prompt_tokens": usage_info.get("input_tokens", 0),
            "candidates_tokens": usage_info.get("output_tokens", 0),
            "total_tokens": usage_info.get("total_tokens", 0),
        }

        return {
            "text": final_text,
            "usage": usage,
            "raw_agy_output": agy_data,
        }
