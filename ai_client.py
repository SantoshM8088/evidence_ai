"""Thin wrapper around the OpenAI SDK.

The rest of the app only needs two operations, `embed` and `generate_json`, so tests
can replace this class with a fake that returns canned responses.
"""
import json
import os

import openai

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 0  # 0 for initial live verification: each call is sent exactly once (SDK default would be 2)
EMBED_BATCH_SIZE = 100


class AIServiceError(Exception):
    """The OpenAI service failed (network, auth, rate limit, malformed output).

    This is deliberately separate from "insufficient evidence": an API failure says
    nothing about whether the documents contain the answer.
    """


def load_config():
    """Return (config dict, list of missing variable names)."""
    names = ["OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "OPENAI_EMBEDDING_MODEL"]
    config = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in config.items() if not value]
    return config, missing


class OpenAIClient:
    def __init__(self, api_key, chat_model, embedding_model):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self._client = openai.OpenAI(
            api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES
        )

    def embed(self, texts):
        """Return one embedding vector per input text (same model for documents and queries)."""
        vectors = []
        try:
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[start:start + EMBED_BATCH_SIZE]
                response = self._client.embeddings.create(model=self.embedding_model, input=batch)
                vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        except openai.OpenAIError as exc:
            raise AIServiceError(_describe_error(exc)) from exc
        return vectors

    def generate_json(self, name, instructions, user_input, schema):
        """Call the chat model with a strict JSON schema and return the parsed object."""
        try:
            response = self._client.responses.create(
                model=self.chat_model,
                instructions=instructions,
                input=user_input,
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            )
        except openai.OpenAIError as exc:
            raise AIServiceError(_describe_error(exc)) from exc

        if getattr(response, "status", None) == "incomplete":
            raise AIServiceError("The model response was incomplete (output was cut off).")
        for item in response.output or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", "") == "refusal":
                    raise AIServiceError("The model refused to answer this request.")
        try:
            return json.loads(response.output_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise AIServiceError("The model returned output that is not valid JSON.") from exc


def _describe_error(exc):
    """Short, secret-free description of an SDK error for display."""
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key (authentication error). Check OPENAI_API_KEY."
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI rate limit or quota exceeded. Wait and retry, or check your billing."
    if isinstance(exc, openai.APITimeoutError):
        return f"OpenAI request timed out after {REQUEST_TIMEOUT_SECONDS}s."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not connect to OpenAI. Check your internet connection."
    if isinstance(exc, openai.NotFoundError):
        return "OpenAI returned 'not found'. Check OPENAI_CHAT_MODEL / OPENAI_EMBEDDING_MODEL."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI API error (HTTP {exc.status_code})."
    return f"OpenAI error: {type(exc).__name__}"
