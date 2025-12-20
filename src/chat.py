import requests
from typing import Dict, List, Optional
import sys
import os
from dotenv import load_dotenv

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import OPENROUTER_URL, DEFAULT_MODEL

SYSTEM_PROMPT = (
    "You are a Tokyo residential real estate market advisor."
    "Use the provided property details and recent market behavior to answer clearly, "
    "note uncertainty when data is thin, and do not fabricate numbers."
)

load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

def _format_property_context(property_context: Optional[Dict]) -> Optional[str]:
    if not property_context:
        return None

    lines = []
    for key, value in property_context.items():
        if value is None or value == "":
            continue
        lines.append(f"- {key}: {value}")

    if not lines:
        return None

    return "Property context to inform your answer:\n" + "\n".join(lines)


def _build_messages(
    history: List[Dict[str, str]],
    property_context: Optional[Dict],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    context_message = _format_property_context(property_context)
    if context_message:
        messages.append({"role": "system", "content": context_message})

    messages.extend(history)
    return messages


def get_chat_completion(
    history: List[Dict[str, str]],
    property_context: Optional[Dict] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    max_tokens: int = 512,
) -> str:
    """
    Calls OpenRouter's chat completions endpoint with the given history and context.
    Raises an exception if the API call fails or returns no content.
    """
    messages = _build_messages(history, property_context)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("OpenRouter returned no choices.")

    content = choices[0].get("message", {}).get("content")
    if not content:
        raise RuntimeError("OpenRouter returned an empty message.")

    return content.strip()
