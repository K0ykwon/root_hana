import os
import json
from typing import Dict, List, Optional
import requests


def call_openai_chat(
    api_key: Optional[str],
    messages: List[Dict[str, str]],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    response_format: Optional[Dict[str, str]] = None,
    timeout_s: int = 120,
) -> Optional[str]:
    if not api_key:
        return None
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if response_format:
        payload["response_format"] = response_format
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


