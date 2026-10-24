"""Structured self-reported confidence, used as the early-exit signal."""

from __future__ import annotations

import json
import re

SYSTEM_PROMPT = (
    "Answer the user's task. Reply with ONLY a JSON object of the form "
    '{"answer": "<your answer>", "confidence": <number from 0 to 1>} where confidence is your honest '
    "probability that the answer is correct and complete. Use a low value if you are unsure or the task "
    "exceeds what you can do reliably."
)


def parse_response(text: str) -> tuple[str, float | None]:
    """Return (answer, confidence). Unparseable output yields confidence None (treated as not confident)."""
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            try:
                conf = float(obj.get("confidence"))
            except (TypeError, ValueError):
                return str(obj["answer"]), None
            return str(obj["answer"]), max(0.0, min(1.0, conf))
    return text.strip(), None
