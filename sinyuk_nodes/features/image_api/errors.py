"""Keep untrusted upstream diagnostics bounded and credential-free."""

from __future__ import annotations

import re
from collections.abc import Iterable


def clean_message(value: object, secrets: Iterable[str] = ()) -> str:
    text = value if isinstance(value, str) else "Upstream returned an invalid error message."
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"data:image/[^\s]+|[A-Za-z0-9+/=]{160,}", "[omitted]", text)
    text = re.sub(r"https?://\S+", "[URL omitted]", text)
    return " ".join(text.split())[:400]


class GrsaiError(RuntimeError):
    def __init__(self, message: str, task_id: str | None = None, index: int | None = None) -> None:
        self.task_id = task_id
        self.index = index
        suffix = f" [task_id={task_id}]" if task_id else ""
        if index is not None:
            suffix += f" [result={index}]"
        super().__init__(message + suffix)
