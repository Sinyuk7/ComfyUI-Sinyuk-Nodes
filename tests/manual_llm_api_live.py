"""Run focused live tests against an OpenAI-compatible multimodal endpoint.

This is an explicit manual test, not part of the normal pytest suite. The API
key is read from ``LLM_API_KEY`` and is never written to the report.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx

_DEFAULT_BASE_URL: Final[str] = "https://sub2.de5.net/v1"
_DEFAULT_IMAGE_DIR: Final[Path] = Path("/Users/sinyuk/Downloads/AI测试数据")
_DEFAULT_REPORT: Final[Path] = Path("tests/manual_llm_api_live_report.json")
_ERROR_BODY_LIMIT: Final[int] = 700


@dataclass(frozen=True)
class ImageInput:
    path: Path
    detail: str


@dataclass(frozen=True)
class TestCase:
    name: str
    prompt: str
    images: tuple[ImageInput, ...] = ()
    response_format: dict[str, object] | None = None
    optional_parameters: dict[str, object] | None = None


def _data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)
    if mime_type is None:
        raise ValueError(f"Unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _content(response_json: object) -> str | None:
    if not isinstance(response_json, dict):
        return None
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    value = message.get("content")
    return value if isinstance(value, str) else None


def _cases(image_dir: Path) -> tuple[TestCase, ...]:
    top = image_dir / "001" / "001_top.jpg"
    bottom = image_dir / "001" / "001_bottom.jpg"
    shoes = image_dir / "001" / "001_shoes.jpg"
    for path in (top, bottom, shoes):
        if not path.is_file():
            raise FileNotFoundError(path)
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["items", "summary"],
        "additionalProperties": False,
    }
    return (
        TestCase("text_defaults", "Reply with exactly: OK."),
        TestCase(
            "configured_parameters",
            "Reply with exactly: OK.",
            optional_parameters={"temperature": 0.2, "top_p": 0.9, "max_tokens": 16},
        ),
        TestCase(
            "multi_outfit",
            (
                "These are pieces from one outfit. List the visible pieces and explain "
                "how they coordinate."
            ),
            (ImageInput(top, "high"), ImageInput(bottom, "high"), ImageInput(shoes, "high")),
        ),
        TestCase(
            "multi_json_schema",
            (
                "Identify the visible outfit pieces across these images and return the "
                "requested structured result."
            ),
            (ImageInput(top, "high"), ImageInput(bottom, "high"), ImageInput(shoes, "high")),
            {
                "type": "json_schema",
                "json_schema": {"name": "outfit", "strict": True, "schema": schema},
            },
        ),
    )


def _payload(model: str, case: TestCase) -> dict[str, object]:
    if not case.images:
        content: str | list[dict[str, object]] = case.prompt
    else:
        content = [{"type": "text", "text": case.prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": _data_url(image.path), "detail": image.detail},
            }
            for image in case.images
        )
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if case.optional_parameters is not None:
        payload.update(case.optional_parameters)
    if case.response_format is not None:
        payload["response_format"] = case.response_format
    return payload


async def _run(
    base_url: str, api_key: str, model: str | None, image_dir: Path
) -> dict[str, object]:
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = httpx.Timeout(120.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        models_response = await client.get(f"{base_url.rstrip('/')}/models", headers=headers)
        models_payload: object = models_response.json()
        model_ids = [
            item["id"]
            for item in models_payload.get("data", [])
            if isinstance(models_payload, dict)
            and isinstance(models_payload.get("data"), list)
            and isinstance(item, dict)
            and isinstance(item.get("id"), str)
        ]
        selected_model = model or (model_ids[0] if model_ids else None)
        if selected_model is None:
            raise RuntimeError("No model ID was returned by /models.")
        results: list[dict[str, object]] = []
        for case in _cases(image_dir):
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=_payload(selected_model, case),
            )
            try:
                body: object = response.json()
            except json.JSONDecodeError:
                body = response.text[:_ERROR_BODY_LIMIT]
            result: dict[str, object] = {
                "name": case.name,
                "status": response.status_code,
                "image_count": len(case.images),
                "image_details": [image.detail for image in case.images],
                "content": _content(body),
            }
            if response.status_code >= 400:
                result["error"] = body
            results.append(result)
        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "base_url": base_url,
            "model": selected_model,
            "available_models": model_ids,
            "image_dir": str(image_dir),
            "results": results,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("LLM_API_BASE_URL", _DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("LLM_API_MODEL"))
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(os.environ.get("LLM_TEST_IMAGE_DIR", _DEFAULT_IMAGE_DIR)),
    )
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    args = parser.parse_args()
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise SystemExit("Set LLM_API_KEY before running this live test.")
    import asyncio

    report = asyncio.run(_run(args.base_url, api_key, args.model, args.image_dir))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
