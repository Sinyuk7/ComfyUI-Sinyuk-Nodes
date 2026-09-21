"""Pytest setup for tests that exercise the ComfyUI V3 boundary."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _add_comfyui_path() -> None:
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        return

    path = str(Path(comfyui_path).expanduser().resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


_add_comfyui_path()
pytest.importorskip("comfy_api")
