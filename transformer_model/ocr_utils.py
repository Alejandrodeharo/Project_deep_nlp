from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


_OCR_CACHE: Dict[str, Any] | None = None
_OCR_CACHE_PATH: str | None = None


def _load_cache(cache_path: str | None) -> Dict[str, Any]:
    global _OCR_CACHE, _OCR_CACHE_PATH

    if cache_path is None:
        return {}

    cache_path = str(cache_path)
    if _OCR_CACHE is not None and _OCR_CACHE_PATH == cache_path:
        return _OCR_CACHE

    path = Path(cache_path)
    if not path.exists():
        _OCR_CACHE = {}
        _OCR_CACHE_PATH = cache_path
        return _OCR_CACHE

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        data = {}

    _OCR_CACHE = data
    _OCR_CACHE_PATH = cache_path
    return _OCR_CACHE


def _record_key_candidates(record: Dict[str, Any]) -> list[str]:
    candidates = []
    for key in ["match_id", "id", "image_name", "image_file", "filename", "file_name"]:
        if key in record and record[key] is not None:
            candidates.append(str(record[key]))
    return candidates


def get_ocr_signal_for_record(
    record: Dict[str, Any],
    image_dir: str | None = None,
    cache_path: str | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    cache = _load_cache(cache_path)

    for candidate in _record_key_candidates(record):
        if candidate in cache:
            cached_value = cache[candidate]
            if isinstance(cached_value, dict):
                signal = dict(cached_value)
                signal.setdefault("source", "cache")
                return signal
            if isinstance(cached_value, str):
                return {
                    "ocr_text": cached_value,
                    "source": "cache",
                    "has_ocr": bool(cached_value.strip()),
                }

    signal: Dict[str, Any] = {
        "ocr_text": "",
        "has_ocr": False,
        "source": "none",
    }

    if image_dir is not None:
        image_path = Path(image_dir)
        for candidate in _record_key_candidates(record):
            direct_path = image_path / candidate
            if direct_path.exists():
                signal["image_found"] = True
                signal["image_path"] = str(direct_path)
                signal["source"] = "image_only"
                return signal

            for extension in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
                candidate_path = image_path / f"{candidate}{extension}"
                if candidate_path.exists():
                    signal["image_found"] = True
                    signal["image_path"] = str(candidate_path)
                    signal["source"] = "image_only"
                    return signal

    return signal


def append_ocr_hint_to_text(text: str, record: Dict[str, Any], signal: Optional[Dict[str, Any]]) -> str:
    if not signal:
        return text

    ocr_text = str(signal.get("ocr_text", "")).strip()
    if not ocr_text:
        return text

    match_id = record.get("match_id")
    prefix = f"[OCR_MATCH_ID={match_id}] " if match_id is not None else "[OCR] "
    return f"{text}\n\n{prefix}{ocr_text}"
