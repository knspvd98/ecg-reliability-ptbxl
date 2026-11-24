from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any


def _comment_key_value(comment: str) -> tuple[str, str] | None:
    text = str(comment).strip()
    if not text:
        return None
    if text.startswith("#"):
        text = text[1:].strip()
    if ":" not in text:
        return None
    key, value = text.split(":", maxsplit=1)
    return key.strip().lower(), value.strip()


def normalize_dx_codes(dx_codes: Any) -> list[str]:
    if dx_codes is None:
        return []
    if isinstance(dx_codes, str):
        stripped = dx_codes.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(code) for code in parsed if str(code).strip()]
            except (SyntaxError, ValueError):
                pass
        return [code.strip() for code in stripped.split(",") if code.strip()]
    if isinstance(dx_codes, Iterable):
        return [str(code) for code in dx_codes if str(code).strip()]
    return [str(dx_codes)]


def parse_snomed_dx_codes(header_comments: Iterable[str] | None) -> list[str]:
    if header_comments is None:
        return []
    for comment in header_comments:
        parsed = _comment_key_value(str(comment))
        if parsed is None:
            continue
        key, value = parsed
        if key == "dx":
            return [code.strip() for code in value.split(",") if code.strip()]
    return []


def is_external_normal(dx_codes: Iterable[str], normal_snomed_codes: Iterable[str]) -> bool:
    normal_codes = {str(code) for code in normal_snomed_codes}
    codes = normalize_dx_codes(dx_codes)
    return bool(codes) and all(code in normal_codes for code in codes)


def binary_abnormal_label(dx_codes: Iterable[str], normal_snomed_codes: Iterable[str]) -> int | None:
    codes = normalize_dx_codes(dx_codes)
    if not codes:
        return None
    return 0 if is_external_normal(codes, normal_snomed_codes) else 1


def optional_snomed_overlap_label_vector(dx_codes: Iterable[str], mapping: dict[str, Any]) -> dict[str, int]:
    labels_payload = mapping.get("labels", mapping)
    code_set = set(normalize_dx_codes(dx_codes))
    result: dict[str, int] = {}
    for label_name, payload in labels_payload.items():
        external_codes = payload.get("external_codes", payload if isinstance(payload, list) else [])
        result[str(label_name)] = int(any(str(code) in code_set for code in external_codes))
    return result
