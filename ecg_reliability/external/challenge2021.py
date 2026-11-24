from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ecg_reliability.external.label_mapping import binary_abnormal_label, parse_snomed_dx_codes

STANDARD_12_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_SOURCE_ENV_MAP = {
    "georgia": "GEORGIA_ROOT",
    "chapman_shaoxing": "CHAPMAN_SHAOXING_ROOT",
    "ningbo": "NINGBO_ROOT",
}


def _require_wfdb():
    try:
        import wfdb
    except ImportError:  # pragma: no cover - exercised in lightweight installs
        return None
    return wfdb


def canonical_lead_name(name: str) -> str:
    normalized = str(name).strip().upper()
    aliases = {
        "AVR": "AVR",
        "AVL": "AVL",
        "AVF": "AVF",
        "A VR": "AVR",
        "A VL": "AVL",
        "A VF": "AVF",
    }
    return aliases.get(normalized, normalized)


def _extract_metadata_from_comments(comments: list[str]) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {"age": None, "sex": None, "patient_id": None}
    for comment in comments:
        text = str(comment).strip()
        if text.startswith("#"):
            text = text[1:].strip()
        if ":" not in text:
            continue
        key, value = text.split(":", maxsplit=1)
        key = key.strip().lower()
        value = value.strip()
        if key == "age":
            metadata["age"] = value
        elif key == "sex":
            metadata["sex"] = value
        elif key in {"patientid", "patient_id"}:
            metadata["patient_id"] = value
    return metadata


def read_challenge2021_header(record_path: str | Path) -> dict[str, Any]:
    wfdb = _require_wfdb()
    if wfdb is not None:
        header = wfdb.rdheader(str(record_path))
        comments = list(getattr(header, "comments", []) or [])
        sampling_rate = float(getattr(header, "fs", 0.0) or 0.0)
        signal_length = int(getattr(header, "sig_len", 0) or 0)
        lead_names = [canonical_lead_name(name) for name in list(getattr(header, "sig_name", []) or [])]
    else:
        sampling_rate, signal_length, lead_names, comments = _parse_header_without_wfdb(record_path)
    meta = _extract_metadata_from_comments(comments)
    return {
        "record_path": str(record_path),
        "sampling_rate_original": sampling_rate,
        "signal_length_original": signal_length,
        "lead_names_original": lead_names,
        "dx_codes": parse_snomed_dx_codes(comments),
        "age": meta["age"],
        "sex": meta["sex"],
        "patient_id": meta["patient_id"],
    }


def _parse_header_without_wfdb(record_path: str | Path) -> tuple[float, int, list[str], list[str]]:
    """Read the small metadata subset needed to build a manifest.

    Signal loading still requires ``wfdb``. Keeping manifest discovery usable
    without the optional dependency makes repository tests and dry runs fast.
    """

    header_path = Path(record_path).with_suffix(".hea")
    lines = header_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty WFDB header: {header_path}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Malformed WFDB header: {header_path}")
    num_leads = int(first[1])
    sampling_rate = float(first[2].split("/", maxsplit=1)[0])
    signal_length = int(first[3])
    lead_names = []
    for line in lines[1 : 1 + num_leads]:
        fields = line.split()
        if fields:
            lead_names.append(canonical_lead_name(fields[-1]))
    comments = [line for line in lines[1 + num_leads :] if line.lstrip().startswith("#")]
    return sampling_rate, signal_length, lead_names, comments


def iter_source_record_paths(source_dir: str | Path) -> list[Path]:
    root = Path(source_dir)
    return sorted(path.with_suffix("") for path in root.rglob("*.hea"))


def resolve_source_directory(
    source: str,
    *,
    source_roots: dict[str, str | Path] | None = None,
    nested_root: str | Path | None = None,
) -> tuple[Path | None, str | None]:
    source_roots = source_roots or {}
    direct_root = source_roots.get(source)
    if direct_root:
        direct_candidates = [Path(direct_root), Path(direct_root) / source]
        for candidate in direct_candidates:
            if candidate.exists() and any(candidate.rglob("*.hea")):
                return candidate, "direct_source_root"

    if nested_root:
        nested_candidates = [
            Path(nested_root) / "training" / source,
            Path(nested_root) / source,
        ]
        for candidate in nested_candidates:
            if candidate.exists() and any(candidate.rglob("*.hea")):
                return candidate, "nested_challenge_root"
    return None, None


def _record_identifier(source: str, source_dir: Path, record_path: Path) -> str:
    relative = record_path.relative_to(source_dir)
    stem = relative.as_posix().replace("/", "__")
    return f"{source}__{stem}"


def build_challenge2021_manifest(
    root: str | Path | None,
    sources: list[str],
    normal_snomed_codes: list[str],
    allow_sources: dict[str, bool] | None = None,
    max_records_per_source: int | None = None,
    source_roots: dict[str, str | Path] | None = None,
) -> pd.DataFrame:
    allow_sources = allow_sources or {}
    nested_root = Path(root) if root else None
    rows: list[dict[str, Any]] = []
    for source in sources:
        if not allow_sources.get(source, True):
            continue
        source_dir, source_root_type = resolve_source_directory(
            source,
            source_roots=source_roots,
            nested_root=nested_root,
        )
        if source_dir is None:
            continue
        record_paths = iter_source_record_paths(source_dir)
        if max_records_per_source is not None:
            record_paths = record_paths[: int(max_records_per_source)]
        for record_path in record_paths:
            metadata = read_challenge2021_header(record_path)
            dx_codes = metadata["dx_codes"]
            label = binary_abnormal_label(dx_codes, normal_snomed_codes)
            warnings: list[str] = []
            status = "ready"
            if label is None:
                status = "invalid_label"
                warnings.append("missing_or_unparsable_dx_codes")
            rows.append(
                {
                    "external_record_id": _record_identifier(source, source_dir, record_path),
                    "source": source,
                    "source_root_type": source_root_type,
                    "source_root": str(source_dir),
                    "record_path": str(record_path),
                    "sampling_rate_original": metadata["sampling_rate_original"],
                    "signal_length_original": metadata["signal_length_original"],
                    "lead_names_original": metadata["lead_names_original"],
                    "dx_codes": dx_codes,
                    "binary_abnormal_label": label,
                    "split": "external_test",
                    "cache_path": None,
                    "status": status,
                    "warnings": "|".join(warnings),
                    "age": metadata["age"],
                    "sex": metadata["sex"],
                    "patient_id": metadata["patient_id"],
                }
            )
    return pd.DataFrame(rows)
