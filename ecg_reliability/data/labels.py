from __future__ import annotations

import ast
from typing import Any

import pandas as pd

TARGET_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
ABNORMAL_LABELS = ["MI", "STTC", "CD", "HYP"]


def parse_scp_codes(raw: Any) -> dict[str, float]:
    """Parse PTB-XL ``scp_codes`` field into a ``{code: likelihood}`` mapping."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Malformed scp_codes string: {raw!r}") from exc
    else:
        raise TypeError(f"Unsupported scp_codes value type: {type(raw)!r}")

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dict from scp_codes, got {type(parsed)!r}")

    result: dict[str, float] = {}
    for code, likelihood in parsed.items():
        try:
            score = float(likelihood)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric likelihood for SCP code {code!r}: {likelihood!r}") from exc
        result[str(code).strip()] = score
    return result


def build_diagnostic_superclass_map(scp_statements: pd.DataFrame) -> dict[str, str]:
    """Build a mapping from diagnostic SCP code to PTB-XL diagnostic superclass."""
    if "diagnostic" not in scp_statements.columns or "diagnostic_class" not in scp_statements.columns:
        raise ValueError("scp_statements must contain 'diagnostic' and 'diagnostic_class' columns.")
    diagnostic_mask = pd.to_numeric(scp_statements["diagnostic"], errors="coerce").fillna(0) == 1
    diagnostic_rows = scp_statements.loc[diagnostic_mask]
    mapping: dict[str, str] = {}
    for code, row in diagnostic_rows.iterrows():
        superclass = row["diagnostic_class"]
        if pd.isna(superclass):
            continue
        superclass_str = str(superclass).strip()
        if superclass_str:
            mapping[str(code).strip()] = superclass_str
    return mapping


def add_target_columns(
    ptbxl_df: pd.DataFrame,
    scp_statements: pd.DataFrame,
    *,
    filter_empty_targets: bool = True,
    target_labels: list[str] | None = None,
    min_likelihood: float = 0.0,
) -> pd.DataFrame:
    """Map PTB-XL ``scp_codes`` to binary diagnostic superclass target columns.

    Preserves the PTB-XL convention where diagnostic statements (``diagnostic == 1``)
    in ``scp_statements.csv`` map to the five diagnostic superclasses:
    ``NORM``, ``MI``, ``STTC``, ``CD``, and ``HYP``.
    """
    labels = list(target_labels or TARGET_LABELS)
    superclass_map = build_diagnostic_superclass_map(scp_statements)
    frame = ptbxl_df.copy()

    parsed_codes = [parse_scp_codes(value) for value in frame["scp_codes"]]
    mapped_sets: list[set[str]] = []
    for code_dict in parsed_codes:
        active: set[str] = set()
        for code, likelihood in code_dict.items():
            if likelihood < float(min_likelihood):
                continue
            superclass = superclass_map.get(code)
            if superclass in labels:
                active.add(superclass)
        mapped_sets.append(active)

    for label in labels:
        frame[label] = [1 if label in active else 0 for active in mapped_sets]

    empty_mask = frame[labels].sum(axis=1) == 0
    empty_count = int(empty_mask.sum())
    frame = frame.loc[~empty_mask].reset_index(drop=True) if filter_empty_targets else frame.reset_index(drop=True)

    frame.attrs["empty_target_count_before_filter"] = empty_count
    frame.attrs["target_labels"] = labels
    return frame
