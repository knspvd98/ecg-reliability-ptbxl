from __future__ import annotations

import pandas as pd

from ecg_reliability.data.labels import TARGET_LABELS, add_target_columns, parse_scp_codes


def test_parse_scp_codes_string() -> None:
    parsed = parse_scp_codes("{'NORM': 100.0, 'AMI': 50.0}")
    assert parsed["NORM"] == 100.0
    assert parsed["AMI"] == 50.0


def test_add_target_columns_maps_superclasses() -> None:
    ptbxl_df = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "scp_codes": ["{'NORM': 100}", "{'IMI': 100, 'LVH': 50}", "{'NST_': 100}"],
            "strat_fold": [1, 9, 10],
        }
    )
    scp_statements = pd.DataFrame(
        {
            "diagnostic": [1, 1, 1, 1],
            "diagnostic_class": ["NORM", "MI", "HYP", "STTC"],
        },
        index=["NORM", "IMI", "LVH", "NST_"],
    )
    result = add_target_columns(ptbxl_df, scp_statements, filter_empty_targets=True)
    assert list(result.columns.intersection(TARGET_LABELS)) == TARGET_LABELS
    row_norm = result.loc[result["ecg_id"] == 1].iloc[0]
    row_multi = result.loc[result["ecg_id"] == 2].iloc[0]
    assert row_norm["NORM"] == 1
    assert row_multi["MI"] == 1
    assert row_multi["HYP"] == 1
