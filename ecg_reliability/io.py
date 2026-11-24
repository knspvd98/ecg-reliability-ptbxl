from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".csv":
        df.to_csv(target, index=False)
        return
    if target.suffix == ".parquet":
        try:
            df.to_parquet(target, index=False)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Saving parquet requires pyarrow or fastparquet.") from exc
        return
    raise ValueError(f"Unsupported table extension: {target.suffix}")


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    if source.suffix == ".parquet":
        try:
            return pd.read_parquet(source)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Reading parquet requires pyarrow or fastparquet.") from exc
    raise ValueError(f"Unsupported table extension: {source.suffix}")
