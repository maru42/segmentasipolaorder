from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DatasetDescription:
    """Dataset profile values used by the dashboard overview."""

    row_count: int
    column_count: int
    missing_values: pd.DataFrame
    dtypes: pd.DataFrame
    columns: list[str]


def load_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    """Return a DataFrame loaded from the uploaded CSV/XLSX file."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Format file tidak didukung. Gunakan CSV, XLSX, atau XLS.")


def describe_dataset(df: pd.DataFrame) -> DatasetDescription:
    """Return row, column, missing-value, dtype, and column-name metadata for a DataFrame."""
    missing = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "kolom", 0: "missing_values"})
    )
    dtypes = (
        df.dtypes.astype(str)
        .reset_index()
        .rename(columns={"index": "kolom", 0: "tipe_data"})
    )
    return DatasetDescription(
        row_count=len(df),
        column_count=len(df.columns),
        missing_values=missing,
        dtypes=dtypes,
        columns=[str(col) for col in df.columns],
    )
