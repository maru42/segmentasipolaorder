from __future__ import annotations

import re
from typing import Any

import pandas as pd


CANONICAL_REPLACEMENTS: dict[str, str] = {
    "non tunai": "NonTunai",
    "nontunai": "NonTunai",
    "non-tunai": "NonTunai",
    "cashless": "NonTunai",
    "tunai": "Tunai",
    "cash": "Tunai",
    "instant": "Instant",
    "instan": "Instant",
    "same day": "Same Day",
    "sameday": "Same Day",
    "regular": "Regular",
    "reguler": "Regular",
    "hemat": "Hemat",
    "express": "Express",
    "ekspres": "Express",
}

CATEGORICAL_FIELDS = {
    "layanan",
    "pembayaran",
    "asal",
    "asal_kota_kabupaten",
    "tujuan",
    "tujuan_kota_kabupaten",
    "sub_layanan",
}
NUMERIC_FIELDS = {"jarak", "tarif", "jumlah_titik_pengambilan", "jumlah_titik_pengantaran"}


def normalize_text_value(value: Any) -> Any:
    """Trim whitespace and standardize common categorical labels."""
    if pd.isna(value):
        return value
    if not isinstance(value, str):
        return value

    compact = re.sub(r"\s+", " ", value.strip())
    lookup_key = compact.lower().replace("_", " ").replace("/", " ")
    lookup_key = re.sub(r"\s+", " ", lookup_key)

    if lookup_key in CANONICAL_REPLACEMENTS:
        return CANONICAL_REPLACEMENTS[lookup_key]

    # Title case keeps category labels readable while reducing inconsistent casing.
    return compact.title()


def fill_missing_values(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Fill missing values according to each column type."""
    result = df.copy()
    numeric_columns = {mapping.get(field) for field in NUMERIC_FIELDS if mapping.get(field)}

    for column in result.columns:
        if column in numeric_columns:
            numeric_series = coerce_numeric_series(result[column])
            median = numeric_series.median()
            result[column] = numeric_series.fillna(0 if pd.isna(median) else median)
        elif pd.api.types.is_numeric_dtype(result[column]):
            median = result[column].median()
            result[column] = result[column].fillna(0 if pd.isna(median) else median)
        else:
            result[column] = result[column].fillna("Tidak Diketahui")

    return result


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Convert flexible currency/distance text into numeric values."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[^0-9,.\-]", "", regex=True)
        .str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def coerce_datetime_series(series: pd.Series) -> pd.Series:
    """Parse mixed user date formats into pandas datetime values."""
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)


def normalize_time_value(value: Any) -> str | pd.NA:
    """Normalize flexible time values into HH:MM strings."""
    if pd.isna(value):
        return pd.NA

    if hasattr(value, "hour") and hasattr(value, "minute"):
        return f"{int(value.hour):02d}:{int(value.minute):02d}"

    if isinstance(value, (int, float)) and not pd.isna(value):
        hour = int(value)
        return f"{hour:02d}:00" if 0 <= hour <= 23 else pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA

    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return f"{int(parsed.hour):02d}:{int(parsed.minute):02d}"

    match = re.search(r"(?<!\d)([01]?\d|2[0-3])(?:[:.](\d{1,2}))?", text)
    if not match:
        return pd.NA

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if minute > 59:
        return pd.NA
    return f"{hour:02d}:{minute:02d}"


def standardize_mapped_dtypes(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
) -> pd.DataFrame:
    """Coerce mapped columns into analysis-ready data types."""
    result = df.copy()

    for field in NUMERIC_FIELDS:
        column = mapping.get(field)
        if column and column in result.columns:
            result[column] = coerce_numeric_series(result[column])

    date_column = mapping.get("tanggal")
    if date_column and date_column in result.columns:
        result[date_column] = coerce_datetime_series(result[date_column])

    time_column = mapping.get("waktu")
    if time_column and time_column in result.columns:
        result[time_column] = result[time_column].map(normalize_time_value).astype("string")
        result[time_column] = result[time_column].fillna("Tidak Diketahui")

    for field in CATEGORICAL_FIELDS:
        column = mapping.get(field)
        if column and column in result.columns:
            result[column] = result[column].astype("string").fillna("Tidak Diketahui")

    return result


def preprocess_dataset(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Run automatic preprocessing while preserving original uploaded columns."""
    result = df.copy()

    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(normalize_text_value)

    result = fill_missing_values(result, mapping)
    result = standardize_mapped_dtypes(result, mapping)
    result = result.drop_duplicates().reset_index(drop=True)

    return result
