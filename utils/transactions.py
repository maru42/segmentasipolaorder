from __future__ import annotations

import re
from datetime import date
from typing import Iterable

import pandas as pd
import streamlit as st


TIME_LABELS = ["Dini Hari", "Pagi", "Siang", "Sore", "Malam", "Tidak Diketahui"]
RAMADAN_2026_START = date(2026, 2, 20)
RAMADAN_2026_END = date(2026, 3, 21)


def _selected_column(mapping: dict[str, str | None], field: str) -> str | None:
    column = mapping.get(field)
    return column if column else None


def _location_part(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat", "tidak diketahui"}:
        return None
    return text


def _build_combined_location(
    df: pd.DataFrame,
    district_column: str | None,
    city_regency_column: str | None,
) -> pd.Series | None:
    """Combine kecamatan with kota/kabupaten without requiring both columns."""
    available_columns = [
        column for column in [district_column, city_regency_column] if column and column in df.columns
    ]
    if not available_columns:
        return None

    def combine(row: pd.Series) -> str:
        district = _location_part(row[district_column]) if district_column in row.index else None
        city_regency = (
            _location_part(row[city_regency_column])
            if city_regency_column in row.index
            else None
        )
        if district and city_regency:
            return f"{district}, {city_regency}"
        return district or city_regency or "Tidak Diketahui"

    return df[available_columns].apply(combine, axis=1)


def _copy_location_level(
    df: pd.DataFrame,
    source_column: str | None,
) -> pd.Series | None:
    if not source_column or source_column not in df.columns:
        return None
    return df[source_column].map(lambda value: _location_part(value) or "Tidak Diketahui")


def _categorize_point_count(value: object, single_label: str, multi_label: str) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "Tidak Diketahui"
    return multi_label if numeric > 1 else single_label


def _extract_hour(value: object) -> int | None:
    """Extract hour from flexible time/date strings or datetime-like values."""
    if pd.isna(value):
        return None
    if hasattr(value, "hour"):
        return int(value.hour)
    if isinstance(value, (int, float)) and not pd.isna(value):
        hour = int(value)
        return hour if 0 <= hour <= 23 else None

    text = str(value).strip()
    if not text:
        return None

    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return int(parsed.hour)

    match = re.search(r"(?<!\d)([01]?\d|2[0-3])(?:[:.]\d{1,2})?", text)
    if match:
        return int(match.group(1))
    return None


def categorize_hour(hour: int | None) -> str:
    """Map an hour value into the research time category."""
    if hour is None:
        return "Tidak Diketahui"
    if 5 <= hour <= 10:
        return "Pagi"
    if 11 <= hour <= 14:
        return "Siang"
    if 15 <= hour <= 18:
        return "Sore"
    if 19 <= hour <= 23:
        return "Malam"
    if 0 <= hour <= 4:
        return "Dini Hari"
    return "Tidak Diketahui"


def enrich_dataset(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Add helper columns needed for filters and Apriori transactions."""
    result = df.copy()
    time_column = _selected_column(mapping, "waktu")
    date_column = _selected_column(mapping, "tanggal")

    if time_column and time_column in result.columns:
        result["_jam_order"] = result[time_column].map(_extract_hour)
    elif date_column and date_column in result.columns:
        result["_jam_order"] = pd.to_datetime(result[date_column], errors="coerce").dt.hour
    else:
        result["_jam_order"] = None

    result["_kategori_waktu"] = result["_jam_order"].map(categorize_hour)

    if date_column and date_column in result.columns:
        result["_tanggal_filter"] = pd.to_datetime(result[date_column], errors="coerce").dt.date
    else:
        result["_tanggal_filter"] = pd.NaT

    result["_periode_ramadan_2026"] = result["_tanggal_filter"].map(classify_ramadan_2026_period)

    origin_location = _build_combined_location(
        result,
        _selected_column(mapping, "asal"),
        _selected_column(mapping, "asal_kota_kabupaten"),
    )
    if origin_location is not None:
        result["_lokasi_asal"] = origin_location

    origin_city_regency = _copy_location_level(
        result,
        _selected_column(mapping, "asal_kota_kabupaten"),
    )
    if origin_city_regency is not None:
        result["_kota_kabupaten_asal"] = origin_city_regency

    destination_location = _build_combined_location(
        result,
        _selected_column(mapping, "tujuan"),
        _selected_column(mapping, "tujuan_kota_kabupaten"),
    )
    if destination_location is not None:
        result["_lokasi_tujuan"] = destination_location

    destination_city_regency = _copy_location_level(
        result,
        _selected_column(mapping, "tujuan_kota_kabupaten"),
    )
    if destination_city_regency is not None:
        result["_kota_kabupaten_tujuan"] = destination_city_regency

    pickup_points_column = _selected_column(mapping, "jumlah_titik_pengambilan")
    if pickup_points_column and pickup_points_column in result.columns:
        result["_jumlah_titik_pengambilan"] = pd.to_numeric(
            result[pickup_points_column],
            errors="coerce",
        )
        result["_kategori_titik_pengambilan"] = result["_jumlah_titik_pengambilan"].map(
            lambda value: _categorize_point_count(value, "Single Pickup", "Multi Pickup")
        )

    dropoff_points_column = _selected_column(mapping, "jumlah_titik_pengantaran")
    if dropoff_points_column and dropoff_points_column in result.columns:
        result["_jumlah_titik_pengantaran"] = pd.to_numeric(
            result[dropoff_points_column],
            errors="coerce",
        )
        result["_kategori_titik_pengantaran"] = result["_jumlah_titik_pengantaran"].map(
            lambda value: _categorize_point_count(value, "Single Dropoff", "Multi Dropoff")
        )

    return result


def classify_ramadan_2026_period(value: object) -> str:
    """Classify a transaction date around Ramadan 2026."""
    if pd.isna(value):
        return "Tanggal Tidak Tersedia"
    if RAMADAN_2026_START <= value <= RAMADAN_2026_END:
        return "Ramadan 2026"
    if value > RAMADAN_2026_END:
        return "Pasca Ramadan 2026"
    return "Pra Ramadan 2026"


def _clean_item(field_label: str, value: object) -> str | None:
    if pd.isna(value):
        return None
    if hasattr(value, "strftime"):
        value = value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    return f"{field_label}={text}"


def _bin_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(["Tidak Diketahui"] * len(series), index=series.index)
    if numeric.nunique(dropna=True) <= 10:
        return numeric.fillna("Tidak Diketahui")

    try:
        binned = pd.qcut(
            numeric,
            q=3,
            labels=["Rendah", "Sedang", "Tinggi"],
            duplicates="drop",
        )
        return binned.astype("string").fillna("Tidak Diketahui")
    except ValueError:
        return numeric.fillna("Tidak Diketahui")


def _transaction_sources_for_selected_columns(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    selected_columns: list[str] | None,
) -> list[tuple[str, str]]:
    """Map user-selected raw columns into analysis-friendly transaction sources."""
    if not selected_columns:
        return [
            ("Waktu", "_kategori_waktu"),
            ("Layanan", _selected_column(mapping, "layanan") or ""),
            ("Pembayaran", _selected_column(mapping, "pembayaran") or ""),
            ("Asal", "_lokasi_asal" if "_lokasi_asal" in df.columns else (_selected_column(mapping, "asal") or "")),
            ("Kota/Kab Asal", "_kota_kabupaten_asal"),
            ("Tujuan", "_lokasi_tujuan" if "_lokasi_tujuan" in df.columns else (_selected_column(mapping, "tujuan") or "")),
            ("Kota/Kab Tujuan", "_kota_kabupaten_tujuan"),
            ("Sub Layanan", _selected_column(mapping, "sub_layanan") or ""),
            ("Titik Pengambilan", "_kategori_titik_pengambilan"),
            ("Titik Pengantaran", "_kategori_titik_pengantaran"),
        ]

    sources: list[tuple[str, str]] = []
    for column in selected_columns:
        if column not in df.columns:
            continue

        if column == mapping.get("waktu") and "_kategori_waktu" in df.columns:
            sources.append(("Waktu", "_kategori_waktu"))
        elif column == mapping.get("tanggal") and "_tanggal_filter" in df.columns:
            sources.append(("Tanggal", "_tanggal_filter"))
        elif column in {mapping.get("asal"), mapping.get("asal_kota_kabupaten")} and "_lokasi_asal" in df.columns:
            sources.append(("Asal", "_lokasi_asal"))
            if column == mapping.get("asal_kota_kabupaten") and "_kota_kabupaten_asal" in df.columns:
                sources.append(("Kota/Kab Asal", "_kota_kabupaten_asal"))
        elif column in {mapping.get("tujuan"), mapping.get("tujuan_kota_kabupaten")} and "_lokasi_tujuan" in df.columns:
            sources.append(("Tujuan", "_lokasi_tujuan"))
            if column == mapping.get("tujuan_kota_kabupaten") and "_kota_kabupaten_tujuan" in df.columns:
                sources.append(("Kota/Kab Tujuan", "_kota_kabupaten_tujuan"))
        elif column == mapping.get("jumlah_titik_pengambilan") and "_kategori_titik_pengambilan" in df.columns:
            sources.append(("Titik Pengambilan", "_kategori_titik_pengambilan"))
        elif column == mapping.get("jumlah_titik_pengantaran") and "_kategori_titik_pengantaran" in df.columns:
            sources.append(("Titik Pengantaran", "_kategori_titik_pengantaran"))
        else:
            sources.append((column, column))

    unique_sources: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for label, column in sources:
        if not column or column not in df.columns or (label, column) in seen:
            continue
        seen.add((label, column))
        unique_sources.append((label, column))
    return unique_sources


def build_apriori_transactions(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    selected_columns: list[str] | None = None,
) -> pd.Series:
    """Create Apriori transaction item lists from mapped columns."""
    unique_item_sources = _transaction_sources_for_selected_columns(df, mapping, selected_columns)
    source_values: dict[tuple[str, str], pd.Series] = {}
    for label, column in unique_item_sources:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            source_values[(label, column)] = _bin_numeric_series(series)
        else:
            source_values[(label, column)] = series

    transactions: list[list[str]] = []
    for row_index in range(len(df)):
        items: list[str] = []
        for label, column in unique_item_sources:
            item = _clean_item(label, source_values[(label, column)].iloc[row_index])
            if item:
                items.append(item)
        transactions.append(items)

    return pd.Series(transactions, name="items_transaksi")


def _filter_multiselect(
    label: str,
    df: pd.DataFrame,
    column: str,
    key: str,
) -> list[object]:
    values = sorted([value for value in df[column].dropna().unique().tolist()], key=lambda item: str(item))
    return st.multiselect(label, values, default=[], key=key)


def _filter_numeric_range(
    label: str,
    df: pd.DataFrame,
    column: str,
    key: str,
) -> tuple[float, float] | None:
    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    if numeric.empty:
        return None
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if min_value == max_value:
        st.caption(f"{label}: semua nilai {min_value:g}")
        return None
    return st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value),
        key=key,
    )


def _filter_target_for_selected_column(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    column: str,
) -> tuple[str, str, str] | None:
    if column not in df.columns:
        return None
    if column == mapping.get("waktu") and "_kategori_waktu" in df.columns:
        return "_kategori_waktu", "Kategori waktu", "categorical"
    if column == mapping.get("tanggal") and "_tanggal_filter" in df.columns:
        return "_tanggal_filter", "Tanggal", "date"
    if column in {mapping.get("asal"), mapping.get("asal_kota_kabupaten")} and "_lokasi_asal" in df.columns:
        return "_lokasi_asal", "Lokasi asal", "categorical"
    if column in {mapping.get("tujuan"), mapping.get("tujuan_kota_kabupaten")} and "_lokasi_tujuan" in df.columns:
        return "_lokasi_tujuan", "Lokasi tujuan", "categorical"
    if column == mapping.get("jumlah_titik_pengambilan") and "_kategori_titik_pengambilan" in df.columns:
        return "_kategori_titik_pengambilan", "Titik pengambilan", "categorical"
    if column == mapping.get("jumlah_titik_pengantaran") and "_kategori_titik_pengantaran" in df.columns:
        return "_kategori_titik_pengantaran", "Titik pengantaran", "categorical"
    if pd.api.types.is_numeric_dtype(df[column]):
        return column, column, "numeric"
    return column, column, "categorical"


def apply_interactive_filters(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    selected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Render filters and return the filtered DataFrame."""
    result = df.copy()

    selected_columns = selected_columns or []
    targets: list[tuple[str, str, str]] = []
    for column in selected_columns:
        target = _filter_target_for_selected_column(result, mapping, column)
        if target:
            targets.append(target)

    if not targets:
        st.info("Belum ada kolom terpilih untuk difilter.")
        return result

    unique_targets: list[tuple[str, str, str]] = []
    seen_filter_columns: set[str] = set()
    for column, label, filter_type in targets:
        if column in seen_filter_columns or column not in result.columns:
            continue
        seen_filter_columns.add(column)
        unique_targets.append((column, label, filter_type))

    for start in range(0, len(unique_targets), 2):
        filter_columns = st.columns(2)
        for offset, (column, label, filter_type) in enumerate(unique_targets[start : start + 2]):
            with filter_columns[offset]:
                result = _apply_single_filter(result, column, label, filter_type)

    return result


def _apply_single_filter(
    result: pd.DataFrame,
    column: str,
    label: str,
    filter_type: str,
) -> pd.DataFrame:
    if filter_type == "date":
        valid_dates = result[column].dropna()
        if valid_dates.empty:
            st.caption(f"{label} belum memiliki data tanggal valid.")
            return result
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        selected_range = st.date_input(
            label,
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"filter_{column}",
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
            result = result[(result[column] >= start_date) & (result[column] <= end_date)]
    elif filter_type == "numeric":
        selected_range = _filter_numeric_range(label, result, column, f"filter_{column}")
        if selected_range is not None:
            start_value, end_value = selected_range
            numeric = pd.to_numeric(result[column], errors="coerce")
            result = result[(numeric >= start_value) & (numeric <= end_value)]
    else:
        selected = _filter_multiselect(label, result, column, f"filter_{column}")
        if selected:
            result = result[result[column].isin(selected)]

    return result


def flatten_itemsets(itemsets: Iterable[frozenset[str]]) -> list[str]:
    """Convert mlxtend frozensets into sorted display strings."""
    return [", ".join(sorted(itemset)) for itemset in itemsets]
