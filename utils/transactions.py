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


def _categorize_multi_point(row: pd.Series) -> str:
    pickup = pd.to_numeric(row.get("_jumlah_titik_pengambilan"), errors="coerce")
    dropoff = pd.to_numeric(row.get("_jumlah_titik_pengantaran"), errors="coerce")
    if pd.isna(pickup) and pd.isna(dropoff):
        return "Tidak Diketahui"
    if (not pd.isna(pickup) and pickup > 1) or (not pd.isna(dropoff) and dropoff > 1):
        return "Multi Titik"
    return "Single Titik"


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

    if "_jumlah_titik_pengambilan" in result.columns or "_jumlah_titik_pengantaran" in result.columns:
        result["_kategori_multi_titik"] = result.apply(_categorize_multi_point, axis=1)

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
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    return f"{field_label}={text}"


def build_apriori_transactions(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
) -> pd.Series:
    """Create Apriori transaction item lists from mapped columns."""
    item_sources: list[tuple[str, str | None]] = [
        ("Waktu", "_kategori_waktu"),
        ("Layanan", _selected_column(mapping, "layanan")),
        ("Pembayaran", _selected_column(mapping, "pembayaran")),
        ("Asal", "_lokasi_asal" if "_lokasi_asal" in df.columns else _selected_column(mapping, "asal")),
        ("Kota/Kab Asal", "_kota_kabupaten_asal"),
        ("Tujuan", "_lokasi_tujuan" if "_lokasi_tujuan" in df.columns else _selected_column(mapping, "tujuan")),
        ("Kota/Kab Tujuan", "_kota_kabupaten_tujuan"),
        ("Sub Layanan", _selected_column(mapping, "sub_layanan")),
        ("Titik Pengambilan", "_kategori_titik_pengambilan"),
        ("Titik Pengantaran", "_kategori_titik_pengantaran"),
        ("Multi Titik", "_kategori_multi_titik"),
    ]

    seen_sources: set[str] = set()
    unique_item_sources: list[tuple[str, str | None]] = []
    for label, column in item_sources:
        if not column or column in seen_sources:
            continue
        seen_sources.add(column)
        unique_item_sources.append((label, column))

    transactions: list[list[str]] = []
    for _, row in df.iterrows():
        items: list[str] = []
        for label, column in unique_item_sources:
            if column and column in df.columns:
                item = _clean_item(label, row[column])
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


def apply_interactive_filters(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Render filters and return the filtered DataFrame."""
    result = df.copy()

    col1, col2 = st.columns(2)
    with col1:
        if "_tanggal_filter" in result.columns and result["_tanggal_filter"].notna().any():
            min_date = result["_tanggal_filter"].dropna().min()
            max_date = result["_tanggal_filter"].dropna().max()
            selected_range = st.date_input(
                "Tanggal",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(selected_range, tuple) and len(selected_range) == 2:
                start_date, end_date = selected_range
                result = result[
                    (result["_tanggal_filter"] >= start_date)
                    & (result["_tanggal_filter"] <= end_date)
                ]
        else:
            st.caption("Filter tanggal belum tersedia karena kolom tanggal tidak dipetakan.")

    with col2:
        selected_time = _filter_multiselect(
            "Kategori waktu", result, "_kategori_waktu", "filter_kategori_waktu"
        )
        if selected_time:
            result = result[result["_kategori_waktu"].isin(selected_time)]

    filter_columns = st.columns(2)
    filter_fields = [
        ("layanan", "Layanan"),
        ("pembayaran", "Pembayaran"),
        ("_lokasi_asal", "Asal"),
        ("_lokasi_tujuan", "Tujuan"),
        ("sub_layanan", "Sub layanan"),
        ("_kategori_multi_titik", "Single/multi titik"),
    ]
    seen_filter_columns: set[str] = set()
    for index, (field, label) in enumerate(filter_fields):
        column = field if field.startswith("_") else _selected_column(mapping, field)
        if not column or column not in result.columns:
            continue
        if column in seen_filter_columns:
            continue
        seen_filter_columns.add(column)
        with filter_columns[index % 2]:
            selected = _filter_multiselect(label, result, column, f"filter_{field}")
        if selected:
            result = result[result[column].isin(selected)]

    return result


def flatten_itemsets(itemsets: Iterable[frozenset[str]]) -> list[str]:
    """Convert mlxtend frozensets into sorted display strings."""
    return [", ".join(sorted(itemset)) for itemset in itemsets]
