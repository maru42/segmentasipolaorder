from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st


MAPPING_FIELDS: dict[str, str] = {
    "waktu": "Kolom waktu",
    "tanggal": "Kolom tanggal",
    "layanan": "Kolom layanan",
    "pembayaran": "Kolom pembayaran",
    "jarak": "Kolom jarak",
    "tarif": "Kolom tarif",
    "asal": "Kolom kecamatan asal",
    "asal_kota_kabupaten": "Kolom kota/kabupaten asal (opsional)",
    "tujuan": "Kolom kecamatan tujuan",
    "tujuan_kota_kabupaten": "Kolom kota/kabupaten tujuan (opsional)",
    "sub_layanan": "Kolom sub layanan",
    "jumlah_titik_pengambilan": "Kolom jumlah titik pengambilan (opsional)",
    "jumlah_titik_pengantaran": "Kolom jumlah titik pengantaran (opsional)",
}


FIELD_KEYWORDS: dict[str, list[str]] = {
    "waktu": ["waktu", "jam", "time", "hour", "pickup_time", "order_time"],
    "tanggal": ["tanggal", "tgl", "date", "order_date", "created_at"],
    "layanan": ["layanan", "service", "produk", "product", "tipe_layanan"],
    "pembayaran": ["pembayaran", "payment", "bayar", "metode", "method"],
    "jarak": ["jarak", "distance", "km", "kilometer"],
    "tarif": ["tarif", "fare", "harga", "price", "biaya", "amount"],
    "asal": [
        "asal",
        "kecamatan_asal",
        "asal_kecamatan",
        "origin_district",
        "pickup_district",
        "kecamatan pickup",
        "kecamatan jemput",
    ],
    "asal_kota_kabupaten": [
        "kota_asal",
        "kabupaten_asal",
        "kota kabupaten asal",
        "asal_kota",
        "asal_kabupaten",
        "origin_city",
        "origin_regency",
        "pickup_city",
    ],
    "tujuan": [
        "tujuan",
        "kecamatan_tujuan",
        "tujuan_kecamatan",
        "destination_district",
        "dropoff_district",
        "kecamatan tujuan",
        "kecamatan antar",
    ],
    "tujuan_kota_kabupaten": [
        "kota_tujuan",
        "kabupaten_tujuan",
        "kota kabupaten tujuan",
        "tujuan_kota",
        "tujuan_kabupaten",
        "destination_city",
        "destination_regency",
        "dropoff_city",
    ],
    "sub_layanan": [
        "sub_layanan",
        "sub layanan",
        "sub_kategori_layanan",
        "sub kategori layanan",
        "sub kategori",
        "sublayanan",
        "sub_service",
        "subservice",
        "detail_layanan",
        "layanan_detail",
        "tipe_layanan",
        "jenis_layanan",
        "service_type",
    ],
    "jumlah_titik_pengambilan": [
        "jumlah_titik_pengambilan",
        "jumlah titik pengambilan",
        "titik_pengambilan",
        "pickup_points",
        "pickup point",
        "jumlah pickup",
        "jumlah penjemputan",
    ],
    "jumlah_titik_pengantaran": [
        "jumlah_titik_pengantaran",
        "jumlah titik pengantaran",
        "titik_pengantaran",
        "dropoff_points",
        "dropoff point",
        "jumlah dropoff",
        "jumlah pengantaran",
    ],
}


def normalize_column_name(name: object) -> str:
    """Normalize a column name only for similarity checks."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def guess_column_mapping(columns: Iterable[object]) -> dict[str, str | None]:
    """Suggest mapping values from uploaded column names without requiring exact names."""
    column_list = [str(col) for col in columns]
    normalized_columns = {str(col): normalize_column_name(col) for col in column_list}
    mapping: dict[str, str | None] = {}

    for field, keywords in FIELD_KEYWORDS.items():
        best_column: str | None = None
        best_score = 0.0
        for column, normalized_column in normalized_columns.items():
            scores = []
            for keyword in keywords:
                normalized_keyword = normalize_column_name(keyword)
                exact_bonus = 1.0 if normalized_keyword in normalized_column else 0.0
                scores.append(max(exact_bonus, _similarity(normalized_column, normalized_keyword)))
            score = max(scores)
            if score > best_score:
                best_score = score
                best_column = column
        mapping[field] = best_column if best_score >= 0.55 else None

    for district_field, city_regency_field in [
        ("asal", "asal_kota_kabupaten"),
        ("tujuan", "tujuan_kota_kabupaten"),
    ]:
        district_column = mapping.get(district_field)
        city_regency_column = mapping.get(city_regency_field)
        if not district_column or district_column != city_regency_column:
            continue

        normalized_column = normalize_column_name(district_column)
        if any(token in normalized_column for token in ["kota", "kab", "kabupaten", "city", "regency"]):
            mapping[district_field] = None
        elif any(token in normalized_column for token in ["kecamatan", "district"]):
            mapping[city_regency_field] = None

    if mapping.get("sub_layanan") == mapping.get("layanan"):
        for column, normalized_column in normalized_columns.items():
            if "sub" in normalized_column and "layanan" in normalized_column:
                mapping["sub_layanan"] = column
                break

    return mapping


def build_column_mapping_ui(
    df: pd.DataFrame,
    default_mapping: dict[str, str | None],
    fields: dict[str, str],
) -> dict[str, str | None]:
    """Render selectbox mapping controls for all required semantic fields."""
    options = ["-- Tidak digunakan --"] + [str(col) for col in df.columns]
    mapping: dict[str, str | None] = {}

    st.caption(
        "Mapping otomatis hanya sebagai saran awal. User tetap dapat mengganti setiap pilihan secara manual."
    )

    col_left, col_right = st.columns(2)
    field_items = list(fields.items())
    for index, (field_key, label) in enumerate(field_items):
        container = col_left if index % 2 == 0 else col_right
        default_column = default_mapping.get(field_key)
        default_index = options.index(default_column) if default_column in options else 0
        with container:
            selected = st.selectbox(
                label,
                options,
                index=default_index,
                key=f"mapping_{field_key}",
                help="Pilih kolom dari dataset yang sesuai dengan konsep ini.",
            )
        mapping[field_key] = None if selected == "-- Tidak digunakan --" else selected

    return mapping
