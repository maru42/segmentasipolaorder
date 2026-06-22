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
        mapping["sub_layanan"] = None
        for column, normalized_column in normalized_columns.items():
            if "sub" in normalized_column and "layanan" in normalized_column:
                mapping["sub_layanan"] = column
                break

    directional_pairs = [
        ("asal", "tujuan", ["asal", "origin", "pickup", "jemput"], ["tujuan", "destination", "dropoff", "antar"]),
        (
            "asal_kota_kabupaten",
            "tujuan_kota_kabupaten",
            ["asal", "origin", "pickup"],
            ["tujuan", "destination", "dropoff"],
        ),
        (
            "jumlah_titik_pengambilan",
            "jumlah_titik_pengantaran",
            ["pengambilan", "pickup", "penjemputan"],
            ["pengantaran", "dropoff", "antar"],
        ),
    ]
    for left_field, right_field, left_tokens, right_tokens in directional_pairs:
        left_column = mapping.get(left_field)
        right_column = mapping.get(right_field)
        if not left_column or left_column != right_column:
            continue

        normalized_column = normalize_column_name(left_column)
        has_left_token = any(token in normalized_column for token in left_tokens)
        has_right_token = any(token in normalized_column for token in right_tokens)
        if has_left_token and not has_right_token:
            mapping[right_field] = None
        elif has_right_token and not has_left_token:
            mapping[left_field] = None

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


def default_selected_columns(df: pd.DataFrame) -> list[str]:
    """Use every uploaded column by default so the user can remove what is not needed."""
    return [str(column) for column in df.columns]


def build_column_selection_ui(
    df: pd.DataFrame,
    selected_columns: list[str] | None,
) -> list[str]:
    """Render a flexible column picker instead of fixed semantic selectboxes."""
    options = [str(column) for column in df.columns]
    valid_defaults = [column for column in (selected_columns or options) if column in options]
    if not valid_defaults:
        valid_defaults = options

    st.caption(
        "Pilih kolom dataset yang ingin dipakai untuk preprocessing, filter, visualisasi, "
        "dan Apriori. Peran kolom seperti waktu, lokasi, tarif, dan layanan akan dideteksi otomatis "
        "dari kolom yang dipilih."
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.multiselect(
            "Kolom yang dipakai",
            options,
            default=valid_defaults,
            help="Hanya kolom yang dipilih di sini yang digunakan pada proses berikutnya.",
        )
    with col2:
        st.metric("Kolom dipilih", len(selected))

    return selected
