from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher
import re

import pandas as pd
import streamlit as st


COLUMN_MAPPING_THRESHOLD = 0.55
FRIENDLY_COLUMN_LABELS = {
    "_jam_order": "Jam Order",
    "_kategori_waktu": "Kategori Waktu",
    "_tanggal_filter": "Tanggal Filter",
    "_periode_ramadan_2026": "Periode Ramadan 2026",
    "_lokasi_asal": "Lokasi Asal",
    "_lokasi_tujuan": "Lokasi Tujuan",
    "_jumlah_titik_pengambilan": "Jumlah Titik\nPengambilan",
    "_jumlah_titik_pengantaran": "Jumlah Titik\nPengantaran",
    "_kategori_titik_pengambilan": "Kategori Titik\nPengambilan",
    "_kategori_titik_pengantaran": "Kategori Titik\nPengantaran",
    "_status_grouped_order": "Status Grouped Order",
    "_kelompok_pembayaran": "Kelompok Pembayaran",
    "_pola_transaksi": "Pola Transaksi",
    "items_transaksi": "Items Transaksi",
    "jumlah_titik": "Jumlah Titik",
    "jumlah_order": "Jumlah Order",
    "jumlah_transaksi": "Jumlah Transaksi",
    "rata_rata_titik_pengambilan": "Rata-rata Titik\nPengambilan",
    "rata_rata_titik_pengantaran": "Rata-rata Titik\nPengantaran",
    "maks_titik_pengambilan": "Maks Titik\nPengambilan",
    "maks_titik_pengantaran": "Maks Titik\nPengantaran",
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
    "tujuan": [
        "tujuan",
        "kecamatan_tujuan",
        "tujuan_kecamatan",
        "destination_district",
        "dropoff_district",
        "kecamatan tujuan",
        "kecamatan antar",
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
    """Return a normalized column name for keyword similarity checks."""
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
    """Return detected semantic roles for the provided dataset column names."""
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
        mapping[field] = best_column if best_score >= COLUMN_MAPPING_THRESHOLD else None

    directional_pairs = [
        ("asal", "tujuan", ["asal", "origin", "pickup", "jemput"], ["tujuan", "destination", "dropoff", "antar"]),
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


def default_selected_columns(df: pd.DataFrame) -> list[str]:
    """Return all DataFrame columns as the default active analysis columns."""
    return [str(column) for column in df.columns]


def format_column_label(column: object, multiline: bool = False) -> str:
    """Return a user-facing label for internal/helper column names."""
    text = str(column)
    if text in FRIENDLY_COLUMN_LABELS:
        label = FRIENDLY_COLUMN_LABELS[text]
    else:
        label = re.sub(r"^_+", "", text)
        label = label.replace("_", " ")
        label = re.sub(r"\s+", " ", label).strip().title()

    if not multiline:
        return label.replace("\n", " ")

    return _apply_multiline_label(label)


def _apply_multiline_label(label: str) -> str:
    """Insert compact line breaks for long table headers."""
    if "\n" in label:
        return label
    replacements = {
        "Jumlah Titik Pengambilan": "Jumlah Titik\nPengambilan",
        "Jumlah Titik Pengantaran": "Jumlah Titik\nPengantaran",
        "Kategori Titik Pengambilan": "Kategori Titik\nPengambilan",
        "Kategori Titik Pengantaran": "Kategori Titik\nPengantaran",
        "Rata-Rata Titik Pengambilan": "Rata-rata Titik\nPengambilan",
        "Rata-Rata Titik Pengantaran": "Rata-rata Titik\nPengantaran",
        "Maks Titik Pengambilan": "Maks Titik\nPengambilan",
        "Maks Titik Pengantaran": "Maks Titik\nPengantaran",
    }
    return replacements.get(label, label)


def friendly_column_config(df: pd.DataFrame) -> dict[str, st.column_config.Column]:
    """Return Streamlit column config with readable labels for a DataFrame."""
    return {
        str(column): st.column_config.Column(format_column_label(column, multiline=True))
        for column in df.columns
    }


def build_column_selection_ui(
    df: pd.DataFrame,
    selected_columns: list[str] | None,
) -> list[str]:
    """Render the active-column picker and return the selected column names."""
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
