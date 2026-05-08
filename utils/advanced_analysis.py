from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.visualizations import PLOT_TEMPLATE, bar_chart, value_count_frame


CASHLESS_PATTERN = re.compile(r"non\s*tunai|nontunai|cashless|e-?wallet|gopay|ovo|dana|qris|saldo", re.IGNORECASE)
CASH_PATTERN = re.compile(r"tunai|cash|cod", re.IGNORECASE)
GROUPED_ORDER_VALUES = {
    "multi instant",
    "same day",
    "food double",
}


def _selected_column(mapping: dict[str, str | None], field: str) -> str | None:
    column = mapping.get(field)
    return column if column else None


def _location_column(df: pd.DataFrame, fallback_column: str | None, helper_column: str) -> str | None:
    if helper_column in df.columns:
        return helper_column
    return fallback_column if fallback_column and fallback_column in df.columns else None


def _missing_info(title: str, fields: str) -> None:
    st.info(f"{title} belum tersedia karena kolom {fields} belum dipetakan.")


def _safe_crosstab(
    df: pd.DataFrame,
    row_column: str,
    column_column: str,
    top_rows: int = 12,
) -> pd.DataFrame:
    data = df[[row_column, column_column]].dropna().astype(str)
    if data.empty:
        return pd.DataFrame()

    top_values = data[row_column].value_counts().head(top_rows).index
    data = data[data[row_column].isin(top_values)]
    return pd.crosstab(data[row_column], data[column_column])


def _render_heatmap(
    matrix: pd.DataFrame,
    title: str,
    x_title: str,
    y_title: str,
) -> None:
    if matrix.empty:
        st.info(f"{title} belum memiliki data yang cukup.")
        return

    fig = px.imshow(
        matrix,
        title=title,
        template=PLOT_TEMPLATE,
        color_continuous_scale="Teal",
        labels=dict(x=x_title, y=y_title, color="Order"),
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _render_stacked_bar(
    data: pd.DataFrame,
    x: str,
    color: str,
    title: str,
) -> None:
    if data.empty:
        st.info(f"{title} belum memiliki data yang cukup.")
        return

    count_df = data.groupby([x, color], dropna=False).size().reset_index(name="jumlah")
    fig = px.bar(
        count_df,
        x=x,
        y="jumlah",
        color=color,
        title=title,
        template=PLOT_TEMPLATE,
        barmode="stack",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), yaxis_title="Jumlah order")
    st.plotly_chart(fig, use_container_width=True)


def _render_point_distribution(df: pd.DataFrame, column: str, title: str) -> None:
    point_counts = (
        pd.to_numeric(df[column], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("jumlah_titik")
        .reset_index(name="jumlah_order")
    )
    if point_counts.empty:
        st.info(f"{title} belum memiliki data yang cukup.")
        return

    fig = px.bar(
        point_counts,
        x="jumlah_titik",
        y="jumlah_order",
        title=title,
        template=PLOT_TEMPLATE,
        color="jumlah_order",
        color_continuous_scale="Teal",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), xaxis_title="Jumlah titik")
    st.plotly_chart(fig, use_container_width=True)


def normalize_payment_group(value: object) -> str:
    if pd.isna(value):
        return "Tidak Diketahui"
    text = str(value).strip()
    if CASHLESS_PATTERN.search(text):
        return "NonTunai"
    if CASH_PATTERN.search(text):
        return "Tunai"
    return "Lainnya"


def derive_grouped_status(series: pd.Series) -> pd.Series:
    def classify(value: object) -> str:
        if pd.isna(value):
            return "Non Grouped/Single"
        normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
        return (
            "Grouped/Multi Order"
            if normalized in GROUPED_ORDER_VALUES
            else "Non Grouped/Single"
        )

    return series.fillna("Tidak Diketahui").astype(str).map(
        classify
    )


def render_relationship_patterns(df: pd.DataFrame, mapping: dict[str, str | None]) -> None:
    st.subheader("Pola Hubungan Antar Variabel")

    service_column = _selected_column(mapping, "layanan")
    sub_service_column = _selected_column(mapping, "sub_layanan")
    payment_column = _selected_column(mapping, "pembayaran")
    origin_column = _location_column(df, _selected_column(mapping, "asal"), "_lokasi_asal")
    destination_column = _location_column(df, _selected_column(mapping, "tujuan"), "_lokasi_tujuan")

    col1, col2 = st.columns(2)
    with col1:
        if service_column and service_column in df.columns and "_kategori_waktu" in df.columns:
            matrix = _safe_crosstab(df, "_kategori_waktu", service_column)
            _render_heatmap(matrix, "Hubungan Kategori Waktu dan Jenis Layanan", "Layanan", "Kategori waktu")
        else:
            _missing_info("Hubungan waktu dan layanan", "waktu/layanan")

    with col2:
        if payment_column and payment_column in df.columns and service_column and service_column in df.columns:
            matrix = _safe_crosstab(df, payment_column, service_column)
            _render_heatmap(matrix, "Hubungan Pembayaran dan Layanan", "Layanan", "Pembayaran")
        else:
            _missing_info("Hubungan pembayaran dan layanan", "pembayaran/layanan")

    col1, col2 = st.columns(2)
    with col1:
        if origin_column and origin_column in df.columns and service_column and service_column in df.columns:
            matrix = _safe_crosstab(df, origin_column, service_column)
            _render_heatmap(matrix, "Hubungan Lokasi Asal dan Layanan", "Layanan", "Lokasi asal")
        else:
            _missing_info("Hubungan lokasi asal dan layanan", "asal/layanan")

    with col2:
        if destination_column and destination_column in df.columns and service_column and service_column in df.columns:
            matrix = _safe_crosstab(df, destination_column, service_column)
            _render_heatmap(matrix, "Hubungan Lokasi Tujuan dan Layanan", "Layanan", "Lokasi tujuan")
        else:
            _missing_info("Hubungan lokasi tujuan dan layanan", "tujuan/layanan")

    if sub_service_column and sub_service_column in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            if "_kategori_waktu" in df.columns:
                matrix = _safe_crosstab(df, "_kategori_waktu", sub_service_column)
                _render_heatmap(matrix, "Hubungan Kategori Waktu dan Sub Layanan", "Sub layanan", "Kategori waktu")
        with col2:
            if payment_column and payment_column in df.columns:
                matrix = _safe_crosstab(df, payment_column, sub_service_column)
                _render_heatmap(matrix, "Hubungan Pembayaran dan Sub Layanan", "Sub layanan", "Pembayaran")


def render_grouped_order_analysis(df: pd.DataFrame, mapping: dict[str, str | None]) -> None:
    st.subheader("Pola Grouped Order / Multi Order")

    group_column = _selected_column(mapping, "sub_layanan")
    if not group_column or group_column not in df.columns:
        _missing_info("Analisis grouped/multi order", "sub layanan")
        return

    data = df.copy()
    data["_status_grouped_order"] = derive_grouped_status(data[group_column])

    status_counts = value_count_frame(data, "_status_grouped_order")
    raw_counts = value_count_frame(data, group_column)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            bar_chart(status_counts, "_status_grouped_order", "jumlah", "Grouped vs Non Grouped Order"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            bar_chart(raw_counts, group_column, "jumlah", "Distribusi Sub Layanan"),
            use_container_width=True,
        )

    if "_kategori_waktu" in data.columns:
        _render_stacked_bar(
            data[["_kategori_waktu", "_status_grouped_order"]],
            "_kategori_waktu",
            "_status_grouped_order",
            "Hubungan Kategori Waktu dengan Grouped Order",
        )

    if "_kategori_multi_titik" in data.columns:
        matrix = pd.crosstab(data["_status_grouped_order"], data["_kategori_multi_titik"])
        _render_heatmap(
            matrix,
            "Validasi Grouped Order terhadap Single/Multi Titik",
            "Kategori titik",
            "Status grouped order",
        )


def render_point_count_analysis(df: pd.DataFrame, mapping: dict[str, str | None]) -> None:
    st.subheader("Pola Jumlah Titik Pengambilan dan Pengantaran")

    pickup_column = _selected_column(mapping, "jumlah_titik_pengambilan")
    dropoff_column = _selected_column(mapping, "jumlah_titik_pengantaran")
    sub_service_column = _selected_column(mapping, "sub_layanan")

    has_pickup = pickup_column and pickup_column in df.columns
    has_dropoff = dropoff_column and dropoff_column in df.columns
    if not has_pickup and not has_dropoff:
        _missing_info("Analisis jumlah titik", "jumlah titik pengambilan/pengantaran")
        return

    multi_point_counts = (
        value_count_frame(df, "_kategori_multi_titik")
        if "_kategori_multi_titik" in df.columns
        else pd.DataFrame()
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total transaksi", f"{len(df):,}".replace(",", "."))

    if "_kategori_multi_titik" in df.columns:
        multi_count = int((df["_kategori_multi_titik"] == "Multi Titik").sum())
        multi_share = (multi_count / len(df) * 100) if len(df) else 0
        metric_cols[1].metric("Multi titik", f"{multi_count:,}".replace(",", "."), f"{multi_share:.1f}%")
    else:
        metric_cols[1].metric("Multi titik", "Belum tersedia")

    avg_pickup = pd.to_numeric(df[pickup_column], errors="coerce").mean() if has_pickup else float("nan")
    avg_dropoff = pd.to_numeric(df[dropoff_column], errors="coerce").mean() if has_dropoff else float("nan")
    metric_cols[2].metric("Rata-rata pickup", "-" if pd.isna(avg_pickup) else f"{avg_pickup:.2f}")
    metric_cols[3].metric("Rata-rata dropoff", "-" if pd.isna(avg_dropoff) else f"{avg_dropoff:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        if has_pickup:
            _render_point_distribution(df, pickup_column, "Distribusi Jumlah Titik Pengambilan")
    with col2:
        if has_dropoff:
            _render_point_distribution(df, dropoff_column, "Distribusi Jumlah Titik Pengantaran")

    col1, col2 = st.columns(2)
    with col1:
        if not multi_point_counts.empty:
            st.plotly_chart(
                bar_chart(multi_point_counts, "_kategori_multi_titik", "jumlah", "Single vs Multi Titik"),
                use_container_width=True,
            )
    with col2:
        if sub_service_column and sub_service_column in df.columns and "_kategori_multi_titik" in df.columns:
            matrix = _safe_crosstab(df, sub_service_column, "_kategori_multi_titik", top_rows=15)
            _render_heatmap(matrix, "Hubungan Sub Layanan dengan Single/Multi Titik", "Kategori titik", "Sub layanan")

    if sub_service_column and sub_service_column in df.columns:
        summary_fields = [sub_service_column]
        if has_pickup:
            summary_fields.append(pickup_column)
        if has_dropoff:
            summary_fields.append(dropoff_column)
        point_summary = df[summary_fields].copy()
        agg_spec: dict[str, tuple[str, str]] = {"jumlah_transaksi": (sub_service_column, "count")}
        if has_pickup:
            agg_spec["rata_rata_titik_pengambilan"] = (pickup_column, "mean")
            agg_spec["maks_titik_pengambilan"] = (pickup_column, "max")
        if has_dropoff:
            agg_spec["rata_rata_titik_pengantaran"] = (dropoff_column, "mean")
            agg_spec["maks_titik_pengantaran"] = (dropoff_column, "max")
        point_summary = (
            point_summary.groupby(sub_service_column)
            .agg(**agg_spec)
            .sort_values("jumlah_transaksi", ascending=False)
            .reset_index()
        )
        st.dataframe(point_summary, use_container_width=True, hide_index=True)


def render_ramadan_comparison(df: pd.DataFrame, mapping: dict[str, str | None]) -> None:
    st.subheader("Perbandingan Ramadan dan Pasca Ramadan 2026")
    st.caption("Ramadan 2026 didefinisikan sebagai 20 Februari 2026 sampai 21 Maret 2026.")

    if "_periode_ramadan_2026" not in df.columns or "_tanggal_filter" not in df.columns:
        _missing_info("Perbandingan Ramadan", "tanggal")
        return

    compare_df = df[df["_periode_ramadan_2026"].isin(["Ramadan 2026", "Pasca Ramadan 2026"])].copy()
    if compare_df.empty:
        st.info("Tidak ada transaksi pada periode Ramadan 2026 atau pasca Ramadan 2026 di data yang sedang difilter.")
        return

    period_counts = value_count_frame(compare_df, "_periode_ramadan_2026")
    service_column = _selected_column(mapping, "layanan")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            bar_chart(period_counts, "_periode_ramadan_2026", "jumlah", "Jumlah Transaksi Ramadan vs Pasca Ramadan"),
            use_container_width=True,
        )
    with col2:
        if service_column and service_column in compare_df.columns:
            _render_stacked_bar(
                compare_df[["_periode_ramadan_2026", service_column]],
                "_periode_ramadan_2026",
                service_column,
                "Komposisi Layanan Ramadan vs Pasca Ramadan",
            )
        else:
            _missing_info("Komposisi layanan Ramadan", "layanan")


def render_payment_tendency(df: pd.DataFrame, mapping: dict[str, str | None]) -> None:
    st.subheader("Kecenderungan Transaksi Tunai dan NonTunai")

    payment_column = _selected_column(mapping, "pembayaran")
    service_column = _selected_column(mapping, "layanan")
    sub_service_column = _selected_column(mapping, "sub_layanan")
    if not payment_column or payment_column not in df.columns:
        _missing_info("Analisis tunai dan non-tunai", "pembayaran")
        return

    data = df.copy()
    data["_kelompok_pembayaran"] = data[payment_column].map(normalize_payment_group)
    payment_counts = value_count_frame(data, "_kelompok_pembayaran")
    payment_counts["persentase"] = (payment_counts["jumlah"] / payment_counts["jumlah"].sum() * 100).round(2)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            bar_chart(payment_counts, "_kelompok_pembayaran", "jumlah", "Tunai vs NonTunai"),
            use_container_width=True,
        )
    with col2:
        st.dataframe(payment_counts, use_container_width=True, hide_index=True)

    if service_column and service_column in data.columns:
        _render_stacked_bar(
            data[[service_column, "_kelompok_pembayaran"]],
            service_column,
            "_kelompok_pembayaran",
            "Hubungan Pembayaran Tunai/NonTunai dengan Layanan",
        )
    if sub_service_column and sub_service_column in data.columns:
        _render_stacked_bar(
            data[[sub_service_column, "_kelompok_pembayaran"]],
            sub_service_column,
            "_kelompok_pembayaran",
            "Hubungan Pembayaran Tunai/NonTunai dengan Sub Layanan",
        )


def render_personal_history_patterns(df: pd.DataFrame, mapping: dict[str, str | None]) -> None:
    st.subheader("Karakteristik Pola Transaksi Historis Pribadi")

    service_column = _selected_column(mapping, "layanan")
    sub_service_column = _selected_column(mapping, "sub_layanan")
    payment_column = _selected_column(mapping, "pembayaran")
    origin_column = _location_column(df, _selected_column(mapping, "asal"), "_lokasi_asal")
    destination_column = _location_column(df, _selected_column(mapping, "tujuan"), "_lokasi_tujuan")
    tariff_column = _selected_column(mapping, "tarif")
    distance_column = _selected_column(mapping, "jarak")

    pattern_columns = [
        column
        for column in [
            "_kategori_waktu",
            service_column,
            sub_service_column,
            payment_column,
            origin_column,
            destination_column,
        ]
        if column and column in df.columns
    ]

    if not pattern_columns:
        st.info("Pola historis belum tersedia karena kolom utama belum dipetakan.")
        return

    data = df.copy()
    data["_pola_transaksi"] = (
        data[pattern_columns]
        .fillna("Tidak Diketahui")
        .astype(str)
        .agg(" | ".join, axis=1)
    )

    pattern_counts = (
        data["_pola_transaksi"]
        .value_counts()
        .head(15)
        .rename_axis("pola_transaksi")
        .reset_index(name="jumlah")
    )

    if tariff_column and tariff_column in df.columns:
        avg_tariff = pd.to_numeric(df[tariff_column], errors="coerce").mean()
    else:
        avg_tariff = float("nan")
    if distance_column and distance_column in df.columns:
        avg_distance = pd.to_numeric(df[distance_column], errors="coerce").mean()
    else:
        avg_distance = float("nan")

    col1, col2 = st.columns(2)
    with col1:
        chart_df = pattern_counts.sort_values("jumlah")
        fig = px.bar(
            chart_df,
            x="jumlah",
            y="pola_transaksi",
            orientation="h",
            title="Top Pola Transaksi Historis",
            template=PLOT_TEMPLATE,
            color="jumlah",
            color_continuous_scale="Teal",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Total transaksi", f"{len(df):,}".replace(",", "."))
        metric_cols[1].metric("Rata-rata tarif", "-" if pd.isna(avg_tariff) else f"{avg_tariff:,.0f}")
        metric_cols[2].metric("Rata-rata jarak", "-" if pd.isna(avg_distance) else f"{avg_distance:,.2f}")
        st.dataframe(pattern_counts, use_container_width=True, hide_index=True)


def render_advanced_pattern_analysis(df: pd.DataFrame, mapping: dict[str, str | None]) -> None:
    st.caption(
        "Analisis ini memakai hasil mapping kolom. Jika sebuah insight belum muncul, cek kembali mapping kolom yang relevan."
    )
    render_relationship_patterns(df, mapping)
    st.divider()
    render_grouped_order_analysis(df, mapping)
    st.divider()
    render_point_count_analysis(df, mapping)
    st.divider()
    render_ramadan_comparison(df, mapping)
    st.divider()
    render_payment_tendency(df, mapping)
    st.divider()
    render_personal_history_patterns(df, mapping)
