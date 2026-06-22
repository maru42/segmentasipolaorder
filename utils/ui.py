from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import DatasetDescription
from utils.visualizations import PLOT_TEMPLATE, bar_chart, histogram, value_count_frame


def inject_global_styles() -> None:
    """Inject compact styles that follow the user's device color scheme."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light dark;
            --page: #f6f8fb;
            --panel: #ffffff;
            --panel-soft: #f8fafc;
            --text: #172033;
            --muted: #64748b;
            --accent: #0f766e;
            --accent-2: #2563eb;
            --line: rgba(15, 23, 42, 0.12);
            --shadow: rgba(15, 23, 42, 0.05);
            --button-bg: #ffffff;
            --sidebar-bg: #ffffff;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --page: #0f172a;
                --panel: #111827;
                --panel-soft: #1f2937;
                --text: #e5e7eb;
                --muted: #94a3b8;
                --accent: #2dd4bf;
                --accent-2: #60a5fa;
                --line: rgba(226, 232, 240, 0.16);
                --shadow: rgba(0, 0, 0, 0.24);
                --button-bg: #111827;
                --sidebar-bg: #0b1120;
            }
        }
        .stApp {
            background: var(--page);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--line);
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 10px 24px var(--shadow);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        .sidebar-menu-title {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            margin: 1rem 0 0.35rem;
            text-transform: uppercase;
        }
        div.stButton > button {
            border-radius: 8px;
            border: 1px solid var(--accent);
            background: var(--button-bg);
            color: var(--accent);
        }
        [data-testid="stSidebar"] div.stButton > button {
            justify-content: flex-start;
            border-color: transparent;
            color: var(--text);
            background: transparent;
            box-shadow: none;
            min-height: 2.45rem;
        }
        [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            border-color: var(--accent);
            color: var(--accent);
            background: color-mix(in srgb, var(--accent) 13%, transparent);
        }
        div.stButton > button:hover {
            border-color: var(--accent-2);
            color: var(--accent-2);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3, h4, h5, h6, p, label, span {
            letter-spacing: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_overview(
    df: pd.DataFrame,
    description: DatasetDescription,
    title: str = "Ringkasan Dataset",
) -> None:
    st.subheader(title)
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah baris", f"{description.row_count:,}".replace(",", "."))
    col2.metric("Jumlah kolom", f"{description.column_count:,}".replace(",", "."))
    col3.metric("Total missing values", f"{int(df.isna().sum().sum()):,}".replace(",", "."))

    st.subheader("Preview Dataset")
    st.dataframe(df, use_container_width=True, height=360)

    st.subheader("Daftar Nama Kolom")
    st.dataframe(pd.DataFrame({"nama_kolom": description.columns}), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Missing Values")
        st.dataframe(description.missing_values, use_container_width=True, height=300)
    with col_right:
        st.subheader("Tipe Data")
        st.dataframe(description.dtypes, use_container_width=True, height=300)


def render_mapping_summary(
    mapping: dict[str, str | None],
    selected_columns: list[str] | None = None,
) -> None:
    st.subheader("Ringkasan Kolom Aktif")
    if selected_columns:
        st.dataframe(
            pd.DataFrame({"kolom_dipakai": selected_columns}),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Peran Kolom Terdeteksi Otomatis")
    detected_rows = [
        {"peran": field, "kolom_dataset": column}
        for field, column in mapping.items()
        if column
    ]
    if detected_rows:
        st.dataframe(pd.DataFrame(detected_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada peran khusus yang terdeteksi dari kolom pilihan.")

    generic_columns = [
        column
        for column in (selected_columns or [])
        if column not in {mapped_column for mapped_column in mapping.values() if mapped_column}
    ]
    if generic_columns:
        st.subheader("Kolom Umum yang Tetap Dipakai")
        st.caption(
            "Kolom ini tidak punya peran khusus, tapi tetap dipakai untuk filter, "
            "visualisasi otomatis, dan transaksi Apriori."
        )
        st.dataframe(
            pd.DataFrame({"kolom_umum": generic_columns}),
            use_container_width=True,
            hide_index=True,
        )


def _first_existing_column(df: pd.DataFrame, columns: list[str | None]) -> str | None:
    for column in columns:
        if column and column in df.columns:
            return column
    return None


def _location_column(df: pd.DataFrame, fallback_column: str | None, helper_column: str) -> str | None:
    if helper_column in df.columns:
        return helper_column
    return fallback_column if fallback_column and fallback_column in df.columns else None


def render_summary_cards(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    rules: pd.DataFrame | None = None,
) -> None:
    tariff_column = _first_existing_column(df, [mapping.get("tarif")])
    distance_column = _first_existing_column(df, [mapping.get("jarak")])
    service_column = _first_existing_column(df, [mapping.get("layanan")])

    total_transactions = len(df)
    avg_tariff = (
        pd.to_numeric(df[tariff_column], errors="coerce").mean()
        if tariff_column
        else float("nan")
    )
    avg_distance = (
        pd.to_numeric(df[distance_column], errors="coerce").mean()
        if distance_column
        else float("nan")
    )
    top_service = (
        df[service_column].astype(str).mode().iloc[0]
        if service_column and not df[service_column].dropna().empty
        else "-"
    )

    best_rule = "Belum ada rule"
    if rules is not None and not rules.empty:
        best = rules.sort_values(["confidence", "lift"], ascending=False).iloc[0]
        best_rule = f"{best['antecedents']} -> {best['consequents']}"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total transaksi", f"{total_transactions:,}".replace(",", "."))
    col2.metric("Rata-rata tarif", "-" if pd.isna(avg_tariff) else f"{avg_tariff:,.0f}")
    col3.metric("Rata-rata jarak", "-" if pd.isna(avg_distance) else f"{avg_distance:,.2f}")
    col4.metric("Layanan terbanyak", top_service)
    col5.metric("Rule terbaik", best_rule[:52] + ("..." if len(best_rule) > 52 else ""))


def render_rule_summary(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    rules: pd.DataFrame,
) -> None:
    render_summary_cards(df, mapping, rules)


def _render_count_chart(df: pd.DataFrame, column: str, title: str) -> None:
    count_df = value_count_frame(df, column)
    if count_df.empty:
        st.info(f"Tidak ada data untuk {title.lower()}.")
        return
    st.plotly_chart(bar_chart(count_df, column, "jumlah", title), use_container_width=True)


def _render_selected_column_chart(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        return

    if pd.api.types.is_numeric_dtype(df[column]):
        st.plotly_chart(histogram(df, column, f"Distribusi {column}"), use_container_width=True)
        return

    if pd.api.types.is_datetime64_any_dtype(df[column]):
        date_counts = (
            df[column]
            .dropna()
            .dt.date
            .value_counts()
            .sort_index()
            .rename_axis("tanggal")
            .reset_index(name="jumlah")
        )
        if date_counts.empty:
            st.info(f"Tidak ada data tanggal valid untuk {column}.")
            return
        fig = px.line(
            date_counts,
            x="tanggal",
            y="jumlah",
            markers=True,
            title=f"Tren {column}",
            template=PLOT_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
        return

    _render_count_chart(df, column, f"Jumlah Data per {column}")


def render_selected_columns_analysis(df: pd.DataFrame, selected_columns: list[str] | None) -> None:
    selected_columns = [column for column in (selected_columns or []) if column in df.columns]
    if not selected_columns:
        return

    st.subheader("Analisis Kolom Terpilih")
    st.caption("Visualisasi berikut dibuat otomatis berdasarkan tipe data setiap kolom yang dipilih.")
    for first, second in zip(selected_columns[0::2], selected_columns[1::2]):
        col1, col2 = st.columns(2)
        with col1:
            _render_selected_column_chart(df, first)
        with col2:
            _render_selected_column_chart(df, second)

    if len(selected_columns) % 2 == 1:
        _render_selected_column_chart(df, selected_columns[-1])


def _top_label_and_count(series: pd.Series) -> tuple[str, int]:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return "Belum tersedia", 0
    return str(counts.index[0]), int(counts.iloc[0])


def _hour_count_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "_jam_order" not in df.columns:
        return pd.DataFrame(columns=["jam", "jam_label", "jumlah"])

    hours = pd.to_numeric(df["_jam_order"], errors="coerce").dropna()
    if hours.empty:
        return pd.DataFrame(columns=["jam", "jam_label", "jumlah"])

    hour_counts = (
        hours.astype(int)
        .value_counts()
        .rename_axis("jam")
        .reset_index(name="jumlah")
        .sort_values("jam")
    )
    hour_counts["jam_label"] = hour_counts["jam"].map(lambda hour: f"{hour:02d}:00")
    return hour_counts


def _route_count_frame(
    df: pd.DataFrame,
    origin_column: str | None,
    destination_column: str | None,
    top_n: int = 15,
) -> pd.DataFrame:
    if not origin_column or not destination_column:
        return pd.DataFrame(columns=["rute", "jumlah"])
    if origin_column not in df.columns or destination_column not in df.columns:
        return pd.DataFrame(columns=["rute", "jumlah"])

    routes = (
        df[[origin_column, destination_column]]
        .fillna("Tidak Diketahui")
        .astype(str)
        .assign(
            rute=lambda data: data[origin_column] + " -> " + data[destination_column]
        )
    )
    return (
        routes["rute"]
        .value_counts()
        .head(top_n)
        .rename_axis("rute")
        .reset_index(name="jumlah")
    )


def render_peak_demand_insights(df: pd.DataFrame, mapping: dict[str, str | None]) -> None:
    """Show direct answers for busy time and busy location questions."""
    st.subheader("Insight Jam & Lokasi Ramai")

    origin_column = _location_column(df, mapping.get("asal"), "_lokasi_asal")
    destination_column = _location_column(df, mapping.get("tujuan"), "_lokasi_tujuan")
    hour_counts = _hour_count_frame(df)
    route_counts = _route_count_frame(df, origin_column, destination_column)

    if hour_counts.empty:
        peak_hour_label, peak_hour_count = "Belum tersedia", 0
    else:
        peak_hour = hour_counts.sort_values("jumlah", ascending=False).iloc[0]
        peak_hour_label = str(peak_hour["jam_label"])
        peak_hour_count = int(peak_hour["jumlah"])

    peak_time_label, peak_time_count = (
        _top_label_and_count(df["_kategori_waktu"])
        if "_kategori_waktu" in df.columns
        else ("Belum tersedia", 0)
    )
    peak_origin_label, peak_origin_count = (
        _top_label_and_count(df[origin_column])
        if origin_column and origin_column in df.columns
        else ("Belum tersedia", 0)
    )
    peak_destination_label, peak_destination_count = (
        _top_label_and_count(df[destination_column])
        if destination_column and destination_column in df.columns
        else ("Belum tersedia", 0)
    )
    peak_route_label = "Belum tersedia" if route_counts.empty else str(route_counts.iloc[0]["rute"])
    peak_route_count = 0 if route_counts.empty else int(route_counts.iloc[0]["jumlah"])

    metric_cols = st.columns(5)
    metric_cols[0].metric("Jam tersibuk", f"{peak_hour_label}", f"{peak_hour_count} order")
    metric_cols[1].metric("Waktu tersibuk", peak_time_label, f"{peak_time_count} order")
    metric_cols[2].metric("Asal tersibuk", peak_origin_label, f"{peak_origin_count} order")
    metric_cols[3].metric("Tujuan tersibuk", peak_destination_label, f"{peak_destination_count} order")
    metric_cols[4].metric("Rute tersibuk", peak_route_label[:42], f"{peak_route_count} order")

    col1, col2 = st.columns(2)
    with col1:
        if hour_counts.empty:
            st.info("Grafik jam ramai belum tersedia karena kolom waktu/tanggal belum dipetakan.")
        else:
            fig = px.bar(
                hour_counts,
                x="jam_label",
                y="jumlah",
                title="Jumlah Order per Jam",
                template=PLOT_TEMPLATE,
                color="jumlah",
                color_continuous_scale="Teal",
            )
            fig.update_layout(
                xaxis_title="Jam",
                yaxis_title="Jumlah order",
                showlegend=False,
                margin=dict(l=10, r=10, t=55, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if route_counts.empty:
            st.info("Grafik rute ramai belum tersedia karena kolom asal/tujuan belum dipetakan.")
        else:
            fig = px.bar(
                route_counts.sort_values("jumlah"),
                x="jumlah",
                y="rute",
                orientation="h",
                title="Top Rute Asal-Tujuan",
                template=PLOT_TEMPLATE,
                color="jumlah",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                xaxis_title="Jumlah order",
                yaxis_title="",
                showlegend=False,
                margin=dict(l=10, r=10, t=55, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    if (
        not hour_counts.empty
        and origin_column
        and origin_column in df.columns
        and "_jam_order" in df.columns
    ):
        heatmap_df = df[[origin_column, "_jam_order"]].copy()
        heatmap_df["_jam_order"] = pd.to_numeric(heatmap_df["_jam_order"], errors="coerce")
        heatmap_df = heatmap_df.dropna(subset=[origin_column, "_jam_order"])
        if not heatmap_df.empty:
            top_origins = heatmap_df[origin_column].astype(str).value_counts().head(10).index
            heatmap_df = heatmap_df[heatmap_df[origin_column].astype(str).isin(top_origins)]
            heatmap_df["jam_label"] = heatmap_df["_jam_order"].astype(int).map(lambda hour: f"{hour:02d}:00")
            pivot = pd.crosstab(heatmap_df[origin_column].astype(str), heatmap_df["jam_label"])
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            fig = px.imshow(
                pivot,
                title="Heatmap Keramaian: Jam x Lokasi Asal",
                template=PLOT_TEMPLATE,
                color_continuous_scale="Teal",
                labels=dict(x="Jam", y="Lokasi asal", color="Order"),
            )
            fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)


def render_descriptive_analysis(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    selected_columns: list[str] | None = None,
) -> None:
    render_peak_demand_insights(df, mapping)
    st.divider()
    render_selected_columns_analysis(df, selected_columns)
    st.divider()

    charts = [
        ("_kategori_waktu", "Jumlah Order per Kategori Waktu"),
        (mapping.get("layanan"), "Jumlah Order per Layanan"),
        (mapping.get("pembayaran"), "Pembayaran Terbanyak"),
        (_location_column(df, mapping.get("asal"), "_lokasi_asal"), "Lokasi Asal Terbanyak"),
        (_location_column(df, mapping.get("tujuan"), "_lokasi_tujuan"), "Lokasi Tujuan Terbanyak"),
    ]

    for first, second in zip(charts[0::2], charts[1::2]):
        col1, col2 = st.columns(2)
        with col1:
            column, title = first
            if column and column in df.columns:
                _render_count_chart(df, column, title)
            else:
                st.info(f"{title} belum tersedia karena kolom belum dipetakan.")
        with col2:
            column, title = second
            if column and column in df.columns:
                _render_count_chart(df, column, title)
            else:
                st.info(f"{title} belum tersedia karena kolom belum dipetakan.")

    remaining = charts[-1]
    if len(charts) % 2 == 1:
        column, title = remaining
        if column and column in df.columns:
            _render_count_chart(df, column, title)

    col1, col2 = st.columns(2)
    with col1:
        tariff_column = mapping.get("tarif")
        if tariff_column and tariff_column in df.columns:
            st.plotly_chart(histogram(df, tariff_column, "Distribusi Tarif"), use_container_width=True)
        else:
            st.info("Distribusi tarif belum tersedia karena kolom tarif belum dipetakan.")
    with col2:
        distance_column = mapping.get("jarak")
        if distance_column and distance_column in df.columns:
            st.plotly_chart(histogram(df, distance_column, "Distribusi Jarak"), use_container_width=True)
        else:
            st.info("Distribusi jarak belum tersedia karena kolom jarak belum dipetakan.")
