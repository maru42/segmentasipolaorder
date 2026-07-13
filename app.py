from __future__ import annotations

import streamlit as st

from utils.advanced_analysis import render_advanced_pattern_analysis
from utils.apriori_utils import run_apriori
from utils.data_loader import describe_dataset, load_uploaded_file
from utils.mapping import (
    build_column_selection_ui,
    default_selected_columns,
    friendly_column_config,
    guess_column_mapping,
)
from utils.preprocessing import preprocess_dataset
from utils.transactions import (
    apply_interactive_filters,
    build_apriori_transactions,
    enrich_dataset,
)
from utils.ui import (
    inject_global_styles,
    render_dataset_overview,
    render_descriptive_analysis,
    render_mapping_summary,
    render_rule_summary,
    render_summary_cards,
)
from utils.visualizations import (
    render_apriori_visualizations,
    render_frequent_itemsets_chart,
)


DEFAULT_PREVIEW_ROWS = 100
FILTER_PREVIEW_ROWS = 250
DEFAULT_TABLE_HEIGHT = 360
COMPACT_TABLE_HEIGHT = 320
RULES_TABLE_HEIGHT = 420
APRIORI_PARAMETER_SPECS = {
    "min_support": {
        "label": "Minimum Support",
        "min": 0.01,
        "max": 1.0,
        "default": 0.05,
        "step": 0.01,
        "format": "%.2f",
    },
    "min_confidence": {
        "label": "Minimum Confidence",
        "min": 0.01,
        "max": 1.0,
        "default": 0.30,
        "step": 0.01,
        "format": "%.2f",
    },
    "min_lift": {
        "label": "Minimum Lift",
        "min": 0.01,
        "max": 10.0,
        "default": 1.0,
        "step": 0.05,
        "format": "%.2f",
    },
}


st.set_page_config(
    page_title="Dashboard Data Mining Ojek Online",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_state() -> None:
    """Prepare Streamlit session keys used by the dashboard."""
    defaults = {
        "raw_df": None,
        "processed_df": None,
        "enriched_df": None,
        "filtered_df": None,
        "apriori_result": None,
        "mapping": {},
        "selected_columns": [],
        "preprocessing_done": False,
        "uploaded_file_name": None,
        "active_section": "1. Upload Dataset",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar() -> str:
    """Render sidebar navigation and return the selected dashboard section."""
    sections = [
        "1. Upload Dataset",
        "2. Pilih Kolom",
        "3. Preprocessing",
        "4. Transformasi & Filter",
        "5. Analisis Deskriptif",
        "6. Analisis Pola Lanjutan",
        "7. Apriori",
        "8. Visualisasi Rules",
    ]

    if st.session_state.active_section not in sections:
        st.session_state.active_section = sections[0]

    st.sidebar.title("Data Mining Ojek Online")
    st.sidebar.caption("Association Rule Mining dengan Apriori")

    st.sidebar.markdown('<div class="sidebar-menu-title">Menu</div>', unsafe_allow_html=True)
    for index, section_name in enumerate(sections):
        is_active = st.session_state.active_section == section_name
        if st.sidebar.button(
            section_name,
            key=f"sidebar_menu_{index}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_section = section_name
            st.rerun()

    st.sidebar.divider()
    st.sidebar.info(
        "Upload CSV/XLSX, pilih kolom yang ingin dipakai, lalu jalankan preprocessing dan Apriori."
    )
    return st.session_state.active_section


def upload_section() -> None:
    """Render upload controls and store the uploaded dataset in session state."""
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload file dataset transaksi ojek online",
        type=["csv", "xlsx", "xls"],
        help="Aplikasi membaca nama kolom dari file dan tidak bergantung pada nama kolom tertentu.",
    )

    if uploaded_file is None:
        st.info("Silakan upload file CSV atau XLSX untuk memulai analisis.")
        return

    try:
        df = load_uploaded_file(uploaded_file)
    except Exception as exc:  # pragma: no cover - displayed in the app
        st.error(f"File tidak dapat dibaca: {exc}")
        return

    file_changed = uploaded_file.name != st.session_state.uploaded_file_name
    if file_changed:
        st.session_state.raw_df = df
        st.session_state.processed_df = None
        st.session_state.enriched_df = None
        st.session_state.filtered_df = None
        st.session_state.apriori_result = None
        st.session_state.preprocessing_done = False
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.selected_columns = default_selected_columns(df)
        st.session_state.mapping = guess_column_mapping(st.session_state.selected_columns)
    else:
        st.session_state.raw_df = df

    st.success(f"Dataset `{uploaded_file.name}` berhasil dimuat.")
    render_dataset_overview(st.session_state.raw_df, describe_dataset(st.session_state.raw_df))


def mapping_section() -> None:
    """Render active-column selection and update detected mapping in session state."""
    st.header("Pemilihan Kolom Analisis")
    if st.session_state.raw_df is None:
        st.warning("Upload dataset terlebih dahulu.")
        return

    st.write("Daftar seluruh nama kolom pada dataset:")
    st.code(", ".join(map(str, st.session_state.raw_df.columns.tolist())))

    current_selected = st.session_state.selected_columns or default_selected_columns(st.session_state.raw_df)
    old_mapping = st.session_state.mapping.copy()
    old_selected = list(current_selected)

    selected_columns = build_column_selection_ui(st.session_state.raw_df, current_selected)
    mapping = guess_column_mapping(selected_columns)

    st.session_state.selected_columns = selected_columns
    st.session_state.mapping = mapping
    if mapping != old_mapping or selected_columns != old_selected:
        st.session_state.processed_df = None
        st.session_state.enriched_df = None
        st.session_state.filtered_df = None
        st.session_state.apriori_result = None
        st.session_state.preprocessing_done = False
        st.info("Pilihan kolom berubah. Jalankan preprocessing ulang agar hasil analisis memakai kolom terbaru.")
    render_mapping_summary(mapping, selected_columns)


def preprocessing_section() -> None:
    """Render preprocessing controls and store processed/enriched data in session state."""
    st.header("Preprocessing Data")
    if st.session_state.raw_df is None:
        st.warning("Upload dataset terlebih dahulu.")
        return

    if not st.session_state.selected_columns:
        st.warning("Pilih minimal satu kolom di menu Mapping Kolom terlebih dahulu.")
        return

    if not st.session_state.mapping:
        st.session_state.mapping = guess_column_mapping(st.session_state.selected_columns)

    source_df = st.session_state.raw_df[st.session_state.selected_columns].copy()

    with st.expander("Dataset sebelum preprocessing", expanded=True):
        st.dataframe(source_df, use_container_width=True, height=DEFAULT_TABLE_HEIGHT)

    if st.button("Lakukan Preprocessing", type="primary", use_container_width=True):
        processed_df = preprocess_dataset(source_df, st.session_state.mapping)
        st.session_state.processed_df = processed_df
        st.session_state.enriched_df = enrich_dataset(processed_df, st.session_state.mapping)
        st.session_state.filtered_df = None
        st.session_state.apriori_result = None
        st.session_state.preprocessing_done = True
        st.success("Preprocessing selesai.")

    if st.session_state.processed_df is not None:
        with st.expander("Dataset sesudah preprocessing", expanded=True):
            st.dataframe(st.session_state.processed_df, use_container_width=True, height=DEFAULT_TABLE_HEIGHT)
        render_dataset_overview(
            st.session_state.processed_df,
            describe_dataset(st.session_state.processed_df),
            title="Ringkasan Dataset Setelah Preprocessing",
        )


def transform_filter_section() -> None:
    """Render transformation previews, Apriori transactions, and interactive filters."""
    st.header("Transformasi Data & Filter Interaktif")
    if st.session_state.enriched_df is None:
        st.warning("Jalankan preprocessing terlebih dahulu.")
        return

    enriched_df = st.session_state.enriched_df
    st.subheader("Kategori Waktu Otomatis")
    preview_candidates = [
        col
        for col in [
            "_kategori_waktu",
            st.session_state.mapping.get("waktu"),
            st.session_state.mapping.get("tanggal"),
            st.session_state.mapping.get("layanan"),
            st.session_state.mapping.get("pembayaran"),
            "_lokasi_asal",
            st.session_state.mapping.get("asal"),
            "_lokasi_tujuan",
            st.session_state.mapping.get("tujuan"),
            st.session_state.mapping.get("jumlah_titik_pengambilan"),
            st.session_state.mapping.get("jumlah_titik_pengantaran"),
            "_kategori_titik_pengambilan",
            "_kategori_titik_pengantaran",
        ]
        if col and col in enriched_df.columns
    ]
    preview_cols = list(dict.fromkeys(preview_candidates))
    st.dataframe(
        enriched_df[preview_cols].head(DEFAULT_PREVIEW_ROWS),
        use_container_width=True,
        height=COMPACT_TABLE_HEIGHT,
        column_config=friendly_column_config(enriched_df[preview_cols]),
    )

    st.subheader("Transaksi Apriori")
    transactions = build_apriori_transactions(
        enriched_df,
        st.session_state.mapping,
        st.session_state.selected_columns,
    )
    st.caption(
        "Setiap baris transaksi dibentuk dari kategori waktu, layanan, pembayaran, asal, tujuan, "
        "dan jumlah titik jika tersedia. Jumlah titik pengambilan/pengantaran "
        "ikut menjadi item Apriori jika kolomnya dipetakan."
    )
    st.dataframe(
        transactions.head(DEFAULT_PREVIEW_ROWS).to_frame(name="items_transaksi"),
        use_container_width=True,
        height=COMPACT_TABLE_HEIGHT,
        column_config=friendly_column_config(transactions.to_frame(name="items_transaksi")),
    )

    st.subheader("Filter Interaktif")
    old_filtered_df = st.session_state.get("filtered_df")
    filtered_df = apply_interactive_filters(
        enriched_df,
        st.session_state.mapping,
        st.session_state.selected_columns,
    )
    st.session_state.filtered_df = filtered_df
    if old_filtered_df is not None and not filtered_df.equals(old_filtered_df):
        st.session_state.apriori_result = None
    st.metric("Total transaksi setelah filter", f"{len(filtered_df):,}".replace(",", "."))
    st.dataframe(
        filtered_df.head(FILTER_PREVIEW_ROWS),
        use_container_width=True,
        height=DEFAULT_TABLE_HEIGHT,
        column_config=friendly_column_config(filtered_df),
    )


def descriptive_section() -> None:
    """Render descriptive dashboard charts for the current filtered dataset."""
    st.header("Analisis Deskriptif")
    df = st.session_state.get("filtered_df")
    if df is None:
        df = st.session_state.enriched_df
    if df is None:
        st.warning("Jalankan preprocessing dan transformasi terlebih dahulu.")
        return

    apriori_result = st.session_state.get("apriori_result")
    rules = apriori_result.rules if apriori_result is not None else None
    render_summary_cards(df, st.session_state.mapping, rules)
    st.divider()
    render_descriptive_analysis(df, st.session_state.mapping, st.session_state.selected_columns)


def _ensure_apriori_parameter_state() -> None:
    """Initialize Apriori parameter state for synced sliders and manual inputs."""
    for parameter, spec in APRIORI_PARAMETER_SPECS.items():
        value_key = f"apriori_{parameter}"
        slider_key = f"{value_key}_slider"
        manual_key = f"{value_key}_manual"
        value = float(st.session_state.get(value_key, spec["default"]))
        st.session_state.setdefault(value_key, value)
        st.session_state.setdefault(slider_key, value)
        st.session_state.setdefault(manual_key, value)


def _sync_apriori_parameter(parameter: str, source: str) -> None:
    """Sync one Apriori parameter after either its slider or manual input changes."""
    value_key = f"apriori_{parameter}"
    source_key = f"{value_key}_{source}"
    target_source = "manual" if source == "slider" else "slider"
    target_key = f"{value_key}_{target_source}"
    value = float(st.session_state[source_key])
    st.session_state[value_key] = value
    st.session_state[target_key] = value


def render_apriori_parameter_controls() -> tuple[float, float, float]:
    """Render Apriori sliders with synced manual number inputs and return parameter values."""
    _ensure_apriori_parameter_state()
    columns = st.columns(3)
    for column, (parameter, spec) in zip(columns, APRIORI_PARAMETER_SPECS.items()):
        value_key = f"apriori_{parameter}"
        with column:
            st.slider(
                spec["label"],
                min_value=spec["min"],
                max_value=spec["max"],
                step=spec["step"],
                key=f"{value_key}_slider",
                on_change=_sync_apriori_parameter,
                args=(parameter, "slider"),
            )
            st.number_input(
                "Input manual",
                min_value=spec["min"],
                max_value=spec["max"],
                step=spec["step"],
                format=spec["format"],
                key=f"{value_key}_manual",
                on_change=_sync_apriori_parameter,
                args=(parameter, "manual"),
            )

    return (
        float(st.session_state["apriori_min_support"]),
        float(st.session_state["apriori_min_confidence"]),
        float(st.session_state["apriori_min_lift"]),
    )


def apriori_section() -> None:
    """Render Apriori parameter controls and association-rule tables."""
    st.header("Implementasi Apriori")
    df = st.session_state.get("filtered_df")
    if df is None:
        df = st.session_state.enriched_df
    if df is None:
        st.warning("Jalankan preprocessing dan transformasi terlebih dahulu.")
        return

    st.subheader("Parameter Apriori")
    min_support, min_confidence, min_lift = render_apriori_parameter_controls()

    transactions = build_apriori_transactions(
        df,
        st.session_state.mapping,
        st.session_state.selected_columns,
    )
    result = run_apriori(
        transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )
    st.session_state.apriori_result = result

    render_rule_summary(df, st.session_state.mapping, result.rules)

    with st.expander("Preview One Hot Encoding", expanded=False):
        if result.encoded_transactions.empty:
            st.info("One hot encoding belum tersedia karena transaksi kosong.")
        else:
            st.dataframe(
                result.encoded_transactions.head(DEFAULT_PREVIEW_ROWS),
                use_container_width=True,
                height=COMPACT_TABLE_HEIGHT,
                column_config=friendly_column_config(result.encoded_transactions),
            )

    st.subheader("Frequent Itemsets")
    if result.frequent_itemsets.empty:
        st.warning("Tidak ada frequent itemset. Turunkan minimum support.")
    else:
        st.dataframe(
            result.frequent_itemsets,
            use_container_width=True,
            height=DEFAULT_TABLE_HEIGHT,
            column_config=friendly_column_config(result.frequent_itemsets),
        )
        render_frequent_itemsets_chart(result.frequent_itemsets)

    st.subheader("Association Rules")
    if result.rules.empty:
        st.warning("Tidak ada association rule. Turunkan support/confidence atau lift.")
    else:
        st.dataframe(
            result.rules,
            use_container_width=True,
            height=RULES_TABLE_HEIGHT,
            column_config=friendly_column_config(result.rules),
        )


def advanced_pattern_section() -> None:
    """Render advanced pattern analysis for the current filtered dataset."""
    st.header("Analisis Pola Lanjutan")
    df = st.session_state.get("filtered_df")
    if df is None:
        df = st.session_state.enriched_df
    if df is None:
        st.warning("Jalankan preprocessing dan transformasi terlebih dahulu.")
        return

    render_advanced_pattern_analysis(df, st.session_state.mapping)


def rules_visualization_section() -> None:
    """Render Apriori rule visualizations from the latest rule-mining result."""
    st.header("Visualisasi Hasil Apriori")
    result = st.session_state.get("apriori_result")
    if result is None:
        st.warning("Jalankan Apriori terlebih dahulu.")
        return

    render_apriori_visualizations(result.frequent_itemsets, result.rules)


def main() -> None:
    """Run the Streamlit dashboard application."""
    initialize_state()
    inject_global_styles()

    section = render_sidebar()
    st.title("Dashboard Data Mining Permintaan Layanan Ojek Online")
    st.caption(
        "Analisis Pola Permintaan Layanan Ojek Online Berdasarkan Waktu, Lokasi, "
        "dan Jenis Layanan Menggunakan Association Rule Mining"
    )

    if section == "1. Upload Dataset":
        upload_section()
    elif section == "2. Pilih Kolom":
        mapping_section()
    elif section == "3. Preprocessing":
        preprocessing_section()
    elif section == "4. Transformasi & Filter":
        transform_filter_section()
    elif section == "5. Analisis Deskriptif":
        descriptive_section()
    elif section == "6. Analisis Pola Lanjutan":
        advanced_pattern_section()
    elif section == "7. Apriori":
        apriori_section()
    elif section == "8. Visualisasi Rules":
        rules_visualization_section()


if __name__ == "__main__":
    main()
