from __future__ import annotations

import math
import re

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.mapping import format_column_label


PLOT_TEMPLATE = None
DEFAULT_VALUE_COUNT_LIMIT = 15
FREQUENT_ITEMSET_CHART_LIMIT = 15
TOP_RULE_CHART_LIMIT = 10
NETWORK_GRAPH_HEIGHT = 620
NETWORK_MIN_RULES = 5
NETWORK_MAX_RULES = 100
NETWORK_DEFAULT_RULES = 30
NETWORK_RULE_STEP = 5
DAY_CATEGORY_ORDER = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
PAYMENT_CATEGORY_ORDER = ["NonTunai", "Tunai", "Campuran", "Lainnya", "Tidak Diketahui"]
KNOWN_CATEGORY_ORDERS = {
    "_kategori_waktu": ["Dini Hari", "Pagi", "Siang", "Sore", "Malam", "Tidak Diketahui"],
    "_status_grouped_order": ["Non Grouped/Single", "Grouped/Multi Order"],
    "_periode_ramadan_2026": [
        "Pra Ramadan 2026",
        "Ramadan 2026",
        "Pasca Ramadan 2026",
        "Tanggal Tidak Tersedia",
    ],
    "_kelompok_pembayaran": PAYMENT_CATEGORY_ORDER,
    "_kategori_titik_pengambilan": ["Single Pickup", "Multi Pickup", "Tidak Diketahui"],
    "_kategori_titik_pengantaran": ["Single Dropoff", "Multi Dropoff", "Tidak Diketahui"],
}


def _normalize_category_key(value: object) -> str:
    """Return a comparable key for category labels with flexible spacing/casing."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _normalized_column_name(column: str) -> str:
    """Return a comparable key for dynamic uploaded column names."""
    return re.sub(r"[^a-z0-9]+", "", str(column).casefold())


def _inferred_category_order(column: str) -> list[str] | None:
    """Return a natural category order inferred from a dynamic column name."""
    normalized_column = _normalized_column_name(column)
    if "hari" in normalized_column or normalized_column in {"day", "dayname", "weekday"}:
        return DAY_CATEGORY_ORDER
    if any(token in normalized_column for token in ["pembayaran", "payment", "bayar", "metode", "method"]):
        return PAYMENT_CATEGORY_ORDER
    return None


def _stable_value_sort_key(value: object) -> tuple[int, str]:
    """Return a stable sort key for category labels."""
    text = str(value)
    return (1 if text == "Tidak Diketahui" else 0, text.casefold())


def _sort_count_frame(
    count_df: pd.DataFrame,
    column: str,
    category_order: list[str] | None,
) -> pd.DataFrame:
    """Return count data ordered by category label instead of count value."""
    if count_df.empty:
        return count_df

    if category_order:
        order_lookup = {_normalize_category_key(value): index for index, value in enumerate(category_order)}
        sorted_df = count_df.assign(
            _order=count_df[column].map(
                lambda value: order_lookup.get(_normalize_category_key(value), len(order_lookup))
            ),
            _label=count_df[column].map(lambda value: str(value).casefold()),
        ).sort_values(["_order", "_label"])
        return sorted_df.drop(columns=["_order", "_label"]).reset_index(drop=True)

    sorted_df = count_df.assign(_sort_key=count_df[column].map(_stable_value_sort_key)).sort_values("_sort_key")
    return sorted_df.drop(columns="_sort_key").reset_index(drop=True)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    orientation: str = "v",
) -> go.Figure:
    """Return a Plotly bar chart for the provided DataFrame columns."""
    category_axis = x if orientation == "v" else y
    category_orders = {}
    if category_axis in df.columns:
        category_orders[category_axis] = df[category_axis].astype(str).drop_duplicates().tolist()

    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        orientation=orientation,
        template=PLOT_TEMPLATE,
        color=y if orientation == "v" else x,
        color_continuous_scale="Tealgrn",
        category_orders=category_orders,
        labels={
            x: format_column_label(x),
            y: format_column_label(y),
        },
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def histogram(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Return a Plotly histogram for a numeric DataFrame column."""
    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    chart_df = pd.DataFrame({column: numeric})
    fig = px.histogram(
        chart_df,
        x=column,
        nbins=30,
        title=title,
        template=PLOT_TEMPLATE,
        color_discrete_sequence=["#43d9ad"],
        labels={column: format_column_label(column)},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    return fig


def value_count_frame(
    df: pd.DataFrame,
    column: str,
    top_n: int = DEFAULT_VALUE_COUNT_LIMIT,
    category_order: list[str] | None = None,
) -> pd.DataFrame:
    """Return top value counts ordered by stable category labels for display."""
    count_df = (
        df[column]
        .fillna("Tidak Diketahui")
        .astype(str)
        .value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="jumlah")
    )
    natural_order = category_order or KNOWN_CATEGORY_ORDERS.get(column) or _inferred_category_order(column)
    return _sort_count_frame(count_df, column, natural_order)


def render_frequent_itemsets_chart(frequent_itemsets: pd.DataFrame) -> None:
    """Render the top frequent itemsets by support."""
    chart_data = frequent_itemsets.head(FREQUENT_ITEMSET_CHART_LIMIT).copy()
    if chart_data.empty:
        return
    chart_data["support"] = chart_data["support"].round(4)
    fig = px.bar(
        chart_data.sort_values("support"),
        x="support",
        y="itemsets",
        orientation="h",
        title="Top Frequent Itemset Berdasarkan Support",
        template=PLOT_TEMPLATE,
        color="support",
        color_continuous_scale="Teal",
        labels={
            "support": "Support",
            "itemsets": "Itemsets",
        },
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_confidence_lift_chart(rules: pd.DataFrame) -> None:
    """Render the confidence-versus-lift scatter plot for association rules."""
    fig = px.scatter(
        rules,
        x="confidence",
        y="lift",
        size="support",
        color="support",
        hover_data=["antecedents", "consequents"],
        title="Confidence vs Lift",
        template=PLOT_TEMPLATE,
        color_continuous_scale="Viridis",
        labels={
            "confidence": "Confidence",
            "lift": "Lift",
            "support": "Support",
        },
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_top_rules_chart(rules: pd.DataFrame) -> None:
    """Render the top association rules ranked by confidence."""
    chart_data = rules.head(TOP_RULE_CHART_LIMIT).copy()
    chart_data["rule"] = chart_data["antecedents"] + " -> " + chart_data["consequents"]
    fig = px.bar(
        chart_data.sort_values("confidence"),
        x="confidence",
        y="rule",
        orientation="h",
        title="Top Association Rules Berdasarkan Confidence",
        template=PLOT_TEMPLATE,
        color="lift",
        color_continuous_scale="Tealrose",
        labels={
            "confidence": "Confidence",
            "rule": "Rule",
            "lift": "Lift",
        },
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_network_graph(rules: pd.DataFrame, max_rules: int = 20) -> None:
    """Render a directed network graph for the first association rules."""
    graph = nx.DiGraph()
    selected_rules = rules.head(max_rules)

    for _, row in selected_rules.iterrows():
        antecedents = [item.strip() for item in str(row["antecedents"]).split(",")]
        consequents = [item.strip() for item in str(row["consequents"]).split(",")]
        for antecedent in antecedents:
            for consequent in consequents:
                graph.add_edge(
                    antecedent,
                    consequent,
                    weight=float(row["confidence"]),
                    lift=float(row["lift"]),
                )

    if graph.number_of_nodes() == 0:
        st.info("Network graph belum dapat dibuat karena rules kosong.")
        return

    layout = nx.spring_layout(graph, seed=42, k=1 / math.sqrt(max(graph.number_of_nodes(), 1)))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for source, target in graph.edges():
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.2, color="#6b7280"),
        hoverinfo="none",
        mode="lines",
    )

    node_x: list[float] = []
    node_y: list[float] = []
    labels: list[str] = []
    sizes: list[int] = []
    for node in graph.nodes():
        x, y = layout[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        sizes.append(16 + 3 * graph.degree(node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=sizes,
            color="#43d9ad",
            line=dict(width=1, color="#f8fafc"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Network Graph Hubungan Item",
        template=PLOT_TEMPLATE,
        showlegend=False,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=NETWORK_GRAPH_HEIGHT,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_apriori_visualizations(
    frequent_itemsets: pd.DataFrame,
    rules: pd.DataFrame,
) -> None:
    """Render all Apriori visualizations for the provided itemsets and rules."""
    if rules.empty:
        st.warning("Belum ada association rules untuk divisualisasikan.")
        return

    render_top_rules_chart(rules)
    render_confidence_lift_chart(rules)
    max_rules = st.slider(
        "Jumlah rule untuk network graph",
        min_value=NETWORK_MIN_RULES,
        max_value=min(NETWORK_MAX_RULES, len(rules)),
        value=min(NETWORK_DEFAULT_RULES, len(rules)),
        step=NETWORK_RULE_STEP,
        help="Naikkan nilai ini jika item tertentu belum terlihat di network graph.",
    )
    render_network_graph(rules, max_rules=max_rules)
    render_frequent_itemsets_chart(frequent_itemsets)
