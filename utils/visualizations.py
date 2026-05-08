from __future__ import annotations

import math

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PLOT_TEMPLATE = "plotly_white"


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    orientation: str = "v",
) -> go.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        orientation=orientation,
        template=PLOT_TEMPLATE,
        color=y if orientation == "v" else x,
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def histogram(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    chart_df = pd.DataFrame({column: numeric})
    fig = px.histogram(
        chart_df,
        x=column,
        nbins=30,
        title=title,
        template=PLOT_TEMPLATE,
        color_discrete_sequence=["#43d9ad"],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    return fig


def value_count_frame(df: pd.DataFrame, column: str, top_n: int = 15) -> pd.DataFrame:
    return (
        df[column]
        .fillna("Tidak Diketahui")
        .astype(str)
        .value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="jumlah")
    )


def render_frequent_itemsets_chart(frequent_itemsets: pd.DataFrame) -> None:
    chart_data = frequent_itemsets.head(15).copy()
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
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_confidence_lift_chart(rules: pd.DataFrame) -> None:
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
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_top_rules_chart(rules: pd.DataFrame) -> None:
    chart_data = rules.head(10).copy()
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
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_network_graph(rules: pd.DataFrame, max_rules: int = 20) -> None:
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
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_apriori_visualizations(
    frequent_itemsets: pd.DataFrame,
    rules: pd.DataFrame,
) -> None:
    if rules.empty:
        st.warning("Belum ada association rules untuk divisualisasikan.")
        return

    render_top_rules_chart(rules)
    render_confidence_lift_chart(rules)
    max_rules = st.slider(
        "Jumlah rule untuk network graph",
        min_value=5,
        max_value=min(100, len(rules)),
        value=min(30, len(rules)),
        step=5,
        help="Naikkan nilai ini jika item tertentu belum terlihat di network graph.",
    )
    render_network_graph(rules, max_rules=max_rules)
    render_frequent_itemsets_chart(frequent_itemsets)
