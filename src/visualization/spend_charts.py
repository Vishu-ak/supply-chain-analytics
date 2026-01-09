"""
Spend analysis visualization module.

Provides a library of Plotly charts for procurement spend dashboards:
  - Spend treemap by category and vendor
  - Pareto (80/20) cumulative spend chart
  - Vendor comparison bar charts
  - Sunburst chart for hierarchical spend
  - Price variance heatmap
  - Maverick spend waterfall chart
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Brand color palette
COLORS = {
    "primary": "#1e3a5f",
    "secondary": "#2980b9",
    "accent": "#27ae60",
    "warning": "#e67e22",
    "danger": "#e74c3c",
    "neutral": "#95a5a6",
    "background": "#f8f9fa",
}

CATEGORY_COLORS = px.colors.qualitative.Set2


class SpendCharts:
    """
    Factory class for supply chain spend visualizations.

    All methods return Plotly Figure objects that can be
    rendered in Streamlit, Jupyter, or exported as HTML/PNG.

    Example
    -------
    >>> charts = SpendCharts()
    >>> fig = charts.spend_treemap(pareto_df, title="Q4 Spend by Category")
    >>> fig.show()
    """

    def __init__(self, template: str = "plotly_white") -> None:
        self.template = template

    # ------------------------------------------------------------------
    # Treemap
    # ------------------------------------------------------------------

    def spend_treemap(
        self,
        df: pd.DataFrame,
        path: list[str] | None = None,
        values_col: str = "total_spend",
        title: str = "Procurement Spend Treemap",
        color_col: str | None = None,
    ) -> go.Figure:
        """
        Render a hierarchical spend treemap.

        Parameters
        ----------
        df : DataFrame with spend and path columns
        path : hierarchy columns for treemap (e.g. ['category', 'vendor_id'])
        values_col : column with spend values
        title : chart title
        color_col : column to color by (e.g. 'pareto_class')

        Returns
        -------
        plotly.graph_objects.Figure
        """
        path = path or ["category"]

        # Ensure path columns exist
        path = [c for c in path if c in df.columns]

        fig = px.treemap(
            df,
            path=[px.Constant("All Spend")] + path,
            values=values_col,
            color=color_col or values_col,
            color_continuous_scale=["#d5e8f7", COLORS["primary"]],
            title=title,
            template=self.template,
        )
        fig.update_traces(
            textinfo="label+value+percent parent",
            texttemplate="%{label}<br>$%{value:,.0f}<br>%{percentParent:.1%}",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            font=dict(family="Inter, sans-serif", size=12),
        )
        return fig

    # ------------------------------------------------------------------
    # Pareto chart
    # ------------------------------------------------------------------

    def pareto_chart(
        self,
        pareto_df: pd.DataFrame,
        x_col: str,
        spend_col: str = "total_spend",
        cumulative_col: str = "cumulative_pct",
        top_n: int = 25,
        title: str = "Pareto Analysis — Cumulative Spend",
    ) -> go.Figure:
        """
        Dual-axis Pareto chart: bars for spend, line for cumulative %.

        Parameters
        ----------
        pareto_df : output from SpendAnalyzer.pareto_analysis()
        x_col : dimension column (vendor_id, category, etc.)
        top_n : number of top entities to show
        """
        df = pareto_df.head(top_n).copy()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Spend bars
        bar_colors = [
            COLORS["primary"] if c == "A"
            else COLORS["secondary"] if c == "B"
            else COLORS["neutral"]
            for c in df.get("pareto_class", ["A"] * len(df))
        ]

        fig.add_trace(
            go.Bar(
                x=df[x_col],
                y=df[spend_col],
                name="Spend (USD)",
                marker_color=bar_colors,
                text=df[spend_col].apply(lambda v: f"${v:,.0f}"),
                textposition="outside",
            ),
            secondary_y=False,
        )

        # Cumulative line
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[cumulative_col] * 100,
                name="Cumulative %",
                mode="lines+markers",
                line=dict(color=COLORS["danger"], width=2),
                marker=dict(size=5),
            ),
            secondary_y=True,
        )

        # 80% reference line
        fig.add_hline(y=80, secondary_y=True, line_dash="dash",
                      line_color=COLORS["warning"],
                      annotation_text="80% threshold",
                      annotation_position="top right")

        fig.update_layout(
            title=title,
            template=self.template,
            showlegend=True,
            font=dict(family="Inter, sans-serif", size=11),
            xaxis=dict(tickangle=-45),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text="Total Spend (USD)", secondary_y=False,
                         tickprefix="$", tickformat=",.0f")
        fig.update_yaxes(title_text="Cumulative Spend %", secondary_y=True,
                         ticksuffix="%", range=[0, 110])
        return fig

    # ------------------------------------------------------------------
    # Vendor comparison
    # ------------------------------------------------------------------

    def vendor_comparison_bar(
        self,
        scorecard_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Vendor Performance Scorecard",
    ) -> go.Figure:
        """
        Horizontal grouped bar chart comparing vendor scores across dimensions.
        """
        df = scorecard_df.nlargest(top_n, "composite_score").copy()
        if "vendor_name" in df.columns:
            df["label"] = df["vendor_name"].str[:30]
        else:
            df["label"] = df.index.astype(str)

        score_cols = [c for c in ["delivery_score", "quality_score", "cost_score",
                                   "responsiveness_score"] if c in df.columns]
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["warning"]]

        fig = go.Figure()
        for col, color in zip(score_cols, colors):
            fig.add_trace(go.Bar(
                y=df["label"],
                x=df[col],
                name=col.replace("_score", "").title(),
                orientation="h",
                marker_color=color,
                opacity=0.85,
            ))

        fig.update_layout(
            title=title,
            barmode="group",
            template=self.template,
            xaxis=dict(title="Score (0–100)", range=[0, 105]),
            yaxis=dict(autorange="reversed"),
            font=dict(family="Inter, sans-serif", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=max(400, top_n * 28),
        )
        return fig

    # ------------------------------------------------------------------
    # Sunburst
    # ------------------------------------------------------------------

    def spend_sunburst(
        self,
        df: pd.DataFrame,
        path: list[str] | None = None,
        values_col: str = "total_amount",
        title: str = "Spend Breakdown",
    ) -> go.Figure:
        """Sunburst chart for multi-level spend hierarchy."""
        path = [c for c in (path or ["category", "vendor_id"]) if c in df.columns]
        if not path:
            return go.Figure()

        agg = df.groupby(path, observed=True)[values_col].sum().reset_index()

        fig = px.sunburst(
            agg,
            path=path,
            values=values_col,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template=self.template,
        )
        fig.update_traces(
            textinfo="label+percent parent",
            insidetextorientation="radial",
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=11),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    # ------------------------------------------------------------------
    # Category spend trend
    # ------------------------------------------------------------------

    def category_spend_trend(
        self,
        df: pd.DataFrame,
        top_n_categories: int = 6,
        title: str = "Category Spend Trend",
    ) -> go.Figure:
        """
        Line chart showing spend trend for top N categories.

        Parameters
        ----------
        df : DataFrame with order_date, category, total_amount columns
        """
        if not all(c in df.columns for c in ["order_date", "category", "total_amount"]):
            return go.Figure()

        df = df.copy()
        df["month"] = pd.to_datetime(df["order_date"]).dt.to_period("M").dt.to_timestamp()

        top_cats = (
            df.groupby("category", observed=True)["total_amount"]
            .sum()
            .nlargest(top_n_categories)
            .index
        )

        monthly = (
            df[df["category"].isin(top_cats)]
            .groupby(["month", "category"], observed=True)["total_amount"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            monthly,
            x="month",
            y="total_amount",
            color="category",
            title=title,
            template=self.template,
            color_discrete_sequence=CATEGORY_COLORS,
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Spend (USD)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            font=dict(family="Inter, sans-serif", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    # ------------------------------------------------------------------
    # KPI cards helper
    # ------------------------------------------------------------------

    def kpi_indicator(
        self,
        value: float,
        reference: float,
        title: str,
        value_format: str = ",.0f",
        prefix: str = "$",
        suffix: str = "",
    ) -> go.Figure:
        """Render a single KPI indicator (gauge / delta)."""
        delta = value - reference
        delta_color = "green" if delta >= 0 else "red"

        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=value,
            number={"prefix": prefix, "suffix": suffix, "valueformat": value_format},
            delta={"reference": reference, "valueformat": ".1%",
                   "increasing": {"color": delta_color}},
            title={"text": title, "font": {"size": 14}},
        ))
        fig.update_layout(
            height=140,
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(family="Inter, sans-serif"),
            template=self.template,
        )
        return fig
