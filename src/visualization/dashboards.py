"""
Dashboard component library for Supply Chain Analytics.

Provides reusable Plotly chart components used by the Streamlit app:
  - KPI header cards
  - Vendor tier performance radar chart
  - Inventory ABC donut chart
  - Procurement cycle time box plots
  - Delivery performance scatter/heatmap
  - Cost breakdown waterfall chart
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

THEME = {
    "primary": "#1e3a5f",
    "secondary": "#2980b9",
    "accent": "#27ae60",
    "warning": "#e67e22",
    "danger": "#e74c3c",
    "neutral": "#95a5a6",
    "light_bg": "#f8f9fa",
    "font": "Inter, system-ui, sans-serif",
}


class DashboardComponents:
    """
    Collection of Plotly figure factories for the Streamlit procurement dashboard.

    Example
    -------
    >>> dc = DashboardComponents()
    >>> fig = dc.vendor_radar(scorecard_df, vendor_id="V00001")
    >>> fig.show()
    """

    def __init__(self, template: str = "plotly_white") -> None:
        self.template = template

    # ------------------------------------------------------------------
    # KPI Summary bar
    # ------------------------------------------------------------------

    def kpi_summary_chart(
        self,
        kpis: dict[str, Any],
        title: str = "Procurement KPI Overview",
    ) -> go.Figure:
        """
        Horizontal bar chart of KPI achievement vs. target.

        Parameters
        ----------
        kpis : dict with keys: name, value, target, unit
        """
        names = list(kpis.keys())
        values = [kpis[k].get("value", 0) for k in names]
        targets = [kpis[k].get("target", 100) for k in names]
        units = [kpis[k].get("unit", "") for k in names]

        pct_of_target = [
            min(v / max(t, 1) * 100, 150)
            for v, t in zip(values, targets)
        ]

        bar_colors = [
            THEME["accent"] if p >= 95 else THEME["warning"] if p >= 80 else THEME["danger"]
            for p in pct_of_target
        ]

        fig = go.Figure(go.Bar(
            y=names,
            x=pct_of_target,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}{u}" for v, u in zip(values, units)],
            textposition="auto",
            hovertemplate="%{y}<br>Value: %{text}<br>% of Target: %{x:.1f}%<extra></extra>",
        ))

        fig.add_vline(x=100, line_dash="dash", line_color=THEME["neutral"],
                      annotation_text="Target", annotation_position="top")

        fig.update_layout(
            title=title,
            xaxis_title="% of Target",
            xaxis_range=[0, 155],
            yaxis=dict(autorange="reversed"),
            template=self.template,
            font=dict(family=THEME["font"], size=11),
            height=max(250, len(names) * 45),
        )
        return fig

    # ------------------------------------------------------------------
    # Vendor radar chart
    # ------------------------------------------------------------------

    def vendor_radar(
        self,
        scorecard: pd.DataFrame,
        vendor_ids: list[str] | None = None,
        max_vendors: int = 5,
        title: str = "Vendor Performance Radar",
    ) -> go.Figure:
        """
        Spider/radar chart comparing vendor performance dimensions.

        Parameters
        ----------
        scorecard : output from VendorScorer.compute_scorecards()
        vendor_ids : specific vendors to show; defaults to top 5 by composite
        max_vendors : maximum vendors to overlay
        """
        score_cols = [c for c in ["delivery_score", "quality_score",
                                   "cost_score", "responsiveness_score"]
                      if c in scorecard.columns]
        if not score_cols:
            return go.Figure()

        if vendor_ids:
            df = scorecard[scorecard.index.isin(vendor_ids)].head(max_vendors)
        else:
            df = scorecard.nlargest(max_vendors, "composite_score")

        categories = [c.replace("_score", "").title() for c in score_cols]
        categories += [categories[0]]  # Close the polygon

        colors = px.colors.qualitative.Set1[:max_vendors]
        fig = go.Figure()

        for i, (vid, row) in enumerate(df.iterrows()):
            values = [row.get(c, 50) for c in score_cols]
            values += [values[0]]
            label = row.get("vendor_name", str(vid))[:20] if "vendor_name" in row else str(vid)

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=label,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)].replace(")", ", 0.1)").replace("rgb", "rgba"),
                opacity=0.85,
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
            ),
            showlegend=True,
            title=title,
            template=self.template,
            font=dict(family=THEME["font"], size=11),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            height=450,
        )
        return fig

    # ------------------------------------------------------------------
    # Inventory ABC donut
    # ------------------------------------------------------------------

    def inventory_abc_donut(
        self,
        inventory_params: pd.DataFrame,
        value_col: str = "total_annual_cost",
        title: str = "Inventory ABC Classification",
    ) -> go.Figure:
        """
        Donut chart showing ABC class distribution by inventory cost.
        """
        if "abc_class" not in inventory_params.columns:
            return go.Figure()

        agg = (
            inventory_params.groupby("abc_class", observed=True)[value_col]
            .sum()
            .reindex(["A", "B", "C"])
            .fillna(0)
            .reset_index()
        )

        fig = go.Figure(go.Pie(
            labels=agg["abc_class"],
            values=agg[value_col],
            hole=0.5,
            marker_colors=[THEME["primary"], THEME["secondary"], THEME["neutral"]],
            textinfo="label+percent+value",
            texttemplate="%{label}<br>%{percent:.0%}<br>$%{value:,.0f}",
            hovertemplate="Class %{label}<br>Cost: $%{value:,.0f}<br>%{percent:.1%}<extra></extra>",
        ))

        fig.update_layout(
            title=title,
            template=self.template,
            font=dict(family=THEME["font"], size=11),
            showlegend=True,
            height=350,
            annotations=[dict(text="ABC", x=0.5, y=0.5, font_size=20, showarrow=False)],
        )
        return fig

    # ------------------------------------------------------------------
    # Cycle time box plot
    # ------------------------------------------------------------------

    def cycle_time_boxplot(
        self,
        cycle_time_df: pd.DataFrame,
        group_col: str = "category",
        value_col: str = "avg_cycle_days",
        title: str = "Procurement Cycle Time Distribution",
    ) -> go.Figure:
        """Box plot of cycle times grouped by category or vendor tier."""
        if cycle_time_df.empty or group_col not in cycle_time_df.columns:
            return go.Figure()

        groups = sorted(cycle_time_df[group_col].unique())
        colors = px.colors.qualitative.Set2

        fig = go.Figure()
        for i, grp in enumerate(groups):
            data = cycle_time_df[cycle_time_df[group_col] == grp][value_col].dropna()
            if data.empty:
                continue
            fig.add_trace(go.Box(
                y=data,
                name=str(grp),
                marker_color=colors[i % len(colors)],
                boxmean="sd",
            ))

        fig.update_layout(
            title=title,
            yaxis_title="Days",
            template=self.template,
            font=dict(family=THEME["font"], size=11),
            showlegend=False,
        )
        return fig

    # ------------------------------------------------------------------
    # Delivery performance heatmap
    # ------------------------------------------------------------------

    def delivery_heatmap(
        self,
        deliveries: pd.DataFrame,
        title: str = "On-Time Delivery Rate by Vendor & Month",
    ) -> go.Figure:
        """
        Heatmap of on-time delivery rate with vendors on y-axis and months on x-axis.
        Shows only top 20 vendors by volume.
        """
        if deliveries.empty:
            return go.Figure()

        df = deliveries.copy()
        df["month"] = pd.to_datetime(df["scheduled_date"], errors="coerce").dt.to_period("M")

        top_vendors = (
            df.groupby("vendor_id", observed=True)["delivery_id"]
            .count()
            .nlargest(20)
            .index
        )
        df = df[df["vendor_id"].isin(top_vendors)]

        pivot = (
            df.groupby(["vendor_id", "month"], observed=True)["on_time"]
            .mean()
            .unstack()
        )
        pivot.columns = [str(c) for c in pivot.columns]

        fig = px.imshow(
            pivot,
            title=title,
            labels=dict(x="Month", y="Vendor ID", color="On-Time Rate"),
            color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
            zmin=0, zmax=1,
            text_auto=".0%",
            template=self.template,
            aspect="auto",
        )
        fig.update_layout(
            font=dict(family=THEME["font"], size=10),
            xaxis=dict(tickangle=-45),
            height=max(400, len(top_vendors) * 25),
        )
        return fig

    # ------------------------------------------------------------------
    # Cost waterfall chart
    # ------------------------------------------------------------------

    def cost_waterfall(
        self,
        tco_df: pd.DataFrame,
        vendor_id: str,
        title: str | None = None,
    ) -> go.Figure:
        """
        Waterfall chart decomposing TCO into cost components for a single vendor.
        """
        row = tco_df[tco_df["vendor_id"] == vendor_id] if "vendor_id" in tco_df.columns else tco_df.head(1)
        if row.empty:
            return go.Figure()

        row = row.iloc[0]
        cost_cols = {
            "Material": "material_cost",
            "Logistics": "total_logistics_cost",
            "Quality": "quality_cost",
            "Admin": "admin_cost",
        }

        measures, x, y, text = ["absolute"], ["Base"], [0], [""]
        for label, col in cost_cols.items():
            if col in row:
                measures.append("relative")
                x.append(label)
                v = float(row[col])
                y.append(v)
                text.append(f"${v:,.0f}")

        measures.append("total")
        x.append("Total TCO")
        y.append(0)
        text.append(f"${float(row.get('total_tco', sum(y[1:]))):,.0f}")

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measures,
            x=x,
            y=y,
            text=text,
            textposition="outside",
            connector=dict(line=dict(color="rgb(63, 63, 63)")),
            increasing=dict(marker_color=THEME["danger"]),
            decreasing=dict(marker_color=THEME["accent"]),
            totals=dict(marker_color=THEME["primary"]),
        ))

        title = title or f"TCO Breakdown — {vendor_id}"
        fig.update_layout(
            title=title,
            yaxis_title="Cost (USD)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
            template=self.template,
            font=dict(family=THEME["font"], size=11),
        )
        return fig
