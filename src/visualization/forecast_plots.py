"""
Forecast visualization module.

Renders time series charts with confidence intervals, trend decomposition,
and model comparison plots using Plotly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from src.analysis.demand_forecast import ForecastResult

COLORS = {
    "historical": "#1e3a5f",
    "forecast": "#2980b9",
    "ci_fill": "rgba(41, 128, 185, 0.2)",
    "actual": "#27ae60",
    "prophet": "#e74c3c",
    "hw": "#e67e22",
}


class ForecastPlots:
    """
    Visualization library for demand forecast results.

    Example
    -------
    >>> fp = ForecastPlots()
    >>> fig = fp.forecast_with_ci(result, historical_series)
    >>> fig.show()
    """

    def __init__(self, template: str = "plotly_white") -> None:
        self.template = template

    # ------------------------------------------------------------------
    # Main forecast chart with CI
    # ------------------------------------------------------------------

    def forecast_with_ci(
        self,
        result: "ForecastResult",
        historical: pd.Series | None = None,
        title: str | None = None,
        show_ci: bool = True,
    ) -> go.Figure:
        """
        Plot a forecast with shaded confidence interval and historical actuals.

        Parameters
        ----------
        result : ForecastResult from DemandForecaster
        historical : optional pd.Series (date index, quantity values) of actuals
        title : chart title; auto-generated if None
        show_ci : whether to shade the confidence interval

        Returns
        -------
        plotly.graph_objects.Figure
        """
        fc = result.forecast_df.copy()
        fc["ds"] = pd.to_datetime(fc["ds"])
        fc_future = fc[fc["ds"] > fc["ds"].quantile(0.7)] if len(fc) > 1 else fc

        title = title or f"Demand Forecast — {result.entity_id} ({result.method.replace('_', ' ').title()})"

        fig = go.Figure()

        # Historical actuals
        if historical is not None and len(historical) > 0:
            hist_df = historical.reset_index()
            hist_df.columns = ["ds", "y"]
            hist_df["ds"] = pd.to_datetime(hist_df["ds"])
            fig.add_trace(go.Scatter(
                x=hist_df["ds"],
                y=hist_df["y"],
                mode="lines",
                name="Historical Demand",
                line=dict(color=COLORS["historical"], width=1.5),
                opacity=0.8,
            ))

        # Forecast line (fitted portion)
        if historical is not None:
            fc_fitted = fc[fc["ds"].isin(pd.to_datetime(historical.index))]
            if not fc_fitted.empty:
                fig.add_trace(go.Scatter(
                    x=fc_fitted["ds"],
                    y=fc_fitted["yhat"],
                    mode="lines",
                    name="Fitted",
                    line=dict(color=COLORS["forecast"], width=1.5, dash="dot"),
                    opacity=0.7,
                ))

        # Future forecast
        future_dates = fc["ds"].max() if not fc.empty else pd.Timestamp.now()
        if historical is not None:
            cutoff = pd.to_datetime(historical.index.max())
            fc_future = fc[fc["ds"] > cutoff]
        else:
            fc_future = fc

        # CI shading
        if show_ci and "yhat_lower" in fc_future.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_future["ds"], fc_future["ds"].iloc[::-1]]),
                y=pd.concat([fc_future["yhat_upper"], fc_future["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor=COLORS["ci_fill"],
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
                showlegend=True,
            ))

        # Forecast line
        if not fc_future.empty:
            fig.add_trace(go.Scatter(
                x=fc_future["ds"],
                y=fc_future["yhat"],
                mode="lines",
                name="Forecast",
                line=dict(color=COLORS["forecast"], width=2.5),
            ))

        # Vertical cutoff line
        if historical is not None and len(historical) > 0:
            cutoff = pd.to_datetime(historical.index.max())
            fig.add_vline(
                x=cutoff.timestamp() * 1000,
                line_dash="dash",
                line_color="gray",
                annotation_text="Forecast start",
                annotation_position="top right",
            )

        # Metrics annotation
        if result.mape > 0:
            fig.add_annotation(
                text=f"MAE: {result.mae:.1f}  |  MAPE: {result.mape:.1f}%",
                xref="paper", yref="paper",
                x=0.01, y=0.98,
                showarrow=False,
                font=dict(size=11, color="gray"),
                align="left",
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Quantity",
            template=self.template,
            font=dict(family="Inter, sans-serif", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------
    # Model comparison chart
    # ------------------------------------------------------------------

    def model_comparison(
        self,
        prophet_result: "ForecastResult",
        hw_result: "ForecastResult",
        historical: pd.Series | None = None,
        title: str = "Prophet vs. Holt-Winters Forecast Comparison",
    ) -> go.Figure:
        """
        Overlay Prophet and Holt-Winters forecasts for visual comparison.
        """
        fig = go.Figure()

        # Historical actuals
        if historical is not None and len(historical) > 0:
            hist_df = historical.reset_index()
            hist_df.columns = ["ds", "y"]
            hist_df["ds"] = pd.to_datetime(hist_df["ds"])
            fig.add_trace(go.Scatter(
                x=hist_df["ds"], y=hist_df["y"],
                mode="lines", name="Actuals",
                line=dict(color=COLORS["historical"], width=1.5),
            ))

        cutoff = pd.to_datetime(historical.index.max()) if historical is not None and len(historical) > 0 else None

        for result, color, name in [
            (prophet_result, COLORS["prophet"], "Prophet"),
            (hw_result, COLORS["hw"], "Holt-Winters"),
        ]:
            fc = result.forecast_df.copy()
            if fc.empty:
                continue
            fc["ds"] = pd.to_datetime(fc["ds"])
            if cutoff is not None:
                fc = fc[fc["ds"] > cutoff]
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat"],
                mode="lines", name=f"{name} (MAPE={result.mape:.1f}%)",
                line=dict(color=color, width=2, dash="solid"),
            ))

            if "yhat_lower" in fc.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
                    y=pd.concat([fc["yhat_upper"], fc["yhat_lower"].iloc[::-1]]),
                    fill="toself",
                    fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba"),
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{name} CI",
                    showlegend=False,
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Quantity",
            template=self.template,
            font=dict(family="Inter, sans-serif", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------
    # Multi-category forecast grid
    # ------------------------------------------------------------------

    def category_forecast_grid(
        self,
        forecast_results: dict[str, "ForecastResult"],
        cols: int = 2,
    ) -> go.Figure:
        """
        Grid of forecast charts, one per category.

        Parameters
        ----------
        forecast_results : dict from DemandForecaster.forecast_all_categories()
        cols : number of columns in grid layout
        """
        categories = list(forecast_results.keys())
        n = len(categories)
        if n == 0:
            return go.Figure()

        rows = (n + cols - 1) // cols
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=categories,
            shared_xaxes=False,
        )

        for i, cat in enumerate(categories):
            row = i // cols + 1
            col = i % cols + 1
            result = forecast_results[cat]
            fc = result.forecast_df.copy()
            if fc.empty:
                continue
            fc["ds"] = pd.to_datetime(fc["ds"])

            fig.add_trace(
                go.Scatter(
                    x=fc["ds"], y=fc["yhat"],
                    mode="lines",
                    line=dict(color=COLORS["forecast"], width=1.5),
                    showlegend=False,
                ),
                row=row, col=col,
            )
            if "yhat_lower" in fc.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
                        y=pd.concat([fc["yhat_upper"], fc["yhat_lower"].iloc[::-1]]),
                        fill="toself",
                        fillcolor=COLORS["ci_fill"],
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                    ),
                    row=row, col=col,
                )

        fig.update_layout(
            title="Category Demand Forecasts",
            template=self.template,
            font=dict(family="Inter, sans-serif", size=10),
            height=300 * rows,
        )
        return fig

    # ------------------------------------------------------------------
    # Seasonality decomposition
    # ------------------------------------------------------------------

    def seasonality_chart(
        self,
        weekly_series: pd.Series,
        title: str = "Weekly Demand Seasonality",
    ) -> go.Figure:
        """
        Decompose and visualize demand seasonality using a heatmap.

        Parameters
        ----------
        weekly_series : weekly demand pd.Series with DatetimeIndex
        """
        df = weekly_series.reset_index()
        df.columns = ["week", "demand"]
        df["week"] = pd.to_datetime(df["week"])
        df["year"] = df["week"].dt.year
        df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)

        pivot = df.pivot_table(
            index="year", columns="week_of_year", values="demand", aggfunc="mean"
        )

        fig = px.imshow(
            pivot,
            title=title,
            labels=dict(x="Week of Year", y="Year", color="Avg Demand"),
            color_continuous_scale="Blues",
            template=self.template,
            aspect="auto",
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=11),
        )
        return fig
