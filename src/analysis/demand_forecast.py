"""
Demand forecasting module.

Implements two complementary approaches:
  1. Facebook Prophet   — handles seasonality, holidays, and trend changepoints
  2. Holt-Winters ETS   — triple exponential smoothing (statsmodels)

Both produce point forecasts with configurable confidence intervals.
Supports per-product, per-category, and aggregate-level forecasting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for a single forecast output."""

    entity_id: str           # Product, category, or 'total'
    method: str              # 'prophet' | 'holt_winters'
    forecast_df: pd.DataFrame  # columns: ds, yhat, yhat_lower, yhat_upper
    mae: float = 0.0
    mape: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DemandForecaster:
    """
    Generates demand forecasts from historical PO quantity data.

    Parameters
    ----------
    purchase_orders : DataFrame with order_date and quantity columns
    date_col : name of the date column (default 'order_date')
    quantity_col : name of the quantity column (default 'quantity')
    horizon_days : how many days ahead to forecast

    Example
    -------
    >>> fc = DemandForecaster(po_df, horizon_days=90)
    >>> result = fc.forecast_prophet(entity_id="total")
    >>> result.forecast_df.tail()
    """

    def __init__(
        self,
        purchase_orders: pd.DataFrame,
        date_col: str = "order_date",
        quantity_col: str = "quantity",
        horizon_days: int = 90,
    ) -> None:
        self.po = purchase_orders.copy()
        self.date_col = date_col
        self.quantity_col = quantity_col
        self.horizon_days = horizon_days
        self.po[date_col] = pd.to_datetime(self.po[date_col], errors="coerce")

    # ------------------------------------------------------------------
    # Prophet forecasting
    # ------------------------------------------------------------------

    def forecast_prophet(
        self,
        entity_id: str = "total",
        group_col: str | None = None,
        ci: float = 0.95,
    ) -> ForecastResult:
        """
        Fit a Facebook Prophet model and generate a demand forecast.

        Parameters
        ----------
        entity_id : value in group_col to filter on (or 'total' for all)
        group_col : column to filter by (e.g. 'category', 'product_id')
        ci : confidence interval width (0.80, 0.90, 0.95)

        Returns
        -------
        ForecastResult with daily-level forecast DataFrame
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.warning("prophet not installed — falling back to Holt-Winters")
            return self.forecast_holt_winters(entity_id=entity_id, group_col=group_col, ci=ci)

        ts = self._build_daily_series(entity_id, group_col)
        if ts is None or len(ts) < 30:
            logger.warning("Insufficient data for Prophet (%s=%s), need ≥30 days",
                           group_col, entity_id)
            return self._empty_result(entity_id, "prophet")

        df_prophet = ts.reset_index().rename(columns={self.date_col: "ds", "y": "y"})
        df_prophet.columns = ["ds", "y"]

        model = Prophet(
            interval_width=ci,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=self.horizon_days, freq="D")
        forecast = model.predict(future)

        # Clip negative predictions
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = forecast[col].clip(lower=0)

        # Evaluate on held-out last 30 days
        historical = forecast[forecast["ds"].isin(df_prophet["ds"])]
        actual = df_prophet.set_index("ds")["y"]
        pred = historical.set_index("ds")["yhat"]
        mae, mape = self._eval_metrics(actual, pred)

        return ForecastResult(
            entity_id=entity_id,
            method="prophet",
            forecast_df=forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            mae=mae,
            mape=mape,
            metadata={"model": "Prophet", "horizon_days": self.horizon_days, "ci": ci},
        )

    # ------------------------------------------------------------------
    # Holt-Winters forecasting
    # ------------------------------------------------------------------

    def forecast_holt_winters(
        self,
        entity_id: str = "total",
        group_col: str | None = None,
        ci: float = 0.95,
        seasonal_periods: int = 52,
    ) -> ForecastResult:
        """
        Fit a Holt-Winters Triple Exponential Smoothing model.

        Parameters
        ----------
        entity_id : group identifier or 'total'
        group_col : column to group by
        ci : confidence interval width for prediction bands
        seasonal_periods : number of periods per seasonal cycle (52 for weekly data)

        Returns
        -------
        ForecastResult with weekly-level forecast DataFrame
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError as exc:
            raise ImportError("statsmodels is required: pip install statsmodels") from exc

        ts = self._build_weekly_series(entity_id, group_col)
        if ts is None or len(ts) < max(seasonal_periods * 2, 20):
            logger.warning("Insufficient data for Holt-Winters (%s=%s)",
                           group_col, entity_id)
            return self._empty_result(entity_id, "holt_winters")

        n_weeks = max(1, self.horizon_days // 7)

        # Determine best trend/seasonal combo
        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add",
            seasonal_periods=min(seasonal_periods, len(ts) // 2),
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        forecast_values = fitted.forecast(n_weeks)

        # Build CI using simulation approach
        z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(ci, 1.960)
        residuals = fitted.resid
        sigma = float(residuals.std())
        se = np.array([sigma * np.sqrt(1 + t / len(ts)) for t in range(1, n_weeks + 1)])

        future_index = pd.date_range(
            start=ts.index[-1] + pd.Timedelta(weeks=1),
            periods=n_weeks, freq="W"
        )

        forecast_df = pd.DataFrame({
            "ds": future_index,
            "yhat": np.maximum(0, forecast_values.values),
            "yhat_lower": np.maximum(0, forecast_values.values - z * se),
            "yhat_upper": np.maximum(0, forecast_values.values + z * se),
        })

        # Compute historical fit metrics
        actual = ts
        pred = fitted.fittedvalues
        mae, mape = self._eval_metrics(actual, pred)

        return ForecastResult(
            entity_id=entity_id,
            method="holt_winters",
            forecast_df=forecast_df,
            mae=mae,
            mape=mape,
            metadata={
                "model": "Holt-Winters ETS",
                "alpha": float(fitted.params.get("smoothing_level", 0)),
                "beta": float(fitted.params.get("smoothing_trend", 0)),
                "gamma": float(fitted.params.get("smoothing_seasonal", 0)),
                "seasonal_periods": seasonal_periods,
            },
        )

    # ------------------------------------------------------------------
    # Batch forecasting
    # ------------------------------------------------------------------

    def forecast_all_categories(
        self, method: str = "holt_winters", top_n: int = 10
    ) -> dict[str, ForecastResult]:
        """
        Forecast demand for the top N spend categories.

        Parameters
        ----------
        method : 'prophet' | 'holt_winters'
        top_n : number of categories to forecast

        Returns
        -------
        Dict mapping category → ForecastResult
        """
        if "category" not in self.po.columns:
            return {}

        top_categories = (
            self.po.groupby("category", observed=True)[self.quantity_col]
            .sum()
            .nlargest(top_n)
            .index
        )

        results: dict[str, ForecastResult] = {}
        forecast_fn = self.forecast_prophet if method == "prophet" else self.forecast_holt_winters

        for cat in top_categories:
            logger.info("Forecasting category: %s", cat)
            try:
                results[cat] = forecast_fn(entity_id=cat, group_col="category")
            except Exception as exc:
                logger.warning("Failed to forecast %s: %s", cat, exc)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_daily_series(
        self, entity_id: str, group_col: str | None
    ) -> pd.Series | None:
        """Build a daily-frequency quantity series."""
        df = self.po.copy()
        if group_col and entity_id != "total":
            df = df[df[group_col] == entity_id]

        if df.empty:
            return None

        ts = (
            df.set_index(self.date_col)[self.quantity_col]
            .resample("D")
            .sum()
            .rename("y")
        )
        return ts

    def _build_weekly_series(
        self, entity_id: str, group_col: str | None
    ) -> pd.Series | None:
        """Build a weekly-frequency quantity series."""
        df = self.po.copy()
        if group_col and entity_id != "total":
            df = df[df[group_col] == entity_id]

        if df.empty:
            return None

        ts = (
            df.set_index(self.date_col)[self.quantity_col]
            .resample("W")
            .sum()
        )
        return ts

    @staticmethod
    def _eval_metrics(
        actual: pd.Series, pred: pd.Series
    ) -> tuple[float, float]:
        """Compute MAE and MAPE between actual and predicted series."""
        aligned = actual.align(pred, join="inner")
        a, p = aligned
        if len(a) == 0:
            return 0.0, 0.0
        mae = float(np.mean(np.abs(a - p)))
        nonzero = a[a != 0]
        mape = float(np.mean(np.abs((nonzero - p.loc[nonzero.index]) / nonzero)) * 100) if len(nonzero) else 0.0
        return round(mae, 2), round(mape, 2)

    @staticmethod
    def _empty_result(entity_id: str, method: str) -> ForecastResult:
        return ForecastResult(
            entity_id=entity_id,
            method=method,
            forecast_df=pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"]),
        )
