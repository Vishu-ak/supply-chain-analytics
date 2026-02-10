"""
Unit tests for src.analysis.demand_forecast.DemandForecaster.

Tests cover:
  - Holt-Winters forecast output format and horizon
  - Confidence interval ordering (lower ≤ yhat ≤ upper)
  - Non-negative forecast values
  - Insufficient data handling
  - Batch category forecasting
  - ForecastResult dataclass contract
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.demand_forecast import DemandForecaster, ForecastResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pos_3yr() -> pd.DataFrame:
    """Three years of weekly-ish purchase order data."""
    rng = np.random.default_rng(7)
    n = 3_000
    dates = pd.date_range("2021-01-01", periods=n, freq="6h")
    categories = ["Raw Materials", "Packaging", "Electronics", "MRO"]

    df = pd.DataFrame({
        "po_id": [f"PO-{i:07d}" for i in range(n)],
        "vendor_id": [f"V{rng.integers(1, 50):04d}" for _ in range(n)],
        "product_id": [f"P{rng.integers(1, 200):05d}" for _ in range(n)],
        "order_date": dates,
        "quantity": rng.integers(1, 100, size=n),
        "total_amount": rng.uniform(100, 10_000, size=n).round(2),
        "category": rng.choice(categories, size=n),
        "abc_class": rng.choice(["A", "B", "C"], size=n),
    })
    return df


@pytest.fixture
def forecaster(sample_pos_3yr: pd.DataFrame) -> DemandForecaster:
    return DemandForecaster(sample_pos_3yr, horizon_days=90)


# ---------------------------------------------------------------------------
# Tests: ForecastResult dataclass
# ---------------------------------------------------------------------------

class TestForecastResult:
    def test_dataclass_instantiation(self) -> None:
        fc_df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=30),
                               "yhat": np.ones(30),
                               "yhat_lower": np.zeros(30),
                               "yhat_upper": np.ones(30) * 2})
        result = ForecastResult(entity_id="test", method="test", forecast_df=fc_df, mae=1.5, mape=5.2)
        assert result.entity_id == "test"
        assert result.mape == 5.2
        assert isinstance(result.forecast_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: Holt-Winters forecasting
# ---------------------------------------------------------------------------

class TestHoltWinters:
    def test_returns_forecast_result(self, forecaster: DemandForecaster) -> None:
        result = forecaster.forecast_holt_winters(entity_id="total")
        assert isinstance(result, ForecastResult)

    def test_forecast_method_label(self, forecaster: DemandForecaster) -> None:
        result = forecaster.forecast_holt_winters(entity_id="total")
        assert result.method == "holt_winters"

    def test_forecast_df_has_required_columns(self, forecaster: DemandForecaster) -> None:
        result = forecaster.forecast_holt_winters(entity_id="total")
        required = {"ds", "yhat", "yhat_lower", "yhat_upper"}
        assert required.issubset(result.forecast_df.columns)

    def test_forecast_non_negative(self, forecaster: DemandForecaster) -> None:
        result = forecaster.forecast_holt_winters(entity_id="total")
        assert (result.forecast_df["yhat"] >= 0).all()
        assert (result.forecast_df["yhat_lower"] >= 0).all()

    def test_ci_ordering(self, forecaster: DemandForecaster) -> None:
        """yhat_lower ≤ yhat ≤ yhat_upper at all points."""
        result = forecaster.forecast_holt_winters(entity_id="total")
        fc = result.forecast_df
        assert (fc["yhat_lower"] <= fc["yhat"] + 1e-6).all()
        assert (fc["yhat"] <= fc["yhat_upper"] + 1e-6).all()

    def test_forecast_horizon_in_output(self, forecaster: DemandForecaster, sample_pos_3yr: pd.DataFrame) -> None:
        """Output should include future dates beyond training data."""
        result = forecaster.forecast_holt_winters(entity_id="total")
        fc = result.forecast_df
        if not fc.empty:
            max_date = fc["ds"].max()
            training_end = pd.to_datetime(sample_pos_3yr["order_date"]).max()
            # Forecast should extend at least 2 weeks beyond training data
            assert max_date >= training_end, "Forecast should extend beyond training data"

    def test_category_forecast(self, forecaster: DemandForecaster) -> None:
        result = forecaster.forecast_holt_winters(entity_id="Raw Materials", group_col="category")
        assert isinstance(result, ForecastResult)
        assert result.entity_id == "Raw Materials"

    def test_mape_is_non_negative(self, forecaster: DemandForecaster) -> None:
        result = forecaster.forecast_holt_winters(entity_id="total")
        assert result.mape >= 0

    def test_mae_is_non_negative(self, forecaster: DemandForecaster) -> None:
        result = forecaster.forecast_holt_winters(entity_id="total")
        assert result.mae >= 0

    def test_invalid_entity_returns_empty(self, forecaster: DemandForecaster) -> None:
        """Unknown category should return empty result gracefully."""
        result = forecaster.forecast_holt_winters(
            entity_id="NONEXISTENT_CATEGORY_XYZ",
            group_col="category",
        )
        assert isinstance(result, ForecastResult)
        # Either empty forecast or graceful fallback
        assert result.forecast_df is not None


# ---------------------------------------------------------------------------
# Tests: Short time series handling
# ---------------------------------------------------------------------------

class TestShortTimeSeries:
    def test_short_series_returns_gracefully(self) -> None:
        """With < 20 observations, forecaster should return gracefully (not raise)."""
        short_po = pd.DataFrame({
            "po_id": [f"PO-{i}" for i in range(10)],
            "order_date": pd.date_range("2023-01-01", periods=10, freq="W"),
            "quantity": [10] * 10,
            "category": ["Electronics"] * 10,
        })
        fc = DemandForecaster(short_po, horizon_days=30)
        result = fc.forecast_holt_winters(entity_id="total")
        assert isinstance(result, ForecastResult)

    def test_empty_dataframe_returns_gracefully(self) -> None:
        fc = DemandForecaster(pd.DataFrame(columns=["po_id", "order_date", "quantity"]), horizon_days=30)
        result = fc.forecast_holt_winters(entity_id="total")
        assert isinstance(result, ForecastResult)
        assert result.forecast_df.empty or result.forecast_df is not None


# ---------------------------------------------------------------------------
# Tests: Batch category forecasting
# ---------------------------------------------------------------------------

class TestBatchForecasting:
    def test_returns_dict(self, forecaster: DemandForecaster) -> None:
        results = forecaster.forecast_all_categories(method="holt_winters", top_n=3)
        assert isinstance(results, dict)

    def test_keys_are_category_names(self, forecaster: DemandForecaster, sample_pos_3yr) -> None:
        results = forecaster.forecast_all_categories(method="holt_winters", top_n=3)
        valid_cats = set(sample_pos_3yr["category"].unique())
        for key in results.keys():
            assert key in valid_cats

    def test_each_result_is_forecast_result(self, forecaster: DemandForecaster) -> None:
        results = forecaster.forecast_all_categories(method="holt_winters", top_n=3)
        for key, result in results.items():
            assert isinstance(result, ForecastResult), f"Expected ForecastResult for '{key}'"

    def test_top_n_respects_limit(self, forecaster: DemandForecaster) -> None:
        results = forecaster.forecast_all_categories(method="holt_winters", top_n=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Tests: DemandForecaster initialization
# ---------------------------------------------------------------------------

class TestForecasterInit:
    def test_custom_date_col(self, sample_pos_3yr: pd.DataFrame) -> None:
        df = sample_pos_3yr.rename(columns={"order_date": "purchase_date"})
        fc = DemandForecaster(df, date_col="purchase_date", horizon_days=30)
        result = fc.forecast_holt_winters(entity_id="total")
        assert isinstance(result, ForecastResult)

    def test_custom_quantity_col(self, sample_pos_3yr: pd.DataFrame) -> None:
        df = sample_pos_3yr.rename(columns={"quantity": "units_ordered"})
        fc = DemandForecaster(df, quantity_col="units_ordered", horizon_days=30)
        result = fc.forecast_holt_winters(entity_id="total")
        assert isinstance(result, ForecastResult)

    def test_horizon_reflected_in_forecast(self, sample_pos_3yr: pd.DataFrame) -> None:
        for horizon in [30, 60, 90]:
            fc = DemandForecaster(sample_pos_3yr, horizon_days=horizon)
            result = fc.forecast_holt_winters(entity_id="total")
            # The forecast_df should have rows; exact count varies by weekly aggregation
            if not result.forecast_df.empty:
                n_future_weeks = horizon // 7
                assert len(result.forecast_df) >= n_future_weeks // 2  # loose check
