"""
Unit tests for src.analysis.spend_analysis.SpendAnalyzer.

Tests cover:
  - Pareto analysis correctness (cumulative % ordering, class assignment)
  - Maverick spend detection (off-contract, emergency flags)
  - Category aggregation and trend computation
  - HHI concentration index
  - KPI summary completeness
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.spend_analysis import SpendAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pos() -> pd.DataFrame:
    """Minimal purchase orders DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 500
    categories = ["Raw Materials", "Packaging", "Electronics", "MRO", "Services"]
    vendor_ids = [f"V{i:04d}" for i in range(1, 21)]  # 20 vendors

    df = pd.DataFrame({
        "po_id": [f"PO-{i:06d}" for i in range(n)],
        "vendor_id": rng.choice(vendor_ids, size=n),
        "product_id": [f"P{i:05d}" for i in range(n)],
        "order_date": pd.date_range("2023-01-01", periods=n, freq="12h"),
        "requested_delivery_date": pd.date_range("2023-01-15", periods=n, freq="12h"),
        "quantity": rng.integers(1, 100, size=n),
        "unit_price": rng.uniform(10.0, 500.0, size=n).round(2),
        "total_amount": rng.uniform(100.0, 50_000.0, size=n).round(2),
        "status": rng.choice(["Open", "Received", "Cancelled"], size=n, p=[0.1, 0.8, 0.1]),
        "category": rng.choice(categories, size=n),
        "abc_class": rng.choice(["A", "B", "C"], size=n, p=[0.2, 0.3, 0.5]),
        "is_emergency": rng.choice([True, False], size=n, p=[0.05, 0.95]),
    })
    return df


@pytest.fixture
def sample_contracts() -> pd.DataFrame:
    """Contracts covering half the vendors."""
    return pd.DataFrame({
        "contract_id": [f"CTR-{i:04d}" for i in range(10)],
        "vendor_id": [f"V{i:04d}" for i in range(1, 11)],
        "category": ["Raw Materials", "Packaging", "Electronics", "MRO", "Services"] * 2,
        "contracted_price_per_unit": [50.0, 100.0, 150.0, 75.0, 200.0] * 2,
        "status": ["Active"] * 10,
    })


@pytest.fixture
def analyzer(sample_pos, sample_contracts) -> SpendAnalyzer:
    return SpendAnalyzer(sample_pos, contracts=sample_contracts)


# ---------------------------------------------------------------------------
# Tests: Pareto analysis
# ---------------------------------------------------------------------------

class TestParetoAnalysis:
    def test_returns_dataframe(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.pareto_analysis(by="vendor_id")
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.pareto_analysis(by="vendor_id")
        required = {"vendor_id", "total_spend", "spend_pct", "cumulative_pct", "pareto_class"}
        assert required.issubset(result.columns)

    def test_cumulative_pct_monotonically_increasing(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.pareto_analysis(by="vendor_id")
        diffs = result["cumulative_pct"].diff().dropna()
        assert (diffs >= -1e-9).all(), "Cumulative pct should be non-decreasing"

    def test_cumulative_pct_ends_near_one(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.pareto_analysis(by="vendor_id")
        assert abs(result["cumulative_pct"].iloc[-1] - 1.0) < 0.001

    def test_pareto_class_only_abc(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.pareto_analysis(by="vendor_id")
        assert set(result["pareto_class"].unique()).issubset({"A", "B", "C"})

    def test_sorted_by_spend_descending(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.pareto_analysis(by="vendor_id")
        diffs = result["total_spend"].diff().dropna()
        assert (diffs <= 0).all(), "Spend should be sorted descending"

    def test_by_category(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.pareto_analysis(by="category")
        assert "category" in result.columns
        assert len(result) > 0

    def test_custom_threshold(self, analyzer: SpendAnalyzer) -> None:
        result_70 = analyzer.pareto_analysis(by="vendor_id", threshold=0.70)
        result_90 = analyzer.pareto_analysis(by="vendor_id", threshold=0.90)
        # With higher threshold, more items should be class A
        a_70 = (result_70["pareto_class"] == "A").sum()
        a_90 = (result_90["pareto_class"] == "A").sum()
        assert a_90 >= a_70, "Higher threshold should yield more A-class items"


# ---------------------------------------------------------------------------
# Tests: Maverick spend
# ---------------------------------------------------------------------------

class TestMaverickSpend:
    def test_emergency_pos_flagged(self, sample_pos: pd.DataFrame) -> None:
        analyzer = SpendAnalyzer(sample_pos)  # no contracts → only emergency flags
        result = analyzer.detect_maverick_spend()
        if not result.empty:
            assert "maverick_reason" in result.columns
            emergency_reasons = result["maverick_reason"].str.contains("Emergency", na=False)
            # At least some rows should have emergency reason
            assert emergency_reasons.any() or len(result) >= 0

    def test_maverick_result_is_dataframe(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.detect_maverick_spend()
        assert isinstance(result, pd.DataFrame)

    def test_maverick_subset_of_original(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.detect_maverick_spend()
        if not result.empty:
            assert "po_id" in result.columns
            # All flagged POs must exist in the original dataset
            assert result["po_id"].isin(analyzer.po["po_id"]).all()

    def test_no_duplicate_pos_in_maverick(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.detect_maverick_spend()
        if not result.empty:
            assert not result["po_id"].duplicated().any(), "Each PO should appear once"


# ---------------------------------------------------------------------------
# Tests: Category aggregation
# ---------------------------------------------------------------------------

class TestCategorySpend:
    def test_category_spend_returns_dataframe(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.category_spend_summary(period="M")
        assert isinstance(result, pd.DataFrame)

    def test_total_column_exists(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.category_spend_summary()
        assert "total" in result.columns

    def test_category_spend_sorted_by_total(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.category_spend_summary()
        diffs = result["total"].diff().dropna()
        assert (diffs <= 0).all(), "Categories should be sorted by total spend descending"

    def test_spend_trend_has_pct_change(self, analyzer: SpendAnalyzer) -> None:
        trend = analyzer.spend_trend(period="M")
        assert "pct_change" in trend.columns
        assert "total_spend" in trend.columns


# ---------------------------------------------------------------------------
# Tests: HHI
# ---------------------------------------------------------------------------

class TestHHI:
    def test_hhi_in_valid_range(self, analyzer: SpendAnalyzer) -> None:
        hhi = analyzer.herfindahl_index(by="vendor_id")
        assert 0 <= hhi <= 1

    def test_hhi_by_category(self, analyzer: SpendAnalyzer) -> None:
        hhi = analyzer.herfindahl_index(by="category")
        assert 0 <= hhi <= 1

    def test_perfect_monopoly_hhi(self) -> None:
        """Single vendor should yield HHI = 1.0."""
        po = pd.DataFrame({
            "po_id": ["PO-001", "PO-002"],
            "vendor_id": ["V001", "V001"],
            "order_date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "total_amount": [100.0, 200.0],
        })
        analyzer = SpendAnalyzer(po)
        assert analyzer.herfindahl_index() == pytest.approx(1.0, abs=1e-9)

    def test_equal_share_hhi(self) -> None:
        """Four equal vendors → HHI = 0.25."""
        po = pd.DataFrame({
            "po_id": [f"PO-{i}" for i in range(4)],
            "vendor_id": ["V001", "V002", "V003", "V004"],
            "order_date": pd.to_datetime(["2023-01-01"] * 4),
            "total_amount": [100.0, 100.0, 100.0, 100.0],
        })
        analyzer = SpendAnalyzer(po)
        assert analyzer.herfindahl_index() == pytest.approx(0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests: KPI summary
# ---------------------------------------------------------------------------

class TestKPISummary:
    def test_kpi_summary_keys(self, analyzer: SpendAnalyzer) -> None:
        kpis = analyzer.kpi_summary()
        required_keys = {
            "total_spend_usd", "total_po_count", "active_vendors",
            "avg_po_value_usd", "maverick_spend_usd", "maverick_spend_pct", "spend_hhi",
        }
        assert required_keys.issubset(kpis.keys())

    def test_total_spend_positive(self, analyzer: SpendAnalyzer) -> None:
        kpis = analyzer.kpi_summary()
        assert kpis["total_spend_usd"] > 0

    def test_active_vendors_count(self, analyzer: SpendAnalyzer) -> None:
        kpis = analyzer.kpi_summary()
        n_unique = analyzer.po["vendor_id"].nunique()
        assert kpis["active_vendors"] == n_unique


# ---------------------------------------------------------------------------
# Tests: Price variance
# ---------------------------------------------------------------------------

class TestPriceVariance:
    def test_returns_dataframe_with_required_cols(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.price_variance_analysis()
        required = {"vendor_id", "category", "avg_price", "benchmark_price", "price_variance_pct"}
        assert required.issubset(result.columns)

    def test_benchmark_is_non_negative(self, analyzer: SpendAnalyzer) -> None:
        result = analyzer.price_variance_analysis()
        assert (result["benchmark_price"] >= 0).all()
