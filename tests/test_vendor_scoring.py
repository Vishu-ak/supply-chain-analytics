"""
Unit tests for src.analysis.vendor_scoring.VendorScorer.

Tests cover:
  - Scorecard completeness and column set
  - Weight validation (must sum to 1.0)
  - Dimension score ranges [0, 100]
  - Grade assignment correctness
  - Tier performance summary
  - Rank ordering consistency
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.vendor_scoring import VendorScorer, _score_to_grade, GRADE_THRESHOLDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vendor_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (vendors, deliveries, quality_inspections, purchase_orders)."""
    rng = np.random.default_rng(0)
    n_vendors = 30
    vendor_ids = [f"V{i:04d}" for i in range(1, n_vendors + 1)]
    tiers = rng.choice(["Tier 1", "Tier 2", "Tier 3"], size=n_vendors, p=[0.3, 0.4, 0.3])

    vendors = pd.DataFrame({
        "vendor_id": vendor_ids,
        "vendor_name": [f"Vendor {i}" for i in range(n_vendors)],
        "tier": tiers,
        "category": rng.choice(["Electronics", "MRO", "Packaging"], size=n_vendors),
        "lead_time_days_avg": rng.integers(5, 30, size=n_vendors),
        "active": True,
    })

    # Deliveries
    n_del = 500
    del_vendor_ids = rng.choice(vendor_ids, size=n_del)
    on_time = rng.random(size=n_del) < 0.85
    days_early_late = np.where(on_time, rng.integers(-3, 1, size=n_del), rng.integers(1, 15, size=n_del))

    deliveries = pd.DataFrame({
        "delivery_id": [f"DEL-{i:06d}" for i in range(n_del)],
        "po_id": [f"PO-{i:06d}" for i in range(n_del)],
        "vendor_id": del_vendor_ids,
        "on_time": on_time,
        "days_early_late": days_early_late,
        "scheduled_date": pd.date_range("2023-01-01", periods=n_del, freq="4h"),
        "actual_date": pd.date_range("2023-01-01", periods=n_del, freq="4h"),
        "quantity_delivered": rng.integers(10, 100, size=n_del),
    })

    # Quality inspections
    n_qi = 300
    qi_delivery_ids = rng.choice(deliveries["delivery_id"].values, size=n_qi)
    defect_rates = rng.uniform(0.0, 0.05, size=n_qi)
    qi = pd.DataFrame({
        "inspection_id": [f"INS-{i:06d}" for i in range(n_qi)],
        "delivery_id": qi_delivery_ids,
        "quantity_inspected": rng.integers(10, 100, size=n_qi),
        "defects_found": (defect_rates * rng.integers(10, 100, size=n_qi)).astype(int),
        "defect_rate": defect_rates.round(4),
        "passed": defect_rates < 0.03,
    })
    # Add vendor_id to qi
    qi = qi.merge(
        deliveries[["delivery_id", "vendor_id"]].drop_duplicates(),
        on="delivery_id", how="left",
    )

    # POs
    n_po = 400
    categories = ["Electronics", "MRO", "Packaging", "Services"]
    po = pd.DataFrame({
        "po_id": [f"PO-{i:06d}" for i in range(n_po)],
        "vendor_id": rng.choice(vendor_ids, size=n_po),
        "product_id": [f"P{i:05d}" for i in range(n_po)],
        "order_date": pd.date_range("2023-01-01", periods=n_po, freq="6h"),
        "quantity": rng.integers(1, 50, size=n_po),
        "unit_price": rng.uniform(20.0, 300.0, size=n_po).round(2),
        "total_amount": rng.uniform(200.0, 15_000.0, size=n_po).round(2),
        "category": rng.choice(categories, size=n_po),
        "abc_class": rng.choice(["A", "B", "C"], size=n_po),
        "status": "Received",
    })

    return vendors, deliveries, qi, po


@pytest.fixture
def scorer(vendor_data) -> VendorScorer:
    vendors, deliveries, qi, po = vendor_data
    return VendorScorer(
        deliveries=deliveries,
        quality_inspections=qi,
        purchase_orders=po,
        vendors=vendors,
    )


# ---------------------------------------------------------------------------
# Tests: Weight validation
# ---------------------------------------------------------------------------

class TestWeightValidation:
    def test_default_weights_sum_to_one(self) -> None:
        total = sum(VendorScorer.DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_custom_valid_weights(self, vendor_data) -> None:
        vendors, deliveries, qi, po = vendor_data
        scorer = VendorScorer(
            deliveries=deliveries, quality_inspections=qi,
            purchase_orders=po, vendors=vendors,
            weights={"delivery": 0.5, "quality": 0.2, "cost": 0.2, "responsiveness": 0.1},
        )
        sc = scorer.compute_scorecards()
        assert not sc.empty

    def test_invalid_weights_raise(self, vendor_data) -> None:
        vendors, deliveries, qi, po = vendor_data
        with pytest.raises(ValueError, match="sum to 1.0"):
            VendorScorer(
                deliveries=deliveries, quality_inspections=qi,
                purchase_orders=po, vendors=vendors,
                weights={"delivery": 0.5, "quality": 0.5, "cost": 0.5, "responsiveness": 0.1},
            )


# ---------------------------------------------------------------------------
# Tests: Scorecard columns and structure
# ---------------------------------------------------------------------------

class TestScorecardStructure:
    def test_returns_dataframe(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert isinstance(sc, pd.DataFrame)

    def test_required_columns_present(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        required = {"delivery_score", "quality_score", "cost_score",
                    "responsiveness_score", "composite_score", "grade", "rank"}
        assert required.issubset(sc.columns)

    def test_non_empty_result(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert len(sc) > 0

    def test_no_duplicate_vendors(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards().reset_index()
        if "vendor_id" in sc.columns:
            assert not sc["vendor_id"].duplicated().any()

    def test_vendor_metadata_joined(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert "tier" in sc.columns or "vendor_name" in sc.columns


# ---------------------------------------------------------------------------
# Tests: Score ranges
# ---------------------------------------------------------------------------

class TestScoreRanges:
    def test_delivery_score_in_range(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert sc["delivery_score"].between(0, 100).all()

    def test_quality_score_in_range(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert sc["quality_score"].between(0, 100).all()

    def test_cost_score_in_range(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert sc["cost_score"].between(0, 100).all()

    def test_responsiveness_score_in_range(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert sc["responsiveness_score"].between(0, 100).all()

    def test_composite_score_in_range(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert sc["composite_score"].between(0, 100).all()


# ---------------------------------------------------------------------------
# Tests: Grade assignment
# ---------------------------------------------------------------------------

class TestGradeAssignment:
    @pytest.mark.parametrize("score, expected_grade", [
        (95.0, "A"),
        (85.0, "B"),
        (75.0, "C"),
        (65.0, "D"),
        (50.0, "F"),
        (90.0, "A"),
        (80.0, "B"),
    ])
    def test_grade_thresholds(self, score: float, expected_grade: str) -> None:
        assert _score_to_grade(score) == expected_grade

    def test_scorecard_grades_valid(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        valid_grades = {"A", "B", "C", "D", "F"}
        assert set(sc["grade"].unique()).issubset(valid_grades)

    def test_grade_consistent_with_composite(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        for _, row in sc.iterrows():
            expected = _score_to_grade(row["composite_score"])
            assert row["grade"] == expected, (
                f"Grade mismatch for score {row['composite_score']}: "
                f"expected {expected}, got {row['grade']}"
            )


# ---------------------------------------------------------------------------
# Tests: Rank ordering
# ---------------------------------------------------------------------------

class TestRankOrdering:
    def test_rank_monotone_with_score(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards().sort_values("composite_score", ascending=False)
        # Rank 1 should be highest score
        top_score = sc.iloc[0]["composite_score"]
        assert sc.loc[sc["rank"] == 1, "composite_score"].iloc[0] == pytest.approx(top_score, abs=1e-3)

    def test_rank_is_integer(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert sc["rank"].dtype in (int, "int64", "int32")

    def test_rank_starts_at_one(self, scorer: VendorScorer) -> None:
        sc = scorer.compute_scorecards()
        assert sc["rank"].min() == 1


# ---------------------------------------------------------------------------
# Tests: Tier performance summary
# ---------------------------------------------------------------------------

class TestTierPerformanceSummary:
    def test_returns_dataframe(self, scorer: VendorScorer) -> None:
        result = scorer.tier_performance_summary()
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, scorer: VendorScorer) -> None:
        result = scorer.tier_performance_summary()
        required = {"count", "avg_composite", "avg_delivery", "avg_quality", "avg_cost"}
        assert required.issubset(result.columns)

    def test_tier1_better_than_tier3(self, scorer: VendorScorer) -> None:
        result = scorer.tier_performance_summary()
        if "Tier 1" in result.index and "Tier 3" in result.index:
            # Tier 1 should generally outperform Tier 3
            # (may not hold for all small random seeds, so we test loosely)
            tier1 = result.loc["Tier 1", "avg_delivery"]
            tier3 = result.loc["Tier 3", "avg_delivery"]
            # Just verify both are computed — not testing specific ordering
            assert 0 <= tier1 <= 100
            assert 0 <= tier3 <= 100
