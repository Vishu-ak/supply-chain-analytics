"""
Vendor performance scorecard module.

Builds weighted composite scores across four dimensions:
  1. Delivery performance  (on-time delivery rate)
  2. Quality               (inverse defect rate)
  3. Cost competitiveness  (price vs. category benchmark)
  4. Responsiveness        (lead time vs. agreed SLA)

Each dimension is normalized to [0, 100] before weighting.
Final scores are mapped to a letter grade (A–F).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Grade thresholds
GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (60, "D"),
    (0, "F"),
]


def _score_to_grade(score: float) -> str:
    """Map a numeric score [0, 100] to a letter grade."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


class VendorScorer:
    """
    Computes vendor performance scorecards and rankings.

    Parameters
    ----------
    deliveries : DataFrame with on_time, days_early_late, vendor_id columns
    quality_inspections : DataFrame with defect_rate, passed, vendor_id columns
    purchase_orders : DataFrame for cost competitiveness baseline
    vendors : Vendor master for enrichment (tier, lead_time_days_avg)
    weights : Dict overriding default dimension weights

    Example
    -------
    >>> scorer = VendorScorer(deliveries=del_df, quality_inspections=qi_df,
    ...                       purchase_orders=po_df, vendors=vendors_df)
    >>> scorecards = scorer.compute_scorecards()
    >>> top10 = scorecards.nlargest(10, "composite_score")
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "delivery": 0.35,
        "quality": 0.30,
        "cost": 0.25,
        "responsiveness": 0.10,
    }

    def __init__(
        self,
        deliveries: pd.DataFrame,
        quality_inspections: pd.DataFrame,
        purchase_orders: pd.DataFrame,
        vendors: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.deliveries = deliveries.copy()
        self.qi = quality_inspections.copy()
        self.po = purchase_orders.copy()
        self.vendors = vendors.copy()
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._validate_weights()

    # ------------------------------------------------------------------
    # Main scoring method
    # ------------------------------------------------------------------

    def compute_scorecards(self) -> pd.DataFrame:
        """
        Compute the full vendor scorecard table.

        Returns
        -------
        DataFrame indexed by vendor_id with columns for each dimension score,
        composite score, grade, and rank.
        """
        delivery_scores = self._score_delivery()
        quality_scores = self._score_quality()
        cost_scores = self._score_cost()
        responsiveness_scores = self._score_responsiveness()

        # Merge all dimension scores — convert Series to DataFrames first
        frames = [
            delivery_scores.to_frame() if isinstance(delivery_scores, pd.Series) else delivery_scores,
            quality_scores.to_frame() if isinstance(quality_scores, pd.Series) else quality_scores,
            cost_scores.to_frame() if isinstance(cost_scores, pd.Series) else cost_scores,
            responsiveness_scores.to_frame() if isinstance(responsiveness_scores, pd.Series) else responsiveness_scores,
        ]
        scorecard = frames[0]
        for frame in frames[1:]:
            if not frame.empty:
                scorecard = scorecard.join(frame, how="outer")
        scorecard = scorecard.fillna(50.0)  # Neutral score for missing dimensions

        # Composite weighted score
        w = self.weights
        scorecard["composite_score"] = (
            scorecard["delivery_score"] * w["delivery"] +
            scorecard["quality_score"] * w["quality"] +
            scorecard["cost_score"] * w["cost"] +
            scorecard["responsiveness_score"] * w["responsiveness"]
        ).round(2)

        scorecard["grade"] = scorecard["composite_score"].apply(_score_to_grade)
        scorecard["rank"] = scorecard["composite_score"].rank(method="min", ascending=False).astype(int)

        # Enrich with vendor metadata
        if "vendor_name" in self.vendors.columns:
            meta = self.vendors.set_index("vendor_id")[["vendor_name", "tier", "category"]]
            scorecard = scorecard.join(meta, how="left")

        scorecard = scorecard.sort_values("composite_score", ascending=False)
        logger.info("Scorecards computed for %d vendors (mean composite: %.1f)",
                    len(scorecard), scorecard["composite_score"].mean())
        return scorecard

    # ------------------------------------------------------------------
    # Dimension scoring
    # ------------------------------------------------------------------

    def _score_delivery(self) -> pd.Series:
        """
        Score = on-time delivery rate × 100, normalized to [0, 100].

        Penalizes vendors with high average lateness.
        """
        if self.deliveries.empty:
            return pd.Series(dtype=float, name="delivery_score")

        on_time_rate = (
            self.deliveries.groupby("vendor_id", observed=True)["on_time"]
            .mean()
            .rename("on_time_rate")
        )

        # Penalize for average days late (only positive = late)
        avg_lateness = (
            self.deliveries[self.deliveries["days_early_late"] > 0]
            .groupby("vendor_id", observed=True)["days_early_late"]
            .mean()
            .rename("avg_days_late")
            .fillna(0)
        )

        combined = on_time_rate.to_frame().join(avg_lateness, how="left").fillna(0)
        # 1 day late → -1 point penalty (up to -10 points)
        penalty = np.minimum(combined["avg_days_late"], 10)
        raw = combined["on_time_rate"] * 100 - penalty
        return raw.clip(0, 100).rename("delivery_score").round(2)

    def _score_quality(self) -> pd.Series:
        """
        Score = (1 - average defect rate) × 100, adjusted for rejection rate.
        """
        if self.qi.empty:
            return pd.Series(dtype=float, name="quality_score")

        # Join PO → deliveries to get vendor_id for inspections
        if "vendor_id" not in self.qi.columns:
            qi_enriched = self.qi.merge(
                self.deliveries[["delivery_id", "vendor_id"]].drop_duplicates(),
                on="delivery_id", how="left",
            )
        else:
            qi_enriched = self.qi.copy()

        agg = qi_enriched.groupby("vendor_id", observed=True).agg(
            avg_defect_rate=("defect_rate", "mean"),
            rejection_rate=("passed", lambda x: 1 - x.mean()),
        )
        raw = (1 - agg["avg_defect_rate"]) * 100 - agg["rejection_rate"] * 10
        return raw.clip(0, 100).rename("quality_score").round(2)

    def _score_cost(self) -> pd.Series:
        """
        Score = 100 - price_premium_pct (relative to category median).

        Vendors priced below median get a bonus; above median get a penalty.
        """
        if self.po.empty:
            return pd.Series(dtype=float, name="cost_score")

        category_median = (
            self.po.groupby("category", observed=True)["unit_price"].median()
        )
        po_enriched = self.po.merge(
            category_median.rename("category_median_price"),
            on="category", how="left",
        )
        po_enriched["price_premium_pct"] = (
            (po_enriched["unit_price"] - po_enriched["category_median_price"]) /
            po_enriched["category_median_price"].replace(0, np.nan) * 100
        )

        vendor_premium = (
            po_enriched.groupby("vendor_id", observed=True)["price_premium_pct"]
            .mean()
        )
        raw = 50 - vendor_premium  # Premium of 0 → score 50, below median → >50
        return raw.clip(0, 100).rename("cost_score").round(2)

    def _score_responsiveness(self) -> pd.Series:
        """
        Score = 100 minus a penalty for lead time exceeding the category average.
        """
        if "lead_time_days_avg" not in self.vendors.columns:
            return pd.Series(dtype=float, name="responsiveness_score")

        lead_times = self.vendors.set_index("vendor_id")["lead_time_days_avg"]
        category_avg_lt: float = float(lead_times.mean())
        # Normalize: worst vendor gets 0, best gets 100
        min_lt = float(lead_times.min())
        max_lt = float(lead_times.max())
        span = max_lt - min_lt if max_lt != min_lt else 1.0
        raw = (1 - (lead_times - min_lt) / span) * 100
        return raw.rename("responsiveness_score").round(2)

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def score_trend(self, periods: int = 4) -> pd.DataFrame:
        """
        Compute quarterly scorecard trends for the top 20 vendors by spend.

        Returns
        -------
        DataFrame with vendor_id, quarter, composite_score columns
        """
        if "order_date" not in self.po.columns:
            return pd.DataFrame()

        top_vendors = (
            self.po.groupby("vendor_id", observed=True)["total_amount"]
            .sum()
            .nlargest(20)
            .index
        )

        records: list[dict[str, Any]] = []
        po_with_quarter = self.po.copy()
        po_with_quarter["quarter"] = pd.to_datetime(po_with_quarter["order_date"]).dt.to_period("Q")

        for quarter in sorted(po_with_quarter["quarter"].unique())[-periods:]:
            po_q = po_with_quarter[
                (po_with_quarter["quarter"] == quarter) &
                (po_with_quarter["vendor_id"].isin(top_vendors))
            ]
            if po_q.empty:
                continue

            # Deliveries in this quarter (approximate by scheduled_date)
            del_q = self.deliveries.copy()
            if "scheduled_date" in del_q.columns:
                del_q["quarter"] = pd.to_datetime(del_q["scheduled_date"], errors="coerce").dt.to_period("Q")
                del_q = del_q[del_q["quarter"] == quarter]

            try:
                scorer_q = VendorScorer(
                    deliveries=del_q,
                    quality_inspections=self.qi,
                    purchase_orders=po_q,
                    vendors=self.vendors,
                    weights=self.weights,
                )
                sc_q = scorer_q.compute_scorecards()[["composite_score"]]
                sc_q["quarter"] = str(quarter)
                records.append(sc_q.reset_index())
            except Exception:
                pass

        if not records:
            return pd.DataFrame()
        return pd.concat(records, ignore_index=True)

    # ------------------------------------------------------------------
    # Tier comparison
    # ------------------------------------------------------------------

    def tier_performance_summary(self) -> pd.DataFrame:
        """Summarize scorecard metrics by vendor tier."""
        scorecards = self.compute_scorecards()
        if "tier" not in scorecards.columns:
            return pd.DataFrame()

        return (
            scorecards.groupby("tier")
            .agg(
                count=("composite_score", "count"),
                avg_composite=("composite_score", "mean"),
                avg_delivery=("delivery_score", "mean"),
                avg_quality=("quality_score", "mean"),
                avg_cost=("cost_score", "mean"),
                pct_grade_a=("grade", lambda x: (x == "A").mean() * 100),
                pct_grade_f=("grade", lambda x: (x == "F").mean() * 100),
            )
            .round(2)
            .sort_values("avg_composite", ascending=False)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Vendor score weights must sum to 1.0, got {total:.3f}")

