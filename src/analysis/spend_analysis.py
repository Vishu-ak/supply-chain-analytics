"""
Procurement spend analysis module.

Implements:
  - ABC / Pareto classification of spend by vendor and category
  - Maverick spend detection (off-contract, unapproved supplier purchases)
  - Category-level spend aggregation and trending
  - Price variance analysis vs. contracted rates
  - Spend concentration index (Herfindahl-Hirschman)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SpendAnalyzer:
    """
    Analyzes procurement spend data to surface savings opportunities,
    concentration risks, and compliance issues.

    Parameters
    ----------
    purchase_orders : DataFrame with PO-level spend records
    contracts : optional DataFrame mapping vendor/category to contracted prices
    vendors : optional vendor master for enrichment

    Example
    -------
    >>> analyzer = SpendAnalyzer(purchase_orders=po_df, contracts=contracts_df)
    >>> pareto = analyzer.pareto_analysis(by="vendor_id")
    >>> maverick = analyzer.detect_maverick_spend()
    """

    def __init__(
        self,
        purchase_orders: pd.DataFrame,
        contracts: pd.DataFrame | None = None,
        vendors: pd.DataFrame | None = None,
    ) -> None:
        self.po = purchase_orders.copy()
        self.contracts = contracts.copy() if contracts is not None else pd.DataFrame()
        self.vendors = vendors.copy() if vendors is not None else pd.DataFrame()
        self._ensure_dates()

    # ------------------------------------------------------------------
    # Pareto / ABC classification
    # ------------------------------------------------------------------

    def pareto_analysis(
        self, by: str = "vendor_id", threshold: float = 0.80
    ) -> pd.DataFrame:
        """
        Compute Pareto (80/20) analysis of spend by a given dimension.

        Parameters
        ----------
        by : column to aggregate spend on ('vendor_id', 'category', 'product_id')
        threshold : cumulative spend fraction that defines the A-class boundary

        Returns
        -------
        DataFrame with columns: [by, total_spend, spend_pct, cumulative_pct, pareto_class]
        """
        agg = (
            self.po.groupby(by, observed=True)["total_amount"]
            .sum()
            .reset_index()
            .rename(columns={"total_amount": "total_spend"})
            .sort_values("total_spend", ascending=False)
        )

        total = agg["total_spend"].sum()
        agg["spend_pct"] = agg["total_spend"] / total
        agg["cumulative_pct"] = agg["spend_pct"].cumsum()

        # Class assignment: A = top 80%, B = next 15%, C = remainder
        conditions = [
            agg["cumulative_pct"] <= threshold,
            (agg["cumulative_pct"] > threshold) & (agg["cumulative_pct"] <= threshold + 0.15),
        ]
        agg["pareto_class"] = np.select(conditions, ["A", "B"], default="C")

        # Rank
        agg = agg.reset_index(drop=True)
        agg.index += 1
        agg.index.name = "rank"

        logger.info("Pareto analysis by '%s': %d A-class, %d B-class, %d C-class",
                    by,
                    (agg["pareto_class"] == "A").sum(),
                    (agg["pareto_class"] == "B").sum(),
                    (agg["pareto_class"] == "C").sum())
        return agg.reset_index()

    # ------------------------------------------------------------------
    # Maverick spend detection
    # ------------------------------------------------------------------

    def detect_maverick_spend(self, threshold: float = 0.15) -> pd.DataFrame:
        """
        Identify maverick (off-contract) spend.

        Maverick spend is flagged when:
          1. A vendor is not in the contracts table for the given category, OR
          2. The purchase price exceeds the contracted price by > threshold %

        Parameters
        ----------
        threshold : price premium above contracted rate to flag as maverick

        Returns
        -------
        DataFrame of flagged PO rows with a 'maverick_reason' column
        """
        flagged_rows: list[pd.DataFrame] = []

        # Flag 1: vendor-category not in contracts
        if not self.contracts.empty and "vendor_id" in self.contracts.columns:
            contracted_pairs = self.contracts.set_index(
                ["vendor_id", "category"]
            ).index if "category" in self.contracts.columns else pd.Index([])

            if len(contracted_pairs):
                po_pairs = pd.MultiIndex.from_frame(
                    self.po[["vendor_id", "category"]].fillna("UNKNOWN")
                )
                mask_off_contract = ~po_pairs.isin(contracted_pairs)
                off_contract = self.po[mask_off_contract].copy()
                off_contract["maverick_reason"] = "Off-contract vendor-category pair"
                flagged_rows.append(off_contract)

        # Flag 2: price premium vs contract
        if not self.contracts.empty and "contracted_price_per_unit" in self.contracts.columns:
            merged = self.po.merge(
                self.contracts[["vendor_id", "contracted_price_per_unit"]].dropna(),
                on="vendor_id",
                how="left",
            )
            price_premium = (
                (merged["unit_price"] - merged["contracted_price_per_unit"]) /
                merged["contracted_price_per_unit"].replace(0, np.nan)
            )
            mask_premium = price_premium > threshold
            premium_rows = self.po[mask_premium.values].copy()
            premium_rows["maverick_reason"] = f"Price >{threshold*100:.0f}% above contracted rate"
            flagged_rows.append(premium_rows)

        # Emergency purchases are always maverick candidates
        if "is_emergency" in self.po.columns:
            emergency = self.po[self.po["is_emergency"] == True].copy()
            emergency["maverick_reason"] = "Emergency / unplanned purchase"
            flagged_rows.append(emergency)

        if not flagged_rows:
            logger.info("No maverick spend flagged (no contract data available)")
            return pd.DataFrame()

        result = pd.concat(flagged_rows, ignore_index=True).drop_duplicates(subset=["po_id"])
        logger.info("Maverick spend: %d POs flagged (USD %.2f M)",
                    len(result),
                    result["total_amount"].sum() / 1e6)
        return result

    # ------------------------------------------------------------------
    # Category spend aggregation
    # ------------------------------------------------------------------

    def category_spend_summary(
        self, period: str = "Q"
    ) -> pd.DataFrame:
        """
        Aggregate spend by category and time period.

        Parameters
        ----------
        period : pandas offset alias ('M'=monthly, 'Q'=quarterly, 'A'=annual)

        Returns
        -------
        Wide-format DataFrame: categories as rows, periods as columns
        """
        df = self.po.copy()
        df["period"] = pd.to_datetime(df["order_date"]).dt.to_period(period)
        pivot = (
            df.groupby(["category", "period"], observed=True)["total_amount"]
            .sum()
            .unstack(fill_value=0)
        )
        pivot.columns = [str(c) for c in pivot.columns]
        pivot["total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("total", ascending=False)
        return pivot

    # ------------------------------------------------------------------
    # Spend concentration index
    # ------------------------------------------------------------------

    def herfindahl_index(self, by: str = "vendor_id") -> float:
        """
        Compute the Herfindahl-Hirschman Index (HHI) for spend concentration.

        HHI = sum of squared market share fractions.
        HHI < 0.15 → unconcentrated  |  0.15–0.25 → moderately concentrated  |  > 0.25 → highly concentrated

        Parameters
        ----------
        by : dimension to compute concentration on

        Returns
        -------
        float in [0, 1]
        """
        shares = self.po.groupby(by, observed=True)["total_amount"].sum()
        shares = shares / shares.sum()
        hhi = float((shares ** 2).sum())
        logger.info("HHI by '%s': %.4f (%s)", by, hhi,
                    "high" if hhi > 0.25 else ("moderate" if hhi > 0.15 else "low"))
        return hhi

    # ------------------------------------------------------------------
    # Price variance analysis
    # ------------------------------------------------------------------

    def price_variance_analysis(self) -> pd.DataFrame:
        """
        Compute price variance per vendor vs. category benchmark (median price).

        Returns
        -------
        DataFrame with vendor-level price premium / discount vs. benchmark
        """
        benchmarks = (
            self.po.groupby("category", observed=True)["unit_price"]
            .median()
            .rename("benchmark_price")
        )
        agg = (
            self.po.groupby(["vendor_id", "category"], observed=True)
            .agg(
                avg_price=("unit_price", "mean"),
                total_spend=("total_amount", "sum"),
                po_count=("po_id", "count"),
            )
            .reset_index()
        )
        agg = agg.merge(benchmarks, on="category")
        agg["price_variance_pct"] = (
            (agg["avg_price"] - agg["benchmark_price"]) / agg["benchmark_price"] * 100
        )
        return agg.sort_values("price_variance_pct", ascending=False)

    # ------------------------------------------------------------------
    # Spend trend analysis
    # ------------------------------------------------------------------

    def spend_trend(self, period: str = "M") -> pd.DataFrame:
        """
        Monthly or quarterly total spend trend with MoM / QoQ change.

        Returns
        -------
        DataFrame with columns: period, total_spend, pct_change
        """
        df = self.po.copy()
        df["period"] = pd.to_datetime(df["order_date"]).dt.to_period(period)
        trend = (
            df.groupby("period", observed=True)["total_amount"]
            .sum()
            .reset_index()
            .rename(columns={"total_amount": "total_spend"})
        )
        trend["pct_change"] = trend["total_spend"].pct_change() * 100
        return trend

    # ------------------------------------------------------------------
    # KPI summary
    # ------------------------------------------------------------------

    def kpi_summary(self) -> dict[str, Any]:
        """Return a dict of high-level spend KPIs."""
        total_spend = self.po["total_amount"].sum()
        po_count = len(self.po)
        n_vendors = self.po["vendor_id"].nunique()
        avg_po_value = total_spend / po_count if po_count else 0

        maverick = self.detect_maverick_spend()
        maverick_spend = maverick["total_amount"].sum() if not maverick.empty else 0

        return {
            "total_spend_usd": round(total_spend, 2),
            "total_po_count": po_count,
            "active_vendors": n_vendors,
            "avg_po_value_usd": round(avg_po_value, 2),
            "maverick_spend_usd": round(maverick_spend, 2),
            "maverick_spend_pct": round(maverick_spend / total_spend * 100, 2) if total_spend else 0,
            "spend_hhi": round(self.herfindahl_index(), 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dates(self) -> None:
        """Coerce date columns to datetime."""
        for col in ["order_date", "requested_delivery_date"]:
            if col in self.po.columns:
                self.po[col] = pd.to_datetime(self.po[col], errors="coerce")
