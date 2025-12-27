"""
Total Cost of Ownership (TCO) and cost optimization module.

Analyzes procurement costs beyond unit price, including:
  - Total Cost of Ownership (TCO): unit price + logistics + quality + admin
  - Make vs. Buy analysis with NPV-based breakeven
  - Vendor consolidation savings opportunities
  - Price benchmarking and volume discount modeling
  - Procurement cycle time analysis and its cost impact
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CostOptimizer:
    """
    Identifies procurement cost reduction opportunities through TCO analysis,
    vendor consolidation, and volume discount modeling.

    Parameters
    ----------
    purchase_orders : PO-level transaction data
    vendors : vendor master (for tier, lead time)
    deliveries : delivery performance (for logistics cost factors)
    quality_inspections : quality data (for quality cost estimation)

    Example
    -------
    >>> opt = CostOptimizer(po_df, vendors_df, del_df, qi_df)
    >>> tco = opt.total_cost_of_ownership()
    >>> savings = opt.consolidation_opportunities(min_savings_usd=50_000)
    """

    # Cost estimation factors (as % of unit price)
    LOGISTICS_COST_FACTOR: float = 0.08       # 8% of spend
    QUALITY_COST_PER_DEFECT: float = 150.0    # USD per defective unit
    LATE_DELIVERY_COST_PER_DAY: float = 250.0 # USD per day late
    ADMIN_COST_PER_PO: float = 75.0           # Processing cost per PO line

    def __init__(
        self,
        purchase_orders: pd.DataFrame,
        vendors: pd.DataFrame,
        deliveries: pd.DataFrame | None = None,
        quality_inspections: pd.DataFrame | None = None,
    ) -> None:
        self.po = purchase_orders.copy()
        self.vendors = vendors.copy()
        self.deliveries = deliveries.copy() if deliveries is not None else pd.DataFrame()
        self.qi = quality_inspections.copy() if quality_inspections is not None else pd.DataFrame()

        self.po["order_date"] = pd.to_datetime(self.po["order_date"], errors="coerce")

    # ------------------------------------------------------------------
    # Total Cost of Ownership
    # ------------------------------------------------------------------

    def total_cost_of_ownership(self) -> pd.DataFrame:
        """
        Compute TCO per vendor, decomposed into cost categories.

        Components:
          - Material cost: sum of PO total_amount
          - Logistics cost: estimated 8% of spend + penalty for late deliveries
          - Quality cost: defective units × cost per defect
          - Admin cost: number of POs × processing cost per PO

        Returns
        -------
        DataFrame with vendor_id and TCO breakdown columns
        """
        # Material cost
        material = (
            self.po.groupby("vendor_id", observed=True)["total_amount"]
            .sum()
            .rename("material_cost")
        )

        # Admin cost
        po_count = (
            self.po.groupby("vendor_id", observed=True)["po_id"]
            .count()
            .rename("po_count")
        )
        admin_cost = (po_count * self.ADMIN_COST_PER_PO).rename("admin_cost")

        # Logistics cost
        logistics = (material * self.LOGISTICS_COST_FACTOR).rename("logistics_cost_base")

        # Late delivery penalty
        late_cost = pd.Series(dtype=float, name="late_delivery_cost")
        if not self.deliveries.empty and "days_early_late" in self.deliveries.columns:
            late_df = self.deliveries[self.deliveries["days_early_late"] > 0].copy()
            late_cost = (
                late_df.groupby("vendor_id", observed=True)["days_early_late"]
                .sum() * self.LATE_DELIVERY_COST_PER_DAY
            ).rename("late_delivery_cost")

        # Quality cost
        quality_cost = pd.Series(dtype=float, name="quality_cost")
        if not self.qi.empty and "defects_found" in self.qi.columns:
            qi_with_vendor = self.qi.copy()
            if "vendor_id" not in qi_with_vendor.columns:
                qi_with_vendor = qi_with_vendor.merge(
                    self.deliveries[["delivery_id", "vendor_id"]].drop_duplicates(),
                    on="delivery_id", how="left",
                )
            quality_cost = (
                qi_with_vendor.groupby("vendor_id", observed=True)["defects_found"]
                .sum() * self.QUALITY_COST_PER_DEFECT
            ).rename("quality_cost")

        # Assemble TCO table
        tco = (
            material.to_frame()
            .join(po_count, how="left")
            .join(admin_cost, how="left")
            .join(logistics, how="left")
            .join(late_cost, how="left")
            .join(quality_cost, how="left")
            .fillna(0)
        )

        tco["total_logistics_cost"] = tco["logistics_cost_base"] + tco["late_delivery_cost"]
        tco["total_tco"] = (
            tco["material_cost"] +
            tco["admin_cost"] +
            tco["total_logistics_cost"] +
            tco["quality_cost"]
        )
        tco["tco_premium_pct"] = (
            (tco["total_tco"] - tco["material_cost"]) / tco["material_cost"].replace(0, np.nan) * 100
        ).round(2)

        # Enrich with vendor metadata
        if "tier" in self.vendors.columns:
            tco = tco.join(
                self.vendors.set_index("vendor_id")[["vendor_name", "tier"]],
                how="left",
            )

        tco = tco.sort_values("total_tco", ascending=False).reset_index()
        logger.info("TCO computed for %d vendors; total = USD %.2f M",
                    len(tco), tco["total_tco"].sum() / 1e6)
        return tco

    # ------------------------------------------------------------------
    # Make vs. Buy
    # ------------------------------------------------------------------

    def make_vs_buy_analysis(
        self,
        product_id: str,
        annual_demand: float,
        buy_unit_cost: float,
        make_fixed_cost: float,
        make_variable_cost: float,
        discount_rate: float = 0.10,
        horizon_years: int = 5,
    ) -> dict[str, Any]:
        """
        Compute NPV-based make vs. buy breakeven analysis.

        Parameters
        ----------
        product_id : product identifier for reference
        annual_demand : expected annual units
        buy_unit_cost : current purchase price per unit
        make_fixed_cost : one-time capital investment to manufacture in-house
        make_variable_cost : variable cost per unit for in-house production
        discount_rate : WACC / hurdle rate for NPV calculation
        horizon_years : analysis horizon

        Returns
        -------
        Dict with NPV comparison, breakeven volume, and recommendation
        """
        years = np.arange(1, horizon_years + 1)
        discount_factors = 1 / (1 + discount_rate) ** years

        # Buy scenario: recurring material cost
        buy_annual = annual_demand * buy_unit_cost
        npv_buy = float(np.sum(buy_annual * discount_factors))

        # Make scenario: capex year 0 + recurring variable cost
        make_annual = annual_demand * make_variable_cost
        npv_make = make_fixed_cost + float(np.sum(make_annual * discount_factors))

        # Breakeven volume (per year) at which make = buy
        # make_fixed_cost + Q × make_variable × PVF = Q × buy_cost × PVF
        # Q = make_fixed_cost / (PVF × (buy_cost - make_variable_cost))
        total_pvf = float(discount_factors.sum())
        cost_diff = buy_unit_cost - make_variable_cost
        if cost_diff > 0 and total_pvf > 0:
            breakeven_annual_volume = make_fixed_cost / (total_pvf * cost_diff)
        else:
            breakeven_annual_volume = float("inf")

        recommendation = (
            "MAKE — in-house production has lower NPV" if npv_make < npv_buy
            else "BUY — purchasing is more cost-effective"
        )

        return {
            "product_id": product_id,
            "annual_demand": annual_demand,
            "npv_buy_usd": round(npv_buy, 2),
            "npv_make_usd": round(npv_make, 2),
            "npv_savings_usd": round(npv_buy - npv_make, 2),
            "breakeven_annual_volume": round(breakeven_annual_volume, 0),
            "recommendation": recommendation,
            "horizon_years": horizon_years,
            "discount_rate": discount_rate,
        }

    # ------------------------------------------------------------------
    # Vendor consolidation
    # ------------------------------------------------------------------

    def consolidation_opportunities(
        self, min_savings_usd: float = 10_000
    ) -> pd.DataFrame:
        """
        Identify categories where spend is fragmented across many vendors,
        estimating savings from consolidating to fewer preferred suppliers.

        Consolidation savings assumption: 5–15% price reduction through
        larger volume commitments to fewer vendors.

        Returns
        -------
        DataFrame with category, current vendor count, recommended vendor count,
        estimated savings USD
        """
        cat_agg = (
            self.po.groupby("category", observed=True)
            .agg(
                total_spend=("total_amount", "sum"),
                vendor_count=("vendor_id", "nunique"),
                po_count=("po_id", "count"),
            )
            .reset_index()
        )

        # Categories with >3 vendors and >$50K spend are candidates
        candidates = cat_agg[
            (cat_agg["vendor_count"] > 3) &
            (cat_agg["total_spend"] >= 50_000)
        ].copy()

        # Savings estimate: consolidating to 2 vendors → ~8% saving
        candidates["recommended_vendors"] = 2
        candidates["saving_rate"] = np.where(
            candidates["vendor_count"] > 10, 0.12,
            np.where(candidates["vendor_count"] > 5, 0.08, 0.05)
        )
        candidates["estimated_savings_usd"] = (
            candidates["total_spend"] * candidates["saving_rate"]
        ).round(2)

        result = candidates[
            candidates["estimated_savings_usd"] >= min_savings_usd
        ].sort_values("estimated_savings_usd", ascending=False)

        logger.info("Consolidation opportunities: %d categories, total savings potential USD %.2f M",
                    len(result), result["estimated_savings_usd"].sum() / 1e6)
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Procurement cycle time
    # ------------------------------------------------------------------

    def procurement_cycle_time(self) -> pd.DataFrame:
        """
        Compute average procurement cycle time (days from PO to delivery) by vendor and category.

        Returns
        -------
        DataFrame with vendor_id, category, avg_cycle_days, median_cycle_days, po_count
        """
        if self.deliveries.empty:
            return pd.DataFrame()

        merged = self.deliveries.merge(
            self.po[["po_id", "order_date", "category"]].drop_duplicates(),
            on="po_id", how="left",
        )
        merged["order_date"] = pd.to_datetime(merged["order_date"], errors="coerce")
        merged["actual_date"] = pd.to_datetime(merged["actual_date"], errors="coerce")
        merged["cycle_days"] = (merged["actual_date"] - merged["order_date"]).dt.days

        cycle_time = (
            merged.groupby(["vendor_id", "category"], observed=True)
            .agg(
                avg_cycle_days=("cycle_days", "mean"),
                median_cycle_days=("cycle_days", "median"),
                po_count=("po_id", "count"),
            )
            .round(1)
            .reset_index()
            .sort_values("avg_cycle_days")
        )
        return cycle_time

    # ------------------------------------------------------------------
    # KPI summary
    # ------------------------------------------------------------------

    def kpi_summary(self) -> dict[str, Any]:
        """Return high-level cost optimization KPIs."""
        tco = self.total_cost_of_ownership()
        consolidation = self.consolidation_opportunities()

        return {
            "total_tco_usd": round(float(tco["total_tco"].sum()), 2),
            "total_quality_cost_usd": round(float(tco["quality_cost"].sum()), 2),
            "total_logistics_cost_usd": round(float(tco["total_logistics_cost"].sum()), 2),
            "avg_tco_premium_pct": round(float(tco["tco_premium_pct"].mean()), 2),
            "consolidation_savings_potential_usd": round(
                float(consolidation["estimated_savings_usd"].sum()), 2
            ) if not consolidation.empty else 0,
            "consolidation_categories": len(consolidation),
        }
