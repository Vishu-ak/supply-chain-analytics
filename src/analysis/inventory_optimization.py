"""
Inventory optimization module.

Implements classical inventory management models:
  - Economic Order Quantity (EOQ) — Wilson formula
  - Reorder Point (ROP) with safety stock
  - Safety stock calculation at target service levels
  - ABC classification reinforcement from PO data
  - Days of Supply / Inventory Turnover KPIs

All calculations follow standard Operations Research models
with configurable cost parameters from project settings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """
    Computes optimal inventory parameters for each product/SKU.

    Parameters
    ----------
    purchase_orders : historical demand data (used to compute avg & std demand)
    products : product master with unit_cost and abc_class
    holding_cost_rate : annual holding cost as fraction of unit cost (e.g. 0.25)
    ordering_cost : fixed cost per purchase order (USD)
    service_level : target service level (e.g. 0.95 = 95th percentile)
    lead_time_days : average supplier lead time in days

    Example
    -------
    >>> opt = InventoryOptimizer(po_df, products_df)
    >>> params = opt.compute_all()
    >>> params[["product_id", "eoq", "safety_stock", "reorder_point"]].head()
    """

    def __init__(
        self,
        purchase_orders: pd.DataFrame,
        products: pd.DataFrame,
        holding_cost_rate: float = 0.25,
        ordering_cost: float = 150.0,
        service_level: float = 0.95,
        lead_time_days: int = 14,
    ) -> None:
        self.po = purchase_orders.copy()
        self.products = products.copy()
        self.holding_cost_rate = holding_cost_rate
        self.ordering_cost = ordering_cost
        self.service_level = service_level
        self.lead_time_days = lead_time_days

        self.po["order_date"] = pd.to_datetime(self.po["order_date"], errors="coerce")
        self._z_score = float(stats.norm.ppf(service_level))  # e.g. 1.645 for 95%

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute_all(self) -> pd.DataFrame:
        """
        Compute EOQ, ROP, safety stock, and ABC class for all products.

        Returns
        -------
        DataFrame with one row per product, all inventory parameters populated.
        """
        demand_stats = self._compute_demand_stats()
        params = self.products.merge(demand_stats, on="product_id", how="left")

        # Fill missing demand with 0
        for col in ["avg_daily_demand", "demand_std_daily", "annual_demand", "total_po_count"]:
            if col in params.columns:
                params[col] = params[col].fillna(0)

        # EOQ
        params["eoq"] = params.apply(
            lambda r: self._eoq(r["annual_demand"], r["unit_cost"]), axis=1
        ).round(0).astype(int)

        # Safety stock
        params["safety_stock"] = params.apply(
            lambda r: self._safety_stock(r["demand_std_daily"]), axis=1
        ).round(0).astype(int)

        # Reorder point
        params["reorder_point"] = params.apply(
            lambda r: self._reorder_point(r["avg_daily_demand"], r["safety_stock"]), axis=1
        ).round(0).astype(int)

        # Max stock
        params["max_stock"] = params["reorder_point"] + params["eoq"]

        # Annual holding cost
        params["annual_holding_cost"] = (
            (params["eoq"] / 2) * params["unit_cost"] * self.holding_cost_rate
        ).round(2)

        # Annual ordering cost
        params["annual_ordering_cost"] = params.apply(
            lambda r: (r["annual_demand"] / max(r["eoq"], 1)) * self.ordering_cost, axis=1
        ).round(2)

        # Total annual inventory cost
        params["total_annual_cost"] = (
            params["annual_holding_cost"] + params["annual_ordering_cost"]
        ).round(2)

        # Days of supply (reorder point / avg daily demand)
        params["days_of_supply"] = np.where(
            params["avg_daily_demand"] > 0,
            params["reorder_point"] / params["avg_daily_demand"],
            0,
        ).round(1)

        # Inventory turnover
        params["inventory_turnover"] = np.where(
            params["eoq"] > 0,
            params["annual_demand"] / (params["eoq"] / 2),
            0,
        ).round(2)

        # Reinforce ABC from spend data
        if "abc_class" not in params.columns:
            params["abc_class"] = self._classify_abc(params)

        logger.info(
            "Inventory parameters computed: %d SKUs | EOQ avg=%.0f | Safety stock avg=%.0f",
            len(params),
            params["eoq"].mean(),
            params["safety_stock"].mean(),
        )
        return params

    # ------------------------------------------------------------------
    # Demand statistics
    # ------------------------------------------------------------------

    def _compute_demand_stats(self) -> pd.DataFrame:
        """
        Compute per-product daily demand statistics from purchase order history.

        Returns
        -------
        DataFrame with product_id, avg_daily_demand, demand_std_daily, annual_demand
        """
        if self.po.empty:
            return pd.DataFrame(columns=["product_id", "avg_daily_demand", "demand_std_daily", "annual_demand"])

        # Aggregate to daily demand per product
        daily = (
            self.po.groupby(["product_id", self.po["order_date"].dt.date], observed=True)["quantity"]
            .sum()
            .reset_index()
            .rename(columns={"order_date": "date"})
        )

        stats_df = daily.groupby("product_id", observed=True).agg(
            avg_daily_demand=("quantity", "mean"),
            demand_std_daily=("quantity", "std"),
            total_po_count=("quantity", "count"),
        ).reset_index()

        # Annual demand = avg daily × 365
        stats_df["annual_demand"] = (stats_df["avg_daily_demand"] * 365).round(0)
        stats_df["demand_std_daily"] = stats_df["demand_std_daily"].fillna(0)
        return stats_df

    # ------------------------------------------------------------------
    # Formula implementations
    # ------------------------------------------------------------------

    def _eoq(self, annual_demand: float, unit_cost: float) -> float:
        """
        Wilson EOQ formula: Q* = sqrt(2 × D × K / h)

        Parameters
        ----------
        annual_demand : D — annual units demanded
        unit_cost     : product unit cost (USD)

        Returns
        -------
        Economic order quantity in units
        """
        if annual_demand <= 0 or unit_cost <= 0:
            return 0.0
        h = unit_cost * self.holding_cost_rate  # annual holding cost per unit
        return float(np.sqrt((2 * annual_demand * self.ordering_cost) / h))

    def _safety_stock(self, demand_std_daily: float) -> float:
        """
        Safety stock = Z × σ_demand × sqrt(lead_time)

        Accounts for demand variability during the replenishment lead time.
        """
        return float(self._z_score * demand_std_daily * np.sqrt(self.lead_time_days))

    def _reorder_point(self, avg_daily_demand: float, safety_stock: float) -> float:
        """
        Reorder point = average demand during lead time + safety stock.
        """
        return avg_daily_demand * self.lead_time_days + safety_stock

    # ------------------------------------------------------------------
    # ABC classification
    # ------------------------------------------------------------------

    def _classify_abc(self, params: pd.DataFrame) -> pd.Series:
        """
        Classify products into ABC tiers based on annual spend (demand × cost).

        A = top 20% by count driving 80% of spend
        B = next 30%
        C = remaining 50%
        """
        params = params.copy()
        params["annual_spend"] = params["annual_demand"] * params["unit_cost"]
        params = params.sort_values("annual_spend", ascending=False)
        total_spend = params["annual_spend"].sum()
        params["cumulative_pct"] = params["annual_spend"].cumsum() / total_spend

        conditions = [
            params["cumulative_pct"] <= 0.80,
            (params["cumulative_pct"] > 0.80) & (params["cumulative_pct"] <= 0.95),
        ]
        return pd.Series(
            np.select(conditions, ["A", "B"], default="C"),
            index=params.index,
        )

    # ------------------------------------------------------------------
    # KPI summary
    # ------------------------------------------------------------------

    def kpi_summary(self, params: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Return high-level inventory KPIs.

        Parameters
        ----------
        params : pre-computed DataFrame from compute_all(); computed internally if None
        """
        if params is None:
            params = self.compute_all()

        return {
            "total_skus": len(params),
            "abc_a_count": int((params["abc_class"] == "A").sum()),
            "abc_b_count": int((params["abc_class"] == "B").sum()),
            "abc_c_count": int((params["abc_class"] == "C").sum()),
            "avg_eoq": round(float(params["eoq"].mean()), 1),
            "avg_safety_stock": round(float(params["safety_stock"].mean()), 1),
            "total_annual_inv_cost_usd": round(float(params["total_annual_cost"].sum()), 2),
            "avg_inventory_turnover": round(float(params["inventory_turnover"].mean()), 2),
            "avg_days_of_supply": round(float(params["days_of_supply"].mean()), 1),
            "service_level_target": self.service_level,
        }

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def eoq_sensitivity(
        self, product_id: str, demand_range: tuple[float, float] = (0.5, 2.0), n_points: int = 50
    ) -> pd.DataFrame:
        """
        Compute EOQ and total cost across a range of demand multipliers.

        Useful for sensitivity analysis: "how does EOQ change if demand grows 2×?"
        """
        product = self.products[self.products["product_id"] == product_id]
        if product.empty:
            return pd.DataFrame()

        unit_cost = float(product["unit_cost"].iloc[0])
        base_demand = float(
            self.po[self.po["product_id"] == product_id]["quantity"].sum()
        )

        multipliers = np.linspace(*demand_range, n_points)
        rows: list[dict[str, float]] = []
        for m in multipliers:
            annual_demand = base_demand * m
            eoq = self._eoq(annual_demand, unit_cost)
            h = unit_cost * self.holding_cost_rate
            if eoq > 0:
                holding = (eoq / 2) * h
                ordering = (annual_demand / eoq) * self.ordering_cost
                total = holding + ordering
            else:
                holding = ordering = total = 0.0
            rows.append({
                "demand_multiplier": m,
                "annual_demand": annual_demand,
                "eoq": eoq,
                "annual_holding_cost": holding,
                "annual_ordering_cost": ordering,
                "total_cost": total,
            })
        return pd.DataFrame(rows)

