"""
KPI definition, computation, and tracking module.

Defines procurement KPI catalog, computes them from processed data,
and tracks period-over-period changes. Designed to feed both
the Streamlit dashboard and the PDF executive report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KPIDefinition:
    """Metadata for a single KPI."""

    key: str
    name: str
    description: str
    unit: str                  # e.g., '%', 'days', 'USD', 'ratio'
    target: float
    direction: str             # 'higher_better' | 'lower_better'
    category: str              # 'spend' | 'vendor' | 'inventory' | 'logistics'
    formula: str               # Human-readable formula


# ---------------------------------------------------------------------------
# KPI catalog
# ---------------------------------------------------------------------------
KPI_CATALOG: list[KPIDefinition] = [
    KPIDefinition(
        key="otd_rate",
        name="On-Time Delivery Rate",
        description="Percentage of deliveries received on or before the scheduled date.",
        unit="%",
        target=95.0,
        direction="higher_better",
        category="logistics",
        formula="(On-time deliveries / Total deliveries) × 100",
    ),
    KPIDefinition(
        key="supplier_defect_rate",
        name="Supplier Defect Rate",
        description="Percentage of inbound units failing quality inspection.",
        unit="%",
        target=1.0,
        direction="lower_better",
        category="vendor",
        formula="(Total defects / Total units inspected) × 100",
    ),
    KPIDefinition(
        key="procurement_cycle_time",
        name="Procurement Cycle Time",
        description="Average calendar days from PO creation to delivery.",
        unit="days",
        target=14.0,
        direction="lower_better",
        category="logistics",
        formula="Mean(delivery_date − order_date) across all completed POs",
    ),
    KPIDefinition(
        key="cost_savings_pct",
        name="Cost Savings %",
        description="Price achieved vs. category benchmark price (negative = savings).",
        unit="%",
        target=5.0,
        direction="higher_better",
        category="spend",
        formula="(Benchmark spend − Actual spend) / Benchmark spend × 100",
    ),
    KPIDefinition(
        key="inventory_turnover",
        name="Inventory Turnover",
        description="Number of times inventory is consumed and replenished annually.",
        unit="×",
        target=12.0,
        direction="higher_better",
        category="inventory",
        formula="Annual demand / Average inventory level (EOQ / 2)",
    ),
    KPIDefinition(
        key="fill_rate",
        name="Fill Rate",
        description="Percentage of order lines fulfilled completely on the first delivery.",
        unit="%",
        target=98.0,
        direction="higher_better",
        category="logistics",
        formula="(Fully fulfilled lines / Total order lines) × 100",
    ),
    KPIDefinition(
        key="maverick_spend_pct",
        name="Maverick Spend %",
        description="Percentage of total spend made outside contracted channels.",
        unit="%",
        target=5.0,
        direction="lower_better",
        category="spend",
        formula="(Off-contract spend / Total spend) × 100",
    ),
    KPIDefinition(
        key="vendor_concentration",
        name="Vendor Concentration (HHI)",
        description="Herfindahl-Hirschman Index measuring spend concentration risk.",
        unit="",
        target=0.10,
        direction="lower_better",
        category="vendor",
        formula="Σ (vendor_spend_share²)",
    ),
    KPIDefinition(
        key="po_cycle_time",
        name="PO Approval Cycle Time",
        description="Average business days from PO creation to approval.",
        unit="days",
        target=2.0,
        direction="lower_better",
        category="spend",
        formula="Mean time from PO draft to approved status",
    ),
    KPIDefinition(
        key="contract_coverage",
        name="Contract Coverage %",
        description="Percentage of total spend covered by active contracts.",
        unit="%",
        target=85.0,
        direction="higher_better",
        category="spend",
        formula="(Contracted spend / Total spend) × 100",
    ),
]


@dataclass
class KPIResult:
    """Computed value for a single KPI."""

    definition: KPIDefinition
    value: float
    period: str
    previous_value: float | None = None
    status: str = "unknown"  # 'on_target' | 'at_risk' | 'off_target'

    @property
    def vs_target(self) -> float:
        """Difference from target (positive = better for higher_better KPIs)."""
        if self.definition.direction == "higher_better":
            return self.value - self.definition.target
        return self.definition.target - self.value

    @property
    def achievement_pct(self) -> float:
        """Value as % of target."""
        if self.definition.target == 0:
            return 100.0
        return round(self.value / self.definition.target * 100, 1)

    @property
    def period_change(self) -> float | None:
        """Change from previous period."""
        if self.previous_value is None:
            return None
        return round(self.value - self.previous_value, 4)

    def compute_status(self) -> str:
        """Assign traffic-light status based on % of target achieved."""
        pct = self.achievement_pct
        if self.definition.direction == "higher_better":
            if pct >= 95:
                return "on_target"
            elif pct >= 80:
                return "at_risk"
            return "off_target"
        else:
            # Lower is better: achievement_pct > 100 means we're OVER the target (bad)
            if pct <= 105:
                return "on_target"
            elif pct <= 125:
                return "at_risk"
            return "off_target"


class KPITracker:
    """
    Computes and tracks procurement KPIs from processed data tables.

    Parameters
    ----------
    purchase_orders : PO transaction data
    deliveries : delivery records
    quality_inspections : QI records
    inventory_params : output from InventoryOptimizer.compute_all()
    contracts : contract master data

    Example
    -------
    >>> tracker = KPITracker(po_df, del_df, qi_df)
    >>> results = tracker.compute_all(period="Q4 2024")
    >>> df = tracker.to_dataframe(results)
    """

    def __init__(
        self,
        purchase_orders: pd.DataFrame,
        deliveries: pd.DataFrame | None = None,
        quality_inspections: pd.DataFrame | None = None,
        inventory_params: pd.DataFrame | None = None,
        contracts: pd.DataFrame | None = None,
    ) -> None:
        self.po = purchase_orders.copy()
        self.deliveries = deliveries.copy() if deliveries is not None else pd.DataFrame()
        self.qi = quality_inspections.copy() if quality_inspections is not None else pd.DataFrame()
        self.inventory_params = inventory_params.copy() if inventory_params is not None else pd.DataFrame()
        self.contracts = contracts.copy() if contracts is not None else pd.DataFrame()

        self._catalog = {kpi.key: kpi for kpi in KPI_CATALOG}

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute_all(
        self,
        period: str = "Current",
        previous_results: dict[str, KPIResult] | None = None,
    ) -> dict[str, KPIResult]:
        """
        Compute all KPIs for a given period.

        Parameters
        ----------
        period : descriptive label (e.g., 'Q4 2024', 'FY 2024')
        previous_results : prior period results for change calculation

        Returns
        -------
        Dict mapping KPI key → KPIResult
        """
        computations = {
            "otd_rate": self._compute_otd_rate,
            "supplier_defect_rate": self._compute_defect_rate,
            "procurement_cycle_time": self._compute_cycle_time,
            "cost_savings_pct": self._compute_cost_savings,
            "inventory_turnover": self._compute_inventory_turnover,
            "fill_rate": self._compute_fill_rate,
            "maverick_spend_pct": self._compute_maverick_spend,
            "vendor_concentration": self._compute_vendor_concentration,
            "contract_coverage": self._compute_contract_coverage,
        }

        results: dict[str, KPIResult] = {}
        for key, fn in computations.items():
            try:
                value = fn()
                prev = previous_results.get(key).value if previous_results and key in previous_results else None
                result = KPIResult(
                    definition=self._catalog[key],
                    value=round(value, 4),
                    period=period,
                    previous_value=prev,
                )
                result.status = result.compute_status()
                results[key] = result
            except Exception as exc:
                logger.warning("Failed to compute KPI '%s': %s", key, exc)

        on_target = sum(1 for r in results.values() if r.status == "on_target")
        logger.info("KPI computation complete: %d/%d on target", on_target, len(results))
        return results

    def to_dataframe(self, results: dict[str, KPIResult]) -> pd.DataFrame:
        """Convert KPI results to a tidy DataFrame for display."""
        rows = []
        for key, result in results.items():
            rows.append({
                "kpi": result.definition.name,
                "category": result.definition.category,
                "value": result.value,
                "unit": result.definition.unit,
                "target": result.definition.target,
                "achievement_pct": result.achievement_pct,
                "vs_target": result.vs_target,
                "status": result.status,
                "period_change": result.period_change,
                "period": result.period,
                "description": result.definition.description,
            })
        return pd.DataFrame(rows).sort_values(["category", "kpi"])

    # ------------------------------------------------------------------
    # Individual KPI computations
    # ------------------------------------------------------------------

    def _compute_otd_rate(self) -> float:
        if self.deliveries.empty or "on_time" not in self.deliveries.columns:
            return 0.0
        return float(self.deliveries["on_time"].mean() * 100)

    def _compute_defect_rate(self) -> float:
        if self.qi.empty:
            return 0.0
        total_inspected = self.qi["quantity_inspected"].sum() if "quantity_inspected" in self.qi.columns else 1
        total_defects = self.qi["defects_found"].sum() if "defects_found" in self.qi.columns else 0
        return float(total_defects / max(total_inspected, 1) * 100)

    def _compute_cycle_time(self) -> float:
        if self.deliveries.empty or self.po.empty:
            return 0.0
        merged = self.deliveries.merge(
            self.po[["po_id", "order_date"]].drop_duplicates(), on="po_id", how="left"
        )
        merged["order_date"] = pd.to_datetime(merged["order_date"], errors="coerce")
        merged["actual_date"] = pd.to_datetime(merged["actual_date"], errors="coerce")
        cycle = (merged["actual_date"] - merged["order_date"]).dt.days
        return float(cycle.dropna().mean())

    def _compute_cost_savings(self) -> float:
        if self.po.empty or "unit_price" not in self.po.columns:
            return 0.0
        if "category" not in self.po.columns:
            return 0.0
        benchmark = self.po.groupby("category", observed=True)["unit_price"].transform("median")
        savings = (benchmark - self.po["unit_price"]) / benchmark.replace(0, np.nan)
        return float(savings.mean() * 100)

    def _compute_inventory_turnover(self) -> float:
        if self.inventory_params.empty or "inventory_turnover" not in self.inventory_params.columns:
            return 0.0
        return float(self.inventory_params["inventory_turnover"].mean())

    def _compute_fill_rate(self) -> float:
        if self.deliveries.empty:
            return 0.0
        if "quantity_delivered" not in self.deliveries.columns:
            return 0.0
        merged = self.deliveries.merge(
            self.po[["po_id", "quantity"]].drop_duplicates(), on="po_id", how="left"
        )
        fully_filled = (merged["quantity_delivered"] >= merged["quantity"]).mean()
        return float(fully_filled * 100)

    def _compute_maverick_spend(self) -> float:
        if self.po.empty:
            return 0.0
        emergency_spend = self.po[self.po.get("is_emergency", pd.Series([False] * len(self.po))) == True]["total_amount"].sum() \
            if "is_emergency" in self.po.columns else 0
        total_spend = self.po["total_amount"].sum()
        return float(emergency_spend / max(total_spend, 1) * 100)

    def _compute_vendor_concentration(self) -> float:
        if self.po.empty:
            return 0.0
        shares = self.po.groupby("vendor_id", observed=True)["total_amount"].sum()
        shares = shares / shares.sum()
        return float((shares ** 2).sum())

    def _compute_contract_coverage(self) -> float:
        if self.contracts.empty or self.po.empty:
            return 0.0
        if "vendor_id" not in self.contracts.columns:
            return 0.0
        contracted_vendors = set(self.contracts["vendor_id"])
        contracted_spend = self.po[self.po["vendor_id"].isin(contracted_vendors)]["total_amount"].sum()
        total_spend = self.po["total_amount"].sum()
        return float(contracted_spend / max(total_spend, 1) * 100)
