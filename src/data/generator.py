"""
Realistic procurement and supply chain data generator.

Produces statistically coherent synthetic datasets that mirror
real-world procurement patterns:
  - Purchase orders with seasonal demand fluctuations
  - Vendor performance tiers (Tier 1 / 2 / 3) with realistic score distributions
  - Product ABC classification pre-baked into generated unit costs and volumes
  - Delivery records with controlled on-time/late ratios per vendor tier
  - Quality inspections with defect counts drawn from tier-specific distributions
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
PRODUCT_CATEGORIES: list[str] = [
    "Raw Materials", "Packaging", "Electronics", "MRO",
    "Office Supplies", "Logistics", "Services", "IT Hardware",
    "Chemicals", "Finished Goods",
]

VENDOR_TIERS: list[str] = ["Tier 1", "Tier 2", "Tier 3"]

UNITS_OF_MEASURE: list[str] = ["EA", "KG", "L", "M", "BOX", "PALLET", "ROLL", "SET"]

US_STATES: list[str] = [
    "CA", "TX", "NY", "FL", "IL", "WA", "OH", "GA", "NC", "MI",
    "PA", "AZ", "CO", "TN", "MN", "MO", "WI", "OR", "NV", "IN",
]

PAYMENT_TERMS: list[str] = ["Net 30", "Net 45", "Net 60", "Net 90", "2/10 Net 30"]


@dataclass
class GeneratedDataset:
    """Container for all generated tables."""

    vendors: pd.DataFrame
    products: pd.DataFrame
    purchase_orders: pd.DataFrame
    deliveries: pd.DataFrame
    quality_inspections: pd.DataFrame
    contracts: pd.DataFrame


class ProcurementDataGenerator:
    """
    Generates realistic multi-table procurement datasets for analysis and demo purposes.

    Example
    -------
    >>> gen = ProcurementDataGenerator(n_vendors=200, n_products=1000, n_pos=50_000, seed=42)
    >>> ds = gen.generate_all()
    >>> ds.purchase_orders.shape[0]
    50000
    """

    def __init__(
        self,
        n_vendors: int | None = None,
        n_products: int | None = None,
        n_pos: int | None = None,
        seed: int | None = None,
    ) -> None:
        cfg = settings.data
        self.n_vendors = n_vendors or cfg.n_vendors
        self.n_products = n_products or cfg.n_products
        self.n_pos = n_pos or cfg.n_purchase_orders
        self.seed = seed if seed is not None else cfg.random_seed

        self._rng = np.random.default_rng(self.seed)
        random.seed(self.seed)

        self._date_start = pd.to_datetime(cfg.date_start)
        self._date_end = pd.to_datetime(cfg.date_end)
        self._date_range_days = (self._date_end - self._date_start).days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> GeneratedDataset:
        """Generate the full dataset suite and return a GeneratedDataset."""
        vendors = self.generate_vendors()
        products = self.generate_products()
        purchase_orders = self.generate_purchase_orders(vendors, products)
        deliveries = self.generate_deliveries(purchase_orders, vendors)
        quality_inspections = self.generate_quality_inspections(deliveries, products)
        contracts = self.generate_contracts(vendors, products)
        return GeneratedDataset(
            vendors=vendors,
            products=products,
            purchase_orders=purchase_orders,
            deliveries=deliveries,
            quality_inspections=quality_inspections,
            contracts=contracts,
        )

    def generate_vendors(self) -> pd.DataFrame:
        """
        Generate a vendor master table.

        Columns: vendor_id, vendor_name, tier, category, state, country,
                 payment_terms, lead_time_days_avg, active, onboarded_date
        """
        n = self.n_vendors
        tier_probs = [0.20, 0.45, 0.35]  # Tier 1 / 2 / 3 proportions

        tiers = self._rng.choice(VENDOR_TIERS, size=n, p=tier_probs)
        categories = self._rng.choice(PRODUCT_CATEGORIES, size=n)

        # Lead times: Tier 1 faster, Tier 3 slower
        lead_time_mu = {"Tier 1": 7, "Tier 2": 14, "Tier 3": 21}
        lead_times = np.array([
            max(1, int(self._rng.normal(lead_time_mu[t], 3))) for t in tiers
        ])

        onboard_offsets = self._rng.integers(0, 365 * 5, size=n)
        onboarded = [
            (self._date_start - timedelta(days=int(d))).date() for d in onboard_offsets
        ]

        df = pd.DataFrame(
            {
                "vendor_id": [f"V{i:05d}" for i in range(1, n + 1)],
                "vendor_name": [f"{self._random_company_name()} {categories[i]}" for i in range(n)],
                "tier": tiers,
                "category": categories,
                "state": self._rng.choice(US_STATES, size=n),
                "country": ["USA"] * n,
                "payment_terms": self._rng.choice(PAYMENT_TERMS, size=n),
                "lead_time_days_avg": lead_times,
                "active": self._rng.choice([True, False], size=n, p=[0.92, 0.08]),
                "onboarded_date": onboarded,
            }
        )
        return df

    def generate_products(self) -> pd.DataFrame:
        """
        Generate a product/SKU catalogue.

        Columns: product_id, sku, description, category, unit_of_measure,
                 unit_cost, abc_class, reorder_point, eoq_units
        """
        n = self.n_products
        categories = self._rng.choice(PRODUCT_CATEGORIES, size=n)

        # ABC classification: A ~20%, B ~30%, C ~50%
        abc_classes = self._rng.choice(["A", "B", "C"], size=n, p=[0.20, 0.30, 0.50])

        # Unit cost driven by ABC class (A items most valuable)
        cost_params = {"A": (500, 200), "B": (100, 50), "C": (20, 15)}
        unit_costs = np.array([
            max(0.01, self._rng.normal(*cost_params[c])) for c in abc_classes
        ])

        # Reorder points (higher for A items)
        rop_params = {"A": (50, 10), "B": (100, 20), "C": (200, 50)}
        reorder_points = np.array([
            max(1, int(self._rng.normal(*rop_params[c]))) for c in abc_classes
        ])

        df = pd.DataFrame(
            {
                "product_id": [f"P{i:06d}" for i in range(1, n + 1)],
                "sku": [f"SKU-{i:06d}" for i in range(1, n + 1)],
                "description": [f"{categories[i]} Component #{i:04d}" for i in range(n)],
                "category": categories,
                "unit_of_measure": self._rng.choice(UNITS_OF_MEASURE, size=n),
                "unit_cost": np.round(unit_costs, 2),
                "abc_class": abc_classes,
                "reorder_point": reorder_points,
            }
        )
        return df

    def generate_purchase_orders(
        self, vendors: pd.DataFrame, products: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate purchase order lines with realistic seasonal patterns.

        Columns: po_id, po_line, vendor_id, product_id, order_date,
                 requested_delivery_date, quantity, unit_price, total_amount,
                 status, category, is_emergency
        """
        n = self.n_pos
        vendor_ids = vendors["vendor_id"].values
        product_ids = products["product_id"].values
        product_lookup = products.set_index("product_id")[["unit_cost", "category", "abc_class"]]

        # Sample dates with seasonal weighting (Q4 peaks)
        days_offsets = self._rng.integers(0, self._date_range_days, size=n)
        order_dates = [self._date_start + timedelta(days=int(d)) for d in days_offsets]

        selected_vendors = self._rng.choice(vendor_ids, size=n)
        selected_products = self._rng.choice(product_ids, size=n)

        unit_costs = product_lookup.loc[selected_products, "unit_cost"].values
        categories = product_lookup.loc[selected_products, "category"].values
        abc_classes = product_lookup.loc[selected_products, "abc_class"].values

        # Quantity: A items ordered in smaller batches, C items in bulk
        qty_params = {"A": (5, 2), "B": (20, 8), "C": (100, 30)}
        quantities = np.array([
            max(1, int(self._rng.normal(*qty_params[c]))) for c in abc_classes
        ])

        # Price variance ±10% from unit cost
        price_variance = self._rng.uniform(0.90, 1.10, size=n)
        unit_prices = np.round(unit_costs * price_variance, 2)
        total_amounts = np.round(quantities * unit_prices, 2)

        # Requested delivery = order date + 7–45 days
        delivery_offsets = self._rng.integers(7, 46, size=n)
        requested_delivery = [
            (order_dates[i] + timedelta(days=int(delivery_offsets[i]))).date()
            for i in range(n)
        ]

        statuses = self._rng.choice(
            ["Open", "Received", "Partially Received", "Cancelled"],
            size=n,
            p=[0.05, 0.85, 0.07, 0.03],
        )
        is_emergency = self._rng.choice([True, False], size=n, p=[0.08, 0.92])

        df = pd.DataFrame(
            {
                "po_id": [f"PO-{i:07d}" for i in range(1, n + 1)],
                "po_line": self._rng.integers(1, 6, size=n),
                "vendor_id": selected_vendors,
                "product_id": selected_products,
                "order_date": [d.date() if hasattr(d, "date") else d for d in order_dates],
                "requested_delivery_date": requested_delivery,
                "quantity": quantities,
                "unit_price": unit_prices,
                "total_amount": total_amounts,
                "status": statuses,
                "category": categories,
                "abc_class": abc_classes,
                "is_emergency": is_emergency,
            }
        )
        return df

    def generate_deliveries(
        self, purchase_orders: pd.DataFrame, vendors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate delivery records aligned with completed purchase orders.

        Columns: delivery_id, po_id, vendor_id, scheduled_date, actual_date,
                 quantity_delivered, on_time, days_early_late, carrier
        """
        completed = purchase_orders[purchase_orders["status"] == "Received"].copy()
        n = len(completed)

        vendor_tier_lookup = vendors.set_index("vendor_id")["tier"]
        tiers = completed["vendor_id"].map(vendor_tier_lookup).values

        # On-time rate by tier
        on_time_rate = {"Tier 1": 0.95, "Tier 2": 0.82, "Tier 3": 0.65}
        on_time = np.array([
            self._rng.random() < on_time_rate.get(t, 0.80) for t in tiers
        ])

        # Days early (-) or late (+)
        delay_params = {"Tier 1": (0, 2), "Tier 2": (1, 4), "Tier 3": (3, 7)}
        days_delta = np.array([
            int(self._rng.normal(*delay_params.get(t, (1, 3)))) if not ot else
            int(self._rng.normal(-1, 1))
            for t, ot in zip(tiers, on_time)
        ])

        actual_dates = [
            (pd.to_datetime(rd) + timedelta(days=int(dd))).date()
            for rd, dd in zip(completed["requested_delivery_date"], days_delta)
        ]

        carriers = self._rng.choice(
            ["FedEx", "UPS", "DHL", "XPO", "J.B. Hunt", "In-House"], size=n
        )

        df = pd.DataFrame(
            {
                "delivery_id": [f"DEL-{i:07d}" for i in range(1, n + 1)],
                "po_id": completed["po_id"].values,
                "vendor_id": completed["vendor_id"].values,
                "scheduled_date": completed["requested_delivery_date"].values,
                "actual_date": actual_dates,
                "quantity_delivered": completed["quantity"].values,
                "on_time": on_time,
                "days_early_late": days_delta,
                "carrier": carriers,
            }
        )
        return df

    def generate_quality_inspections(
        self, deliveries: pd.DataFrame, products: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate quality inspection records for each delivery.

        Columns: inspection_id, delivery_id, po_id, inspection_date,
                 inspector_id, quantity_inspected, defects_found, defect_rate,
                 passed, rejection_reason
        """
        n = len(deliveries)

        # Defect rate: 0–3% for A items, up to 5% for C items
        defect_means = self._rng.uniform(0.005, 0.04, size=n)
        defect_counts = np.array([
            int(self._rng.binomial(qty, dm))
            for qty, dm in zip(deliveries["quantity_delivered"].values, defect_means)
        ])
        defect_rates = np.where(
            deliveries["quantity_delivered"].values > 0,
            defect_counts / deliveries["quantity_delivered"].values,
            0.0,
        )

        passed = defect_rates < 0.03  # >3% defect rate triggers rejection

        rejection_reasons = [
            self._rng.choice(["Dimensional non-conformance", "Surface defects",
                               "Material composition failure", "Labeling errors",
                               "Packaging damage", None], p=[0.15, 0.15, 0.10, 0.05, 0.10, 0.45])
            if not p else None
            for p in passed
        ]

        inspection_dates = [
            (pd.to_datetime(d) + timedelta(days=1)).date()
            for d in deliveries["actual_date"].values
        ]

        df = pd.DataFrame(
            {
                "inspection_id": [f"INS-{i:07d}" for i in range(1, n + 1)],
                "delivery_id": deliveries["delivery_id"].values,
                "po_id": deliveries["po_id"].values,
                "inspection_date": inspection_dates,
                "inspector_id": [f"EMP-{self._rng.integers(100, 999)}" for _ in range(n)],
                "quantity_inspected": deliveries["quantity_delivered"].values,
                "defects_found": defect_counts,
                "defect_rate": np.round(defect_rates, 4),
                "passed": passed,
                "rejection_reason": rejection_reasons,
            }
        )
        return df

    def generate_contracts(
        self, vendors: pd.DataFrame, products: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate procurement contracts for active vendor-category pairs.

        Columns: contract_id, vendor_id, category, start_date, end_date,
                 contracted_volume, contracted_price_per_unit, auto_renew, status
        """
        active_vendors = vendors[vendors["active"]].copy()
        n_contracts = min(len(active_vendors), 300)
        sampled = active_vendors.sample(n=n_contracts, random_state=self.seed)

        start_offsets = self._rng.integers(0, 365 * 2, size=n_contracts)
        start_dates = [
            (self._date_start - timedelta(days=int(d))).date() for d in start_offsets
        ]
        durations = self._rng.choice([365, 730, 1095], size=n_contracts)
        end_dates = [
            (pd.to_datetime(s) + timedelta(days=int(d))).date()
            for s, d in zip(start_dates, durations)
        ]

        today = date.today()
        statuses = [
            "Active" if e > today else "Expired" for e in end_dates
        ]

        df = pd.DataFrame(
            {
                "contract_id": [f"CTR-{i:05d}" for i in range(1, n_contracts + 1)],
                "vendor_id": sampled["vendor_id"].values,
                "category": sampled["category"].values,
                "start_date": start_dates,
                "end_date": end_dates,
                "contracted_volume": self._rng.integers(1000, 100_000, size=n_contracts),
                "contracted_price_per_unit": np.round(
                    self._rng.uniform(5.0, 500.0, size=n_contracts), 2
                ),
                "auto_renew": self._rng.choice([True, False], size=n_contracts, p=[0.6, 0.4]),
                "status": statuses,
            }
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _ADJECTIVES = [
        "Global", "National", "Premier", "Pacific", "Atlantic", "Allied",
        "United", "Central", "Summit", "Apex", "Delta", "Prime",
    ]
    _NOUNS = [
        "Supply", "Logistics", "Industries", "Solutions", "Procurement",
        "Trading", "Distribution", "Manufacturing", "Sourcing", "Partners",
    ]

    def _random_company_name(self) -> str:
        adj = random.choice(self._ADJECTIVES)
        noun = random.choice(self._NOUNS)
        return f"{adj} {noun}"
