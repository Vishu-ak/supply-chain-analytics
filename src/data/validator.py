"""
Data quality validation for procurement datasets.

Implements schema checks, referential integrity verification,
business rule validation, and a summary quality report.
Designed to integrate into the ETL pipeline before analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A single data quality finding."""

    severity: str  # 'ERROR' | 'WARNING' | 'INFO'
    table: str
    column: str | None
    rule: str
    message: str
    affected_rows: int = 0

    def __str__(self) -> str:
        col_part = f".{self.column}" if self.column else ""
        return f"[{self.severity}] {self.table}{col_part} — {self.rule}: {self.message}"


@dataclass
class ValidationReport:
    """Aggregated results from a validation run."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def passed(self) -> bool:
        """Returns True if no ERROR-level issues exist."""
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Validation {'PASSED' if self.passed else 'FAILED'}",
            f"  Errors:   {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export issues to a tidy DataFrame for reporting."""
        return pd.DataFrame(
            [
                {
                    "severity": i.severity,
                    "table": i.table,
                    "column": i.column or "",
                    "rule": i.rule,
                    "message": i.message,
                    "affected_rows": i.affected_rows,
                }
                for i in self.issues
            ]
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Validates procurement DataFrames against schema, business rules,
    and referential integrity constraints.

    Example
    -------
    >>> validator = DataValidator()
    >>> report = validator.validate_all(purchase_orders=po_df, vendors=vendors_df)
    >>> print(report.summary())
    """

    # Expected columns with nullable flag
    _REQUIRED_COLUMNS: dict[str, dict[str, bool]] = {
        "purchase_orders": {
            "po_id": False,
            "vendor_id": False,
            "product_id": False,
            "order_date": False,
            "quantity": False,
            "unit_price": False,
            "total_amount": False,
            "status": False,
        },
        "vendors": {
            "vendor_id": False,
            "vendor_name": False,
            "tier": False,
            "active": False,
        },
        "products": {
            "product_id": False,
            "unit_cost": False,
            "abc_class": False,
        },
        "deliveries": {
            "delivery_id": False,
            "po_id": False,
            "vendor_id": False,
            "quantity_delivered": False,
            "on_time": False,
        },
        "quality_inspections": {
            "inspection_id": False,
            "delivery_id": False,
            "defects_found": False,
            "defect_rate": False,
        },
    }

    def validate_all(self, **tables: pd.DataFrame) -> ValidationReport:
        """
        Run all validations across supplied tables.

        Parameters
        ----------
        **tables : keyword args mapping table_name → DataFrame

        Returns
        -------
        ValidationReport with all findings
        """
        report = ValidationReport()

        for name, df in tables.items():
            if df is None or df.empty:
                report.issues.append(
                    ValidationIssue("WARNING", name, None, "empty_table",
                                    f"Table '{name}' is empty or None")
                )
                continue

            report.issues.extend(self._check_schema(name, df))
            report.issues.extend(self._check_nulls(name, df))
            report.issues.extend(self._check_duplicates(name, df))
            report.issues.extend(self._check_value_ranges(name, df))

        # Cross-table referential integrity
        if "purchase_orders" in tables and "vendors" in tables:
            report.issues.extend(
                self._check_foreign_key(
                    tables["purchase_orders"], "vendor_id",
                    tables["vendors"], "vendor_id",
                    "purchase_orders", "vendors",
                )
            )
        if "deliveries" in tables and "purchase_orders" in tables:
            report.issues.extend(
                self._check_foreign_key(
                    tables["deliveries"], "po_id",
                    tables["purchase_orders"], "po_id",
                    "deliveries", "purchase_orders",
                )
            )

        # Business rules
        if "purchase_orders" in tables:
            report.issues.extend(self._check_po_business_rules(tables["purchase_orders"]))
        if "quality_inspections" in tables:
            report.issues.extend(self._check_inspection_rules(tables["quality_inspections"]))

        logger.info("Validation complete: %d errors, %d warnings",
                    len(report.errors), len(report.warnings))
        return report

    # ------------------------------------------------------------------
    # Schema checks
    # ------------------------------------------------------------------

    def _check_schema(self, table_name: str, df: pd.DataFrame) -> list[ValidationIssue]:
        """Verify expected columns are present."""
        issues: list[ValidationIssue] = []
        required = self._REQUIRED_COLUMNS.get(table_name, {})
        missing = [col for col in required if col not in df.columns]
        if missing:
            issues.append(ValidationIssue(
                "ERROR", table_name, None, "missing_columns",
                f"Missing required columns: {missing}",
            ))
        return issues

    # ------------------------------------------------------------------
    # Null checks
    # ------------------------------------------------------------------

    def _check_nulls(self, table_name: str, df: pd.DataFrame) -> list[ValidationIssue]:
        """Identify unexpected null values in required fields."""
        issues: list[ValidationIssue] = []
        required = {
            col for col, nullable in self._REQUIRED_COLUMNS.get(table_name, {}).items()
            if not nullable and col in df.columns
        }
        for col in required:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                pct = null_count / len(df) * 100
                severity = "ERROR" if pct > 5 else "WARNING"
                issues.append(ValidationIssue(
                    severity, table_name, col, "null_values",
                    f"{null_count} null values ({pct:.1f}%)",
                    affected_rows=null_count,
                ))
        return issues

    # ------------------------------------------------------------------
    # Duplicate checks
    # ------------------------------------------------------------------

    def _check_duplicates(self, table_name: str, df: pd.DataFrame) -> list[ValidationIssue]:
        """Check for duplicate primary keys."""
        issues: list[ValidationIssue] = []
        pk_map = {
            "purchase_orders": "po_id",
            "vendors": "vendor_id",
            "products": "product_id",
            "deliveries": "delivery_id",
            "quality_inspections": "inspection_id",
            "contracts": "contract_id",
        }
        pk = pk_map.get(table_name)
        if pk and pk in df.columns:
            dup_count = int(df[pk].duplicated().sum())
            if dup_count > 0:
                issues.append(ValidationIssue(
                    "ERROR", table_name, pk, "duplicate_pk",
                    f"{dup_count} duplicate primary key values",
                    affected_rows=dup_count,
                ))
        return issues

    # ------------------------------------------------------------------
    # Value range checks
    # ------------------------------------------------------------------

    def _check_value_ranges(self, table_name: str, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate numeric ranges and categorical allowlists."""
        issues: list[ValidationIssue] = []

        numeric_bounds: dict[str, tuple[float, float]] = {
            "quantity": (0, 1_000_000),
            "unit_price": (0, 1_000_000),
            "total_amount": (0, 100_000_000),
            "defect_rate": (0.0, 1.0),
            "unit_cost": (0, 1_000_000),
        }

        for col, (lo, hi) in numeric_bounds.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            out_of_range = int(((series < lo) | (series > hi)).sum())
            if out_of_range:
                issues.append(ValidationIssue(
                    "WARNING", table_name, col, "out_of_range",
                    f"{out_of_range} values outside [{lo}, {hi}]",
                    affected_rows=out_of_range,
                ))

        categorical_allowlists: dict[str, set[str]] = {
            "abc_class": {"A", "B", "C"},
            "tier": {"Tier 1", "Tier 2", "Tier 3"},
            "status": {"Open", "Received", "Partially Received", "Cancelled"},
        }
        for col, allowed in categorical_allowlists.items():
            if col not in df.columns:
                continue
            invalid_mask = ~df[col].isin(allowed) & df[col].notna()
            invalid_count = int(invalid_mask.sum())
            if invalid_count:
                issues.append(ValidationIssue(
                    "ERROR", table_name, col, "invalid_category",
                    f"{invalid_count} values not in allowed set {allowed}",
                    affected_rows=invalid_count,
                ))

        return issues

    # ------------------------------------------------------------------
    # Referential integrity
    # ------------------------------------------------------------------

    def _check_foreign_key(
        self,
        child: pd.DataFrame,
        fk_col: str,
        parent: pd.DataFrame,
        pk_col: str,
        child_name: str,
        parent_name: str,
    ) -> list[ValidationIssue]:
        """Detect FK violations (child values missing in parent)."""
        issues: list[ValidationIssue] = []
        if fk_col not in child.columns or pk_col not in parent.columns:
            return issues
        orphaned = ~child[fk_col].isin(parent[pk_col])
        orphan_count = int(orphaned.sum())
        if orphan_count:
            issues.append(ValidationIssue(
                "ERROR", child_name, fk_col, "fk_violation",
                f"{orphan_count} values not found in {parent_name}.{pk_col}",
                affected_rows=orphan_count,
            ))
        return issues

    # ------------------------------------------------------------------
    # Business rule checks
    # ------------------------------------------------------------------

    def _check_po_business_rules(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate procurement-specific business rules for POs."""
        issues: list[ValidationIssue] = []

        # Total amount should equal quantity × unit_price
        if all(c in df.columns for c in ["quantity", "unit_price", "total_amount"]):
            expected = df["quantity"] * df["unit_price"]
            tolerance = 0.02  # 2% tolerance for rounding
            mismatched = (np.abs(df["total_amount"] - expected) / expected.replace(0, 1) > tolerance).sum()
            if mismatched:
                issues.append(ValidationIssue(
                    "WARNING", "purchase_orders", "total_amount", "amount_mismatch",
                    f"{mismatched} rows where total_amount ≠ quantity × unit_price (>2% tolerance)",
                    affected_rows=int(mismatched),
                ))

        # Delivery date must be after order date
        if all(c in df.columns for c in ["order_date", "requested_delivery_date"]):
            order_dt = pd.to_datetime(df["order_date"], errors="coerce")
            delivery_dt = pd.to_datetime(df["requested_delivery_date"], errors="coerce")
            bad_dates = (delivery_dt < order_dt).sum()
            if bad_dates:
                issues.append(ValidationIssue(
                    "ERROR", "purchase_orders", "requested_delivery_date", "delivery_before_order",
                    f"{bad_dates} POs where delivery date precedes order date",
                    affected_rows=int(bad_dates),
                ))

        return issues

    def _check_inspection_rules(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate quality inspection business rules."""
        issues: list[ValidationIssue] = []
        if all(c in df.columns for c in ["defects_found", "quantity_inspected"]):
            exceeds = (df["defects_found"] > df["quantity_inspected"]).sum()
            if exceeds:
                issues.append(ValidationIssue(
                    "ERROR", "quality_inspections", "defects_found", "defects_exceed_qty",
                    f"{exceeds} rows where defects_found > quantity_inspected",
                    affected_rows=int(exceeds),
                ))
        return issues
