"""
ETL pipeline for procurement data processing.

Orchestrates the full data flow:
  Extract  → load raw data from files or generated datasets
  Transform → clean, enrich, type-cast, and derive features
  Load     → persist processed tables to Parquet / SQLite

Designed to run in batch mode (scheduled) or interactively for notebooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ETLRunResult:
    """Summary of an ETL pipeline run."""

    status: str  # 'success' | 'partial' | 'failed'
    tables_processed: list[str] = field(default_factory=list)
    rows_processed: dict[str, int] = field(default_factory=dict)
    validation_issues: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"ETL Run: {self.status.upper()} ({self.duration_seconds:.1f}s)",
            f"  Tables: {', '.join(self.tables_processed)}",
            f"  Rows: {self.rows_processed}",
            f"  Validation issues: {self.validation_issues}",
        ]
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)


class ProcurementETL:
    """
    End-to-end ETL pipeline for procurement data.

    Supports two data sources:
      1. Generated (synthetic) data via ProcurementDataGenerator
      2. File-based (CSV/Parquet) data via DataLoader

    Parameters
    ----------
    output_dir : directory to write processed Parquet files
    use_generated_data : if True, generate synthetic data instead of loading files
    raw_data_dir : directory containing raw CSV/Parquet files (if use_generated_data=False)

    Example
    -------
    >>> etl = ProcurementETL(use_generated_data=True)
    >>> result = etl.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        use_generated_data: bool = True,
        raw_data_dir: Path | str | None = None,
    ) -> None:
        from config.settings import settings

        self.output_dir = Path(output_dir) if output_dir else settings.data.processed_dir
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else settings.data.raw_dir
        self.use_generated_data = use_generated_data

        self._tables: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        validate: bool = True,
        save_output: bool = True,
        fmt: str = "parquet",
    ) -> ETLRunResult:
        """
        Execute the full ETL pipeline.

        Parameters
        ----------
        validate : run DataValidator on extracted data
        save_output : persist processed tables to output_dir
        fmt : output format ('parquet' | 'csv')

        Returns
        -------
        ETLRunResult with status and metrics
        """
        start_time = time.time()
        result = ETLRunResult(status="success")

        # ------------ EXTRACT ------------
        logger.info("ETL: Starting EXTRACT phase")
        try:
            self._tables = self._extract()
            result.tables_processed = list(self._tables.keys())
            result.rows_processed = {k: len(v) for k, v in self._tables.items()}
            logger.info("ETL EXTRACT complete: %s", result.rows_processed)
        except Exception as exc:
            result.status = "failed"
            result.errors.append(f"Extract failed: {exc}")
            result.duration_seconds = time.time() - start_time
            logger.error("ETL Extract failed: %s", exc, exc_info=True)
            return result

        # ------------ VALIDATE ------------
        if validate:
            logger.info("ETL: Starting VALIDATE phase")
            try:
                n_issues = self._validate()
                result.validation_issues = n_issues
            except Exception as exc:
                logger.warning("ETL Validation error (non-fatal): %s", exc)

        # ------------ TRANSFORM ------------
        logger.info("ETL: Starting TRANSFORM phase")
        try:
            self._tables = self._transform(self._tables)
        except Exception as exc:
            result.status = "partial"
            result.errors.append(f"Transform failed: {exc}")
            logger.error("ETL Transform failed: %s", exc, exc_info=True)

        # ------------ LOAD ------------
        if save_output:
            logger.info("ETL: Starting LOAD phase")
            try:
                self._load(self._tables, fmt=fmt)
            except Exception as exc:
                result.status = "partial"
                result.errors.append(f"Load failed: {exc}")
                logger.error("ETL Load failed: %s", exc, exc_info=True)

        result.duration_seconds = time.time() - start_time
        logger.info("ETL complete: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def _extract(self) -> dict[str, pd.DataFrame]:
        """Extract raw data from generator or file system."""
        if self.use_generated_data:
            return self._extract_generated()
        return self._extract_files()

    def _extract_generated(self) -> dict[str, pd.DataFrame]:
        """Generate synthetic procurement dataset."""
        from src.data.generator import ProcurementDataGenerator

        logger.info("Generating synthetic dataset...")
        gen = ProcurementDataGenerator()
        ds = gen.generate_all()
        return {
            "vendors": ds.vendors,
            "products": ds.products,
            "purchase_orders": ds.purchase_orders,
            "deliveries": ds.deliveries,
            "quality_inspections": ds.quality_inspections,
            "contracts": ds.contracts,
        }

    def _extract_files(self) -> dict[str, pd.DataFrame]:
        """Load raw data from CSV/Parquet files."""
        from src.data.loader import DataLoader

        loader = DataLoader(data_dir=self.raw_data_dir)
        tables = loader.load_all_tables(directory=self.raw_data_dir, fmt="csv")
        if not tables:
            raise FileNotFoundError(
                f"No data files found in {self.raw_data_dir}. "
                "Run 'python scripts/generate_data.py' first."
            )
        return tables

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def _validate(self) -> int:
        """Run DataValidator and return total issue count."""
        from src.data.validator import DataValidator

        validator = DataValidator()
        report = validator.validate_all(**self._tables)
        if not report.passed:
            logger.warning("Validation found %d errors:\n%s", len(report.errors), report.summary())
        return len(report.issues)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform(self, tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Apply all transformation steps:
          - Date parsing and normalization
          - Derived columns (month, quarter, year, fiscal_quarter)
          - Enrichment joins (PO ← vendor tier, category)
          - Feature engineering for ML/analysis
        """
        transformed = {}

        if "purchase_orders" in tables:
            transformed["purchase_orders"] = self._transform_pos(
                tables["purchase_orders"]
            )

        if "deliveries" in tables:
            transformed["deliveries"] = self._transform_deliveries(
                tables["deliveries"]
            )

        if "quality_inspections" in tables:
            transformed["quality_inspections"] = self._transform_qi(
                tables["quality_inspections"],
                transformed.get("deliveries", tables.get("deliveries", pd.DataFrame())),
            )

        # Pass through tables that don't need transformation
        for name in ["vendors", "products", "contracts"]:
            if name in tables:
                transformed[name] = tables[name].copy()

        return transformed

    def _transform_pos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich purchase orders with time dimensions and flags."""
        df = df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["requested_delivery_date"] = pd.to_datetime(df["requested_delivery_date"], errors="coerce")

        df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
        df["order_quarter"] = df["order_date"].dt.to_period("Q").astype(str)
        df["order_year"] = df["order_date"].dt.year

        # Requested lead time
        df["requested_lead_days"] = (
            df["requested_delivery_date"] - df["order_date"]
        ).dt.days

        # Price tier flag
        if "unit_price" in df.columns:
            df["price_tier"] = pd.qcut(
                df["unit_price"],
                q=[0, 0.25, 0.75, 1.0],
                labels=["Low", "Mid", "High"],
                duplicates="drop",
            )

        return df

    def _transform_deliveries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich deliveries with lateness buckets."""
        df = df.copy()
        df["scheduled_date"] = pd.to_datetime(df["scheduled_date"], errors="coerce")
        df["actual_date"] = pd.to_datetime(df["actual_date"], errors="coerce")

        if "days_early_late" in df.columns:
            bins = [-float("inf"), -3, 0, 3, 7, float("inf")]
            labels = ["Early (>3d)", "Early (≤3d)", "Late (1-3d)", "Late (4-7d)", "Late (>7d)"]
            df["delivery_bucket"] = pd.cut(
                df["days_early_late"], bins=bins, labels=labels, right=True
            )

        return df

    def _transform_qi(
        self, df: pd.DataFrame, deliveries: pd.DataFrame
    ) -> pd.DataFrame:
        """Enrich quality inspections with vendor_id from deliveries."""
        df = df.copy()
        if "vendor_id" not in df.columns and not deliveries.empty and "vendor_id" in deliveries.columns:
            df = df.merge(
                deliveries[["delivery_id", "vendor_id"]].drop_duplicates(),
                on="delivery_id",
                how="left",
            )

        if "defect_rate" in df.columns:
            df["quality_tier"] = pd.cut(
                df["defect_rate"],
                bins=[0, 0.01, 0.03, 0.05, 1.0],
                labels=["Excellent", "Acceptable", "At Risk", "Reject"],
                right=True,
                include_lowest=True,
            )

        return df

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self, tables: dict[str, pd.DataFrame], fmt: str = "parquet") -> None:
        """Persist processed tables to output_dir."""
        from src.data.loader import DataLoader

        loader = DataLoader(data_dir=self.output_dir)
        loader.save_tables(tables, output_dir=self.output_dir, fmt=fmt)
        logger.info("ETL LOAD complete: %d tables → %s", len(tables), self.output_dir)

    # ------------------------------------------------------------------
    # Property accessors
    # ------------------------------------------------------------------

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        """Return processed tables (populated after run())."""
        return self._tables

    def get_table(self, name: str) -> pd.DataFrame:
        """Return a specific processed table by name."""
        if name not in self._tables:
            raise KeyError(f"Table '{name}' not found. Available: {list(self._tables.keys())}")
        return self._tables[name]
