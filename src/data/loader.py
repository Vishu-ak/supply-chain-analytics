"""
Data loading utilities for procurement and supply chain data.

Supports CSV, Excel, Parquet, and SQLite/PostgreSQL sources.
Implements a unified DataFrame-based interface with consistent
column naming, date parsing, and basic type coercion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema definitions (expected column names per table)
# ---------------------------------------------------------------------------
SCHEMA_PURCHASE_ORDERS: dict[str, str] = {
    "po_id": "object",
    "vendor_id": "object",
    "product_id": "object",
    "order_date": "datetime64[ns]",
    "requested_delivery_date": "datetime64[ns]",
    "quantity": "int64",
    "unit_price": "float64",
    "total_amount": "float64",
    "status": "object",
    "category": "object",
}

SCHEMA_VENDORS: dict[str, str] = {
    "vendor_id": "object",
    "vendor_name": "object",
    "tier": "object",
    "category": "object",
    "active": "bool",
}

SCHEMA_DELIVERIES: dict[str, str] = {
    "delivery_id": "object",
    "po_id": "object",
    "vendor_id": "object",
    "scheduled_date": "datetime64[ns]",
    "actual_date": "datetime64[ns]",
    "quantity_delivered": "int64",
    "on_time": "bool",
    "days_early_late": "int64",
}

SCHEMA_QUALITY_INSPECTIONS: dict[str, str] = {
    "inspection_id": "object",
    "delivery_id": "object",
    "defects_found": "int64",
    "defect_rate": "float64",
    "passed": "bool",
}


class DataLoader:
    """
    Unified data loader for procurement datasets from multiple file formats.

    Parameters
    ----------
    data_dir : Path | str | None
        Base directory to resolve relative file paths.

    Example
    -------
    >>> loader = DataLoader(data_dir="data/sample")
    >>> po = loader.load_csv("purchase_orders.csv", date_cols=["order_date"])
    >>> po.dtypes
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        from config.settings import settings

        self.data_dir: Path = (
            Path(data_dir) if data_dir else settings.data.processed_dir
        )

    # ------------------------------------------------------------------
    # CSV / Parquet / Excel
    # ------------------------------------------------------------------

    def load_csv(
        self,
        filename: str | Path,
        date_cols: list[str] | None = None,
        dtype_map: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load a CSV file, optionally parsing date columns.

        Parameters
        ----------
        filename : path relative to data_dir or absolute
        date_cols : column names to parse as datetime
        dtype_map : explicit dtype overrides
        **kwargs : forwarded to pd.read_csv

        Returns
        -------
        pd.DataFrame
        """
        path = self._resolve(filename)
        logger.info("Loading CSV: %s", path)
        df = pd.read_csv(
            path,
            parse_dates=date_cols or [],
            dtype=dtype_map,
            **kwargs,
        )
        logger.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)
        return df

    def load_excel(
        self,
        filename: str | Path,
        sheet_name: str | int = 0,
        date_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load an Excel workbook sheet into a DataFrame."""
        path = self._resolve(filename)
        logger.info("Loading Excel: %s (sheet=%s)", path, sheet_name)
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        if date_cols:
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        logger.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)
        return df

    def load_parquet(
        self,
        filename: str | Path,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load a Parquet file, optionally projecting specific columns."""
        path = self._resolve(filename)
        logger.info("Loading Parquet: %s", path)
        df = pd.read_parquet(path, columns=columns, **kwargs)
        logger.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)
        return df

    # ------------------------------------------------------------------
    # Database (SQLite / PostgreSQL via SQLAlchemy)
    # ------------------------------------------------------------------

    def load_from_db(
        self,
        query: str,
        connection_url: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Parameters
        ----------
        query : SQL SELECT statement
        connection_url : SQLAlchemy URL; defaults to settings.db.sqlite_url
        params : query parameters (for parameterized queries)
        """
        try:
            from sqlalchemy import create_engine, text
        except ImportError as exc:
            raise ImportError("sqlalchemy is required for DB loading: pip install sqlalchemy") from exc

        from config.settings import settings

        url = connection_url or settings.db.sqlite_url
        engine = create_engine(url)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params or {})
        logger.info("DB query returned %d rows × %d cols", *df.shape)
        return df

    # ------------------------------------------------------------------
    # Multi-table convenience loaders
    # ------------------------------------------------------------------

    def load_all_tables(
        self, directory: Path | str | None = None, fmt: str = "csv"
    ) -> dict[str, pd.DataFrame]:
        """
        Load all standard procurement tables from a directory.

        Supports 'csv', 'parquet', and 'excel' formats.

        Returns
        -------
        dict mapping table name → DataFrame
        """
        dir_path = Path(directory) if directory else self.data_dir
        tables: dict[str, pd.DataFrame] = {}

        table_configs = {
            "purchase_orders": {"date_cols": ["order_date", "requested_delivery_date"]},
            "vendors": {"date_cols": []},
            "products": {"date_cols": []},
            "deliveries": {"date_cols": ["scheduled_date", "actual_date"]},
            "quality_inspections": {"date_cols": ["inspection_date"]},
            "contracts": {"date_cols": ["start_date", "end_date"]},
        }

        for table, cfg in table_configs.items():
            ext_map = {"csv": ".csv", "parquet": ".parquet", "excel": ".xlsx"}
            ext = ext_map.get(fmt, ".csv")
            file_path = dir_path / f"{table}{ext}"

            if not file_path.exists():
                logger.warning("File not found, skipping: %s", file_path)
                continue

            try:
                if fmt == "csv":
                    tables[table] = self.load_csv(file_path, **cfg)
                elif fmt == "parquet":
                    tables[table] = self.load_parquet(file_path)
                elif fmt == "excel":
                    tables[table] = self.load_excel(file_path)
            except Exception as exc:
                logger.error("Failed to load %s: %s", table, exc)

        return tables

    def save_tables(
        self,
        tables: dict[str, pd.DataFrame],
        output_dir: Path | str | None = None,
        fmt: str = "csv",
    ) -> None:
        """
        Persist a dict of DataFrames to disk.

        Parameters
        ----------
        tables : mapping of table_name → DataFrame
        output_dir : directory to write files; defaults to data_dir
        fmt : 'csv' | 'parquet'
        """
        out_dir = Path(output_dir) if output_dir else self.data_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, df in tables.items():
            if fmt == "parquet":
                path = out_dir / f"{name}.parquet"
                df.to_parquet(path, index=False)
            else:
                path = out_dir / f"{name}.csv"
                df.to_csv(path, index=False)
            logger.info("Saved %s → %s (%d rows)", name, path, len(df))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, filename: str | Path) -> Path:
        """Resolve a filename to an absolute path."""
        path = Path(filename)
        if not path.is_absolute():
            path = self.data_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path
