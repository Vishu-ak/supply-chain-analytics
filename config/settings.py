"""
Configuration settings for the Supply Chain Analytics platform.

Uses pydantic-settings for environment-based configuration management
with sensible defaults for local development.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="supply_chain", description="Database name")
    user: str = Field(default="analyst", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=5, description="Connection pool size")
    echo_sql: bool = Field(default=False, description="Echo SQL queries for debugging")

    @property
    def url(self) -> str:
        """Construct SQLAlchemy connection URL."""
        if self.password:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.name}"

    @property
    def sqlite_url(self) -> str:
        """Fallback SQLite URL for local development."""
        return f"sqlite:///{DATA_DIR / 'supply_chain.db'}"


class DataSettings(BaseSettings):
    """Data generation and storage settings."""

    model_config = SettingsConfigDict(env_prefix="DATA_", extra="ignore")

    raw_dir: Path = Field(default=DATA_DIR / "raw", description="Raw data directory")
    processed_dir: Path = Field(default=DATA_DIR / "processed", description="Processed data directory")
    sample_dir: Path = Field(default=DATA_DIR / "sample", description="Sample dataset directory")

    # Generation parameters
    n_vendors: int = Field(default=200, ge=10, le=5000, description="Number of vendors to generate")
    n_products: int = Field(default=1000, ge=10, le=50000, description="Number of products to generate")
    n_purchase_orders: int = Field(default=50_000, ge=100, le=1_000_000, description="Number of POs to generate")
    date_start: str = Field(default="2022-01-01", description="Start date for historical data (YYYY-MM-DD)")
    date_end: str = Field(default="2024-12-31", description="End date for historical data (YYYY-MM-DD)")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    @field_validator("raw_dir", "processed_dir", "sample_dir", mode="before")
    @classmethod
    def coerce_path(cls, v: object) -> Path:
        return Path(str(v))


class AnalysisSettings(BaseSettings):
    """Analysis configuration."""

    model_config = SettingsConfigDict(env_prefix="ANALYSIS_", extra="ignore")

    # Spend analysis
    pareto_threshold: float = Field(default=0.80, ge=0.0, le=1.0, description="Pareto cutoff (80/20 rule)")
    maverick_spend_threshold: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Spend fraction above which maverick spend is flagged"
    )

    # Vendor scoring weights (must sum to 1.0)
    vendor_weight_delivery: float = Field(default=0.35, description="Weight for on-time delivery score")
    vendor_weight_quality: float = Field(default=0.30, description="Weight for quality/defect score")
    vendor_weight_cost: float = Field(default=0.25, description="Weight for cost competitiveness score")
    vendor_weight_responsiveness: float = Field(default=0.10, description="Weight for responsiveness score")

    # Inventory
    service_level: float = Field(default=0.95, ge=0.5, le=0.9999, description="Target service level for safety stock")
    holding_cost_rate: float = Field(default=0.25, ge=0.0, le=1.0, description="Annual holding cost as fraction of unit cost")
    ordering_cost: float = Field(default=150.0, ge=0.0, description="Fixed cost per purchase order (USD)")

    # Forecasting
    forecast_horizon_days: int = Field(default=90, ge=1, le=365, description="Demand forecast horizon in days")
    forecast_confidence: float = Field(default=0.95, ge=0.5, le=0.999, description="Confidence interval for forecast bands")

    @field_validator("vendor_weight_delivery", "vendor_weight_quality", "vendor_weight_cost", "vendor_weight_responsiveness")
    @classmethod
    def weights_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Vendor score weight must be non-negative")
        return v


class ReportingSettings(BaseSettings):
    """Reporting and export settings."""

    model_config = SettingsConfigDict(env_prefix="REPORT_", extra="ignore")

    output_dir: Path = Field(default=REPORTS_DIR, description="Report output directory")
    logo_path: Path | None = Field(default=None, description="Company logo for PDF reports")
    company_name: str = Field(default="Acme Corp", description="Company name in reports")
    currency: str = Field(default="USD", description="Display currency symbol")
    date_format: str = Field(default="%Y-%m-%d", description="Date format for reports")
    pdf_page_size: Literal["A4", "Letter"] = Field(default="Letter", description="PDF page size")

    @field_validator("output_dir", "logo_path", mode="before")
    @classmethod
    def coerce_path(cls, v: object) -> Path | None:
        if v is None:
            return None
        return Path(str(v))


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "staging", "production"] = Field(
        default="development", description="Runtime environment"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Nested settings (instantiated once)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)

    def ensure_directories(self) -> None:
        """Create all required directories if they do not exist."""
        dirs = [
            self.data.raw_dir,
            self.data.processed_dir,
            self.data.sample_dir,
            self.reporting.output_dir,
            LOGS_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Return the singleton AppSettings instance (lazy init)."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
        _settings.ensure_directories()
    return _settings


# Convenience alias used throughout the codebase
settings = get_settings()
