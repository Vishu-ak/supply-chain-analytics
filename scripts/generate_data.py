"""
Generate and persist the sample procurement dataset.

Creates CSV and Parquet files in data/sample/ for use in demos,
notebooks, and testing without running the full ETL pipeline.

Usage:
    python scripts/generate_data.py [--n-pos 50000] [--output-dir data/sample]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic procurement dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-vendors", type=int, default=200, help="Number of vendors")
    parser.add_argument("--n-products", type=int, default=1000, help="Number of products")
    parser.add_argument("--n-pos", type=int, default=50_000, help="Number of purchase orders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="data/sample",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--format", choices=["csv", "parquet", "both"], default="csv",
        help="Output file format"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating dataset: %d vendors, %d products, %d POs, seed=%d",
        args.n_vendors, args.n_products, args.n_pos, args.seed,
    )
    t0 = time.time()

    from src.data.generator import ProcurementDataGenerator

    gen = ProcurementDataGenerator(
        n_vendors=args.n_vendors,
        n_products=args.n_products,
        n_pos=args.n_pos,
        seed=args.seed,
    )
    dataset = gen.generate_all()

    tables = {
        "vendors": dataset.vendors,
        "products": dataset.products,
        "purchase_orders": dataset.purchase_orders,
        "deliveries": dataset.deliveries,
        "quality_inspections": dataset.quality_inspections,
        "contracts": dataset.contracts,
    }

    # Save to disk
    for name, df in tables.items():
        if args.format in ("csv", "both"):
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info("  ✓ %s.csv (%d rows, %.1f MB)", name, len(df), path.stat().st_size / 1e6)
        if args.format in ("parquet", "both"):
            path = output_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
            logger.info("  ✓ %s.parquet (%d rows, %.1f MB)", name, len(df), path.stat().st_size / 1e6)

    elapsed = time.time() - t0
    logger.info("Dataset generated in %.1f seconds → %s", elapsed, output_dir)

    # Print dataset summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for name, df in tables.items():
        print(f"  {name:<25} {len(df):>8,} rows  |  {len(df.columns):>3} columns")
    print(f"\nTotal spend (POs):  ${dataset.purchase_orders['total_amount'].sum() / 1e6:,.1f} M")
    print(f"Date range:         {dataset.purchase_orders['order_date'].min()} → {dataset.purchase_orders['order_date'].max()}")
    print(f"Output directory:   {output_dir.resolve()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
