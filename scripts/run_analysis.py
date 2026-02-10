"""
Run the full procurement analytics pipeline and print summary results.

Executes the ETL pipeline, then runs all analysis modules and prints
a tabular summary to stdout. Optionally exports results to CSV.

Usage:
    python scripts/run_analysis.py [--output-dir results/] [--format csv]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run procurement analytics pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for analysis output files"
    )
    parser.add_argument(
        "--format", choices=["csv", "parquet"], default="csv",
        help="Output file format"
    )
    parser.add_argument(
        "--skip-forecast", action="store_true",
        help="Skip demand forecasting (faster)"
    )
    return parser.parse_args()


def section_header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ----------------------------------------------------------------
    # 1. ETL
    # ----------------------------------------------------------------
    section_header("1. ETL Pipeline")
    from src.pipeline.etl import ProcurementETL

    etl = ProcurementETL(use_generated_data=True)
    etl_result = etl.run(validate=True, save_output=False)
    print(etl_result.summary())

    if etl_result.status == "failed":
        logger.error("ETL failed — aborting.")
        return 1

    tables = etl.tables
    po = tables["purchase_orders"]
    vendors = tables.get("vendors", None)
    products = tables.get("products", None)
    deliveries = tables.get("deliveries", None)
    qi = tables.get("quality_inspections", None)
    contracts = tables.get("contracts", None)

    # ----------------------------------------------------------------
    # 2. Spend Analysis
    # ----------------------------------------------------------------
    section_header("2. Spend Analysis")
    from src.analysis.spend_analysis import SpendAnalyzer

    analyzer = SpendAnalyzer(po, contracts=contracts, vendors=vendors)
    spend_kpis = analyzer.kpi_summary()
    print(f"  Total Spend:         ${spend_kpis['total_spend_usd'] / 1e6:,.1f} M")
    print(f"  Total PO Count:      {spend_kpis['total_po_count']:,}")
    print(f"  Active Vendors:      {spend_kpis['active_vendors']}")
    print(f"  Maverick Spend:      ${spend_kpis['maverick_spend_usd'] / 1e6:,.1f} M ({spend_kpis['maverick_spend_pct']:.1f}%)")
    print(f"  HHI (Concentration): {spend_kpis['spend_hhi']:.4f}")

    pareto = analyzer.pareto_analysis(by="vendor_id")
    a_count = (pareto["pareto_class"] == "A").sum()
    print(f"  Pareto A-class vendors: {a_count} ({a_count / len(pareto) * 100:.1f}%) drive 80% of spend")

    if args.format == "csv":
        pareto.to_csv(output_dir / "pareto_vendors.csv", index=False)
        analyzer.pareto_analysis(by="category").to_csv(output_dir / "pareto_categories.csv", index=False)

    # ----------------------------------------------------------------
    # 3. Vendor Scoring
    # ----------------------------------------------------------------
    section_header("3. Vendor Scorecards")
    from src.analysis.vendor_scoring import VendorScorer

    scorer = VendorScorer(
        deliveries=deliveries,
        quality_inspections=qi,
        purchase_orders=po,
        vendors=vendors,
    )
    scorecards = scorer.compute_scorecards().reset_index()
    print(f"  Vendors scored: {len(scorecards)}")
    print(f"  Avg composite score: {scorecards['composite_score'].mean():.1f}")
    print(f"  Grade distribution:")
    for grade in ["A", "B", "C", "D", "F"]:
        n = (scorecards["grade"] == grade).sum()
        print(f"    {grade}: {n} vendors ({n / len(scorecards) * 100:.1f}%)")

    tier_summary = scorer.tier_performance_summary()
    print("\n  Tier Performance Summary:")
    print(tier_summary.to_string())

    if args.format == "csv":
        scorecards.to_csv(output_dir / "vendor_scorecards.csv", index=False)
        tier_summary.to_csv(output_dir / "tier_performance.csv")

    # ----------------------------------------------------------------
    # 4. Inventory Optimization
    # ----------------------------------------------------------------
    section_header("4. Inventory Optimization")
    from src.analysis.inventory_optimization import InventoryOptimizer

    opt = InventoryOptimizer(purchase_orders=po, products=products)
    inv_params = opt.compute_all()
    inv_kpis = opt.kpi_summary(params=inv_params)

    print(f"  Total SKUs:             {inv_kpis['total_skus']:,}")
    print(f"  A / B / C items:        {inv_kpis['abc_a_count']} / {inv_kpis['abc_b_count']} / {inv_kpis['abc_c_count']}")
    print(f"  Avg EOQ:                {inv_kpis['avg_eoq']:.0f} units")
    print(f"  Avg Safety Stock:       {inv_kpis['avg_safety_stock']:.0f} units")
    print(f"  Avg Inventory Turnover: {inv_kpis['avg_inventory_turnover']:.1f}×")
    print(f"  Total Annual Inv Cost:  ${inv_kpis['total_annual_inv_cost_usd'] / 1e6:,.1f} M")

    if args.format == "csv":
        inv_params.to_csv(output_dir / "inventory_parameters.csv", index=False)

    # ----------------------------------------------------------------
    # 5. Demand Forecast (optional)
    # ----------------------------------------------------------------
    if not args.skip_forecast:
        section_header("5. Demand Forecasting")
        from src.analysis.demand_forecast import DemandForecaster

        fc = DemandForecaster(po, horizon_days=90)
        result = fc.forecast_holt_winters(entity_id="total")
        print(f"  Method: {result.method}")
        print(f"  Horizon: 90 days")
        print(f"  MAE:  {result.mae:.2f}")
        print(f"  MAPE: {result.mape:.2f}%")
        print(f"  Forecast range: {result.forecast_df['ds'].min()} → {result.forecast_df['ds'].max()}")

        if args.format == "csv":
            result.forecast_df.to_csv(output_dir / "demand_forecast_total.csv", index=False)

    # ----------------------------------------------------------------
    # 6. KPIs
    # ----------------------------------------------------------------
    section_header("6. KPI Scorecard")
    from src.reporting.kpi_tracker import KPITracker

    tracker = KPITracker(
        purchase_orders=po,
        deliveries=deliveries,
        quality_inspections=qi,
        inventory_params=inv_params,
        contracts=contracts,
    )
    kpi_results = tracker.compute_all(period="Full Period")
    kpi_df = tracker.to_dataframe(kpi_results)

    on_target = (kpi_df["status"] == "on_target").sum()
    print(f"  KPIs on target: {on_target}/{len(kpi_df)}")
    print()
    print(kpi_df[["kpi", "value", "unit", "target", "achievement_pct", "status"]].to_string(index=False))

    if args.format == "csv":
        kpi_df.to_csv(output_dir / "kpi_scorecard.csv", index=False)

    # ----------------------------------------------------------------
    # 7. Cost Optimization
    # ----------------------------------------------------------------
    section_header("7. Cost Optimization")
    from src.analysis.cost_optimizer import CostOptimizer

    cost_opt = CostOptimizer(po, vendors, deliveries, qi)
    cost_kpis = cost_opt.kpi_summary()
    consolidation = cost_opt.consolidation_opportunities()

    print(f"  Total TCO:                   ${cost_kpis['total_tco_usd'] / 1e6:,.1f} M")
    print(f"  Avg TCO Premium:             {cost_kpis['avg_tco_premium_pct']:.1f}%")
    print(f"  Consolidation Opportunities: {cost_kpis['consolidation_categories']}")
    print(f"  Savings Potential:           ${cost_kpis['consolidation_savings_potential_usd'] / 1e6:,.1f} M")

    if args.format == "csv":
        consolidation.to_csv(output_dir / "consolidation_opportunities.csv", index=False)

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    elapsed = time.time() - total_start
    section_header("ANALYSIS COMPLETE")
    print(f"  Total runtime:    {elapsed:.1f} seconds")
    print(f"  Output directory: {output_dir.resolve()}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
