"""
Generate a PDF executive procurement report.

Runs the analytics pipeline and produces a formatted PDF
with KPI scorecard, spend analysis, vendor performance,
inventory parameters, and action items.

Usage:
    python scripts/create_report.py [--output reports/procurement_report.pdf]
"""

from __future__ import annotations

import argparse
import logging
import sys
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
        description="Generate PDF procurement executive report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", type=str, default="reports/procurement_executive_report.pdf",
        help="Output PDF file path"
    )
    parser.add_argument(
        "--company", type=str, default="Acme Corp",
        help="Company name for the report header"
    )
    parser.add_argument(
        "--title", type=str, default="Procurement Analytics Executive Report",
        help="Report title"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Supply Chain Analytics — Executive Report Generator")
    print("=" * 55)

    # ----------------------------------------------------------------
    # 1. Run ETL
    # ----------------------------------------------------------------
    print("\n[1/6] Running ETL pipeline...")
    from src.pipeline.etl import ProcurementETL

    etl = ProcurementETL(use_generated_data=True)
    etl_result = etl.run(validate=True, save_output=False)
    if etl_result.status == "failed":
        print(f"  ERROR: ETL failed — {etl_result.errors}")
        return 1
    print(f"  ✓ {etl_result.rows_processed}")

    tables = etl.tables
    po = tables["purchase_orders"]
    vendors = tables.get("vendors")
    products = tables.get("products")
    deliveries = tables.get("deliveries")
    qi = tables.get("quality_inspections")
    contracts = tables.get("contracts")

    # ----------------------------------------------------------------
    # 2. Spend analysis
    # ----------------------------------------------------------------
    print("\n[2/6] Running spend analysis...")
    from src.analysis.spend_analysis import SpendAnalyzer

    analyzer = SpendAnalyzer(po, contracts=contracts, vendors=vendors)
    spend_kpis = analyzer.kpi_summary()
    pareto_vendors = analyzer.pareto_analysis(by="vendor_id").head(20)
    pareto_categories = analyzer.pareto_analysis(by="category")
    print(f"  ✓ Total spend: ${spend_kpis['total_spend_usd'] / 1e6:,.1f} M")

    # ----------------------------------------------------------------
    # 3. Vendor scoring
    # ----------------------------------------------------------------
    print("\n[3/6] Computing vendor scorecards...")
    from src.analysis.vendor_scoring import VendorScorer

    scorer = VendorScorer(
        deliveries=deliveries, quality_inspections=qi,
        purchase_orders=po, vendors=vendors,
    )
    tier_summary = scorer.tier_performance_summary()
    print(f"  ✓ {len(tier_summary)} tiers scored")

    # ----------------------------------------------------------------
    # 4. Inventory
    # ----------------------------------------------------------------
    print("\n[4/6] Running inventory optimization...")
    from src.analysis.inventory_optimization import InventoryOptimizer

    opt = InventoryOptimizer(purchase_orders=po, products=products)
    inv_params = opt.compute_all()
    inv_kpis = opt.kpi_summary(params=inv_params)
    abc_summary = inv_params.groupby("abc_class", observed=True).agg(
        count=("product_id", "count"),
        avg_eoq=("eoq", "mean"),
        total_cost=("total_annual_cost", "sum"),
    ).round(2)
    print(f"  ✓ {inv_kpis['total_skus']:,} SKUs optimized")

    # ----------------------------------------------------------------
    # 5. KPIs
    # ----------------------------------------------------------------
    print("\n[5/6] Computing KPIs...")
    from src.reporting.kpi_tracker import KPITracker

    tracker = KPITracker(
        purchase_orders=po, deliveries=deliveries,
        quality_inspections=qi, inventory_params=inv_params, contracts=contracts,
    )
    kpi_results = tracker.compute_all(period="Current Period")
    kpi_df = tracker.to_dataframe(kpi_results)
    on_target = (kpi_df["status"] == "on_target").sum()
    print(f"  ✓ {on_target}/{len(kpi_df)} KPIs on target")

    # ----------------------------------------------------------------
    # 6. Generate PDF
    # ----------------------------------------------------------------
    print("\n[6/6] Generating PDF report...")
    from src.reporting.pdf_report import ProcurementPDFReport

    recommendations = [
        {
            "priority": "High",
            "area": "Vendor Management",
            "action": "Consolidate Tier 3 vendors — reduce from current count to 2 preferred suppliers per category",
            "expected_impact": "8–12% cost reduction (~$2.1 M)",
        },
        {
            "priority": "High",
            "area": "Spend Compliance",
            "action": f"Implement contract compliance controls to reduce maverick spend from {spend_kpis['maverick_spend_pct']:.1f}% to <5%",
            "expected_impact": "5–8% savings on off-contract spend",
        },
        {
            "priority": "Medium",
            "area": "Inventory",
            "action": "Implement EOQ-based replenishment for A-class items; reduce safety stock for C-class items",
            "expected_impact": f"Reduce annual holding cost by ~15% (${inv_kpis['total_annual_inv_cost_usd'] * 0.15:,.0f})",
        },
        {
            "priority": "Medium",
            "area": "Procurement Process",
            "action": "Deploy automated PO approval workflow to reduce cycle time to <2 days for orders <$10K",
            "expected_impact": "30% reduction in admin processing time",
        },
        {
            "priority": "Low",
            "area": "Supplier Development",
            "action": "Launch quarterly business reviews (QBRs) with top 20 vendors by spend",
            "expected_impact": "Improve OTD rate by 3–5 percentage points over 12 months",
        },
    ]

    report = ProcurementPDFReport(
        company_name=args.company,
        report_title=args.title,
    )
    report.add_kpi_section(kpi_df)
    report.add_spend_section(
        spend_summary=spend_kpis,
        top_vendors=pareto_vendors,
        top_categories=pareto_categories,
    )
    report.add_vendor_section(tier_summary=tier_summary)
    report.add_inventory_section(
        inventory_kpis=inv_kpis,
        abc_summary=abc_summary,
    )
    report.add_recommendations(recommendations)

    output_path = Path(args.output)
    saved_path = report.save(output_path)

    print(f"\n  ✓ Report saved: {saved_path}")
    print(f"     Size: {saved_path.stat().st_size / 1024:.0f} KB")
    print("\nReport generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
