"""
Supply Chain & Procurement Analytics — Streamlit Dashboard.

Interactive dashboard providing:
  - KPI overview with traffic-light status cards
  - Spend analysis: Pareto, treemaps, category trends
  - Vendor scorecards and performance radar charts
  - Demand forecasting with Prophet and Holt-Winters
  - Inventory optimization (EOQ, safety stock, ABC)
  - PDF report export

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/vishnukumarak2002/supply-chain-analytics",
        "About": "Supply Chain & Procurement Analytics Platform — Vishnu Kumar A.K.",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Brand color variables */
    :root {
        --primary: #1e3a5f;
        --secondary: #2980b9;
        --accent: #27ae60;
        --warning: #e67e22;
        --danger: #e74c3c;
        --neutral: #95a5a6;
        --bg-light: #f8f9fa;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; font-size: 0.9rem; }

    /* KPI card */
    .kpi-card {
        background: white;
        border-left: 4px solid var(--secondary);
        padding: 1rem 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 0.5rem;
    }
    .kpi-card.green  { border-left-color: var(--accent); }
    .kpi-card.orange { border-left-color: var(--warning); }
    .kpi-card.red    { border-left-color: var(--danger); }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: var(--primary); }
    .kpi-label { font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-delta { font-size: 0.80rem; margin-top: 2px; }

    /* Section headers */
    h2.section-title {
        color: var(--primary);
        border-bottom: 2px solid var(--secondary);
        padding-bottom: 6px;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading procurement data…", ttl=3600)
def load_data() -> dict[str, pd.DataFrame]:
    """Run ETL pipeline and return processed tables."""
    from src.pipeline.etl import ProcurementETL
    etl = ProcurementETL(use_generated_data=True)
    result = etl.run(validate=True, save_output=False)
    if result.status == "failed":
        st.error(f"ETL pipeline failed: {result.errors}")
        return {}
    return etl.tables


@st.cache_data(show_spinner="Computing vendor scorecards…", ttl=3600)
def compute_scorecards(
    _po: pd.DataFrame, _del: pd.DataFrame, _qi: pd.DataFrame, _v: pd.DataFrame
) -> pd.DataFrame:
    from src.analysis.vendor_scoring import VendorScorer
    scorer = VendorScorer(
        deliveries=_del, quality_inspections=_qi,
        purchase_orders=_po, vendors=_v
    )
    return scorer.compute_scorecards().reset_index()


@st.cache_data(show_spinner="Running inventory optimization…", ttl=3600)
def compute_inventory(
    _po: pd.DataFrame, _products: pd.DataFrame
) -> pd.DataFrame:
    from src.analysis.inventory_optimization import InventoryOptimizer
    opt = InventoryOptimizer(purchase_orders=_po, products=_products)
    return opt.compute_all()


@st.cache_data(show_spinner="Computing KPIs…", ttl=3600)
def compute_kpis(_po, _del, _qi, _inv) -> tuple[dict, pd.DataFrame]:
    from src.reporting.kpi_tracker import KPITracker
    tracker = KPITracker(
        purchase_orders=_po,
        deliveries=_del,
        quality_inspections=_qi,
        inventory_params=_inv,
    )
    results = tracker.compute_all(period="Current")
    return results, tracker.to_dataframe(results)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Render sidebar filters and return active filter state."""
    st.sidebar.markdown("""
    <div style="background:#1e3a5f;padding:1rem;border-radius:8px;margin-bottom:1rem;">
      <span style="color:white;font-weight:700;font-size:1.1rem;">⚙ Filters</span>
    </div>
    """, unsafe_allow_html=True)

    filters: dict[str, object] = {}
    po = tables.get("purchase_orders", pd.DataFrame())

    if not po.empty:
        # Date range
        if "order_date" in po.columns:
            dates = pd.to_datetime(po["order_date"], errors="coerce").dropna()
            min_d, max_d = dates.min().date(), dates.max().date()
            filters["date_range"] = st.sidebar.date_input(
                "Order Date Range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
            )

        # Category
        if "category" in po.columns:
            cats = sorted(po["category"].dropna().unique())
            filters["categories"] = st.sidebar.multiselect(
                "Categories", options=cats, default=[]
            )

        # ABC class
        if "abc_class" in po.columns:
            filters["abc_classes"] = st.sidebar.multiselect(
                "ABC Class", options=["A", "B", "C"], default=["A", "B", "C"]
            )

        # Vendor tier
        vendors = tables.get("vendors", pd.DataFrame())
        if not vendors.empty and "tier" in vendors.columns:
            tiers = sorted(vendors["tier"].dropna().unique())
            filters["tiers"] = st.sidebar.multiselect(
                "Vendor Tier", options=tiers, default=tiers
            )

    return filters


def apply_filters(
    po: pd.DataFrame, filters: dict[str, object]
) -> pd.DataFrame:
    """Apply sidebar filters to the PO DataFrame."""
    df = po.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    if "date_range" in filters and len(filters["date_range"]) == 2:
        start, end = filters["date_range"]
        df = df[(df["order_date"].dt.date >= start) & (df["order_date"].dt.date <= end)]

    if filters.get("categories"):
        df = df[df["category"].isin(filters["categories"])]

    if filters.get("abc_classes"):
        df = df[df["abc_class"].isin(filters["abc_classes"])]

    return df


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def page_overview(tables: dict, filters: dict) -> None:
    """KPI overview page."""
    st.markdown("""
    <div class='main-header'>
      <h1>📊 Supply Chain Analytics Dashboard</h1>
      <p>Procurement performance insights — powered by Python & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

    po = apply_filters(tables.get("purchase_orders", pd.DataFrame()), filters)
    inv = compute_inventory(po, tables.get("products", pd.DataFrame()))
    kpi_results, kpi_df = compute_kpis(
        po,
        tables.get("deliveries", pd.DataFrame()),
        tables.get("quality_inspections", pd.DataFrame()),
        inv,
    )

    # KPI cards row
    kpi_order = [
        ("otd_rate", "On-Time Delivery", "%", "green"),
        ("supplier_defect_rate", "Defect Rate", "%", "red"),
        ("procurement_cycle_time", "Cycle Time", "days", "orange"),
        ("cost_savings_pct", "Cost Savings", "%", "green"),
        ("inventory_turnover", "Inv. Turnover", "×", "green"),
        ("fill_rate", "Fill Rate", "%", "green"),
    ]

    cols = st.columns(len(kpi_order))
    for col, (key, label, unit, color) in zip(cols, kpi_order):
        if key in kpi_results:
            r = kpi_results[key]
            delta_text = ""
            if r.period_change is not None:
                sign = "↑" if r.period_change >= 0 else "↓"
                delta_text = f"{sign} {abs(r.period_change):.2f} vs prior"
            col.markdown(f"""
            <div class='kpi-card {color}'>
              <div class='kpi-value'>{r.value:.1f}{unit}</div>
              <div class='kpi-label'>{label}</div>
              <div class='kpi-delta'>{delta_text}</div>
            </div>
            """, unsafe_allow_html=True)

    # KPI status table
    st.subheader("KPI Status Summary")
    if not kpi_df.empty:
        st.dataframe(
            kpi_df[["kpi", "value", "unit", "target", "achievement_pct", "status", "description"]],
            use_container_width=True,
            hide_index=True,
        )


def page_spend_analysis(tables: dict, filters: dict) -> None:
    """Spend analysis page."""
    from src.analysis.spend_analysis import SpendAnalyzer
    from src.visualization.spend_charts import SpendCharts

    st.header("💰 Spend Analysis")
    po = apply_filters(tables.get("purchase_orders", pd.DataFrame()), filters)
    if po.empty:
        st.warning("No data matches the current filters.")
        return

    charts = SpendCharts()
    analyzer = SpendAnalyzer(po, vendors=tables.get("vendors", pd.DataFrame()))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spend Treemap by Category")
        cat_spend = po.groupby("category", observed=True)["total_amount"].sum().reset_index()
        cat_spend.columns = ["category", "total_spend"]
        st.plotly_chart(charts.spend_treemap(cat_spend, title="Spend by Category"), use_container_width=True)

    with col2:
        st.subheader("Category Spend Trend")
        st.plotly_chart(charts.category_spend_trend(po), use_container_width=True)

    # Pareto
    st.subheader("Vendor Pareto Analysis (80/20)")
    pareto_by = st.radio("Analyze by:", ["vendor_id", "category"], horizontal=True, key="pareto_by")
    pareto = analyzer.pareto_analysis(by=pareto_by)
    st.plotly_chart(
        charts.pareto_chart(pareto, x_col=pareto_by, top_n=30,
                            title=f"Pareto — Top 30 {pareto_by.replace('_id', '').title()}s"),
        use_container_width=True,
    )

    # Maverick spend
    st.subheader("Maverick Spend")
    maverick = analyzer.detect_maverick_spend()
    if not maverick.empty:
        total_maverick = maverick["total_amount"].sum()
        total_spend = po["total_amount"].sum()
        st.metric(
            "Maverick Spend",
            f"${total_maverick:,.0f}",
            delta=f"{total_maverick/total_spend*100:.1f}% of total",
            delta_color="inverse",
        )
        st.dataframe(
            maverick[["po_id", "vendor_id", "category", "total_amount", "maverick_reason"]].head(20),
            use_container_width=True,
        )
    else:
        st.info("No maverick spend detected with current data/filters.")

    # Spend KPIs
    kpis = analyzer.kpi_summary()
    st.subheader("Spend KPIs")
    mcols = st.columns(4)
    mcols[0].metric("Total Spend", f"${kpis['total_spend_usd']:,.0f}")
    mcols[1].metric("Total POs", f"{kpis['total_po_count']:,}")
    mcols[2].metric("Active Vendors", kpis["active_vendors"])
    mcols[3].metric("HHI (Concentration)", f"{kpis['spend_hhi']:.4f}")


def page_vendor_performance(tables: dict, filters: dict) -> None:
    """Vendor performance page."""
    from src.visualization.spend_charts import SpendCharts
    from src.visualization.dashboards import DashboardComponents

    st.header("🏭 Vendor Performance")
    po = apply_filters(tables.get("purchase_orders", pd.DataFrame()), filters)
    if po.empty:
        st.warning("No data matches the current filters.")
        return

    charts = SpendCharts()
    dc = DashboardComponents()

    scorecards = compute_scorecards(
        po,
        tables.get("deliveries", pd.DataFrame()),
        tables.get("quality_inspections", pd.DataFrame()),
        tables.get("vendors", pd.DataFrame()),
    )

    st.subheader("Vendor Scorecard Rankings")
    top_n = st.slider("Show top N vendors:", min_value=5, max_value=50, value=20)
    st.plotly_chart(
        charts.vendor_comparison_bar(scorecards.set_index("vendor_id") if "vendor_id" in scorecards.columns else scorecards, top_n=top_n),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Radar (Top 5)")
        sc_indexed = scorecards.set_index("vendor_id") if "vendor_id" in scorecards.columns else scorecards
        st.plotly_chart(dc.vendor_radar(sc_indexed), use_container_width=True)

    with col2:
        st.subheader("Delivery Heatmap")
        deliveries = tables.get("deliveries", pd.DataFrame())
        if not deliveries.empty:
            st.plotly_chart(dc.delivery_heatmap(deliveries), use_container_width=True)

    # Full scorecard table
    st.subheader("Full Scorecard Table")
    display_cols = [c for c in ["vendor_id", "vendor_name", "tier", "composite_score",
                                 "delivery_score", "quality_score", "cost_score",
                                 "responsiveness_score", "grade", "rank"]
                    if c in scorecards.columns]
    st.dataframe(scorecards[display_cols].head(50), use_container_width=True)


def page_demand_forecast(tables: dict, filters: dict) -> None:
    """Demand forecasting page."""
    from src.analysis.demand_forecast import DemandForecaster
    from src.visualization.forecast_plots import ForecastPlots

    st.header("📈 Demand Forecasting")
    po = apply_filters(tables.get("purchase_orders", pd.DataFrame()), filters)
    if po.empty:
        st.warning("No data matches the current filters.")
        return

    fp = ForecastPlots()

    col1, col2 = st.columns([1, 2])
    with col1:
        horizon = st.select_slider("Forecast horizon:", options=[30, 60, 90, 120, 180], value=90)
        method = st.radio("Method:", ["holt_winters", "prophet"], horizontal=True)
        if "category" in po.columns:
            cats = ["total"] + sorted(po["category"].dropna().unique().tolist())
            selected_cat = st.selectbox("Category:", cats)
        else:
            selected_cat = "total"

    with col2:
        st.subheader(f"{method.replace('_', ' ').title()} Forecast — {selected_cat}")
        with st.spinner("Running forecast…"):
            fc = DemandForecaster(po, horizon_days=horizon)
            group_col = "category" if selected_cat != "total" else None
            if method == "prophet":
                result = fc.forecast_prophet(entity_id=selected_cat, group_col=group_col)
            else:
                result = fc.forecast_holt_winters(entity_id=selected_cat, group_col=group_col)

            # Build historical series
            if selected_cat != "total" and "category" in po.columns:
                hist = po[po["category"] == selected_cat].set_index("order_date")["quantity"].resample("W").sum()
            else:
                hist = po.set_index("order_date")["quantity"].resample("W").sum()

            fig = fp.forecast_with_ci(result, historical=hist)
            st.plotly_chart(fig, use_container_width=True)

        if result.mape > 0:
            st.metric("Model MAPE", f"{result.mape:.2f}%")
            st.metric("Model MAE", f"{result.mae:.2f}")


def page_inventory(tables: dict, filters: dict) -> None:
    """Inventory optimization page."""
    from src.visualization.dashboards import DashboardComponents

    st.header("📦 Inventory Optimization")
    po = apply_filters(tables.get("purchase_orders", pd.DataFrame()), filters)
    products = tables.get("products", pd.DataFrame())
    if po.empty or products.empty:
        st.warning("No data available.")
        return

    dc = DashboardComponents()
    inv = compute_inventory(po, products)

    # KPI metrics
    from src.analysis.inventory_optimization import InventoryOptimizer
    opt = InventoryOptimizer(purchase_orders=po, products=products)
    kpis = opt.kpi_summary(params=inv)

    kcols = st.columns(5)
    kcols[0].metric("Total SKUs", f"{kpis['total_skus']:,}")
    kcols[1].metric("A-Class Items", kpis["abc_a_count"])
    kcols[2].metric("Avg EOQ", f"{kpis['avg_eoq']:.0f} units")
    kcols[3].metric("Avg Safety Stock", f"{kpis['avg_safety_stock']:.0f} units")
    kcols[4].metric("Inv. Turnover", f"{kpis['avg_inventory_turnover']:.1f}×")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ABC Classification")
        st.plotly_chart(dc.inventory_abc_donut(inv), use_container_width=True)

    with col2:
        st.subheader("EOQ Distribution by ABC Class")
        import plotly.express as px
        if "abc_class" in inv.columns and "eoq" in inv.columns:
            fig = px.box(inv[inv["eoq"] > 0], x="abc_class", y="eoq",
                         color="abc_class", title="EOQ Distribution",
                         template="plotly_white",
                         category_orders={"abc_class": ["A", "B", "C"]})
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("SKU-Level Inventory Parameters")
    display_cols = [c for c in ["product_id", "sku", "abc_class", "unit_cost",
                                 "eoq", "safety_stock", "reorder_point",
                                 "annual_demand", "inventory_turnover", "total_annual_cost"]
                    if c in inv.columns]
    st.dataframe(inv[display_cols].head(200), use_container_width=True)


def page_cost_optimization(tables: dict, filters: dict) -> None:
    """Cost optimization page."""
    from src.analysis.cost_optimizer import CostOptimizer
    from src.visualization.dashboards import DashboardComponents

    st.header("💡 Cost Optimization")
    po = apply_filters(tables.get("purchase_orders", pd.DataFrame()), filters)
    vendors = tables.get("vendors", pd.DataFrame())
    deliveries = tables.get("deliveries", pd.DataFrame())
    qi = tables.get("quality_inspections", pd.DataFrame())

    if po.empty:
        st.warning("No data matches the current filters.")
        return

    optimizer = CostOptimizer(po, vendors, deliveries, qi)
    dc = DashboardComponents()

    # TCO summary
    st.subheader("Total Cost of Ownership by Vendor")
    with st.spinner("Computing TCO…"):
        tco = optimizer.total_cost_of_ownership()
        kpis = optimizer.kpi_summary()

    kcols = st.columns(3)
    kcols[0].metric("Total TCO", f"${kpis['total_tco_usd']:,.0f}")
    kcols[1].metric("Quality Cost", f"${kpis['total_quality_cost_usd']:,.0f}")
    kcols[2].metric("Avg TCO Premium", f"{kpis['avg_tco_premium_pct']:.1f}%")

    st.dataframe(
        tco[["vendor_id", "material_cost", "logistics_cost_base", "quality_cost",
              "admin_cost", "total_tco", "tco_premium_pct"]].head(30),
        use_container_width=True,
    )

    # Consolidation
    st.subheader("Vendor Consolidation Opportunities")
    consolidation = optimizer.consolidation_opportunities()
    if not consolidation.empty:
        st.metric(
            "Total Savings Potential",
            f"${consolidation['estimated_savings_usd'].sum():,.0f}",
        )
        st.dataframe(consolidation, use_container_width=True)

    # Cycle time
    st.subheader("Procurement Cycle Time by Category")
    cycle = optimizer.procurement_cycle_time()
    if not cycle.empty:
        import plotly.express as px
        fig = px.bar(
            cycle.groupby("category", observed=True)["avg_cycle_days"].mean().reset_index(),
            x="category", y="avg_cycle_days",
            title="Average Cycle Days by Category",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # Load data
    with st.spinner("Initializing Supply Chain Analytics Platform…"):
        tables = load_data()

    if not tables:
        st.error("Failed to load data. Please check the logs.")
        st.stop()

    # Sidebar navigation + filters
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            label="",
            options=[
                "📊 Overview",
                "💰 Spend Analysis",
                "🏭 Vendor Performance",
                "📈 Demand Forecasting",
                "📦 Inventory Optimization",
                "💡 Cost Optimization",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        filters = render_sidebar(tables)

        st.markdown("---")
        st.caption(
            "Supply Chain Analytics Platform\n"
            "Built with Python · Streamlit · Plotly\n"
            "Vishnu Kumar A.K."
        )

    # Route to page
    page_map = {
        "📊 Overview": page_overview,
        "💰 Spend Analysis": page_spend_analysis,
        "🏭 Vendor Performance": page_vendor_performance,
        "📈 Demand Forecasting": page_demand_forecast,
        "📦 Inventory Optimization": page_inventory,
        "💡 Cost Optimization": page_cost_optimization,
    }

    renderer = page_map.get(page, page_overview)
    renderer(tables, filters)


if __name__ == "__main__":
    main()
