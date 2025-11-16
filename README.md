# Supply Chain & Procurement Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/charts-Plotly-3F4F75?style=flat-square&logo=plotly)](https://plotly.com/python/)
[![pandas](https://img.shields.io/badge/data-pandas-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?style=flat-square&logo=pytest)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?style=flat-square)](https://github.com/psyf00/black)

> A production-grade Python platform for supply chain cost reduction, vendor performance management, and procurement data intelligence — built on real-world patterns from procurement system implementations.

---

## Overview

This platform addresses the core analytical challenges of modern procurement operations: **Where is spend concentrated? Which vendors are underperforming? How much safety stock should we hold?** It was designed around procurement problems encountered in practice — the kind that arise when you're managing hundreds of vendors, tens of thousands of purchase orders, and need data-driven decisions instead of gut feel.

The codebase demonstrates:
- **End-to-end data pipelines** from raw PO data to executive PDF reports
- **Statistical models** — EOQ inventory optimization, Holt-Winters exponential smoothing, Pareto classification
- **Interactive dashboards** built with Streamlit and Plotly
- **Software engineering standards** — type hints, docstrings, unit tests, pydantic configuration

---

## Dashboard Screenshots

| Overview KPIs | Spend Treemap |
|---|---|
| ![Overview](docs/screenshots/overview.png) | ![Spend](docs/screenshots/spend_treemap.png) |

| Vendor Scorecard | Demand Forecast |
|---|---|
| ![Vendor](docs/screenshots/vendor_radar.png) | ![Forecast](docs/screenshots/demand_forecast.png) |

> **Run `make streamlit` to launch the live dashboard in your browser.**

---

## Key Features

### Spend Analysis
- **Pareto/ABC Classification** — identify the 20% of vendors driving 80% of spend with cumulative Pareto charts and class assignments (A/B/C)
- **Maverick Spend Detection** — flag off-contract purchases, emergency buys, and price premiums vs. contracted rates
- **Spend Concentration Index** — Herfindahl-Hirschman Index (HHI) to quantify supplier concentration risk
- **Category Trend Analysis** — monthly/quarterly spend trends per procurement category
- **Price Variance Analysis** — vendor price vs. category benchmark across all SKUs

### Vendor Performance Scorecards
- **Weighted composite score** across 4 dimensions: delivery (35%), quality (30%), cost (25%), responsiveness (10%)
- **Dimension scoring**:
  - Delivery: on-time delivery rate penalized by average days-late
  - Quality: inverse defect rate adjusted for inspection rejection rate
  - Cost: price premium/discount relative to category median price
  - Responsiveness: lead time vs. category average
- **Letter grade mapping** (A–F) with configurable thresholds
- **Tier benchmarking** — compare Tier 1/2/3 supplier cohort performance
- **Quarterly trend tracking** — detect score improvements or degradation over time

### Demand Forecasting
- **Facebook Prophet** — handles seasonality, holiday effects, and structural trend changes
- **Holt-Winters ETS** — triple exponential smoothing with additive trend and seasonality
- **Configurable confidence intervals** (80%, 90%, 95%) with shaded CI bands on charts
- **Per-category forecasting** — batch forecast for all top procurement categories
- **Model evaluation** — MAE and MAPE reported for each forecast

### Inventory Optimization
- **Economic Order Quantity (EOQ)** — Wilson formula: Q* = √(2DK/h)
- **Safety Stock** — Z × σ_demand × √(lead_time), parametrized by service level target
- **Reorder Points** — average demand during lead time + safety stock
- **ABC Classification** — A/B/C tiers by annual spend value (80/95/100% cumulative)
- **Inventory Turnover & Days of Supply** — computed per SKU and rolled up
- **EOQ Sensitivity Analysis** — how does optimal order quantity shift with demand changes?

### Cost Optimization
- **Total Cost of Ownership (TCO)** — material cost + logistics + quality cost + admin processing cost
- **Make vs. Buy Analysis** — NPV-based breakeven over configurable horizon and discount rate
- **Vendor Consolidation Opportunities** — categories with fragmented spend and estimated savings
- **Procurement Cycle Time** — average calendar days from PO creation to delivery, by vendor/category

### Reporting & Dashboard
- **Interactive Streamlit dashboard** — 6 pages with date/category/tier filters
- **PDF Executive Report** — auto-generated multi-page report with KPI scorecard, spend summary, vendor tiers, inventory parameters, and recommendations
- **Plotly visualizations** — treemaps, sunbursts, radar charts, waterfall charts, heatmaps, Pareto charts
- **KPI status tracking** — traffic-light status for 10 procurement KPIs

---

## KPI Definitions

| KPI | Formula | Target | Direction |
|---|---|---|---|
| **On-Time Delivery Rate** | On-time deliveries / Total deliveries × 100 | 95% | Higher better |
| **Supplier Defect Rate** | Total defects / Total units inspected × 100 | < 1% | Lower better |
| **Procurement Cycle Time** | Mean(delivery_date − order_date) across all POs | ≤ 14 days | Lower better |
| **Cost Savings %** | (Benchmark spend − Actual spend) / Benchmark spend × 100 | ≥ 5% | Higher better |
| **Inventory Turnover** | Annual demand / (EOQ / 2) | ≥ 12× | Higher better |
| **Fill Rate** | Fully fulfilled lines / Total order lines × 100 | ≥ 98% | Higher better |
| **Maverick Spend %** | Off-contract spend / Total spend × 100 | < 5% | Lower better |
| **Vendor Concentration (HHI)** | Σ(vendor_spend_share²) | < 0.10 | Lower better |
| **Contract Coverage %** | Contracted spend / Total spend × 100 | ≥ 85% | Higher better |
| **PO Approval Cycle Time** | Mean PO draft → approved status (business days) | ≤ 2 days | Lower better |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/vishnukumarak2002/supply-chain-analytics.git
cd supply-chain-analytics
pip install -r requirements.txt
```

### 2. Generate the sample dataset

```bash
python scripts/generate_data.py --n-pos 50000
# Creates 50K POs, 200 vendors, 1000 products in data/sample/
```

### 3. Run the analytics pipeline

```bash
python scripts/run_analysis.py
# Prints KPI scorecard, spend Pareto, vendor scores, inventory params
```

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
# Opens http://localhost:8501
```

### 5. Generate a PDF executive report

```bash
python scripts/create_report.py --output reports/procurement_report.pdf
```

### Using Make

```bash
make install           # Install dependencies
make generate-data     # Generate 50K POs
make streamlit         # Launch dashboard
make run-analysis      # Full CLI analysis
make create-report     # Generate PDF
make test              # Run unit tests
```

---

## Project Structure

```
supply-chain-analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── Makefile
├── config/
│   └── settings.py              # Pydantic-settings configuration
├── src/
│   ├── data/
│   │   ├── generator.py         # Synthetic procurement data generation
│   │   ├── loader.py            # CSV / Excel / Parquet / DB loader
│   │   └── validator.py         # Schema + business rule validation
│   ├── analysis/
│   │   ├── spend_analysis.py    # Pareto, maverick spend, HHI
│   │   ├── vendor_scoring.py    # Weighted vendor scorecards
│   │   ├── demand_forecast.py   # Prophet + Holt-Winters forecasting
│   │   ├── inventory_optimization.py  # EOQ, safety stock, ABC
│   │   └── cost_optimizer.py    # TCO, make vs buy, consolidation
│   ├── visualization/
│   │   ├── dashboards.py        # Radar, heatmap, waterfall, donut charts
│   │   ├── spend_charts.py      # Treemap, Pareto, sunburst, trend lines
│   │   └── forecast_plots.py    # Forecast CI charts, model comparison
│   ├── reporting/
│   │   ├── pdf_report.py        # ReportLab PDF generation
│   │   └── kpi_tracker.py       # KPI catalog + computation + status
│   └── pipeline/
│       └── etl.py               # Extract → Validate → Transform → Load
├── app/
│   └── streamlit_app.py         # Interactive 6-page dashboard
├── scripts/
│   ├── generate_data.py         # Generate sample datasets
│   ├── run_analysis.py          # Full CLI analytics pipeline
│   └── create_report.py         # PDF executive report
└── tests/
    ├── test_spend_analysis.py
    ├── test_vendor_scoring.py
    └── test_forecast.py
```

---

## Data Schema

### purchase_orders
| Column | Type | Description |
|---|---|---|
| `po_id` | string | Unique PO identifier (e.g. PO-0001234) |
| `po_line` | int | Line number within PO |
| `vendor_id` | string | Foreign key to vendors |
| `product_id` | string | Foreign key to products |
| `order_date` | date | Date PO was issued |
| `requested_delivery_date` | date | Requested delivery date |
| `quantity` | int | Units ordered |
| `unit_price` | float | Price per unit (USD) |
| `total_amount` | float | quantity × unit_price |
| `status` | string | Open / Received / Partially Received / Cancelled |
| `category` | string | Procurement category |
| `abc_class` | string | ABC classification (A/B/C) |
| `is_emergency` | bool | Emergency / unplanned purchase flag |

### vendors
| Column | Type | Description |
|---|---|---|
| `vendor_id` | string | Unique vendor identifier |
| `vendor_name` | string | Vendor display name |
| `tier` | string | Tier 1 / Tier 2 / Tier 3 |
| `category` | string | Primary supply category |
| `state` | string | US state |
| `payment_terms` | string | Net 30 / Net 45 / etc. |
| `lead_time_days_avg` | int | Average lead time (days) |
| `active` | bool | Vendor active status |

### deliveries
| Column | Type | Description |
|---|---|---|
| `delivery_id` | string | Unique delivery identifier |
| `po_id` | string | Foreign key to purchase_orders |
| `vendor_id` | string | Foreign key to vendors |
| `scheduled_date` | date | Originally requested delivery date |
| `actual_date` | date | Actual delivery date |
| `quantity_delivered` | int | Units received |
| `on_time` | bool | True if actual_date ≤ scheduled_date |
| `days_early_late` | int | Negative = early, positive = late |

### quality_inspections
| Column | Type | Description |
|---|---|---|
| `inspection_id` | string | Unique inspection identifier |
| `delivery_id` | string | Foreign key to deliveries |
| `quantity_inspected` | int | Units inspected |
| `defects_found` | int | Count of defective units |
| `defect_rate` | float | defects_found / quantity_inspected |
| `passed` | bool | True if defect_rate < 3% |
| `rejection_reason` | string | Reason for rejection (nullable) |

---

## Analysis Methodology

### Pareto / ABC Classification
Vendors and categories are sorted by descending spend. Cumulative spend % is calculated as a running sum of each entity's spend fraction. Class A = entities within the first 80% of cumulative spend; Class B = 80–95%; Class C = remainder. This maps directly to procurement prioritization: A-class suppliers receive formal contracts and quarterly business reviews.

### Vendor Scoring
Each dimension is independently normalized to [0, 100] before weighting:
- **Delivery**: `on_time_rate × 100 − avg_days_late_penalty`
- **Quality**: `(1 − avg_defect_rate) × 100 − rejection_penalty`
- **Cost**: `50 − price_premium_pct_vs_benchmark` (vendors below benchmark score >50)
- **Responsiveness**: inverse lead time, normalized to [0, 100] within the vendor population

Composite = weighted sum. Default weights: Delivery 35% · Quality 30% · Cost 25% · Responsiveness 10%.

### Inventory Optimization (EOQ Model)
The Wilson EOQ formula minimizes total annual inventory cost:

```
Q* = √(2 × D × K / h)
```

Where D = annual demand units, K = fixed ordering cost per PO, h = unit cost × annual holding rate.

Safety stock is calculated as `Z × σ_demand × √(lead_time)` where Z is the standard normal quantile for the target service level (e.g., Z = 1.645 for 95%). Reorder point = `avg_daily_demand × lead_time + safety_stock`.

### Demand Forecasting
**Holt-Winters** fits a state space model with additive trend and seasonal components, optimizing smoothing parameters α (level), β (trend), and γ (seasonal). Confidence intervals are derived from residual variance: `CI = forecast ± Z × σ × √(1 + t/n)`.

**Facebook Prophet** decomposes the time series as `y(t) = trend(t) + seasonality(t) + ε(t)`, handling change points in growth trend and configurable weekly/yearly seasonality Fourier components.

---

## Configuration

Environment variables (or `.env` file) override defaults:

```bash
# Data generation
DATA_N_VENDORS=200
DATA_N_PRODUCTS=1000
DATA_N_PURCHASE_ORDERS=50000
DATA_RANDOM_SEED=42

# Analysis
ANALYSIS_SERVICE_LEVEL=0.95
ANALYSIS_HOLDING_COST_RATE=0.25
ANALYSIS_ORDERING_COST=150.0
ANALYSIS_PARETO_THRESHOLD=0.80

# Reporting
REPORT_COMPANY_NAME="Acme Corp"
REPORT_CURRENCY=USD
```

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| Data manipulation | pandas | ≥ 2.1 |
| Numerical computing | numpy, scipy | ≥ 1.26 |
| Dashboard | Streamlit | ≥ 1.30 |
| Visualization | Plotly | ≥ 5.18 |
| Forecasting | statsmodels (Holt-Winters) | ≥ 0.14 |
| Forecasting | Prophet | ≥ 1.1 |
| Configuration | pydantic-settings | ≥ 2.1 |
| PDF generation | ReportLab | ≥ 4.0 |
| Data storage | PyArrow (Parquet) | ≥ 14.0 |
| Testing | pytest | ≥ 7.4 |

---

## Running Tests

```bash
# All tests
make test

# With coverage report
make test-cov

# Specific module
pytest tests/test_spend_analysis.py -v
pytest tests/test_vendor_scoring.py -v
pytest tests/test_forecast.py -v
```

Test coverage targets:
- `SpendAnalyzer` — Pareto correctness, HHI, maverick detection, edge cases
- `VendorScorer` — weight validation, score ranges [0,100], grade consistency
- `DemandForecaster` — forecast format, CI ordering, short-series handling

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Author

**Vishnu Kumar A.K.**  
Business Data Analyst | Python · SQL · Procurement Analytics  
[vishnukumarak2002@gmail.com](mailto:vishnukumarak2002@gmail.com)
