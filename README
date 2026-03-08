# DJC Jewellers — Business Data Analytics

Data-driven analysis of **DJC Jewellers**, a retail jewellery business operating in Chikodi, Karnataka, India. The project tackles three core business problems using 3 years of transactional data (2022–2024): inventory optimization, seasonal demand forecasting, and pricing strategy.

## Business Context

DJC Jewellers deals in gold and silver jewellery across categories like chains, bangles, rings, coins, and more. The business serves walk-in, wedding, and wholesale customers across towns in the Belagavi district. This analysis covers ~41,000 line items across ~11,700 transactions, 4,000 customers, and daily metal rate movements over 1,096 days.

## Key Findings

**Inventory:** 58.7% of inventory capital (₹73.6L of ₹1.25Cr) is blocked in excess and dead stock. ABC analysis shows 21 Category-A items drive 80% of revenue, while 22 dead-stock items hold ₹32.8L in locked capital.

**Demand:** Revenue follows strong seasonal patterns — November (Diwali/Dhanteras) peaks at ~4× the July trough. A SARIMA(1,1,1)(1,1,1,12) model forecasts the next 12 months with festival-adjusted procurement recommendations.

**Pricing:** Annual revenue leakage of ₹1.56L from inconsistent discounting. Wholesale customers average 11.6% discounts (max 17.9%) against a recommended cap of 15%. Standardized discount slabs by customer type are proposed.

## Project Structure

```
djc-bdm-main/
├── data/
│   ├── raw/                            # Core business data
│   │   ├── transactions.csv                 41,349 line items (2022-2024)
│   │   ├── customers.csv                     4,000 customer records
│   │   ├── daily_metal_rates.csv             1,096 daily gold & silver prices
│   │   ├── current_inventory.csv               180 SKUs with stock status
│   │   ├── monthly_sales_summary.csv            36 monthly aggregates
│   │   └── monthly_metal_rates_summary.csv      36 monthly rate averages
│   └── processed/                      # Cleaned item master
│       ├── gold_items.csv                    82 gold product lines
│       ├── silver_items.csv                  98 silver product lines
│       └── all_items.csv                    180 combined items
├── scripts/
│   └── analysis/
│       ├── config.py                        Paths and logging setup
│       ├── 07_inventory_analysis.py         ABC, dead stock, turnover, reorder points
│       ├── 08_demand_forecasting.py         SARIMA, seasonality, procurement calendar
│       ├── 09_pricing_analysis.py           Discounts, leakage, policy recommendations
│       ├── 10_final_report_analysis.py      Forecast validation, RFM segmentation, EOQ
│       ├── generate_additional_charts.py    Extended visualizations (charts 10–24)
│       └── run_all_analysis.py              Master runner for all analysis scripts
└── outputs/
    ├── visualizations/                 24 analysis charts (PNG)
    └── reports/                        Analysis reports and CSVs
```

## Datasets

| Dataset | Records | Description |
|---|---|---|
| `transactions.csv` | 41,349 | Every sale line item — date, customer, item, weight, metal rate, making charges, GST, MRP, discount, final price, payment mode |
| `customers.csv` | 4,000 | Customer master — type (walk-in / wedding / wholesale), location, purchase history, preferences |
| `daily_metal_rates.csv` | 1,096 | Daily gold (24K, 22K) and silver rates per gram with daily change % |
| `current_inventory.csv` | 180 | Stock snapshot — quantities, values, status (critical / low / normal / excess / dead), reorder points |
| `monthly_sales_summary.csv` | 36 | Monthly revenue, weight sold, discount %, customer counts, metal rate averages |

## Analysis Modules

### 1. Inventory Management (`07_inventory_analysis.py`)
ABC/Pareto classification, dead stock identification, inventory turnover ratios, capital blocked analysis, and reorder point recommendations using lead time and average monthly sales.

### 2. Demand Forecasting (`08_demand_forecasting.py`)
Time series decomposition, ADF stationarity testing, SARIMA model fitting, 12-month revenue forecast, festival impact quantification (Diwali +126%, Akshaya Tritiya +75%), and a month-by-month procurement calendar.

### 3. Pricing Strategy (`09_pricing_analysis.py`)
Discount pattern analysis by customer type, revenue leakage estimation, discount-vs-volume correlation, making charges analysis, and a recommended standardized discount policy.

### 4. Extended Analytics (`10_final_report_analysis.py`)
ACF/PACF plots for SARIMA parameter selection, model diagnostics, forecast-vs-actual validation with accuracy metrics (RMSE, MAE, MAPE), RFM customer segmentation, and EOQ/reorder point analysis.

## Usage

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scipy statsmodels

# Run all analyses
python scripts/analysis/run_all_analysis.py

# Or run individually
python scripts/analysis/07_inventory_analysis.py
python scripts/analysis/08_demand_forecasting.py
python scripts/analysis/09_pricing_analysis.py
python scripts/analysis/10_final_report_analysis.py
python scripts/analysis/generate_additional_charts.py
```

Outputs are saved to `outputs/visualizations/` (charts) and `outputs/reports/` (text and CSV reports).

## Tech Stack

Python 3.10+ with pandas, NumPy, matplotlib, seaborn, SciPy, and statsmodels.

## License

MIT — see [LICENSE](LICENSE).
