"""
Seasonal Demand Forecasting for DJC Jewellers
Problem 2: Time Series Analysis and Forecasting

Analyses:
- Time series decomposition
- Stationarity tests (ADF)
- SARIMA model building
- 12-month demand forecast
- Festival impact quantification
- Procurement calendar generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.insert(0, str(Path(__file__).parent))
from config import setup_logging, DATA_DIR, VIZ_DIR, REPORTS_DIR

logger = setup_logging("08_demand_forecasting")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data():
    """Load required datasets."""
    logger.info("Loading data...")

    monthly = pd.read_csv(DATA_DIR / "monthly_sales_summary.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    transactions['date'] = pd.to_datetime(transactions['date'])

    logger.info(f"  Loaded {len(monthly)} monthly records")
    logger.info(f"  Loaded {len(transactions)} transactions")

    return monthly, transactions


def prepare_time_series(monthly: pd.DataFrame) -> pd.Series:
    """Prepare time series data for analysis."""
    logger.info("-" * 50)
    logger.info("PREPARING TIME SERIES DATA")
    logger.info("-" * 50)

    # Create datetime index
    monthly['date'] = pd.to_datetime(monthly['year'].astype(str) + '-' +
                                      monthly['month'].astype(str) + '-01')
    monthly = monthly.sort_values('date').set_index('date')

    # Use total revenue as the primary metric
    ts = monthly['total_revenue']

    logger.info(f"Time series range: {ts.index.min()} to {ts.index.max()}")
    logger.info(f"Number of observations: {len(ts)}")
    logger.info(f"Mean revenue: Rs. {ts.mean():,.0f}")
    logger.info(f"Std deviation: Rs. {ts.std():,.0f}")

    return ts, monthly


def test_stationarity(ts: pd.Series):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    logger.info("-" * 50)
    logger.info("STATIONARITY TEST (ADF)")
    logger.info("-" * 50)

    result = adfuller(ts, autolag='AIC')

    logger.info(f"ADF Statistic: {result[0]:.4f}")
    logger.info(f"p-value: {result[1]:.4f}")
    logger.info("Critical Values:")
    for key, value in result[4].items():
        logger.info(f"  {key}: {value:.4f}")

    if result[1] < 0.05:
        logger.info("Result: Series is STATIONARY (reject null hypothesis)")
        is_stationary = True
    else:
        logger.info("Result: Series is NON-STATIONARY (fail to reject null)")
        is_stationary = False

    return is_stationary, result


def decompose_time_series(ts: pd.Series):
    """Perform seasonal decomposition."""
    logger.info("-" * 50)
    logger.info("TIME SERIES DECOMPOSITION")
    logger.info("-" * 50)

    # Multiplicative decomposition (typical for revenue data)
    decomposition = seasonal_decompose(ts, model='multiplicative', period=12)

    logger.info("Decomposition completed (multiplicative model, period=12)")

    # Seasonal factors
    seasonal_factors = decomposition.seasonal[:12]
    logger.info("\nMonthly Seasonal Factors:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, (month, factor) in enumerate(zip(months, seasonal_factors)):
        pct = (factor - 1) * 100
        direction = "+" if pct > 0 else ""
        logger.info(f"  {month}: {factor:.3f} ({direction}{pct:.1f}%)")

    return decomposition


def create_decomposition_plot(decomposition, ts: pd.Series):
    """Create time series decomposition visualization."""
    logger.info("Creating decomposition plot...")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Original
    axes[0].plot(ts.index, ts.values / 1e5, color='steelblue', linewidth=1.5)
    axes[0].set_ylabel('Revenue (₹ Lakhs)')
    axes[0].set_title('Original Time Series', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Trend
    axes[1].plot(ts.index, decomposition.trend / 1e5, color='darkgreen', linewidth=2)
    axes[1].set_ylabel('Trend (₹ Lakhs)')
    axes[1].set_title('Trend Component', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Seasonal
    axes[2].plot(ts.index, decomposition.seasonal, color='darkorange', linewidth=1.5)
    axes[2].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Seasonal Factor')
    axes[2].set_title('Seasonal Component (Multiplicative)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Residual
    axes[3].plot(ts.index, decomposition.resid, color='red', linewidth=1, alpha=0.7)
    axes[3].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Residual')
    axes[3].set_title('Residual Component', fontweight='bold')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)

    plt.suptitle('Time Series Decomposition - DJC Jewellers Revenue\n(Multiplicative Model)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = VIZ_DIR / "04_time_series_decomposition.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def create_seasonal_pattern_chart(monthly: pd.DataFrame):
    """Create seasonal pattern visualization."""
    logger.info("Creating seasonal pattern chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Average revenue by month
    monthly_avg = monthly.groupby(monthly.index.month)['total_revenue'].mean()

    colors = ['#3498db'] * 12
    colors[10] = '#2ecc71'  # November - highest
    colors[6] = '#e74c3c'   # July - lowest

    axes[0].bar(months, monthly_avg.values / 1e5, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Month', fontsize=12)
    axes[0].set_ylabel('Average Revenue (₹ Lakhs)', fontsize=12)
    axes[0].set_title('Average Monthly Revenue Pattern', fontsize=12, fontweight='bold')
    axes[0].axhline(y=monthly_avg.mean() / 1e5, color='red', linestyle='--',
                    label=f'Mean: ₹{monthly_avg.mean()/1e5:.1f}L')
    axes[0].legend()

    # Year-over-year comparison
    for year in [2022, 2023, 2024]:
        year_data = monthly[monthly.index.year == year]
        axes[1].plot(range(1, len(year_data) + 1), year_data['total_revenue'].values / 1e5,
                     marker='o', label=str(year), linewidth=2, markersize=6)

    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(months, rotation=45)
    axes[1].set_xlabel('Month', fontsize=12)
    axes[1].set_ylabel('Revenue (₹ Lakhs)', fontsize=12)
    axes[1].set_title('Year-over-Year Revenue Comparison', fontsize=12, fontweight='bold')
    axes[1].legend(title='Year')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Seasonal Revenue Patterns - DJC Jewellers', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = VIZ_DIR / "05_seasonal_patterns.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def build_sarima_model(ts: pd.Series):
    """Build and fit SARIMA model."""
    logger.info("-" * 50)
    logger.info("BUILDING SARIMA MODEL")
    logger.info("-" * 50)

    # SARIMA parameters: (p,d,q)(P,D,Q,s)
    # Based on the seasonal nature with period 12
    # Using (1,1,1)(1,1,1,12) as a common starting point

    logger.info("Fitting SARIMA(1,1,1)(1,1,1,12) model...")

    model = SARIMAX(ts,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    results = model.fit(disp=False)

    logger.info(f"\nModel Summary:")
    logger.info(f"  AIC: {results.aic:.2f}")
    logger.info(f"  BIC: {results.bic:.2f}")
    logger.info(f"  Log Likelihood: {results.llf:.2f}")

    return results


def generate_forecast(model_results, ts: pd.Series, periods: int = 12):
    """Generate forecast for future periods."""
    logger.info("-" * 50)
    logger.info(f"GENERATING {periods}-MONTH FORECAST")
    logger.info("-" * 50)

    # Forecast
    forecast = model_results.get_forecast(steps=periods)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Create forecast dates
    last_date = ts.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                    periods=periods, freq='MS')
    forecast_mean.index = forecast_dates
    forecast_ci.index = forecast_dates

    logger.info("\nForecast Results:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for date, value in forecast_mean.items():
        month_name = months[date.month - 1]
        logger.info(f"  {date.strftime('%Y-%m')} ({month_name}): Rs. {value:,.0f}")

    total_forecast = forecast_mean.sum()
    logger.info(f"\nTotal Forecasted Revenue (12 months): Rs. {total_forecast:,.0f}")
    logger.info(f"Average Monthly Forecast: Rs. {total_forecast/12:,.0f}")

    return forecast_mean, forecast_ci


def create_forecast_plot(ts: pd.Series, forecast_mean: pd.Series, forecast_ci: pd.DataFrame):
    """Create forecast visualization."""
    logger.info("Creating forecast plot...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Historical data
    ax.plot(ts.index, ts.values / 1e5, color='steelblue', linewidth=2,
            label='Historical Revenue', marker='o', markersize=4)

    # Forecast
    ax.plot(forecast_mean.index, forecast_mean.values / 1e5, color='darkgreen',
            linewidth=2, label='Forecast', marker='s', markersize=6)

    # Confidence interval
    ax.fill_between(forecast_ci.index,
                    forecast_ci.iloc[:, 0] / 1e5,
                    forecast_ci.iloc[:, 1] / 1e5,
                    color='green', alpha=0.2, label='95% Confidence Interval')

    # Add vertical line at forecast start
    ax.axvline(x=ts.index[-1], color='red', linestyle='--', alpha=0.7,
               label='Forecast Start')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Revenue (₹ Lakhs)', fontsize=12)
    ax.set_title('Revenue Forecast - DJC Jewellers\nSARIMA(1,1,1)(1,1,1,12) Model',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = VIZ_DIR / "06_revenue_forecast.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def quantify_festival_impact(transactions: pd.DataFrame):
    """Quantify the impact of major festivals on sales."""
    logger.info("-" * 50)
    logger.info("FESTIVAL IMPACT ANALYSIS")
    logger.info("-" * 50)

    transactions['month'] = transactions['date'].dt.month
    transactions['year'] = transactions['date'].dt.year

    # Define festival periods (approximate)
    festivals = {
        'Diwali/Dhanteras (Oct-Nov)': [10, 11],
        'Akshaya Tritiya (Apr-May)': [4, 5],
        'Wedding Season (Nov-Feb)': [11, 12, 1, 2],
        'Ganesh Chaturthi (Aug-Sep)': [8, 9],
    }

    # Non-festival months for baseline
    non_festival_months = [3, 6, 7]

    baseline_avg = transactions[transactions['month'].isin(non_festival_months)].groupby(
        ['year', 'month'])['final_price'].sum().mean()

    logger.info(f"Baseline Monthly Revenue (Mar, Jun, Jul avg): Rs. {baseline_avg:,.0f}")
    logger.info("\nFestival Impact:")

    festival_impacts = []
    for festival, months in festivals.items():
        festival_rev = transactions[transactions['month'].isin(months)].groupby(
            ['year', 'month'])['final_price'].sum().mean()
        impact = ((festival_rev / baseline_avg) - 1) * 100
        festival_impacts.append({
            'festival': festival,
            'avg_monthly_revenue': festival_rev,
            'impact_pct': impact
        })
        logger.info(f"  {festival}: Rs. {festival_rev:,.0f} ({impact:+.1f}%)")

    return pd.DataFrame(festival_impacts)


def create_procurement_calendar(forecast_mean: pd.Series, monthly: pd.DataFrame):
    """Generate procurement calendar based on forecast."""
    logger.info("-" * 50)
    logger.info("PROCUREMENT CALENDAR")
    logger.info("-" * 50)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Calculate average lead time (assume 7 days for gold, 10 days for silver)
    lead_time_days = 10

    calendar = []
    for date, revenue in forecast_mean.items():
        # Estimate quantity needed based on revenue
        # Average item value ~Rs. 3,000 based on our data
        avg_item_value = 3000
        estimated_items = revenue / avg_item_value

        # Order should be placed lead_time_days before
        order_date = date - pd.DateOffset(days=lead_time_days)

        calendar.append({
            'sales_month': date.strftime('%Y-%m'),
            'month_name': months[date.month - 1],
            'forecasted_revenue': revenue,
            'estimated_items': int(estimated_items),
            'order_by_date': order_date.strftime('%Y-%m-%d'),
            'priority': 'HIGH' if date.month in [10, 11, 4, 5] else 'NORMAL'
        })

    calendar_df = pd.DataFrame(calendar)

    logger.info("\nProcurement Calendar (2025):")
    for _, row in calendar_df.iterrows():
        logger.info(f"  {row['month_name']} 2025: Order by {row['order_by_date']}, "
                   f"~{row['estimated_items']} items, Priority: {row['priority']}")

    return calendar_df


def generate_report(ts: pd.Series, forecast_mean: pd.Series,
                   festival_impacts: pd.DataFrame, calendar_df: pd.DataFrame):
    """Generate comprehensive forecasting report."""
    logger.info("-" * 50)
    logger.info("GENERATING FORECAST REPORT")
    logger.info("-" * 50)

    report = []
    report.append("=" * 70)
    report.append("DJC JEWELLERS - DEMAND FORECASTING REPORT")
    report.append("=" * 70)
    report.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)
    historical_annual = ts.sum() / 3
    forecast_annual = forecast_mean.sum()
    growth = ((forecast_annual / historical_annual) - 1) * 100

    report.append(f"Historical Average Annual Revenue: Rs. {historical_annual:,.0f}")
    report.append(f"Forecasted Revenue (Next 12 months): Rs. {forecast_annual:,.0f}")
    report.append(f"Projected Growth: {growth:+.1f}%")
    report.append("")

    # Seasonal Insights
    report.append("SEASONAL INSIGHTS")
    report.append("-" * 40)
    report.append("Peak Months: November (Diwali), May (Akshaya Tritiya)")
    report.append("Low Months: July (Adhik Maas), August (Shravan)")
    report.append("Peak-to-Trough Ratio: ~4x")
    report.append("")

    # Festival Impact
    report.append("FESTIVAL IMPACT")
    report.append("-" * 40)
    for _, row in festival_impacts.iterrows():
        report.append(f"{row['festival']}: {row['impact_pct']:+.1f}% vs baseline")
    report.append("")

    # Recommendations
    report.append("KEY RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("1. STOCK UP: Increase inventory 2 months before Diwali (Sep-Oct)")
    report.append("2. PROMOTIONS: Plan Akshaya Tritiya promotions by March")
    report.append("3. REDUCE: Minimize procurement during July-August")
    report.append("4. CASH FLOW: Prepare for revenue dip in Q3 (Jul-Sep)")
    report.append("5. STAFFING: Plan additional staff for Nov-Dec peak")
    report.append("")

    # Save report
    report_text = "\n".join(report)
    report_path = REPORTS_DIR / "02_demand_forecast_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"Report saved: {report_path}")

    # Save forecast to CSV
    forecast_df = pd.DataFrame({
        'date': forecast_mean.index,
        'forecasted_revenue': forecast_mean.values
    })
    forecast_df.to_csv(REPORTS_DIR / "02_revenue_forecast.csv", index=False)

    # Save procurement calendar
    calendar_df.to_csv(REPORTS_DIR / "02_procurement_calendar.csv", index=False)

    return report_text


def main():
    logger.info("=" * 60)
    logger.info("DEMAND FORECASTING - DJC JEWELLERS")
    logger.info("=" * 60)

    # Load data
    monthly, transactions = load_data()

    # Prepare time series
    ts, monthly_indexed = prepare_time_series(monthly)

    # Stationarity test
    is_stationary, adf_result = test_stationarity(ts)

    # Decomposition
    decomposition = decompose_time_series(ts)
    create_decomposition_plot(decomposition, ts)
    create_seasonal_pattern_chart(monthly_indexed)

    # Build SARIMA model
    model_results = build_sarima_model(ts)

    # Generate forecast
    forecast_mean, forecast_ci = generate_forecast(model_results, ts, periods=12)
    create_forecast_plot(ts, forecast_mean, forecast_ci)

    # Festival impact
    festival_impacts = quantify_festival_impact(transactions)

    # Procurement calendar
    calendar_df = create_procurement_calendar(forecast_mean, monthly_indexed)

    # Generate report
    report = generate_report(ts, forecast_mean, festival_impacts, calendar_df)

    logger.info("=" * 60)
    logger.info("DEMAND FORECASTING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Visualizations saved to: {VIZ_DIR}")
    logger.info(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
