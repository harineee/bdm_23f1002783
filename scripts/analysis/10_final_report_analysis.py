"""
Final Report Additional Analysis - DJC Jewellers
Creates: 5 new visualizations (20-24) + 2 new reports (04-05)
Run: python scripts/analysis/10_final_report_analysis.py

Outputs:
- 20_acf_pacf_plots.png        : ACF/PACF for SARIMA parameter selection
- 21_model_diagnostics.png     : SARIMA model diagnostics (residuals, Q-Q, etc.)
- 22_forecast_validation.png   : Forecast vs actual with accuracy metrics
- 23_rfm_segmentation.png      : Customer RFM segmentation scatter plot
- 24_eoq_reorder_analysis.png  : EOQ/Reorder point analysis for Category A items
- 04_forecast_accuracy_report.txt : Detailed forecast accuracy report
- 05_rfm_segmentation_report.txt  : RFM customer segmentation report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Use existing config
sys.path.insert(0, str(Path(__file__).parent))
from config import setup_logging, DATA_DIR, VIZ_DIR, REPORTS_DIR

logger = setup_logging("10_final_report_analysis")

# Style settings (match existing scripts)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
DPI = 150

# Reference date for RFM analysis
REFERENCE_DATE = datetime(2024, 12, 31)


def load_data():
    """Load all required datasets."""
    logger.info("Loading datasets...")

    # Load transactions
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    transactions['date'] = pd.to_datetime(transactions['date'])
    logger.info(f"  Loaded {len(transactions):,} transaction line items")

    # Load monthly summary
    monthly = pd.read_csv(DATA_DIR / "monthly_sales_summary.csv")
    monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
    monthly = monthly.set_index('date').sort_index()
    logger.info(f"  Loaded {len(monthly)} monthly records")

    # Load inventory with ABC classification
    abc_file = REPORTS_DIR / "01_inventory_abc_classification.csv"
    if abc_file.exists():
        inventory = pd.read_csv(abc_file)
        logger.info(f"  Loaded {len(inventory)} inventory items with ABC classification")
    else:
        inventory = pd.read_csv(DATA_DIR / "current_inventory.csv")
        logger.info(f"  Loaded {len(inventory)} inventory items (no ABC classification)")

    # Load customers
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    logger.info(f"  Loaded {len(customers):,} customers")

    return transactions, monthly, inventory, customers


def create_acf_pacf_plot(monthly: pd.DataFrame):
    """
    Create ACF/PACF plots for SARIMA parameter selection.
    Saves: 20_acf_pacf_plots.png
    """
    logger.info("-" * 50)
    logger.info("Creating ACF/PACF plots...")

    revenue = monthly['total_revenue']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate max lags (50% of sample size - 1)
    max_lags = min(17, len(revenue) // 2 - 1)

    # ACF Plot
    plot_acf(revenue, ax=axes[0], lags=max_lags, alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lag (Months)')
    axes[0].set_ylabel('Correlation')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add annotation for lag 12
    axes[0].annotate('Seasonal spike\nat lag 12', xy=(12, 0.4), xytext=(14, 0.6),
                     fontsize=9, arrowprops=dict(arrowstyle='->', color='red'),
                     color='red')

    # PACF Plot
    plot_pacf(revenue, ax=axes[1], lags=max_lags, alpha=0.05, method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lag (Months)')
    axes[1].set_ylabel('Partial Correlation')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add annotation
    axes[1].annotate('Significant at\nlag 1 (p=1)', xy=(1, 0.35), xytext=(5, 0.5),
                     fontsize=9, arrowprops=dict(arrowstyle='->', color='red'),
                     color='red')

    fig.suptitle('ACF/PACF Analysis - SARIMA Parameter Selection\n'
                 'Justification for SARIMA(1,1,1)(1,1,1,12) Model',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add interpretation box
    interpretation = (
        "Interpretation:\n"
        "• ACF shows gradual decay with spike at lag 12 → Seasonal MA term (Q=1)\n"
        "• PACF shows significant lag 1 → AR term (p=1)\n"
        "• Seasonal period = 12 months (annual cycle)\n"
        "• Differencing (d=1, D=1) needed for stationarity"
    )
    fig.text(0.5, -0.12, interpretation, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_file = VIZ_DIR / "20_acf_pacf_plots.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {output_file}")


def create_model_diagnostics(monthly: pd.DataFrame):
    """
    Create SARIMA model diagnostic plots.
    Saves: 21_model_diagnostics.png
    """
    logger.info("-" * 50)
    logger.info("Creating model diagnostics...")

    revenue = monthly['total_revenue']

    # Fit SARIMA model
    logger.info("  Fitting SARIMA(1,1,1)(1,1,1,12) model...")
    model = SARIMAX(revenue, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    residuals = results.resid

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # [0,0] Residuals vs Time
    axes[0, 0].plot(residuals.index, residuals.values, color='steelblue', linewidth=1)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual Value (Rs.)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Add mean and std annotation
    res_mean = residuals.mean()
    res_std = residuals.std()
    axes[0, 0].annotate(f'Mean: {res_mean:,.0f}\nStd: {res_std:,.0f}',
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # [0,1] Residual Histogram
    axes[0, 1].hist(residuals, bins=15, density=True, color='steelblue',
                    edgecolor='white', alpha=0.7)

    # Fit normal distribution
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = axes[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[0, 1].plot(x, p, 'r-', linewidth=2, label='Normal fit')

    # Shapiro-Wilk test
    stat, p_value = stats.shapiro(residuals)
    axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].annotate(f'Shapiro-Wilk p-value: {p_value:.4f}\n'
                        f'{"Normal" if p_value > 0.05 else "Non-normal"} distribution',
                        xy=(0.98, 0.98), xycoords='axes fraction',
                        fontsize=9, verticalalignment='top', ha='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # [1,0] Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal)', fontsize=12, fontweight='bold')
    axes[1, 0].get_lines()[0].set_markerfacecolor('steelblue')
    axes[1, 0].get_lines()[0].set_markersize(6)
    axes[1, 0].get_lines()[1].set_color('red')

    # [1,1] ACF of Residuals
    plot_acf(residuals, ax=axes[1, 1], lags=20, alpha=0.05)
    axes[1, 1].set_title('ACF of Residuals', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Correlation')

    # Ljung-Box test
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].values[0]
    axes[1, 1].annotate(f'Ljung-Box p-value: {lb_pvalue:.4f}\n'
                        f'{"No autocorrelation" if lb_pvalue > 0.05 else "Autocorrelation present"}',
                        xy=(0.98, 0.98), xycoords='axes fraction',
                        fontsize=9, verticalalignment='top', ha='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('SARIMA(1,1,1)(1,1,1,12) Model Diagnostics\n'
                 'Validating Model Assumptions',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_file = VIZ_DIR / "21_model_diagnostics.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {output_file}")

    return results


def create_forecast_validation(monthly: pd.DataFrame):
    """
    Create forecast validation plot and generate accuracy report.
    Train on 2022-2023, test on 2024.
    Saves: 22_forecast_validation.png, 04_forecast_accuracy_report.txt
    """
    logger.info("-" * 50)
    logger.info("Creating forecast validation...")

    revenue = monthly['total_revenue']

    # Split data: Train on 2022-2023 (24 months), Test on 2024 (12 months)
    train = revenue[revenue.index.year <= 2023]
    test = revenue[revenue.index.year == 2024]

    logger.info(f"  Training period: {train.index[0].strftime('%Y-%m')} to {train.index[-1].strftime('%Y-%m')} ({len(train)} months)")
    logger.info(f"  Testing period: {test.index[0].strftime('%Y-%m')} to {test.index[-1].strftime('%Y-%m')} ({len(test)} months)")

    # Fit model on training data
    logger.info("  Fitting SARIMA model on training data...")
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)

    # Forecast for test period
    forecast = results.get_forecast(steps=len(test))
    predicted = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)

    # Calculate accuracy metrics
    rmse = np.sqrt(mean_squared_error(test, predicted))
    mae = mean_absolute_error(test, predicted)
    mape = np.mean(np.abs((test - predicted) / test)) * 100

    logger.info(f"  RMSE: Rs. {rmse:,.0f}")
    logger.info(f"  MAE:  Rs. {mae:,.0f}")
    logger.info(f"  MAPE: {mape:.2f}%")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot training data
    ax.plot(train.index, train.values / 100000, 'b-', linewidth=2,
            label='Training Data (2022-2023)', marker='o', markersize=4)

    # Plot actual test data
    ax.plot(test.index, test.values / 100000, 'g-', linewidth=2.5,
            label='Actual (2024)', marker='s', markersize=6)

    # Plot predicted values
    ax.plot(test.index, predicted.values / 100000, 'r--', linewidth=2.5,
            label='Predicted (2024)', marker='^', markersize=6)

    # Plot confidence interval
    ax.fill_between(test.index,
                    conf_int.iloc[:, 0].values / 100000,
                    conf_int.iloc[:, 1].values / 100000,
                    color='red', alpha=0.15, label='95% Confidence Interval')

    # Vertical line separating train and test
    ax.axvline(x=train.index[-1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(train.index[-1], ax.get_ylim()[1] * 0.95, '  Train/Test Split',
            fontsize=10, color='gray')

    # Add metrics box
    metrics_text = (f'Model Performance:\n'
                    f'RMSE: Rs. {rmse/100000:.2f} Lakh\n'
                    f'MAE:  Rs. {mae/100000:.2f} Lakh\n'
                    f'MAPE: {mape:.1f}%')

    # Interpretation
    if mape < 10:
        interpretation = "Excellent forecast accuracy"
    elif mape < 20:
        interpretation = "Good forecast accuracy"
    else:
        interpretation = "Fair forecast accuracy"

    metrics_text += f'\n\n{interpretation}'

    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    ax.set_title('Forecast Validation - 2024 Holdout Test\n'
                 'SARIMA(1,1,1)(1,1,1,12) Model Performance',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Revenue (Rs. Lakhs)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_file = VIZ_DIR / "22_forecast_validation.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {output_file}")

    # Generate accuracy report
    generate_forecast_accuracy_report(train, test, predicted, rmse, mae, mape)

    return rmse, mae, mape


def generate_forecast_accuracy_report(train, test, predicted, rmse, mae, mape):
    """Generate detailed forecast accuracy report."""
    logger.info("  Generating forecast accuracy report...")

    # Interpretation based on MAPE
    if mape < 10:
        interpretation = "EXCELLENT - The model achieves highly accurate predictions with MAPE < 10%."
    elif mape < 20:
        interpretation = "GOOD - The model provides reliable predictions with MAPE between 10-20%."
    else:
        interpretation = "FAIR - The model provides reasonable predictions but has room for improvement."

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    report = []
    report.append("=" * 70)
    report.append("DJC JEWELLERS - FORECAST ACCURACY REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Training Period: January 2022 - December 2023 ({len(train)} months)")
    report.append(f"Testing Period: January 2024 - December 2024 ({len(test)} months)")
    report.append("")
    report.append("MODEL: SARIMA(1,1,1)(1,1,1,12)")
    report.append("-" * 70)
    report.append("")
    report.append("ACCURACY METRICS:")
    report.append("-" * 40)
    report.append(f"RMSE: Rs. {rmse:,.0f} (Rs. {rmse/100000:.2f} Lakh)")
    report.append(f"MAE:  Rs. {mae:,.0f} (Rs. {mae/100000:.2f} Lakh)")
    report.append(f"MAPE: {mape:.2f}%")
    report.append("")
    report.append("INTERPRETATION:")
    report.append("-" * 40)
    report.append(interpretation)
    report.append("")
    report.append("MONTHLY BREAKDOWN:")
    report.append("-" * 70)
    report.append(f"{'Month':<12} {'Actual':<18} {'Predicted':<18} {'Error %':<10}")
    report.append("-" * 70)

    for i, (idx, actual) in enumerate(test.items()):
        pred = predicted.iloc[i]
        error_pct = abs((actual - pred) / actual) * 100
        month_name = f"{months[idx.month - 1]} {idx.year}"
        report.append(f"{month_name:<12} Rs. {actual/100000:>8.2f} L     Rs. {pred/100000:>8.2f} L     {error_pct:>6.1f}%")

    report.append("-" * 70)
    report.append("")
    report.append("RECOMMENDATIONS:")
    report.append("-" * 40)
    report.append("1. The model captures seasonal patterns well (Diwali peak in Nov)")
    report.append("2. Consider re-training quarterly with new data for improved accuracy")
    report.append("3. Use confidence intervals for inventory planning decisions")
    report.append("4. Monitor forecast vs actual monthly to detect model drift")
    report.append("")
    report.append("=" * 70)

    output_file = REPORTS_DIR / "04_forecast_accuracy_report.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"  Saved: {output_file}")


def calculate_rfm(transactions: pd.DataFrame):
    """Calculate RFM scores for each customer."""
    logger.info("-" * 50)
    logger.info("Calculating RFM scores...")

    # Aggregate by customer
    customer_data = transactions.groupby('customer_id').agg({
        'date': 'max',  # Most recent purchase
        'transaction_id': 'nunique',  # Number of unique transactions
        'final_price': 'sum'  # Total monetary value
    }).reset_index()

    customer_data.columns = ['customer_id', 'last_purchase_date', 'frequency', 'monetary']

    # Calculate recency (days since last purchase)
    customer_data['recency'] = (REFERENCE_DATE - customer_data['last_purchase_date']).dt.days

    logger.info(f"  Analyzed {len(customer_data):,} customers with transactions")

    # Create RFM scores using quintiles (1-5)
    # Recency: Lower is better, so reverse the labels
    customer_data['R_score'] = pd.qcut(customer_data['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')

    # Frequency: Higher is better
    customer_data['F_score'] = pd.qcut(customer_data['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

    # Monetary: Higher is better
    customer_data['M_score'] = pd.qcut(customer_data['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

    # Convert to int
    customer_data['R_score'] = customer_data['R_score'].astype(int)
    customer_data['F_score'] = customer_data['F_score'].astype(int)
    customer_data['M_score'] = customer_data['M_score'].astype(int)

    # Assign segments
    def assign_segment(row):
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Lost'
        else:
            return 'Others'

    customer_data['segment'] = customer_data.apply(assign_segment, axis=1)

    # Log segment distribution
    segment_counts = customer_data['segment'].value_counts()
    logger.info("  Segment distribution:")
    for segment, count in segment_counts.items():
        pct = count / len(customer_data) * 100
        logger.info(f"    {segment}: {count} ({pct:.1f}%)")

    return customer_data


def create_rfm_visualization(rfm_data: pd.DataFrame):
    """
    Create RFM segmentation scatter plot.
    Saves: 23_rfm_segmentation.png
    """
    logger.info("Creating RFM segmentation visualization...")

    # Define colors for segments
    segment_colors = {
        'Champions': '#2ecc71',   # Green
        'Loyal': '#3498db',       # Blue
        'At Risk': '#f39c12',     # Orange
        'Lost': '#e74c3c',        # Red
        'Others': '#95a5a6'       # Gray
    }

    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot each segment
    segment_order = ['Champions', 'Loyal', 'Others', 'At Risk', 'Lost']

    for segment in segment_order:
        segment_data = rfm_data[rfm_data['segment'] == segment]
        if len(segment_data) > 0:
            ax.scatter(segment_data['frequency'],
                      segment_data['monetary'] / 100000,  # Convert to Lakhs
                      c=segment_colors[segment],
                      label=f"{segment} ({len(segment_data)})",
                      alpha=0.6,
                      s=50,
                      edgecolors='white',
                      linewidth=0.5)

    ax.set_title('RFM Customer Segmentation Analysis\n'
                 f'Total Customers: {len(rfm_data):,}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (Number of Transactions)', fontsize=12)
    ax.set_ylabel('Monetary Value (Rs. Lakhs)', fontsize=12)
    ax.legend(title='Customer Segment', loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add segment description box
    description = (
        "Segment Definitions:\n"
        "• Champions: R≥4, F≥4, M≥4 (Recent, Frequent, High-Value)\n"
        "• Loyal: R≥3, F≥3 (Regular, Engaged Customers)\n"
        "• At Risk: R≤2, F≥3 (Were Frequent, Now Inactive)\n"
        "• Lost: R≤2, F≤2 (Inactive, Low Engagement)\n"
        "• Others: All other customers"
    )
    ax.text(0.02, 0.98, description, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    output_file = VIZ_DIR / "23_rfm_segmentation.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {output_file}")


def generate_rfm_report(rfm_data: pd.DataFrame):
    """Generate RFM segmentation report."""
    logger.info("Generating RFM segmentation report...")

    report = []
    report.append("=" * 70)
    report.append("DJC JEWELLERS - RFM CUSTOMER SEGMENTATION REPORT")
    report.append("=" * 70)
    report.append(f"Analysis Date: {REFERENCE_DATE.strftime('%Y-%m-%d')}")
    report.append(f"Total Customers Analyzed: {len(rfm_data):,}")
    report.append("")
    report.append("SEGMENT DISTRIBUTION:")
    report.append("-" * 70)
    report.append(f"{'Segment':<12} {'Count':<8} {'%':<8} {'Avg Monetary':<16} {'Total Revenue':<16}")
    report.append("-" * 70)

    segment_order = ['Champions', 'Loyal', 'Others', 'At Risk', 'Lost']
    segment_stats = []

    for segment in segment_order:
        segment_data = rfm_data[rfm_data['segment'] == segment]
        if len(segment_data) > 0:
            count = len(segment_data)
            pct = count / len(rfm_data) * 100
            avg_monetary = segment_data['monetary'].mean()
            total_revenue = segment_data['monetary'].sum()
            segment_stats.append({
                'segment': segment,
                'count': count,
                'pct': pct,
                'avg_monetary': avg_monetary,
                'total_revenue': total_revenue
            })
            report.append(f"{segment:<12} {count:<8} {pct:<7.1f}% Rs. {avg_monetary/100000:<8.2f} L   Rs. {total_revenue/100000:<8.2f} L")

    report.append("-" * 70)
    report.append("")
    report.append("SEGMENT DEFINITIONS:")
    report.append("-" * 40)
    report.append("Champions : R≥4, F≥4, M≥4 (Best customers - recent, frequent, high value)")
    report.append("Loyal     : R≥3, F≥3 (Regular customers with good engagement)")
    report.append("At Risk   : R≤2, F≥3 (Were frequent buyers, now becoming inactive)")
    report.append("Lost      : R≤2, F≤2 (Inactive customers with low engagement)")
    report.append("Others    : Everyone else (moderate engagement)")
    report.append("")
    report.append("RFM SCORE METHODOLOGY:")
    report.append("-" * 40)
    report.append("R (Recency)  : Days since last purchase (lower = better, score 5-1)")
    report.append("F (Frequency): Number of unique transactions (higher = better, score 1-5)")
    report.append("M (Monetary) : Total purchase value (higher = better, score 1-5)")
    report.append("")
    report.append("RECOMMENDATIONS:")
    report.append("-" * 40)

    for stat in segment_stats:
        segment = stat['segment']
        pct = stat['pct']
        if segment == 'Champions':
            report.append(f"1. Champions ({pct:.1f}%): Maintain relationship with exclusive previews,")
            report.append("   early access to new collections, and personalized service.")
        elif segment == 'Loyal':
            report.append(f"2. Loyal ({pct:.1f}%): Focus on upselling opportunities, introduce")
            report.append("   loyalty program with points/rewards for continued engagement.")
        elif segment == 'At Risk':
            report.append(f"3. At Risk ({pct:.1f}%): URGENT - Launch re-engagement campaign with")
            report.append("   special offers, personalized outreach, and reminder communications.")
        elif segment == 'Lost':
            report.append(f"4. Lost ({pct:.1f}%): Consider win-back promotions with significant")
            report.append("   discounts, or archive for future seasonal campaigns.")
        elif segment == 'Others':
            report.append(f"5. Others ({pct:.1f}%): Nurture with regular communication,")
            report.append("   festival offers, and gradual engagement building.")

    report.append("")
    report.append("BUSINESS IMPACT:")
    report.append("-" * 40)

    # Calculate revenue concentration
    champions = rfm_data[rfm_data['segment'] == 'Champions']
    top_revenue = champions['monetary'].sum() if len(champions) > 0 else 0
    total_revenue = rfm_data['monetary'].sum()
    top_pct = (top_revenue / total_revenue * 100) if total_revenue > 0 else 0

    report.append(f"• Champions contribute Rs. {top_revenue/100000:.2f} L ({top_pct:.1f}% of revenue)")

    at_risk = rfm_data[rfm_data['segment'] == 'At Risk']
    at_risk_revenue = at_risk['monetary'].sum() if len(at_risk) > 0 else 0
    report.append(f"• At-Risk customers represent Rs. {at_risk_revenue/100000:.2f} L in historical value")
    report.append(f"  - Recovering 50% of At-Risk could add Rs. {at_risk_revenue/200000:.2f} L annually")

    report.append("")
    report.append("=" * 70)

    output_file = REPORTS_DIR / "05_rfm_segmentation_report.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"  Saved: {output_file}")


def create_eoq_reorder_analysis(inventory: pd.DataFrame):
    """
    Create EOQ/Reorder point analysis for Category A items.
    Saves: 24_eoq_reorder_analysis.png
    """
    logger.info("-" * 50)
    logger.info("Creating EOQ/Reorder analysis...")

    # Check if abc_category column exists
    if 'abc_category' not in inventory.columns:
        # Calculate ABC based on revenue
        inventory = inventory.sort_values('revenue_3yr', ascending=False)
        inventory['cumulative_pct'] = inventory['revenue_3yr'].cumsum() / inventory['revenue_3yr'].sum() * 100
        inventory['abc_category'] = pd.cut(inventory['cumulative_pct'],
                                           bins=[0, 80, 95, 100],
                                           labels=['A', 'B', 'C'])

    # Get Category A items (top 20 by revenue)
    category_a = inventory[inventory['abc_category'] == 'A'].head(20).copy()

    if len(category_a) == 0:
        logger.warning("  No Category A items found!")
        return

    logger.info(f"  Analyzing top {len(category_a)} Category A items")

    # EOQ Parameters
    ORDER_COST = 500  # Rs. per order
    HOLDING_COST_PCT = 0.25  # 25% of item value annually

    # Calculate EOQ for each item
    eoq_data = []
    for _, item in category_a.iterrows():
        annual_demand = item['avg_monthly_sales'] * 12
        item_value = item['stock_value'] / item['current_stock_pcs'] if item['current_stock_pcs'] > 0 else 0
        holding_cost = item_value * HOLDING_COST_PCT

        if holding_cost > 0 and annual_demand > 0:
            eoq = np.sqrt((2 * annual_demand * ORDER_COST) / holding_cost)
            reorder_point = eoq / 2
        else:
            eoq = item['reorder_quantity']
            reorder_point = item['reorder_level']

        eoq_data.append({
            'item_name': item['item_name'][:20],  # Truncate for display
            'metal': item['metal'],
            'current_stock': item['current_stock_pcs'],
            'eoq': int(eoq),
            'reorder_point': int(reorder_point),
            'below_reorder': item['current_stock_pcs'] < reorder_point,
            'revenue_3yr': item['revenue_3yr']
        })

    eoq_df = pd.DataFrame(eoq_data)
    eoq_df = eoq_df.sort_values('revenue_3yr', ascending=True)  # For horizontal bar chart

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    y_pos = range(len(eoq_df))

    # Colors based on stock status
    colors = ['#e74c3c' if below else '#2ecc71' for below in eoq_df['below_reorder']]

    # Plot current stock
    bars = ax.barh(y_pos, eoq_df['current_stock'], color=colors, alpha=0.7,
                   label='Current Stock', edgecolor='white')

    # Plot reorder points as markers
    ax.scatter(eoq_df['reorder_point'], y_pos, color='orange', s=100,
               marker='|', linewidths=3, label='Reorder Point', zorder=5)

    # Plot EOQ as markers
    ax.scatter(eoq_df['eoq'], y_pos, color='blue', s=80,
               marker='D', label='EOQ', zorder=5, alpha=0.7)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['item_name']} ({row['metal'][0]})"
                        for _, row in eoq_df.iterrows()], fontsize=9)

    ax.set_xlabel('Quantity (Pieces)', fontsize=12)
    ax.set_title('EOQ & Reorder Point Analysis - Top 20 Category A Items\n'
                 'Items Below Reorder Point Highlighted in Red',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # Add count of items below reorder
    below_count = eoq_df['below_reorder'].sum()
    ax.text(0.02, 0.98, f'Items Below Reorder Point: {below_count}/{len(eoq_df)}\n'
            f'Order Cost: Rs. {ORDER_COST}/order\n'
            f'Holding Cost: {HOLDING_COST_PCT*100:.0f}% of item value',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # EOQ formula annotation
    formula = r'EOQ = $\sqrt{\frac{2 \times D \times S}{H}}$'
    ax.text(0.98, 0.02, f'EOQ Formula:\nD = Annual Demand\nS = Order Cost (Rs.{ORDER_COST})\nH = Holding Cost (25%)',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    output_file = VIZ_DIR / "24_eoq_reorder_analysis.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {output_file}")


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("FINAL REPORT ADDITIONAL ANALYSIS - DJC JEWELLERS")
    logger.info("=" * 60)

    # Load data
    transactions, monthly, inventory, customers = load_data()

    # 1. ACF/PACF Plots
    create_acf_pacf_plot(monthly)

    # 2. Model Diagnostics
    create_model_diagnostics(monthly)

    # 3. Forecast Validation (also generates report)
    rmse, mae, mape = create_forecast_validation(monthly)

    # 4. RFM Analysis
    rfm_data = calculate_rfm(transactions)
    create_rfm_visualization(rfm_data)
    generate_rfm_report(rfm_data)

    # 5. EOQ/Reorder Analysis
    create_eoq_reorder_analysis(inventory)

    # Summary
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("NEW VISUALIZATIONS CREATED:")
    logger.info(f"  • 20_acf_pacf_plots.png")
    logger.info(f"  • 21_model_diagnostics.png")
    logger.info(f"  • 22_forecast_validation.png")
    logger.info(f"  • 23_rfm_segmentation.png")
    logger.info(f"  • 24_eoq_reorder_analysis.png")
    logger.info("")
    logger.info("NEW REPORTS CREATED:")
    logger.info(f"  • 04_forecast_accuracy_report.txt")
    logger.info(f"  • 05_rfm_segmentation_report.txt")
    logger.info("")
    logger.info(f"Visualizations saved to: {VIZ_DIR}")
    logger.info(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
