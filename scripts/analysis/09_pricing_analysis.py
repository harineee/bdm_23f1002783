"""
Pricing and Discount Analysis for DJC Jewellers
Problem 3: Pricing Strategy Optimization

Analyses:
- Discount patterns by customer type
- Revenue leakage calculation
- Discount vs volume correlation
- Making charges analysis
- Standardized discount policy recommendation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import setup_logging, DATA_DIR, VIZ_DIR, REPORTS_DIR

logger = setup_logging("09_pricing_analysis")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data():
    """Load required datasets."""
    logger.info("Loading data...")

    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    transactions['date'] = pd.to_datetime(transactions['date'])
    customers = pd.read_csv(DATA_DIR / "customers.csv")

    logger.info(f"  Loaded {len(transactions)} transaction records")
    logger.info(f"  Loaded {len(customers)} customers")

    return transactions, customers


def analyze_discount_patterns(transactions: pd.DataFrame):
    """Analyze discount patterns by customer type."""
    logger.info("-" * 50)
    logger.info("DISCOUNT PATTERN ANALYSIS")
    logger.info("-" * 50)

    # Overall statistics
    total_mrp = transactions['mrp'].sum()
    total_discount = transactions['discount_amount'].sum()
    total_revenue = transactions['final_price'].sum()
    avg_discount_pct = (total_discount / total_mrp) * 100

    logger.info(f"\nOverall Pricing Summary:")
    logger.info(f"  Total MRP: Rs. {total_mrp:,.0f}")
    logger.info(f"  Total Discounts Given: Rs. {total_discount:,.0f}")
    logger.info(f"  Net Revenue: Rs. {total_revenue:,.0f}")
    logger.info(f"  Average Discount: {avg_discount_pct:.2f}%")

    # By customer type
    logger.info(f"\nDiscount Patterns by Customer Type:")
    customer_stats = transactions.groupby('customer_type').agg({
        'mrp': 'sum',
        'discount_amount': 'sum',
        'discount_pct': ['mean', 'min', 'max', 'std'],
        'final_price': 'sum',
        'transaction_id': 'nunique'
    }).round(2)

    for ctype in ['Walk-in', 'Wedding', 'Wholesale']:
        mrp = customer_stats.loc[ctype, ('mrp', 'sum')]
        disc = customer_stats.loc[ctype, ('discount_amount', 'sum')]
        avg_disc = customer_stats.loc[ctype, ('discount_pct', 'mean')]
        min_disc = customer_stats.loc[ctype, ('discount_pct', 'min')]
        max_disc = customer_stats.loc[ctype, ('discount_pct', 'max')]
        txns = customer_stats.loc[ctype, ('transaction_id', 'nunique')]

        logger.info(f"\n  {ctype}:")
        logger.info(f"    Transactions: {txns:,}")
        logger.info(f"    MRP Total: Rs. {mrp:,.0f}")
        logger.info(f"    Discount Given: Rs. {disc:,.0f}")
        logger.info(f"    Avg Discount: {avg_disc:.1f}% (Range: {min_disc:.1f}% - {max_disc:.1f}%)")

    return customer_stats


def calculate_revenue_leakage(transactions: pd.DataFrame):
    """Calculate revenue leakage from excessive discounts."""
    logger.info("-" * 50)
    logger.info("REVENUE LEAKAGE ANALYSIS")
    logger.info("-" * 50)

    # Define standard discount thresholds
    standard_discounts = {
        'Walk-in': 3.0,      # Max 3%
        'Wedding': 8.0,      # Max 8%
        'Wholesale': 12.0    # Max 12%
    }

    transactions['standard_discount'] = transactions['customer_type'].map(standard_discounts)
    transactions['excess_discount_pct'] = np.maximum(0,
        transactions['discount_pct'] - transactions['standard_discount'])
    transactions['excess_discount_amt'] = transactions['mrp'] * transactions['excess_discount_pct'] / 100

    # Calculate leakage
    total_leakage = transactions['excess_discount_amt'].sum()
    transactions_with_excess = (transactions['excess_discount_pct'] > 0).sum()
    pct_with_excess = transactions_with_excess / len(transactions) * 100

    logger.info(f"\nRevenue Leakage Summary:")
    logger.info(f"  Standard Discount Policy:")
    for ctype, disc in standard_discounts.items():
        logger.info(f"    {ctype}: Max {disc}%")

    logger.info(f"\n  Transactions with Excess Discount: {transactions_with_excess:,} ({pct_with_excess:.1f}%)")
    logger.info(f"  Total Revenue Leakage: Rs. {total_leakage:,.0f}")
    logger.info(f"  Annual Leakage: Rs. {total_leakage/3:,.0f}")

    # By customer type
    logger.info(f"\n  Leakage by Customer Type:")
    for ctype in ['Walk-in', 'Wedding', 'Wholesale']:
        ctype_data = transactions[transactions['customer_type'] == ctype]
        ctype_leakage = ctype_data['excess_discount_amt'].sum()
        ctype_excess_txns = (ctype_data['excess_discount_pct'] > 0).sum()
        logger.info(f"    {ctype}: Rs. {ctype_leakage:,.0f} ({ctype_excess_txns} transactions)")

    return total_leakage, transactions


def create_discount_distribution_chart(transactions: pd.DataFrame):
    """Create discount distribution visualization."""
    logger.info("Creating discount distribution charts...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Discount distribution by customer type
    customer_types = ['Walk-in', 'Wedding', 'Wholesale']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for ctype, color in zip(customer_types, colors):
        data = transactions[transactions['customer_type'] == ctype]['discount_pct']
        axes[0, 0].hist(data, bins=20, alpha=0.6, label=ctype, color=color, edgecolor='black')

    axes[0, 0].set_xlabel('Discount %', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Discount Distribution by Customer Type', fontsize=12, fontweight='bold')
    axes[0, 0].legend()

    # 2. Box plot of discounts
    transactions.boxplot(column='discount_pct', by='customer_type', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Customer Type', fontsize=11)
    axes[0, 1].set_ylabel('Discount %', fontsize=11)
    axes[0, 1].set_title('Discount Range by Customer Type', fontsize=12, fontweight='bold')
    plt.suptitle('')  # Remove automatic title

    # 3. Average discount trend over time
    transactions['month'] = transactions['date'].dt.to_period('M')
    monthly_disc = transactions.groupby(['month', 'customer_type'])['discount_pct'].mean().unstack()
    monthly_disc.index = monthly_disc.index.astype(str)

    for ctype, color in zip(customer_types, colors):
        if ctype in monthly_disc.columns:
            axes[1, 0].plot(range(len(monthly_disc)), monthly_disc[ctype],
                           label=ctype, color=color, marker='o', markersize=3)

    axes[1, 0].set_xlabel('Month', fontsize=11)
    axes[1, 0].set_ylabel('Average Discount %', fontsize=11)
    axes[1, 0].set_title('Discount Trends Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    # Show only every 6th label
    ticks = axes[1, 0].get_xticks()
    axes[1, 0].set_xticks(ticks[::6])

    # 4. Discount vs Transaction Value scatter
    sample = transactions.sample(min(5000, len(transactions)), random_state=42)
    for ctype, color in zip(customer_types, colors):
        ctype_data = sample[sample['customer_type'] == ctype]
        axes[1, 1].scatter(ctype_data['final_price'] / 1000, ctype_data['discount_pct'],
                          alpha=0.4, label=ctype, color=color, s=20)

    axes[1, 1].set_xlabel('Transaction Value (₹ Thousands)', fontsize=11)
    axes[1, 1].set_ylabel('Discount %', fontsize=11)
    axes[1, 1].set_title('Discount vs Transaction Value', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 100)

    plt.suptitle('Discount Analysis - DJC Jewellers', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = VIZ_DIR / "07_discount_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def analyze_making_charges(transactions: pd.DataFrame):
    """Analyze making charges by category."""
    logger.info("-" * 50)
    logger.info("MAKING CHARGES ANALYSIS")
    logger.info("-" * 50)

    # By category
    category_mc = transactions.groupby('item_category').agg({
        'making_charges_pct': ['mean', 'min', 'max'],
        'making_charges_amt': 'sum',
        'base_metal_value': 'sum'
    }).round(2)

    logger.info(f"\nMaking Charges by Category:")
    for cat in category_mc.index:
        avg_mc = category_mc.loc[cat, ('making_charges_pct', 'mean')]
        min_mc = category_mc.loc[cat, ('making_charges_pct', 'min')]
        max_mc = category_mc.loc[cat, ('making_charges_pct', 'max')]
        total_mc = category_mc.loc[cat, ('making_charges_amt', 'sum')]
        logger.info(f"  {cat}: Avg {avg_mc:.1f}% (Range: {min_mc:.1f}%-{max_mc:.1f}%), "
                   f"Total: Rs. {total_mc:,.0f}")

    # By metal
    logger.info(f"\nMaking Charges by Metal:")
    for metal in ['Gold', 'Silver']:
        metal_data = transactions[transactions['metal'] == metal]
        avg_mc = metal_data['making_charges_pct'].mean()
        total_mc = metal_data['making_charges_amt'].sum()
        logger.info(f"  {metal}: Avg {avg_mc:.1f}%, Total: Rs. {total_mc:,.0f}")

    return category_mc


def create_making_charges_chart(transactions: pd.DataFrame):
    """Create making charges visualization."""
    logger.info("Creating making charges chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By category
    category_mc = transactions.groupby('item_category')['making_charges_pct'].mean().sort_values()

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(category_mc)))
    bars = axes[0].barh(category_mc.index, category_mc.values, color=colors)
    axes[0].set_xlabel('Average Making Charges %', fontsize=11)
    axes[0].set_ylabel('Category', fontsize=11)
    axes[0].set_title('Making Charges by Category', fontsize=12, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, category_mc.values):
        axes[0].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=9)

    # Making charges contribution to revenue
    mc_contribution = transactions.groupby('item_category').agg({
        'making_charges_amt': 'sum',
        'final_price': 'sum'
    })
    mc_contribution['mc_pct_of_revenue'] = (mc_contribution['making_charges_amt'] /
                                            mc_contribution['final_price'] * 100)
    mc_contribution = mc_contribution.sort_values('mc_pct_of_revenue')

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(mc_contribution)))
    bars = axes[1].barh(mc_contribution.index, mc_contribution['mc_pct_of_revenue'], color=colors)
    axes[1].set_xlabel('Making Charges as % of Revenue', fontsize=11)
    axes[1].set_ylabel('Category', fontsize=11)
    axes[1].set_title('Making Charges Contribution to Revenue', fontsize=12, fontweight='bold')

    plt.suptitle('Making Charges Analysis - DJC Jewellers', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = VIZ_DIR / "08_making_charges.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def analyze_discount_effectiveness(transactions: pd.DataFrame):
    """Analyze correlation between discount and purchase volume."""
    logger.info("-" * 50)
    logger.info("DISCOUNT EFFECTIVENESS ANALYSIS")
    logger.info("-" * 50)

    # Group by transaction to get transaction-level metrics
    txn_summary = transactions.groupby('transaction_id').agg({
        'customer_type': 'first',
        'final_price': 'sum',
        'discount_pct': 'mean',
        'discount_amount': 'sum',
        'quantity': 'sum',
        'item_name': 'count'  # Number of items
    }).rename(columns={'item_name': 'num_items'})

    # Correlation analysis
    logger.info(f"\nCorrelation Analysis:")

    # Discount vs Transaction Value
    corr_value = txn_summary['discount_pct'].corr(txn_summary['final_price'])
    logger.info(f"  Discount % vs Transaction Value: r = {corr_value:.3f}")

    # Discount vs Number of Items
    corr_items = txn_summary['discount_pct'].corr(txn_summary['num_items'])
    logger.info(f"  Discount % vs Number of Items: r = {corr_items:.3f}")

    # By customer type
    logger.info(f"\nDiscount Impact by Customer Type:")
    for ctype in ['Walk-in', 'Wedding', 'Wholesale']:
        ctype_data = txn_summary[txn_summary['customer_type'] == ctype]
        avg_value = ctype_data['final_price'].mean()
        avg_items = ctype_data['num_items'].mean()
        avg_disc = ctype_data['discount_pct'].mean()
        logger.info(f"  {ctype}: Avg Transaction Rs. {avg_value:,.0f}, "
                   f"Items: {avg_items:.1f}, Discount: {avg_disc:.1f}%")

    return txn_summary


def create_revenue_leakage_chart(transactions: pd.DataFrame):
    """Create revenue leakage visualization."""
    logger.info("Creating revenue leakage chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Standard vs Actual discounts
    standard = {'Walk-in': 3.0, 'Wedding': 8.0, 'Wholesale': 12.0}
    actual = transactions.groupby('customer_type')['discount_pct'].mean()

    x = np.arange(3)
    width = 0.35

    axes[0].bar(x - width/2, [standard['Walk-in'], standard['Wedding'], standard['Wholesale']],
                width, label='Standard Policy', color='#2ecc71', alpha=0.8)
    axes[0].bar(x + width/2, [actual['Walk-in'], actual['Wedding'], actual['Wholesale']],
                width, label='Actual Average', color='#e74c3c', alpha=0.8)

    axes[0].set_ylabel('Discount %', fontsize=11)
    axes[0].set_title('Standard vs Actual Discounts', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Walk-in', 'Wedding', 'Wholesale'])
    axes[0].legend()

    # Revenue breakdown: Retained vs Lost
    total_mrp = transactions['mrp'].sum()
    total_revenue = transactions['final_price'].sum()
    total_discount = transactions['discount_amount'].sum()

    # Standard discount amount
    transactions['standard_discount_amt'] = transactions.apply(
        lambda row: row['mrp'] * standard[row['customer_type']] / 100, axis=1
    )
    standard_discount_total = transactions['standard_discount_amt'].sum()
    excess_discount = total_discount - standard_discount_total

    labels = ['Net Revenue', 'Standard Discount', 'Excess Discount\n(Leakage)']
    sizes = [total_revenue, standard_discount_total, max(0, excess_discount)]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0, 0, 0.1)

    axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct=lambda p: f'₹{p*sum(sizes)/100/1e5:.1f}L',
                startangle=90, textprops={'fontsize': 10})
    axes[1].set_title('Revenue Breakdown', fontsize=12, fontweight='bold')

    plt.suptitle('Revenue Leakage Analysis - DJC Jewellers', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = VIZ_DIR / "09_revenue_leakage.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def generate_discount_policy(transactions: pd.DataFrame, leakage: float):
    """Generate recommended discount policy."""
    logger.info("-" * 50)
    logger.info("RECOMMENDED DISCOUNT POLICY")
    logger.info("-" * 50)

    policy = {
        'Walk-in': {
            'base': 2.0,
            'max': 5.0,
            'volume_bonus': 'Additional 1% for purchases > Rs. 50,000'
        },
        'Wedding': {
            'base': 5.0,
            'max': 10.0,
            'volume_bonus': 'Additional 2% for purchases > Rs. 2,00,000'
        },
        'Wholesale': {
            'base': 10.0,
            'max': 15.0,
            'volume_bonus': 'Negotiable for orders > Rs. 5,00,000'
        }
    }

    logger.info("\nRecommended Discount Policy:")
    for ctype, rules in policy.items():
        logger.info(f"\n  {ctype}:")
        logger.info(f"    Base Discount: {rules['base']}%")
        logger.info(f"    Maximum Discount: {rules['max']}%")
        logger.info(f"    Volume Bonus: {rules['volume_bonus']}")

    logger.info(f"\n  Special Occasions:")
    logger.info(f"    Dhanteras/Diwali: Additional 2% on all categories")
    logger.info(f"    Akshaya Tritiya: Additional 1% on gold items")
    logger.info(f"    First Purchase: Additional 1% for new customers")

    annual_savings = leakage / 3
    logger.info(f"\n  Estimated Annual Savings from Policy: Rs. {annual_savings:,.0f}")

    return policy


def generate_report(transactions: pd.DataFrame, customer_stats: pd.DataFrame,
                   leakage: float, policy: dict):
    """Generate comprehensive pricing report."""
    logger.info("-" * 50)
    logger.info("GENERATING PRICING REPORT")
    logger.info("-" * 50)

    total_mrp = transactions['mrp'].sum()
    total_discount = transactions['discount_amount'].sum()
    total_revenue = transactions['final_price'].sum()

    report = []
    report.append("=" * 70)
    report.append("DJC JEWELLERS - PRICING & DISCOUNT ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append(f"Total MRP (3 Years): Rs. {total_mrp:,.0f}")
    report.append(f"Total Discounts Given: Rs. {total_discount:,.0f}")
    report.append(f"Net Revenue: Rs. {total_revenue:,.0f}")
    report.append(f"Average Discount Rate: {total_discount/total_mrp*100:.2f}%")
    report.append(f"Revenue Leakage (Excess Discounts): Rs. {leakage:,.0f}")
    report.append(f"Annual Leakage: Rs. {leakage/3:,.0f}")
    report.append("")

    # Current Discount Patterns
    report.append("CURRENT DISCOUNT PATTERNS")
    report.append("-" * 40)
    for ctype in ['Walk-in', 'Wedding', 'Wholesale']:
        ctype_data = transactions[transactions['customer_type'] == ctype]
        avg_disc = ctype_data['discount_pct'].mean()
        max_disc = ctype_data['discount_pct'].max()
        report.append(f"{ctype}: Avg {avg_disc:.1f}%, Max {max_disc:.1f}%")
    report.append("")

    # Recommended Policy
    report.append("RECOMMENDED DISCOUNT POLICY")
    report.append("-" * 40)
    for ctype, rules in policy.items():
        report.append(f"{ctype}: {rules['base']}% - {rules['max']}%")
    report.append("")

    # Key Recommendations
    report.append("KEY RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("1. Implement standardized discount slabs by customer type")
    report.append("2. Require manager approval for discounts exceeding policy")
    report.append("3. Track discount patterns by salesperson for training")
    report.append("4. Offer volume-based bonuses instead of ad-hoc discounts")
    report.append("5. Create festival-specific promotional campaigns")
    report.append(f"\nEstimated Annual Savings: Rs. {leakage/3:,.0f}")
    report.append("")

    # Save report
    report_text = "\n".join(report)
    report_path = REPORTS_DIR / "03_pricing_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"Report saved: {report_path}")

    # Save discount analysis CSV
    discount_summary = transactions.groupby('customer_type').agg({
        'discount_pct': ['mean', 'min', 'max', 'std'],
        'discount_amount': 'sum',
        'mrp': 'sum',
        'final_price': 'sum'
    }).round(2)
    discount_summary.to_csv(REPORTS_DIR / "03_discount_summary.csv")

    return report_text


def main():
    logger.info("=" * 60)
    logger.info("PRICING & DISCOUNT ANALYSIS - DJC JEWELLERS")
    logger.info("=" * 60)

    # Load data
    transactions, customers = load_data()

    # Discount Pattern Analysis
    customer_stats = analyze_discount_patterns(transactions)
    create_discount_distribution_chart(transactions)

    # Revenue Leakage
    leakage, transactions = calculate_revenue_leakage(transactions)
    create_revenue_leakage_chart(transactions)

    # Making Charges Analysis
    category_mc = analyze_making_charges(transactions)
    create_making_charges_chart(transactions)

    # Discount Effectiveness
    txn_summary = analyze_discount_effectiveness(transactions)

    # Generate Policy Recommendations
    policy = generate_discount_policy(transactions, leakage)

    # Generate Report
    report = generate_report(transactions, customer_stats, leakage, policy)

    logger.info("=" * 60)
    logger.info("PRICING ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Visualizations saved to: {VIZ_DIR}")
    logger.info(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
