"""
Inventory Analysis for DJC Jewellers
Problem 1: Inventory Management Optimization

Analyses:
- ABC Analysis (Pareto classification)
- Dead stock identification
- Inventory turnover ratios
- Reorder point recommendations
- Capital blocked analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import setup_logging, DATA_DIR, VIZ_DIR, REPORTS_DIR

logger = setup_logging("07_inventory_analysis")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data():
    """Load required datasets."""
    logger.info("Loading data...")

    inventory = pd.read_csv(DATA_DIR / "current_inventory.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    transactions['date'] = pd.to_datetime(transactions['date'])

    logger.info(f"  Loaded {len(inventory)} inventory items")
    logger.info(f"  Loaded {len(transactions)} transactions")

    return inventory, transactions


def abc_analysis(inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Perform ABC Analysis based on revenue contribution.
    A: Top 80% revenue (typically 20% items)
    B: Next 15% revenue (typically 30% items)
    C: Bottom 5% revenue (typically 50% items)
    """
    logger.info("-" * 50)
    logger.info("PERFORMING ABC ANALYSIS")
    logger.info("-" * 50)

    # Calculate revenue contribution from 3-year sales
    df = inventory.copy()
    df['revenue_contribution'] = df['revenue_3yr']

    # Sort by revenue descending
    df = df.sort_values('revenue_contribution', ascending=False).reset_index(drop=True)

    # Calculate cumulative percentage
    total_revenue = df['revenue_contribution'].sum()
    df['cumulative_revenue'] = df['revenue_contribution'].cumsum()
    df['cumulative_pct'] = (df['cumulative_revenue'] / total_revenue) * 100

    # Assign ABC category
    def assign_abc(cum_pct):
        if cum_pct <= 80:
            return 'A'
        elif cum_pct <= 95:
            return 'B'
        else:
            return 'C'

    df['abc_category'] = df['cumulative_pct'].apply(assign_abc)

    # Summary statistics
    abc_summary = df.groupby('abc_category').agg({
        'item_id': 'count',
        'revenue_contribution': 'sum',
        'stock_value': 'sum'
    }).rename(columns={'item_id': 'num_items'})

    abc_summary['revenue_pct'] = (abc_summary['revenue_contribution'] / total_revenue * 100).round(1)
    abc_summary['items_pct'] = (abc_summary['num_items'] / len(df) * 100).round(1)

    logger.info("\nABC Analysis Summary:")
    for cat in ['A', 'B', 'C']:
        if cat in abc_summary.index:
            row = abc_summary.loc[cat]
            logger.info(f"  Category {cat}: {row['num_items']} items ({row['items_pct']}%), "
                       f"Revenue: {row['revenue_pct']}%, Stock Value: ₹{row['stock_value']:,.0f}")

    return df


def create_pareto_chart(df: pd.DataFrame):
    """Create Pareto chart for ABC analysis."""
    logger.info("Creating Pareto chart...")

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar chart for individual revenue
    colors = df['abc_category'].map({'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'})
    bars = ax1.bar(range(len(df)), df['revenue_contribution'] / 1e5, color=colors, alpha=0.7)
    ax1.set_xlabel('Items (sorted by revenue)', fontsize=12)
    ax1.set_ylabel('Revenue (₹ Lakhs)', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Line chart for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(range(len(df)), df['cumulative_pct'], color='darkred', linewidth=2, marker='')
    ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% (A cutoff)')
    ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% (B cutoff)')
    ax2.set_ylabel('Cumulative Revenue %', fontsize=12, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='center right')

    # Add legend for ABC categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='A - High Value'),
        Patch(facecolor='#f39c12', alpha=0.7, label='B - Medium Value'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='C - Low Value')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')

    plt.title('ABC Analysis - Pareto Chart\nDJC Jewellers Inventory', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = VIZ_DIR / "01_abc_pareto_chart.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def analyze_dead_stock(inventory: pd.DataFrame):
    """Analyze dead stock and blocked capital."""
    logger.info("-" * 50)
    logger.info("DEAD STOCK ANALYSIS")
    logger.info("-" * 50)

    # Filter by stock status
    dead_stock = inventory[inventory['stock_status'] == 'Dead Stock']
    excess_stock = inventory[inventory['stock_status'] == 'Excess']

    total_value = inventory['stock_value'].sum()
    dead_value = dead_stock['stock_value'].sum()
    excess_value = excess_stock['stock_value'].sum()
    blocked_capital = dead_value + excess_value

    logger.info(f"\nTotal Inventory Value: ₹{total_value:,.0f}")
    logger.info(f"Dead Stock Value: ₹{dead_value:,.0f} ({dead_value/total_value*100:.1f}%)")
    logger.info(f"Excess Stock Value: ₹{excess_value:,.0f} ({excess_value/total_value*100:.1f}%)")
    logger.info(f"Total Blocked Capital: ₹{blocked_capital:,.0f} ({blocked_capital/total_value*100:.1f}%)")

    # Top dead stock items
    logger.info(f"\nTop 10 Dead Stock Items by Value:")
    top_dead = dead_stock.nlargest(10, 'stock_value')[['item_name', 'metal', 'stock_value', 'months_of_stock', 'total_sold_3yr']]
    for _, row in top_dead.iterrows():
        logger.info(f"  {row['item_name']} ({row['metal']}): ₹{row['stock_value']:,.0f}, "
                   f"{row['months_of_stock']:.0f} months stock, {row['total_sold_3yr']} sold in 3yr")

    return dead_stock, excess_stock


def create_stock_status_chart(inventory: pd.DataFrame):
    """Create stock status distribution charts."""
    logger.info("Creating stock status charts...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Status distribution by count
    status_order = ['Critical', 'Low', 'Normal', 'Excess', 'Dead Stock']
    status_colors = {'Critical': '#e74c3c', 'Low': '#f39c12', 'Normal': '#2ecc71',
                     'Excess': '#3498db', 'Dead Stock': '#9b59b6'}

    status_counts = inventory['stock_status'].value_counts().reindex(status_order)
    colors = [status_colors[s] for s in status_counts.index]

    axes[0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[0].set_title('Stock Status by Item Count', fontsize=12, fontweight='bold')

    # Status distribution by value
    status_value = inventory.groupby('stock_status')['stock_value'].sum().reindex(status_order)
    colors = [status_colors[s] for s in status_value.index]

    axes[1].pie(status_value, labels=status_value.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Stock Status by Value', fontsize=12, fontweight='bold')

    plt.suptitle('Inventory Stock Status Distribution\nDJC Jewellers', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = VIZ_DIR / "02_stock_status_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def calculate_turnover_ratios(inventory: pd.DataFrame, transactions: pd.DataFrame):
    """Calculate inventory turnover ratios."""
    logger.info("-" * 50)
    logger.info("INVENTORY TURNOVER ANALYSIS")
    logger.info("-" * 50)

    # Calculate annual cost of goods sold (approximation using metal value)
    annual_cogs = transactions.groupby(
        transactions['date'].dt.year
    )['base_metal_value'].sum()

    avg_annual_cogs = annual_cogs.mean()
    current_inventory_value = inventory['stock_value'].sum()

    # Overall turnover ratio
    turnover_ratio = avg_annual_cogs / current_inventory_value
    days_in_inventory = 365 / turnover_ratio if turnover_ratio > 0 else float('inf')

    logger.info(f"\nOverall Inventory Metrics:")
    logger.info(f"  Average Annual COGS: ₹{avg_annual_cogs:,.0f}")
    logger.info(f"  Current Inventory Value: ₹{current_inventory_value:,.0f}")
    logger.info(f"  Inventory Turnover Ratio: {turnover_ratio:.2f}")
    logger.info(f"  Days in Inventory (DIO): {days_in_inventory:.0f} days")

    # By metal type
    logger.info(f"\nTurnover by Metal Type:")
    for metal in ['Gold', 'Silver']:
        metal_cogs = transactions[transactions['metal'] == metal]['base_metal_value'].sum() / 3
        metal_inv = inventory[inventory['metal'] == metal]['stock_value'].sum()
        metal_turnover = metal_cogs / metal_inv if metal_inv > 0 else 0
        metal_dio = 365 / metal_turnover if metal_turnover > 0 else float('inf')
        logger.info(f"  {metal}: Turnover = {metal_turnover:.2f}, DIO = {metal_dio:.0f} days")

    # By category (top 5)
    logger.info(f"\nTurnover by Category (Top 10):")
    cat_turnover = []
    for cat in inventory['item_category'].unique():
        cat_cogs = transactions[transactions['item_category'] == cat]['base_metal_value'].sum() / 3
        cat_inv = inventory[inventory['item_category'] == cat]['stock_value'].sum()
        if cat_inv > 0:
            cat_turnover.append({
                'category': cat,
                'turnover': cat_cogs / cat_inv,
                'inventory_value': cat_inv
            })

    cat_df = pd.DataFrame(cat_turnover).sort_values('turnover', ascending=False)
    for _, row in cat_df.head(10).iterrows():
        logger.info(f"  {row['category']}: Turnover = {row['turnover']:.2f}")

    return turnover_ratio, days_in_inventory


def create_turnover_chart(inventory: pd.DataFrame, transactions: pd.DataFrame):
    """Create turnover ratio visualization."""
    logger.info("Creating turnover charts...")

    # Calculate turnover by category
    cat_data = []
    for cat in inventory['item_category'].unique():
        cat_cogs = transactions[transactions['item_category'] == cat]['base_metal_value'].sum() / 3
        cat_inv = inventory[inventory['item_category'] == cat]['stock_value'].sum()
        if cat_inv > 0:
            cat_data.append({
                'category': cat,
                'turnover': cat_cogs / cat_inv,
                'inventory_value': cat_inv / 1e5
            })

    cat_df = pd.DataFrame(cat_data).sort_values('turnover', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#e74c3c' if t < 1 else '#f39c12' if t < 2 else '#2ecc71'
              for t in cat_df['turnover']]

    bars = ax.barh(cat_df['category'], cat_df['turnover'], color=colors, alpha=0.8)

    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Low turnover (<1)')
    ax.axvline(x=2, color='orange', linestyle='--', alpha=0.7, label='Medium turnover')
    ax.axvline(x=3, color='green', linestyle='--', alpha=0.7, label='Good turnover (>3)')

    ax.set_xlabel('Inventory Turnover Ratio (Annual)', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    ax.set_title('Inventory Turnover by Category\nDJC Jewellers', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')

    # Add value labels
    for bar, val in zip(bars, cat_df['turnover']):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    save_path = VIZ_DIR / "03_inventory_turnover.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def generate_reorder_recommendations(inventory: pd.DataFrame):
    """Generate reorder point recommendations."""
    logger.info("-" * 50)
    logger.info("REORDER RECOMMENDATIONS")
    logger.info("-" * 50)

    # Items needing immediate reorder
    critical = inventory[inventory['stock_status'] == 'Critical'].copy()
    low = inventory[inventory['stock_status'] == 'Low'].copy()

    logger.info(f"\nCRITICAL - Immediate Reorder Required ({len(critical)} items):")
    for _, row in critical.iterrows():
        logger.info(f"  {row['item_name']} ({row['metal']}): "
                   f"Stock={row['current_stock_pcs']}pcs, "
                   f"Reorder Qty={row['reorder_quantity']}pcs, "
                   f"Lead Time={row['supplier_lead_days']} days")

    logger.info(f"\nLOW STOCK - Reorder Soon ({len(low)} items):")
    for _, row in low.iterrows():
        logger.info(f"  {row['item_name']} ({row['metal']}): "
                   f"Stock={row['current_stock_pcs']}pcs, "
                   f"Monthly Sales={row['avg_monthly_sales']:.1f}pcs")

    # Calculate optimal safety stock (using simple formula)
    # Safety Stock = Z * σ * √L where Z=1.65 for 95% service level
    inventory['safety_stock'] = np.ceil(1.65 * inventory['avg_monthly_sales'] * 0.3 *
                                         np.sqrt(inventory['supplier_lead_days'] / 30))

    # Reorder point = (Daily demand * Lead time) + Safety stock
    inventory['optimal_reorder_point'] = np.ceil(
        (inventory['avg_monthly_sales'] / 30) * inventory['supplier_lead_days'] +
        inventory['safety_stock']
    )

    return critical, low


def generate_report(inventory: pd.DataFrame, abc_df: pd.DataFrame,
                   dead_stock: pd.DataFrame, turnover_ratio: float):
    """Generate comprehensive inventory report."""
    logger.info("-" * 50)
    logger.info("GENERATING INVENTORY REPORT")
    logger.info("-" * 50)

    report = []
    report.append("=" * 70)
    report.append("DJC JEWELLERS - INVENTORY ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)
    total_value = inventory['stock_value'].sum()
    blocked = inventory[inventory['stock_status'].isin(['Excess', 'Dead Stock'])]['stock_value'].sum()
    report.append(f"Total Inventory Value: Rs. {total_value:,.0f}")
    report.append(f"Capital Blocked (Excess + Dead): Rs. {blocked:,.0f} ({blocked/total_value*100:.1f}%)")
    report.append(f"Inventory Turnover Ratio: {turnover_ratio:.2f}")
    report.append(f"Days in Inventory: {365/turnover_ratio:.0f} days")
    report.append("")

    # ABC Analysis
    report.append("ABC ANALYSIS RESULTS")
    report.append("-" * 40)
    for cat in ['A', 'B', 'C']:
        cat_items = abc_df[abc_df['abc_category'] == cat]
        cat_rev = cat_items['revenue_contribution'].sum()
        report.append(f"Category {cat}: {len(cat_items)} items, Revenue: Rs. {cat_rev:,.0f}")
    report.append("")

    # Stock Status
    report.append("STOCK STATUS DISTRIBUTION")
    report.append("-" * 40)
    for status in ['Critical', 'Low', 'Normal', 'Excess', 'Dead Stock']:
        status_items = inventory[inventory['stock_status'] == status]
        status_value = status_items['stock_value'].sum()
        report.append(f"{status}: {len(status_items)} items, Value: Rs. {status_value:,.0f}")
    report.append("")

    # Recommendations
    report.append("KEY RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("1. IMMEDIATE ACTION: Reorder 12 critical stock items")
    report.append(f"2. LIQUIDATE: Consider discounting {len(dead_stock)} dead stock items")
    report.append(f"   Potential capital recovery: Rs. {dead_stock['stock_value'].sum():,.0f}")
    report.append("3. REDUCE: Review excess stock procurement policies")
    report.append("4. FOCUS: Prioritize Category A items for stock optimization")
    report.append("")

    # Save report
    report_text = "\n".join(report)
    report_path = REPORTS_DIR / "01_inventory_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"Report saved: {report_path}")

    # Save detailed CSV
    abc_df.to_csv(REPORTS_DIR / "01_inventory_abc_classification.csv", index=False)
    logger.info(f"ABC classification saved to CSV")

    return report_text


def main():
    logger.info("=" * 60)
    logger.info("INVENTORY ANALYSIS - DJC JEWELLERS")
    logger.info("=" * 60)

    # Load data
    inventory, transactions = load_data()

    # ABC Analysis
    abc_df = abc_analysis(inventory)
    create_pareto_chart(abc_df)

    # Dead Stock Analysis
    dead_stock, excess_stock = analyze_dead_stock(inventory)
    create_stock_status_chart(inventory)

    # Turnover Analysis
    turnover_ratio, dio = calculate_turnover_ratios(inventory, transactions)
    create_turnover_chart(inventory, transactions)

    # Reorder Recommendations
    critical, low = generate_reorder_recommendations(inventory)

    # Generate Report
    report = generate_report(inventory, abc_df, dead_stock, turnover_ratio)

    logger.info("=" * 60)
    logger.info("INVENTORY ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Visualizations saved to: {VIZ_DIR}")
    logger.info(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
