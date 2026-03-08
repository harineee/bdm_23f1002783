
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
VIZ_DIR = BASE_DIR / 'outputs' / 'visualizations'

# Load data
print("Loading data...")
transactions = pd.read_csv(DATA_DIR / 'transactions.csv')
inventory = pd.read_csv(DATA_DIR / 'current_inventory.csv')
transactions['date'] = pd.to_datetime(transactions['date'])
transactions['year'] = transactions['date'].dt.year
transactions['month'] = transactions['date'].dt.month
transactions['day_of_week'] = transactions['date'].dt.day_name()
transactions['year_month'] = transactions['date'].dt.to_period('M')

print(f"Loaded {len(transactions)} transactions, {len(inventory)} inventory items")

# Colors
GOLD_COLOR = '#FFD700'
SILVER_COLOR = '#C0C0C0'
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95C623', '#5C4D7D', '#E84855']


def chart_10_gold_vs_silver():
    """10. Gold vs Silver Comparison - Two panels"""
    print("Creating chart 10: Gold vs Silver comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): Pie chart - Revenue share
    metal_revenue = transactions.groupby('metal')['final_price'].sum()
    colors = [GOLD_COLOR, SILVER_COLOR]

    wedges, texts, autotexts = axes[0].pie(
        metal_revenue,
        labels=['Gold', 'Silver'],
        autopct='%1.1f%%',
        colors=colors,
        explode=(0.02, 0),
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    axes[0].set_title('(a) Revenue Share by Metal Type', fontsize=14, fontweight='bold', pad=15)

    # Panel (b): Bar chart comparing metrics
    metal_stats = transactions.groupby('metal').agg({
        'final_price': 'sum',
        'weight_grams': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    metal_stats.columns = ['Metal', 'Revenue', 'Weight', 'Transactions']

    # Normalize for comparison
    x = np.arange(3)
    width = 0.35

    gold_data = metal_stats[metal_stats['Metal'] == 'Gold'].iloc[0]
    silver_data = metal_stats[metal_stats['Metal'] == 'Silver'].iloc[0]

    # Convert to comparable units
    gold_vals = [gold_data['Revenue']/1e7, gold_data['Weight']/1000, gold_data['Transactions']/1000]
    silver_vals = [silver_data['Revenue']/1e7, silver_data['Weight']/1000, silver_data['Transactions']/1000]

    bars1 = axes[1].bar(x - width/2, gold_vals, width, label='Gold', color=GOLD_COLOR, edgecolor='black')
    bars2 = axes[1].bar(x + width/2, silver_vals, width, label='Silver', color=SILVER_COLOR, edgecolor='black')

    axes[1].set_xlabel('Metric', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('(b) Gold vs Silver: Revenue, Weight, Transactions', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Revenue\n(Rs. Cr)', 'Weight\n(Kg)', 'Transactions\n(Thousands)'])
    axes[1].legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '10_gold_vs_silver_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 10_gold_vs_silver_comparison.png")


def chart_11_category_revenue():
    """11. Category Revenue Breakdown - Horizontal bar chart"""
    print("Creating chart 11: Category revenue breakdown...")

    fig, ax = plt.subplots(figsize=(12, 8))

    category_revenue = transactions.groupby('item_category')['final_price'].sum().sort_values(ascending=True)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(category_revenue)))

    bars = ax.barh(category_revenue.index, category_revenue.values / 1e5, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Revenue (Rs. Lakhs)', fontsize=12)
    ax.set_ylabel('Item Category', fontsize=12)
    ax.set_title('Revenue by Item Category (2022-2024)', fontsize=14, fontweight='bold', pad=15)

    # Add value labels
    for bar, val in zip(bars, category_revenue.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'Rs. {val/1e5:.1f}L', va='center', fontsize=9)

    ax.set_xlim(0, max(category_revenue.values)/1e5 * 1.15)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '11_category_revenue_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 11_category_revenue_breakdown.png")


def chart_12_yoy_growth():
    """12. Year-over-Year Growth - Bar chart with annotations"""
    print("Creating chart 12: YoY growth...")

    fig, ax = plt.subplots(figsize=(10, 7))

    yearly_revenue = transactions.groupby('year')['final_price'].sum() / 1e7
    years = yearly_revenue.index.tolist()
    revenues = yearly_revenue.values.tolist()

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(years, revenues, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Revenue (Rs. Crore)', fontsize=12)
    ax.set_title('Year-over-Year Revenue Growth (2022-2024)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(years)

    # Add revenue labels on bars
    for bar, rev in zip(bars, revenues):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'Rs. {rev:.2f} Cr', ha='center', fontsize=12, fontweight='bold')

    # Add growth annotations
    growth_2023 = ((revenues[1] - revenues[0]) / revenues[0]) * 100
    growth_2024 = ((revenues[2] - revenues[1]) / revenues[1]) * 100

    # Arrow annotations for growth
    ax.annotate(f'+{growth_2023:.0f}%', xy=(2022.5, (revenues[0] + revenues[1])/2),
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='green'))

    ax.annotate(f'+{growth_2024:.0f}%', xy=(2023.5, (revenues[1] + revenues[2])/2),
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='green'))

    ax.set_ylim(0, max(revenues) * 1.25)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '12_yoy_growth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 12_yoy_growth.png")


def chart_13_location_revenue():
    """13. Location Revenue - Bar chart top 8"""
    print("Creating chart 13: Location revenue...")

    fig, ax = plt.subplots(figsize=(12, 7))

    location_revenue = transactions.groupby('customer_location')['final_price'].sum().sort_values(ascending=False).head(8)

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(location_revenue)))[::-1]

    bars = ax.bar(location_revenue.index, location_revenue.values / 1e5, color=colors, edgecolor='black', linewidth=1)

    ax.set_xlabel('Customer Location', fontsize=12)
    ax.set_ylabel('Revenue (Rs. Lakhs)', fontsize=12)
    ax.set_title('Revenue by Customer Location (Top 8)', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, val in zip(bars, location_revenue.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'Rs. {val/1e5:.0f}L', ha='center', fontsize=10, fontweight='bold')

    # Add percentage annotation for Chikodi
    total = transactions['final_price'].sum()
    chikodi_pct = (location_revenue.iloc[0] / total) * 100
    ax.text(0, location_revenue.iloc[0]/1e5 * 0.5, f'{chikodi_pct:.1f}%\nof total',
            ha='center', fontsize=11, fontweight='bold', color='white')

    ax.set_ylim(0, max(location_revenue.values)/1e5 * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '13_location_revenue.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 13_location_revenue.png")


def chart_14_payment_analysis():
    """14. Payment Analysis - Two panels"""
    print("Creating chart 14: Payment analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): Pie chart - Revenue by payment mode
    payment_revenue = transactions.groupby('payment_mode')['final_price'].sum()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    wedges, texts, autotexts = axes[0].pie(
        payment_revenue,
        labels=payment_revenue.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )
    axes[0].set_title('(a) Revenue Share by Payment Mode', fontsize=14, fontweight='bold', pad=15)

    # Panel (b): Bar chart - Average discount by payment mode
    payment_discount = transactions.groupby('payment_mode')['discount_pct'].mean().sort_values(ascending=False)

    bars = axes[1].bar(payment_discount.index, payment_discount.values, color=colors, edgecolor='black')
    axes[1].set_xlabel('Payment Mode', fontsize=12)
    axes[1].set_ylabel('Average Discount (%)', fontsize=12)
    axes[1].set_title('(b) Average Discount by Payment Mode', fontsize=14, fontweight='bold', pad=15)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')

    axes[1].set_ylim(0, max(payment_discount.values) * 1.2)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '14_payment_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 14_payment_analysis.png")


def chart_15_day_of_week():
    """15. Day of Week Analysis - Bar chart highlighting weekends"""
    print("Creating chart 15: Day of week analysis...")

    fig, ax = plt.subplots(figsize=(11, 7))

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_revenue = transactions.groupby('day_of_week')['final_price'].sum().reindex(day_order)

    # Color weekends differently
    colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c']

    bars = ax.bar(day_order, day_revenue.values / 1e5, color=colors, edgecolor='black', linewidth=1)

    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Revenue (Rs. Lakhs)', fontsize=12)
    ax.set_title('Revenue by Day of Week (Weekends Highlighted)', fontsize=14, fontweight='bold', pad=15)

    # Add value labels
    for bar, val in zip(bars, day_revenue.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'Rs. {val/1e5:.0f}L', ha='center', fontsize=10, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', edgecolor='black', label='Weekdays'),
                      Patch(facecolor='#e74c3c', edgecolor='black', label='Weekends')]
    ax.legend(handles=legend_elements, loc='upper left')

    # Calculate weekend premium
    weekday_avg = day_revenue.iloc[:5].mean()
    weekend_avg = day_revenue.iloc[5:].mean()
    premium = ((weekend_avg - weekday_avg) / weekday_avg) * 100

    ax.text(0.95, 0.95, f'Weekend Premium: +{premium:.1f}%', transform=ax.transAxes,
            fontsize=12, fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'))

    ax.set_ylim(0, max(day_revenue.values)/1e5 * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '15_day_of_week.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 15_day_of_week.png")


def chart_16_gold_price_vs_revenue():
    """16. Gold Price vs Revenue - Dual-axis chart"""
    print("Creating chart 16: Gold price vs revenue...")

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Monthly revenue
    monthly_revenue = transactions.groupby(transactions['date'].dt.to_period('M'))['final_price'].sum()
    months = monthly_revenue.index.to_timestamp()

    # Monthly average gold rate (22K)
    monthly_gold = transactions[transactions['metal'] == 'Gold'].groupby(
        transactions[transactions['metal'] == 'Gold']['date'].dt.to_period('M')
    )['gold_rate_per_gram'].mean()
    gold_months = monthly_gold.index.to_timestamp()

    # Bar chart for revenue
    ax1.bar(months, monthly_revenue.values / 1e5, color='#3498db', alpha=0.7, label='Revenue', width=20)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Revenue (Rs. Lakhs)', fontsize=12, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.tick_params(axis='x', rotation=45)

    # Line chart for gold price on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(gold_months, monthly_gold.values, color='#FFD700', linewidth=3, marker='o',
             markersize=5, label='Gold 22K Price', markeredgecolor='black')
    ax2.set_ylabel('Gold 22K Rate (Rs./gram)', fontsize=12, color='#B8860B')
    ax2.tick_params(axis='y', labelcolor='#B8860B')

    # Title and legend
    ax1.set_title('Monthly Revenue vs Gold Price Trend (2022-2024)', fontsize=14, fontweight='bold', pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '16_gold_price_vs_revenue.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 16_gold_price_vs_revenue.png")


def chart_17_customer_segmentation():
    """17. Customer Segmentation - Three panels"""
    print("Creating chart 17: Customer segmentation...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    customer_types = ['Walk-in', 'Wedding', 'Wholesale']

    # Panel (a): Revenue share pie
    type_revenue = transactions.groupby('customer_type')['final_price'].sum().reindex(customer_types)

    wedges, texts, autotexts = axes[0].pie(
        type_revenue,
        labels=customer_types,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )
    axes[0].set_title('(a) Revenue Share', fontsize=13, fontweight='bold', pad=10)

    # Panel (b): Transaction count bars
    type_txn = transactions.groupby('customer_type')['transaction_id'].nunique().reindex(customer_types)

    bars = axes[1].bar(customer_types, type_txn.values, color=colors, edgecolor='black')
    axes[1].set_xlabel('Customer Type', fontsize=11)
    axes[1].set_ylabel('Number of Transactions', fontsize=11)
    axes[1].set_title('(b) Transaction Count', fontsize=13, fontweight='bold', pad=10)

    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f'{int(height):,}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')

    # Panel (c): Average discount bars
    type_discount = transactions.groupby('customer_type')['discount_pct'].mean().reindex(customer_types)

    bars = axes[2].bar(customer_types, type_discount.values, color=colors, edgecolor='black')
    axes[2].set_xlabel('Customer Type', fontsize=11)
    axes[2].set_ylabel('Average Discount (%)', fontsize=11)
    axes[2].set_title('(c) Average Discount', fontsize=13, fontweight='bold', pad=10)

    for bar in bars:
        height = bar.get_height()
        axes[2].annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('Customer Segmentation Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '17_customer_segmentation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 17_customer_segmentation.png")


def chart_18_inventory_blocked_capital():
    """18. Inventory Blocked Capital - Two panels"""
    print("Creating chart 18: Inventory blocked capital...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): Pie chart by stock status
    status_value = inventory.groupby('stock_status')['stock_value'].sum()
    status_order = ['Critical', 'Low', 'Normal', 'Excess', 'Dead Stock']
    status_value = status_value.reindex(status_order)

    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

    wedges, texts, autotexts = axes[0].pie(
        status_value,
        labels=status_order,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10}
    )
    axes[0].set_title('(a) Inventory Value by Stock Status', fontsize=13, fontweight='bold', pad=15)

    # Panel (b): Active vs Blocked capital
    active_status = ['Critical', 'Low', 'Normal']
    blocked_status = ['Excess', 'Dead Stock']

    active_capital = status_value[active_status].sum() / 1e5
    blocked_capital = status_value[blocked_status].sum() / 1e5

    bars = axes[1].bar(['Active\n(Can be sold soon)', 'Blocked\n(Excess + Dead Stock)'],
                       [active_capital, blocked_capital],
                       color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)

    axes[1].set_ylabel('Capital (Rs. Lakhs)', fontsize=12)
    axes[1].set_title('(b) Active vs Blocked Capital', fontsize=13, fontweight='bold', pad=15)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f'Rs. {height:.1f}L', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')

    # Add percentage annotation
    total = active_capital + blocked_capital
    blocked_pct = (blocked_capital / total) * 100
    axes[1].text(1, blocked_capital * 0.5, f'{blocked_pct:.1f}%\nblocked',
                ha='center', fontsize=11, fontweight='bold', color='white')

    axes[1].set_ylim(0, max(active_capital, blocked_capital) * 1.2)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '18_inventory_blocked_capital.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 18_inventory_blocked_capital.png")


def chart_19_monthly_heatmap():
    """19. Monthly Revenue Heatmap"""
    print("Creating chart 19: Monthly revenue heatmap...")

    fig, ax = plt.subplots(figsize=(14, 5))

    # Create pivot table: year x month
    transactions['month_name'] = transactions['date'].dt.month
    monthly_pivot = transactions.pivot_table(
        values='final_price',
        index='year',
        columns='month_name',
        aggfunc='sum'
    ) / 1e5  # Convert to lakhs

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pivot.columns = month_names

    # Create heatmap
    sns.heatmap(monthly_pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Revenue (Rs. Lakhs)'},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    ax.set_title('Monthly Revenue Heatmap (Rs. Lakhs) - Nov Peak, Jul Trough', fontsize=14, fontweight='bold', pad=15)

    # Highlight max and min
    max_val = monthly_pivot.values.max()
    min_val = monthly_pivot.values.min()

    # Find positions
    for i, year in enumerate(monthly_pivot.index):
        for j, month in enumerate(monthly_pivot.columns):
            val = monthly_pivot.loc[year, month]
            if val == max_val:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='green', lw=3))
            elif val == min_val:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    # Add legend for highlights
    ax.text(13.5, 0.5, 'Peak', fontsize=10, fontweight='bold', color='green')
    ax.text(13.5, 1.5, 'Trough', fontsize=10, fontweight='bold', color='blue')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / '19_monthly_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 19_monthly_heatmap.png")


def main():
    print("=" * 60)
    print("GENERATING ADDITIONAL VISUALIZATIONS (10-19)")
    print("=" * 60)

    chart_10_gold_vs_silver()
    chart_11_category_revenue()
    chart_12_yoy_growth()
    chart_13_location_revenue()
    chart_14_payment_analysis()
    chart_15_day_of_week()
    chart_16_gold_price_vs_revenue()
    chart_17_customer_segmentation()
    chart_18_inventory_blocked_capital()
    chart_19_monthly_heatmap()

    print("=" * 60)
    print("ALL 10 ADDITIONAL CHARTS GENERATED SUCCESSFULLY!")
    print(f"Output directory: {VIZ_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
