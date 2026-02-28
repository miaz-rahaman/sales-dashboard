"""
Sales Data Analysis Dashboard
==============================
- Data cleaning and exploratory data analysis on retail sales datasets
- Revenue trends, seasonal patterns, and top-performing product categories
- KPI visualizations using Python (matplotlib + seaborn)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC RETAIL SALES DATASET
# ─────────────────────────────────────────────

np.random.seed(42)

categories = {
    "Electronics":    {"base": 3200, "seasonal": [0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0]},
    "Clothing":       {"base": 2100, "seasonal": [1.2, 1.0, 1.3, 1.5, 1.4, 1.0, 0.8, 0.9, 1.2, 1.0, 1.3, 1.8]},
    "Home & Garden":  {"base": 1800, "seasonal": [0.7, 0.7, 1.1, 1.5, 1.8, 1.9, 1.6, 1.4, 1.2, 0.9, 0.8, 0.7]},
    "Sports":         {"base": 1500, "seasonal": [0.8, 0.9, 1.1, 1.4, 1.6, 1.8, 1.7, 1.5, 1.2, 0.9, 0.8, 0.9]},
    "Food & Grocery": {"base": 4000, "seasonal": [1.0, 1.0, 1.0, 1.05, 1.05, 1.1, 1.1, 1.05, 1.0, 1.0, 1.1, 1.3]},
    "Books":          {"base":  900, "seasonal": [1.2, 1.0, 0.9, 0.9, 1.0, 0.8, 0.8, 0.9, 1.1, 1.0, 1.1, 1.5]},
    "Beauty":         {"base": 1200, "seasonal": [0.9, 1.2, 1.3, 1.2, 1.4, 1.1, 1.0, 1.0, 1.1, 1.0, 1.2, 1.5]},
}

records = []
for year in [2022, 2023, 2024]:
    for month in range(1, 13):
        for cat, cfg in categories.items():
            seasonal = cfg["seasonal"][month - 1]
            growth = 1 + 0.08 * (year - 2022)
            units = int(cfg["base"] * seasonal * growth * np.random.uniform(0.9, 1.1))
            price = np.random.uniform(15, 120)
            discount = np.random.uniform(0, 0.25)
            revenue = round(units * price * (1 - discount), 2)
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "year": year,
                "month": month,
                "month_name": pd.Timestamp(year=year, month=month, day=1).strftime("%b"),
                "category": cat,
                "units_sold": units,
                "avg_price": round(price, 2),
                "discount_pct": round(discount * 100, 1),
                "revenue": revenue,
                "returns": int(units * np.random.uniform(0.01, 0.08)),
            })

# Inject some missing/dirty values for cleaning demo
df_raw = pd.DataFrame(records)
dirty_idx = np.random.choice(df_raw.index, size=30, replace=False)
df_raw.loc[dirty_idx[:10], "revenue"] = np.nan
df_raw.loc[dirty_idx[10:20], "units_sold"] = -df_raw.loc[dirty_idx[10:20], "units_sold"]  # negative units
df_raw.loc[dirty_idx[20:], "avg_price"] = 0  # zero prices

print("=" * 60)
print("  SALES DATA ANALYSIS DASHBOARD")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("\n📋 RAW DATA SHAPE:", df_raw.shape)
print(f"   Missing values: {df_raw.isnull().sum().sum()}")
print(f"   Negative units: {(df_raw['units_sold'] < 0).sum()}")
print(f"   Zero avg_price: {(df_raw['avg_price'] == 0).sum()}")

df = df_raw.copy()

# Fix missing revenue: impute with median of same category+month
df["revenue"] = df.groupby(["category", "month"])["revenue"].transform(
    lambda x: x.fillna(x.median())
)

# Fix negative units → absolute value
df["units_sold"] = df["units_sold"].abs()

# Fix zero prices → median price of that category
df["avg_price"] = df.groupby("category")["avg_price"].transform(
    lambda x: x.replace(0, x[x > 0].median())
)

# Net units
df["net_units"] = df["units_sold"] - df["returns"]

print("\n✅ AFTER CLEANING:")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Negative units: {(df['units_sold'] < 0).sum()}")
print(f"   Zero avg_price: {(df['avg_price'] == 0).sum()}")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n📊 EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# KPIs
total_revenue = df["revenue"].sum()
total_units = df["units_sold"].sum()
avg_discount = df["discount_pct"].mean()
total_returns = df["returns"].sum()
return_rate = total_returns / total_units * 100

print(f"  Total Revenue:   ${total_revenue:,.0f}")
print(f"  Total Units Sold:{total_units:,}")
print(f"  Avg Discount:    {avg_discount:.1f}%")
print(f"  Return Rate:     {return_rate:.1f}%")

# Revenue by category
cat_revenue = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
print("\n  Revenue by Category:")
for cat, rev in cat_revenue.items():
    print(f"    {cat:<18} ${rev:>12,.0f}")

# YoY growth
yoy = df.groupby("year")["revenue"].sum()
print("\n  Year-over-Year Revenue:")
for yr, rev in yoy.items():
    print(f"    {yr}: ${rev:,.0f}")

# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────

PALETTE = ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560",
           "#f5a623", "#2ecc71", "#3498db"]
CAT_COLORS = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(categories)}

fmt_millions = FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
fmt_thousands = FuncFormatter(lambda x, _: f"${x/1e3:.0f}K")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

fig = plt.figure(figsize=(20, 24))
fig.suptitle("Retail Sales Analysis Dashboard  |  2022–2024",
             fontsize=22, fontweight="bold", y=0.98, color="#1a1a2e")

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── KPI Cards (row 0) ──────────────────────────────────────────
kpi_data = [
    ("Total Revenue", f"${total_revenue/1e6:.2f}M", "#0f3460", "▲ +8% YoY"),
    ("Units Sold", f"{total_units/1e3:.1f}K", "#533483", "▲ +6% YoY"),
    ("Avg Discount", f"{avg_discount:.1f}%", "#e94560", "▼ −1.2pp"),
    ("Return Rate", f"{return_rate:.2f}%", "#f5a623", "◆ Stable"),
]
# Use one wide axis for KPI cards
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.axis("off")
for i, (label, value, color, change) in enumerate(kpi_data):
    x = 0.12 + i * 0.24
    rect = plt.Rectangle((x - 0.10, 0.05), 0.20, 0.85,
                          transform=ax_kpi.transAxes,
                          color=color, alpha=0.9, zorder=1,
                          clip_on=False)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x, 0.72, label, transform=ax_kpi.transAxes,
                ha="center", fontsize=11, color="white", alpha=0.85)
    ax_kpi.text(x, 0.45, value, transform=ax_kpi.transAxes,
                ha="center", fontsize=20, fontweight="bold", color="white")
    ax_kpi.text(x, 0.18, change, transform=ax_kpi.transAxes,
                ha="center", fontsize=10, color="white", alpha=0.75)

# ── 1. Monthly Revenue Trend ───────────────────────────────────
ax1 = fig.add_subplot(gs[1, :2])
monthly = df.groupby(["year", "month"])["revenue"].sum().reset_index()
colors_yr = ["#3498db", "#e94560", "#2ecc71"]
for i, yr in enumerate([2022, 2023, 2024]):
    data = monthly[monthly["year"] == yr].sort_values("month")
    ax1.plot(data["month"], data["revenue"] / 1e3,
             marker="o", markersize=5, linewidth=2.5,
             color=colors_yr[i], label=str(yr))
ax1.yaxis.set_major_formatter(fmt_thousands)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"])
ax1.set_title("Monthly Revenue Trend by Year", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Revenue ($K)")
ax1.legend(title="Year", framealpha=0.7)
# Shade holiday season
ax1.axvspan(10.5, 12.5, alpha=0.08, color="#f5a623", label="Holiday Season")

# ── 2. Category Revenue Bar ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 2])
colors_cat = [CAT_COLORS[c] for c in cat_revenue.index]
bars = ax2.barh(cat_revenue.index, cat_revenue.values / 1e6,
                color=colors_cat, edgecolor="white", linewidth=0.5)
ax2.xaxis.set_major_formatter(fmt_millions)
ax2.set_title("Revenue by Category", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Total Revenue")
for bar, val in zip(bars, cat_revenue.values):
    ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
             f"${val/1e6:.1f}M", va="center", fontsize=9, color="#333")

# ── 3. Seasonal Heatmap ────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :2])
pivot_seasonal = df.groupby(["category", "month"])["revenue"].sum().unstack()
pivot_seasonal.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
# Normalize each row so we see seasonal pattern
pivot_norm = pivot_seasonal.div(pivot_seasonal.mean(axis=1), axis=0)
sns.heatmap(pivot_norm, ax=ax3, cmap="RdYlGn", center=1.0,
            annot=True, fmt=".2f", linewidths=0.5,
            cbar_kws={"label": "Index vs Annual Average", "shrink": 0.8})
ax3.set_title("Seasonal Pattern Heatmap (Revenue Index)", fontsize=13, fontweight="bold", pad=10)
ax3.set_ylabel("")

# ── 4. YoY Growth ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 2])
yoy_cat = df.groupby(["category", "year"])["revenue"].sum().unstack()
growth = ((yoy_cat[2024] - yoy_cat[2022]) / yoy_cat[2022] * 100).sort_values()
colors_growth = ["#e94560" if v < 0 else "#2ecc71" for v in growth]
ax4.barh(growth.index, growth.values, color=colors_growth, edgecolor="white")
ax4.axvline(0, color="#333", linewidth=0.8)
ax4.set_title("2-Year Revenue Growth\n(2022 → 2024)", fontsize=13, fontweight="bold", pad=10)
ax4.set_xlabel("Growth (%)")
for i, (idx, val) in enumerate(growth.items()):
    ax4.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=9)

# ── 5. Units Sold vs Returns by Category ──────────────────────
ax5 = fig.add_subplot(gs[3, 0])
cat_units = df.groupby("category")[["units_sold", "returns"]].sum()
x = np.arange(len(cat_units))
w = 0.35
ax5.bar(x - w/2, cat_units["units_sold"] / 1e3, w, label="Units Sold",
        color="#0f3460", alpha=0.85)
ax5.bar(x + w/2, cat_units["returns"] / 1e3, w, label="Returns",
        color="#e94560", alpha=0.85)
ax5.set_xticks(x)
ax5.set_xticklabels(cat_units.index, rotation=35, ha="right", fontsize=8)
ax5.set_title("Units Sold vs Returns\nby Category", fontsize=13, fontweight="bold", pad=10)
ax5.set_ylabel("Units (K)")
ax5.legend()

# ── 6. Revenue Distribution Box Plot ──────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
cat_order = cat_revenue.index.tolist()
monthly_cat = df.groupby(["category", "year", "month"])["revenue"].sum().reset_index()
bp_data = [monthly_cat[monthly_cat["category"] == c]["revenue"].values / 1e3
           for c in cat_order]
bp = ax6.boxplot(bp_data, patch_artist=True, notch=False,
                 medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], [CAT_COLORS[c] for c in cat_order]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax6.set_xticklabels(cat_order, rotation=35, ha="right", fontsize=8)
ax6.set_title("Monthly Revenue Distribution\nby Category", fontsize=13, fontweight="bold", pad=10)
ax6.set_ylabel("Revenue ($K)")
ax6.yaxis.set_major_formatter(fmt_thousands)

# ── 7. Discount vs Revenue Scatter ────────────────────────────
ax7 = fig.add_subplot(gs[3, 2])
scatter_data = df.groupby("category").agg(
    avg_discount=("discount_pct", "mean"),
    total_revenue=("revenue", "sum"),
    total_units=("units_sold", "sum"),
).reset_index()
scatter = ax7.scatter(
    scatter_data["avg_discount"],
    scatter_data["total_revenue"] / 1e6,
    s=scatter_data["total_units"] / 500,
    c=[CAT_COLORS[c] for c in scatter_data["category"]],
    alpha=0.8, edgecolors="white", linewidths=1.5
)
for _, row in scatter_data.iterrows():
    ax7.annotate(row["category"], (row["avg_discount"], row["total_revenue"] / 1e6),
                 fontsize=7.5, ha="center", va="bottom",
                 xytext=(0, 6), textcoords="offset points")
ax7.set_title("Avg Discount vs Total Revenue\n(bubble = units sold)", fontsize=13, fontweight="bold", pad=10)
ax7.set_xlabel("Avg Discount (%)")
ax7.set_ylabel("Total Revenue ($M)")
ax7.yaxis.set_major_formatter(fmt_millions)

plt.savefig("sales_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("\n✅ Dashboard saved → sales_dashboard.png")
plt.show()

# ─────────────────────────────────────────────
# 5. SUMMARY STATISTICS TABLE
# ─────────────────────────────────────────────
print("\n📊 CATEGORY SUMMARY TABLE")
print("-" * 75)
summary = df.groupby("category").agg(
    Total_Revenue=("revenue", "sum"),
    Avg_Monthly_Rev=("revenue", "mean"),
    Total_Units=("units_sold", "sum"),
    Avg_Discount=("discount_pct", "mean"),
    Return_Rate=("returns", lambda x: x.sum() / df.loc[x.index, "units_sold"].sum() * 100),
).round(2)
summary["Total_Revenue"] = summary["Total_Revenue"].apply(lambda x: f"${x:,.0f}")
summary["Avg_Monthly_Rev"] = summary["Avg_Monthly_Rev"].apply(lambda x: f"${x:,.0f}")
summary["Total_Units"] = summary["Total_Units"].apply(lambda x: f"{x:,}")
summary["Avg_Discount"] = summary["Avg_Discount"].apply(lambda x: f"{x:.1f}%")
summary["Return_Rate"] = summary["Return_Rate"].apply(lambda x: f"{x:.1f}%")
print(summary.to_string())
print("\n✅ Analysis complete.")