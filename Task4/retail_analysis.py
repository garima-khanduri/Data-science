"""
Task 4 — Real-World Data Project: Retail Sales Analytics

End-to-end analysis of retail transaction data.
The bundled dataset is synthetic and reproducible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "retail_transactions.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    # -----------------------------
    # 1. Load and validate
    # -----------------------------
    df = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])

    required_columns = {
        "InvoiceID", "InvoiceDate", "CustomerID", "Country",
        "Category", "Product", "Quantity", "UnitPrice",
        "Discount", "Status", "Revenue"
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    print("=" * 70)
    print("RETAIL SALES ANALYTICS")
    print("=" * 70)
    print("Dataset shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))

    # -----------------------------
    # 2. Prepare completed sales
    # -----------------------------
    completed = df[df["Status"] == "Completed"].copy()
    cancelled = df[df["Status"] == "Cancelled"].copy()

    completed["GrossSales"] = completed["Quantity"] * completed["UnitPrice"]
    completed["NetRevenue"] = completed["GrossSales"] * (1 - completed["Discount"])

    # -----------------------------
    # 3. KPI summary
    # -----------------------------
    total_revenue = completed["NetRevenue"].sum()
    total_invoices = completed["InvoiceID"].nunique()
    total_customers = completed["CustomerID"].nunique()
    total_units = completed["Quantity"].sum()
    average_order_value = total_revenue / total_invoices
    cancellation_rate = len(cancelled) / len(df) * 100

    kpis = pd.DataFrame({
        "Metric": [
            "Total completed revenue",
            "Completed invoices",
            "Unique customers",
            "Units sold",
            "Average order value",
            "Cancellation rate (%)",
        ],
        "Value": [
            total_revenue,
            total_invoices,
            total_customers,
            total_units,
            average_order_value,
            cancellation_rate,
        ],
    })
    kpis.to_csv(OUTPUT_DIR / "kpi_summary.csv", index=False)

    print("\nKPI Summary:")
    print(kpis.to_string(index=False))

    # -----------------------------
    # 4. Monthly revenue
    # -----------------------------
    completed["Month"] = completed["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = completed.groupby("Month", as_index=False)["NetRevenue"].sum()

    plt.figure(figsize=(11, 5))
    plt.plot(monthly["Month"], monthly["NetRevenue"], marker="o")
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    save_plot(OUTPUT_DIR / "monthly_revenue.png")

    # -----------------------------
    # 5. Category analysis
    # -----------------------------
    category_summary = (
        completed.groupby("Category")
        .agg(
            Revenue=("NetRevenue", "sum"),
            Units=("Quantity", "sum"),
            Orders=("InvoiceID", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    category_summary["RevenueShare_%"] = (
        category_summary["Revenue"] / category_summary["Revenue"].sum() * 100
    )
    category_summary.to_csv(OUTPUT_DIR / "category_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(category_summary["Category"], category_summary["Revenue"])
    plt.title("Revenue by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue")
    plt.xticks(rotation=20)
    save_plot(OUTPUT_DIR / "category_revenue.png")

    # -----------------------------
    # 6. Country analysis
    # -----------------------------
    country_summary = (
        completed.groupby("Country")
        .agg(
            Revenue=("NetRevenue", "sum"),
            Units=("Quantity", "sum"),
            Orders=("InvoiceID", "nunique"),
            Customers=("CustomerID", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    country_summary["RevenueShare_%"] = (
        country_summary["Revenue"] / country_summary["Revenue"].sum() * 100
    )
    country_summary.to_csv(OUTPUT_DIR / "country_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(country_summary["Country"], country_summary["Revenue"])
    plt.title("Revenue by Country")
    plt.xlabel("Country")
    plt.ylabel("Revenue")
    plt.xticks(rotation=20)
    save_plot(OUTPUT_DIR / "country_revenue.png")

    # -----------------------------
    # 7. Top products
    # -----------------------------
    product_summary = (
        completed.groupby("Product")
        .agg(
            Revenue=("NetRevenue", "sum"),
            Units=("Quantity", "sum"),
        )
        .sort_values("Revenue", ascending=False)
        .head(10)
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    plt.barh(product_summary["Product"], product_summary["Revenue"])
    plt.title("Top 10 Products by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Product")
    plt.gca().invert_yaxis()
    save_plot(OUTPUT_DIR / "top_products.png")

    # -----------------------------
    # 8. Customer analysis
    # -----------------------------
    customer_summary = (
        completed.groupby("CustomerID")
        .agg(
            Revenue=("NetRevenue", "sum"),
            Orders=("InvoiceID", "nunique"),
            Units=("Quantity", "sum"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    customer_summary["AverageOrderValue"] = (
        customer_summary["Revenue"] / customer_summary["Orders"]
    )
    customer_summary.to_csv(OUTPUT_DIR / "customer_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(customer_summary["Revenue"], bins=40)
    plt.title("Distribution of Customer Revenue")
    plt.xlabel("Customer Revenue")
    plt.ylabel("Number of Customers")
    save_plot(OUTPUT_DIR / "customer_revenue_distribution.png")

    # -----------------------------
    # 9. Discount vs revenue
    # -----------------------------
    discount_summary = (
        completed.groupby("Discount")
        .agg(
            Revenue=("NetRevenue", "sum"),
            Orders=("InvoiceID", "nunique"),
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(
        discount_summary["Discount"],
        discount_summary["Revenue"],
        s=discount_summary["Orders"] / discount_summary["Orders"].max() * 500 + 50,
    )
    plt.title("Discount Level vs Total Revenue")
    plt.xlabel("Discount")
    plt.ylabel("Revenue")
    save_plot(OUTPUT_DIR / "discount_vs_revenue.png")

    # -----------------------------
    # 10. Print analytical findings
    # -----------------------------
    best_category = category_summary.iloc[0]
    best_country = country_summary.iloc[0]
    best_product = product_summary.iloc[0]
    top_customer = customer_summary.iloc[0]
    peak_month = monthly.loc[monthly["NetRevenue"].idxmax()]

    print("\nKEY FINDINGS")
    print("-" * 70)
    print(
        f"Highest-revenue category: {best_category['Category']} "
        f"({best_category['Revenue']:.2f})"
    )
    print(
        f"Highest-revenue country: {best_country['Country']} "
        f"({best_country['Revenue']:.2f})"
    )
    print(
        f"Top product by revenue: {best_product['Product']} "
        f"({best_product['Revenue']:.2f})"
    )
    print(
        f"Highest-revenue month: {peak_month['Month']} "
        f"({peak_month['NetRevenue']:.2f})"
    )
    print(
        f"Highest-revenue customer: {top_customer['CustomerID']} "
        f"({top_customer['Revenue']:.2f})"
    )
    print(
        f"Cancellation rate: {cancellation_rate:.2f}%"
    )

    print("\nAnalysis completed successfully.")
    print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
