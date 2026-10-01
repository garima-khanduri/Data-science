# Task 4 — Real-World Data Project: Retail Sales Analytics

## Internship Task
Work on a domain-specific dataset and perform end-to-end data analysis or prediction, presenting findings with visualizations and conclusions.

## Domain
**Retail / E-commerce**

## Objective
Analyze transaction-level retail data to understand:
- Revenue trends over time
- Category and product performance
- Geographic sales patterns
- Cancellation/return behavior
- Customer purchasing behavior
- Key business metrics and actionable insights

## Dataset Note
The bundled dataset is a **synthetic retail transaction dataset** created specifically for this project because the execution environment could not retrieve the public external retail dataset reliably.

It is intentionally structured like real transaction data, with invoice IDs, dates, customers, countries, products, categories, quantities, discounts, status and revenue.

The analysis code is written so that a real CSV with the same columns can replace `data/retail_transactions.csv`.

## Project Structure

```text
Task_4_Real_World_Retail_Project/
├── data/
│   └── retail_transactions.csv
├── outputs/
│   ├── monthly_revenue.png
│   ├── category_revenue.png
│   ├── country_revenue.png
│   ├── top_products.png
│   ├── customer_revenue_distribution.png
│   ├── discount_vs_revenue.png
│   ├── category_summary.csv
│   ├── country_summary.csv
│   ├── customer_summary.csv
│   └── kpi_summary.csv
├── src/
│   └── retail_analysis.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Run:
```bash
python src/retail_analysis.py
```

## Analysis Workflow

1. Load and validate transaction data
2. Check missing values and duplicates
3. Separate completed sales from cancelled transactions
4. Calculate revenue and transaction KPIs
5. Analyze monthly revenue
6. Compare category performance
7. Compare country performance
8. Identify top products
9. Analyze customer revenue
10. Explore discounts and revenue
11. Generate visualizations and business insights

## Business KPIs
- Total revenue
- Completed revenue
- Number of invoices
- Number of customers
- Units sold
- Average order value
- Cancellation rate

## Important Interpretation Note
The dataset is synthetic, so numerical findings should be treated as examples of analytical methodology rather than claims about an actual retailer's business.
