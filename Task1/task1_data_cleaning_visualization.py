# Task 1 - Data Cleaning & Visualization Project
# Data Science & AI Internship
#
# Run:
#     pip install -r requirements.txt
#     python task1_data_cleaning_visualization.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RAW_FILE = "data/raw_dataset.csv"
CLEAN_FILE = "data/cleaned_dataset.csv"
PLOT_DIR = "visualizations"

os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(RAW_FILE)

print("=" * 60)
print("INITIAL DATASET INFORMATION")
print("=" * 60)
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())
print("\nExact duplicate rows:", df.duplicated().sum())

# ============================================================
# 2. CONVERT DATA TYPES
# ============================================================
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
df["Join_Date"] = pd.to_datetime(df["Join_Date"], errors="coerce")

# ============================================================
# 3. MISSING-VALUE REPORT
# ============================================================
missing_report = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": (df.isna().mean() * 100).round(2)
})

print("\nMissing-value report:")
print(missing_report)

# Missing-value heatmap
plt.figure(figsize=(8, 5))
plt.imshow(df.isna(), aspect="auto", interpolation="nearest")
plt.title("Missing Values Before Cleaning")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.xticks(range(len(df.columns)), df.columns, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "missing_values_before.png"), dpi=300)
plt.close()

# ============================================================
# 4. REMOVE EXACT DUPLICATES
# ============================================================
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

print("\nExact duplicate rows removed:", duplicates_removed)

# ============================================================
# 5. DETECT SALARY OUTLIERS BEFORE IMPUTATION
#    Missing salaries are excluded automatically by quantile().
# ============================================================
q1 = df["Salary"].quantile(0.25)
q3 = df["Salary"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask = (
    (df["Salary"] < lower_bound) |
    (df["Salary"] > upper_bound)
)

outlier_count = int(outlier_mask.sum())

print("\nSalary outlier analysis:")
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower bound:", round(lower_bound, 2))
print("Upper bound:", round(upper_bound, 2))
print("Number of salary outliers:", outlier_count)

if outlier_count > 0:
    print("\nDetected salary outliers:")
    print(df.loc[outlier_mask, ["Name", "Salary", "Department"]])

# Salary boxplot before treatment
plt.figure(figsize=(8, 5))
plt.boxplot(df["Salary"].dropna())
plt.title("Salary Distribution Before Outlier Treatment")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "salary_boxplot_before.png"), dpi=300)
plt.close()

# ============================================================
# 6. TREAT SALARY OUTLIERS USING IQR CAPPING
# ============================================================
# We cap extreme values instead of deleting employee records.
# This keeps the observations while reducing the effect of
# unusually large salary values.
df["Salary"] = df["Salary"].clip(
    lower=lower_bound,
    upper=upper_bound
)

# ============================================================
# 7. HANDLE MISSING VALUES
#    Department-level median is used after outlier treatment.
#    This avoids using extreme salaries when calculating the
#    replacement value.
# ============================================================
df["Age"] = df.groupby("Department")["Age"].transform(
    lambda s: s.fillna(s.median())
)

df["Salary"] = df.groupby("Department")["Salary"].transform(
    lambda s: s.fillna(s.median())
)

# Fallback for any remaining missing numeric values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

# ============================================================
# 8. FINAL DUPLICATE CHECK
# ============================================================
# Capping/imputation can sometimes make two rows identical.
# Remove any such exact duplicates from the final cleaned data.
duplicates_after_cleaning = int(df.duplicated().sum())

if duplicates_after_cleaning > 0:
    df = df.drop_duplicates().copy()

print("\nAdditional exact duplicates after cleaning:", duplicates_after_cleaning)

# ============================================================
# 9. FINAL DATA FORMATTING
# ============================================================
df["Age"] = df["Age"].round().astype(int)
df["Salary"] = df["Salary"].round(2)

# ============================================================
# 10. FINAL VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("FINAL DATASET INFORMATION")
print("=" * 60)
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isna().sum())
print("\nExact duplicate rows:", df.duplicated().sum())

print("\nSummary statistics:")
print(df.describe())

# ============================================================
# 11. SAVE CLEANED DATASET
# ============================================================
df.to_csv(CLEAN_FILE, index=False)

print(f"\nCleaned dataset saved to: {CLEAN_FILE}")

# ============================================================
# 12. VISUALIZATIONS
# ============================================================

# 12.1 Employee count by department
department_counts = df["Department"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(department_counts.index, department_counts.values)
plt.title("Number of Employees by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "employees_by_department.png"), dpi=300)
plt.close()

# 12.2 Salary distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Salary"], bins=10, edgecolor="black")
plt.title("Salary Distribution After Cleaning")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "salary_distribution.png"), dpi=300)
plt.close()

# 12.3 Age distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=10, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "age_distribution.png"), dpi=300)
plt.close()

# 12.4 Average salary by department
department_salary = (
    df.groupby("Department")["Salary"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
plt.bar(department_salary.index, department_salary.values)
plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "average_salary_by_department.png"), dpi=300)
plt.close()

# 12.5 Salary distribution by department
departments = list(df["Department"].dropna().unique())
salary_groups = [
    df.loc[df["Department"] == department, "Salary"]
    for department in departments
]

plt.figure(figsize=(8, 5))
plt.boxplot(salary_groups, tick_labels=departments)
plt.title("Salary Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "salary_by_department.png"), dpi=300)
plt.close()

# 12.6 Correlation heatmap using Matplotlib
corr = df[["Age", "Salary"]].corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr, interpolation="nearest", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha="center", va="center")

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()

# ============================================================
# 13. KEY FINDINGS
# ============================================================
highest_paid = df.loc[df["Salary"].idxmax()]
average_salary = df["Salary"].mean()
average_age = df["Age"].mean()
largest_department = df["Department"].value_counts().idxmax()

print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print(f"Average employee age: {average_age:.2f} years")
print(f"Average salary after cleaning: {average_salary:.2f}")
print(f"Largest department: {largest_department}")
print(
    f"Highest salary after outlier treatment: "
    f"{highest_paid['Salary']:.2f} ({highest_paid['Name']})"
)

print("\nAll visualizations were saved in the 'visualizations' folder.")
print("Project completed successfully.")
