# Task 1 - Data Cleaning & Visualization Project

## Objective
Clean a raw employee dataset, handle missing values and duplicate records, detect and treat salary outliers, and create visualizations to communicate useful insights.

## Dataset
The dataset contains:
- Name
- Age
- Salary
- Join_Date
- Department

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib

## Data Cleaning Steps
1. Load the raw CSV.
2. Inspect shape, data types, missing values, and duplicates.
3. Convert Age and Salary to numeric values.
4. Convert Join_Date to datetime.
5. Remove exact duplicate records.
6. Detect salary outliers using the IQR method before imputing missing salaries.
7. Cap salary outliers using IQR boundaries instead of deleting employee records.
8. Fill missing Age and Salary values using department-level medians.
9. Check for duplicates again because transformations can sometimes create identical rows.
10. Validate and export the cleaned dataset.

## Visualizations
The project generates:
- Missing-value visualization
- Salary boxplot before outlier treatment
- Employee count by department
- Salary distribution
- Age distribution
- Average salary by department
- Salary distribution by department
- Correlation heatmap

## How to Run

```bash
pip install -r requirements.txt
python task1_data_cleaning_visualization.py
```

The cleaned dataset is saved in `data/cleaned_dataset.csv`, and charts are saved in `visualizations/`.

## Project Structure

```text
Task1_Data_Cleaning_Visualization/
│
├── data/
│   ├── raw_dataset.csv
│   └── cleaned_dataset.csv
│
├── visualizations/
│   ├── missing_values_before.png
│   ├── salary_boxplot_before.png
│   ├── employees_by_department.png
│   ├── salary_distribution.png
│   ├── age_distribution.png
│   ├── average_salary_by_department.png
│   ├── salary_by_department.png
│   └── correlation_heatmap.png
│
├── task1_data_cleaning_visualization.py
├── requirements.txt
└── README.md
```

## Cleaning Philosophy
The raw dataset is kept unchanged. All cleaning is performed on the loaded DataFrame, and the cleaned version is exported separately.
