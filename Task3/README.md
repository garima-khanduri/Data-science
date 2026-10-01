# Task 3 — Exploratory Data Analysis (EDA)

## Internship Task
Analyze a dataset to uncover patterns and trends using statistical summaries and visualizations.

## Objective
This project performs a complete Exploratory Data Analysis (EDA) on the Wisconsin Breast Cancer dataset. It examines:
- Dataset structure and data quality
- Descriptive statistics
- Class distribution
- Feature distributions
- Relationships between selected variables
- Correlations between numerical features
- Key patterns and analytical insights

## Dataset
The project uses the Wisconsin Breast Cancer dataset provided by scikit-learn and stores a CSV copy in `data/breast_cancer.csv`.

Target encoding:
- `0` = malignant
- `1` = benign

There are 569 observations and 30 numerical predictor features.

## Project Structure

```text
EDA_Task_3_GitHub_Project/
├── data/
│   └── breast_cancer.csv
├── outputs/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── feature_boxplots.png
│   ├── radius_texture_scatter.png
│   ├── summary_statistics.csv
│   └── top_correlations.csv
├── src/
│   └── eda_analysis.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

### 1. Create and activate a virtual environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the EDA
```bash
python src/eda_analysis.py
```

The script creates all tables and plots in the `outputs/` directory.

## Main EDA Questions
1. What is the shape and structure of the dataset?
2. Are there missing values or duplicate rows?
3. How are malignant and benign cases distributed?
4. Which variables have large differences in distribution between the two classes?
5. Which numerical variables are strongly correlated?
6. What relationships can be observed from scatter plots and boxplots?

## Key Analytical Insight
The analysis identifies strong relationships among several tumor-measurement variables. In particular, measurements describing related physical characteristics tend to move together, while some features show visibly different distributions between malignant and benign cases.

**Important:** Correlation describes association, not causation.

## Technologies
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn.

## Expected Learning Outcome
The project demonstrates a structured EDA workflow: inspect → clean/check → summarize → visualize → analyze relationships → communicate insights.
