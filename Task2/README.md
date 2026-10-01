# Predictive Modeling Using Machine Learning

## Internship Task
Build a supervised machine learning model to predict outcomes from given data and evaluate model performance.

## Project Objective
This project predicts whether a breast tumor is **malignant (1)** or **benign (0)** using diagnostic measurements from the Wisconsin Breast Cancer dataset.

The project demonstrates:
- Data loading and inspection
- Train/test splitting
- Feature scaling
- Logistic Regression
- Decision Tree
- Random Forest
- Accuracy, precision, recall, F1-score and ROC-AUC
- Confusion matrices
- ROC curves
- Saving evaluation results and plots

## Dataset
The dataset is the scikit-learn Wisconsin Breast Cancer dataset. A CSV copy is included in `data/breast_cancer.csv` so the project is reproducible without downloading external data.

**Target:** `target`
- `0` = malignant
- `1` = benign

## Project Structure

```text
Predictive_Modeling_ML_GitHub_Project/
├── data/
│   └── breast_cancer.csv
├── outputs/
│   └── generated after running the script
├── src/
│   └── train_models.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

### 1. Clone the repository
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Predictive_Modeling_ML_GitHub_Project
```

### 2. Create a virtual environment
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

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the model
```bash
python src/train_models.py
```

### 5. Check the outputs
The script creates:
- `outputs/model_metrics.csv`
- `outputs/confusion_matrices.png`
- `outputs/roc_curves.png`

## Expected Evaluation
The script prints a comparison table for all three models. Exact values can vary slightly if the modeling setup is changed.

## Technologies
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Learning Outcome
This task demonstrates the complete supervised machine learning workflow from data preparation through model training and evaluation.
