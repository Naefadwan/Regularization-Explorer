from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Setup
DATA_PATH = Path("data/telecom_churn.csv")
OUTPUT_DIR = Path("artifacts")
PLOT_PATH = OUTPUT_DIR / "learning_curve.png"
ANALYSIS_PATH = OUTPUT_DIR / "learning_curve_analysis.md"

def get_model():
    """Create a pipeline with preprocessing and logistic regression."""
    # Assuming numeric/categorical logic similar to original script
    # We'll need to know column names for full pipeline.
    # For now, let's load data to get columns.
    df = pd.read_csv(DATA_PATH)
    numeric_cols = df.select_dtypes(include=["number"]).drop(columns=["churned"], errors="ignore").columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).drop(columns=["customer_id"], errors="ignore").columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="saga", max_iter=5000, random_state=42))
    ])

def run_learning_curve():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["churned", "customer_id"], errors="ignore")
    y = df["churned"]
    
    model = get_model()
    
    # Stratified CV for imbalance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 5+ training set sizes
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    # Using f1 score as it's better for churn than accuracy
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring="f1"
    )
    
    # Calculate means and stds
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training score", color="darkorange")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="navy")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy")
    
    plt.title("Learning Curves (Logistic Regression)")
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    
    # Analysis
    analysis = """# Learning Curve Analysis
Based on the learning curves, the model appears to... [TO BE FILLED]
"""
    ANALYSIS_PATH.write_text(analysis)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_learning_curve()
