# -*- coding: utf-8 -*-

"""
Employee Recruitment Prediction Pipeline
======================================

Senior-level refactor focusing on:
- Clear structure and separation of concerns
- Reproducibility and configurability
- Robust preprocessing via sklearn Pipelines
- Minimal side effects (no notebook-only commands)
- PEP8-compliant naming and documentation
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
DATA_PATH: str = "data-employee-recruitment.csv"

DT_PARAM_GRID: Dict = {
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}

RF_PARAM_GRID: Dict = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}

# -----------------------------------------------------------------------------
# Data Loading & Cleaning
# -----------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """Load dataset and drop identifier column."""
    df = pd.read_csv(path)
    return df.iloc[:, 1:]


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ordinal variables using explicit, documented mappings."""

    df = df.copy()

    df["last_new_job"] = df["last_new_job"].replace({"never": 0, ">4": 5}).astype(int)
    df["experience"] = df["experience"].replace({"<1": 0, ">20": 21}).astype(int)

    company_size_map = {
        "<10": 5,
        "10/49": 25,
        "50-99": 75,
        "100-500": 250,
        "500-999": 750,
        "1000-4999": 2500,
        "5000-9999": 7500,
        "10000+": 25000,
    }
    df["company_size"] = df["company_size"].map(company_size_map)

    education_map = {"Graduate": 0, "Masters": 1, "Phd": 2}
    df["education_level"] = df["education_level"].map(education_map)

    return df


def clean_training_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows with negative training hours."""
    return df[df["training_hours"] >= 0]


# -----------------------------------------------------------------------------
# Exploratory Data Analysis (Optional / Non-blocking)
# -----------------------------------------------------------------------------


def plot_distributions(df: pd.DataFrame) -> None:
    numerical_cols: List[str] = df.select_dtypes(include=["int64", "float64"]).columns.drop("target")
    df[numerical_cols].hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()

    categorical_cols: List[str] = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Preprocessing & Modeling
# -----------------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


# -----------------------------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------------------------


def train_model(pipeline: Pipeline, param_grid: Dict, X_train, y_train) -> GridSearchCV:
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


# -----------------------------------------------------------------------------
# Feature Importance & Visualization
# -----------------------------------------------------------------------------


def plot_feature_importance(model: Pipeline, X: pd.DataFrame, top_n: int = 6) -> None:
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(X.select_dtypes(include="object").columns)
    num_features = X.select_dtypes(include=["int64", "float64"]).columns

    feature_names = list(num_features) + list(cat_features)

    importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    importance_series.head(top_n).plot(kind="barh", figsize=(8, 4))
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importances (Random Forest)")
    plt.show()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = encode_ordinal_features(df)
    df = clean_training_hours(df)

    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)

    dt_pipeline = build_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE))
    rf_pipeline = build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE))

    dt_grid = train_model(dt_pipeline, DT_PARAM_GRID, X_train, y_train)
    rf_grid = train_model(rf_pipeline, RF_PARAM_GRID, X_train, y_train)

    print("Best Decision Tree params:", dt_grid.best_params_)
    print("DT Test Accuracy:", dt_grid.best_estimator_.score(X_test, y_test))

    print("Best Random Forest params:", rf_grid.best_params_)
    print("RF Test Accuracy:", rf_grid.best_estimator_.score(X_test, y_test))

    plot_feature_importance(rf_grid.best_estimator_, X)
