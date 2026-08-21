"""
train.py
--------
Trains a Random Forest regression model to predict AQI and saves it
to disk. Run this after preprocessing + feature engineering.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import load_clean_data
from src.feature_engineering import build_features


def build_pipeline() -> Pipeline:
    """
    Bundles preprocessing (one-hot encode categoricals) + model into a
    single sklearn Pipeline, so train/predict always apply identical
    transformations.
    """
    categorical_cols = ["City", "Season"]
    numeric_cols = [c for c in config.ALL_FEATURES if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",  # numeric columns pass through untouched
    )

    model = RandomForestRegressor(**config.MODEL_PARAMS)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def get_train_test_split(df: pd.DataFrame):
    X = df[config.ALL_FEATURES]
    y = df[config.TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )


def train_model():
    df = load_clean_data()
    df = build_features(df)

    X_train, X_test, y_train, y_test = get_train_test_split(df)

    pipeline = build_pipeline()
    print("Training model...")
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    return pipeline, X_test, y_test


if __name__ == "__main__":
    train_model()
