"""
config.py
---------
Central place for file paths, column names, and model settings.
Change values here instead of hunting through every script.
"""

import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "city_day.csv")          # raw Kaggle file
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "city_day_clean.csv")  # after preprocessing
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
PRED_PLOT_PATH = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")

# ---- Columns ----
TARGET_COLUMN = "AQI"

POLLUTANT_FEATURES = [
    "PM2.5", "PM10", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

DATE_COLUMN = "Date"
CITY_COLUMN = "City"

# Engineered/derived features created in feature_engineering.py
ENGINEERED_FEATURES = ["Month", "DayOfWeek", "Season", "AQI_lag1"]

ALL_FEATURES = POLLUTANT_FEATURES + ["City"] + ENGINEERED_FEATURES

# ---- Train/test split ----
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---- Model hyperparameters (Random Forest baseline) ----
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
