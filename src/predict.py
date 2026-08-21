"""
predict.py
----------
Loads the trained model and predicts AQI for new input rows.
Use this for single predictions or batch scoring of new data,
without re-running the whole training pipeline.
"""

import joblib
import pandas as pd
from src import config


def load_model():
    return joblib.load(config.MODEL_PATH)


def predict_aqi(input_df: pd.DataFrame) -> pd.Series:
    """
    input_df must contain the same feature columns used in training:
    config.ALL_FEATURES (pollutant readings, City, Month, DayOfWeek,
    Season, AQI_lag1).
    """
    model = load_model()
    missing = set(config.ALL_FEATURES) - set(input_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for prediction: {missing}")

    preds = model.predict(input_df[config.ALL_FEATURES])
    return pd.Series(preds, index=input_df.index, name="Predicted_AQI")


if __name__ == "__main__":
    # Example: predict AQI for one sample row
    sample = pd.DataFrame([{
        "PM2.5": 120, "PM10": 180, "NO2": 40, "NOx": 45, "NH3": 20,
        "CO": 1.2, "SO2": 15, "O3": 30, "Benzene": 2, "Toluene": 5, "Xylene": 1,
        "City": "Delhi", "Month": 12, "DayOfWeek": 2, "Season": "Winter",
        "AQI_lag1": 250,
    }])
    result = predict_aqi(sample)
    print(result)
