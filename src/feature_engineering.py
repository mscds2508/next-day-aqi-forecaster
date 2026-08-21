"""
feature_engineering.py
-----------------------
Turns raw/clean columns into model-ready features:
- calendar features (Month, DayOfWeek, Season)
- lag feature (previous day's AQI per city)
"""

import pandas as pd
from src import config


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Month"] = df[config.DATE_COLUMN].dt.month
    df["DayOfWeek"] = df[config.DATE_COLUMN].dt.dayofweek

    def month_to_season(m):
        if m in (12, 1, 2):
            return "Winter"
        elif m in (3, 4, 5):
            return "Summer"
        elif m in (6, 7, 8, 9):
            return "Monsoon"
        else:
            return "PostMonsoon"

    df["Season"] = df["Month"].apply(month_to_season)
    return df


def add_lag_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Previous day's AQI for the same city — strong predictive signal
    since pollution levels are highly autocorrelated day-to-day."""
    df = df.sort_values([config.CITY_COLUMN, config.DATE_COLUMN])
    df["AQI_lag1"] = df.groupby(config.CITY_COLUMN)[config.TARGET_COLUMN].shift(1)
    # first day per city has no lag; fill with that city's median AQI
    df["AQI_lag1"] = df.groupby(config.CITY_COLUMN)["AQI_lag1"].transform(
        lambda s: s.fillna(s.median())
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_feature(df)
    return df


if __name__ == "__main__":
    from src.data_loader import load_clean_data

    df = load_clean_data()
    df = build_features(df)
    print(df[config.ENGINEERED_FEATURES].head())
