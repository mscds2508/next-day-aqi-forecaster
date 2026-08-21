"""
preprocessing.py
-----------------
Cleans the raw dataset: handles missing values, drops rows with no
target, and writes a clean CSV that the rest of the pipeline uses.
"""

import pandas as pd
from src import config
from src.data_loader import load_raw_data


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with no AQI value can't be used for training or evaluation."""
    before = len(df)
    df = df.dropna(subset=[config.TARGET_COLUMN])
    print(f"Dropped {before - len(df)} rows with missing {config.TARGET_COLUMN}")
    return df


def impute_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing pollutant readings using city-level median.
    Falls back to global median if a city has no data for that pollutant.
    """
    for col in config.POLLUTANT_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df.groupby(config.CITY_COLUMN)[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[config.CITY_COLUMN, config.DATE_COLUMN])
    print(f"Dropped {before - len(df)} duplicate rows")
    return df


def run_preprocessing() -> pd.DataFrame:
    df = load_raw_data()
    df = remove_duplicates(df)
    df = drop_missing_target(df)
    df = impute_pollutants(df)
    df = df.sort_values([config.CITY_COLUMN, config.DATE_COLUMN]).reset_index(drop=True)

    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    print(f"Saved clean data to {config.CLEAN_DATA_PATH}")
    return df


if __name__ == "__main__":
    run_preprocessing()
