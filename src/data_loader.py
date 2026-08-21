"""
data_loader.py
---------------
Loads the raw Kaggle "Air Quality Data in India" CSV into a DataFrame.
Keep all file-reading logic here so the rest of the pipeline never
touches raw paths directly.
"""

import pandas as pd
from src import config


def load_raw_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw city_day.csv file from Kaggle's Air Quality Data in India dataset.

    Expected columns include: City, Date, PM2.5, PM10, NO2, NOx, NH3,
    CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket.
    """
    df = pd.read_csv(path, parse_dates=[config.DATE_COLUMN])
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_clean_data(path: str = config.CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load the already-cleaned dataset produced by preprocessing.py."""
    df = pd.read_csv(path, parse_dates=[config.DATE_COLUMN])
    print(f"Loaded clean data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
