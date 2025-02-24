import pandas as pd
from loguru import logger
from pathlib import Path

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Loads a CSV file and logs information about the dataset.

    Parameters:
        file_path (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    logger.info(f"Attempting to load data from: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        logger.info(f"Columns in dataset: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data. Error: {e}")
        raise
