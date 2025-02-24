import pandas as pd
from loguru import logger

def remove_duplicates(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Removes duplicate rows based on a specific column while keeping the first occurrence.
    Logs the number of duplicates removed.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column to check for duplicates.

    Returns:
        pd.DataFrame: The cleaned DataFrame with duplicates removed.
    """
    # Count total duplicates before removal
    num_duplicates = df.duplicated(subset=[column_name], keep="first").sum()
    logger.info(f"Number of duplicate rows in '{column_name}' before removal: {num_duplicates}")

    # Remove duplicates while keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset=[column_name], keep="first")

    # Log the new shape of the dataset
    logger.info(f"Dataset shape after duplicate removal: {df_cleaned.shape}")

    return df_cleaned
