import pandas as pd
from loguru import logger
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.data.path.append("C:/Users/Lenovo/AppData/Roaming/nltk_data")

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
def remove_byte_prefix(text):
    # Remove leading b' or b" or any variation
    # Examples: b'This is text' → This is text
    return re.sub(r"^b['\"]|['\"]$", "", text.strip())
def normalize_quotes(text):
    # Replace fancy quotes with standard quotes
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    return text
def remove_special_chars(text):
    # Remove line breaks, tabs, or repeated spaces
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Optionally remove ASCII junk like `\x..` if present
    text = re.sub(r"\s+", " ", text).strip()
    return text
def to_lowercase(text):
    return text.lower()
def remove_urls(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove typical HTML entities
    text = re.sub(r"&\S+;", "", text)
    return text
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def tokenize_lemmatize(text):
    tokens = word_tokenize(text)
    cleaned_tokens = []
    for word in tokens:
        # Remove stopwords and short tokens
        if word not in stop_words and len(word) > 2:
            cleaned_tokens.append(lemmatizer.lemmatize(word))
    return " ".join(cleaned_tokens)