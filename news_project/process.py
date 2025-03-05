# import pandas as pd
# from loguru import logger
# import re
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import nltk
# # nltk.data.path.append("C:/Users/Lenovo/AppData/Roaming/nltk_data")

# def remove_duplicates(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
#     """
#     Removes duplicate rows based on a specific column while keeping the first occurrence.
#     Logs the number of duplicates removed.

#     Parameters:
#         df (pd.DataFrame): The input DataFrame.
#         column_name (str): The column to check for duplicates.

#     Returns:
#         pd.DataFrame: The cleaned DataFrame with duplicates removed.
#     """
#     # Count total duplicates before removal
#     num_duplicates = df.duplicated(subset=[column_name], keep="first").sum()
#     logger.info(f"Number of duplicate rows in '{column_name}' before removal: {num_duplicates}")

#     # Remove duplicates while keeping the first occurrence
#     df_cleaned = df.drop_duplicates(subset=[column_name], keep="first")

#     # Log the new shape of the dataset
#     logger.info(f"Dataset shape after duplicate removal: {df_cleaned.shape}")

#     return df_cleaned
# def remove_byte_prefix(text):
#     # Remove leading b' or b" or any variation
#     # Examples: b'This is text' → This is text
#     return re.sub(r"^b['\"]|['\"]$", "", text.strip())
# def normalize_quotes(text):
#     # Replace fancy quotes with standard quotes
#     text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
#     return text
# def remove_special_chars(text):
#     # Remove line breaks, tabs, or repeated spaces
#     text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
#     # Optionally remove ASCII junk like `\x..` if present
#     text = re.sub(r"\s+", " ", text).strip()
#     return text
# def to_lowercase(text):
#     return text.lower()
# def remove_urls(text):
#     # Remove URLs
#     text = re.sub(r"http\S+|www\S+|https\S+", "", text)
#     # Remove typical HTML entities
#     text = re.sub(r"&\S+;", "", text)
#     return text
# stop_words = set(stopwords.words("english"))
# lemmatizer = WordNetLemmatizer()

# def tokenize_lemmatize(text):
#     tokens = word_tokenize(text)
#     cleaned_tokens = []
#     for word in tokens:
#         # Remove stopwords and short tokens
#         if word not in stop_words and len(word) > 2:
#             cleaned_tokens.append(lemmatizer.lemmatize(word))
#     return " ".join(cleaned_tokens)
import pandas as pd
import re
import string
import emoji
import nltk

from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# Ensure required NLTK resources are downloaded
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Initialize stopwords & lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

### 1. Remove Duplicates
def remove_duplicates(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Removes duplicate rows based on a specific column."""
    num_duplicates = df.duplicated(subset=[column_name], keep="first").sum()
    logger.info(f"Number of duplicate rows in '{column_name}' before removal: {num_duplicates}")

    df_cleaned = df.drop_duplicates(subset=[column_name], keep="first")
    logger.info(f"Dataset shape after duplicate removal: {df_cleaned.shape}")
    return df_cleaned

### 2. Text Normalization Functions
def remove_byte_prefix(text: str) -> str:
    """Removes leading byte prefixes like b' or b" from text."""
    return re.sub(r"^b['\"]|['\"]$", "", text.strip())

def normalize_quotes(text: str) -> str:
    """Replaces fancy quotes with standard ones."""
    return text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

def remove_special_chars(text: str) -> str:
    """Removes line breaks, tabs, and extra spaces."""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()

def to_lowercase(text: str) -> str:
    """Converts text to lowercase."""
    return text.lower()

### 3. Handling Mentions, URLs, and Hashtags
def check_mentions(text: str):
    """Finds mentions (@username)."""
    return re.findall(r"@\w+", text)

def remove_mentions(text: str) -> str:
    """Removes all mentions from the text."""
    return re.sub(r"@\w+", "", text)

def check_urls(text: str):
    """Finds URLs."""
    return re.findall(r"http\S+|www.\S+", text)

def remove_urls(text: str) -> str:
    """Removes all URLs from the text."""
    return re.sub(r"http\S+|www.\S+", "", text)

def check_hashtags(text: str):
    """Finds hashtags (#topic)."""
    return re.findall(r"#\w+", text)

def remove_hashtags(text: str) -> str:
    """Removes all hashtags from the text."""
    return re.sub(r"#\w+", "", text)

### 4. Handling Emojis
def check_emoji(text: str) -> bool:
    """Checks if text contains emojis."""
    return any(char in emoji.EMOJI_DATA for char in text)

def replace_emojis(text: str) -> str:
    """Replaces emojis with their text representation (e.g., :thumbs_up:)."""
    return emoji.demojize(text, delimiters=(" ", " "))

### 5. Tokenization, Stopword Removal, and Punctuation Cleaning
def tokenize(text: str):
    """Tokenizes text into words."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Removes stopwords from a tokenized list."""
    return [word for word in tokens if word.lower() not in stop_words]

def remove_punctuation(tokens):
    """Removes punctuation from a tokenized list."""
    punctuation = set(string.punctuation)
    return [word for word in tokens if word not in punctuation]

### 6. Lemmatization with POS tagging
def get_wordnet_pos(tag):
    """Maps NLTK POS tags to WordNet POS tags."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def lemmatize_tokens(tokens):
    """Lemmatizes tokens using their POS tags."""
    tagged_tokens = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens]

### 7. Master Cleaning Function
def clean_news_data(df: pd.DataFrame, text_column: str = "news") -> pd.DataFrame:
    """
    Cleans a DataFrame of news text by:
    1. Removing duplicates.
    2. Removing mentions, URLs, hashtags, and emojis.
    3. Tokenizing and processing text.
    4. Removing stopwords, punctuation, and performing lemmatization.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame with a 'cleaned_text' column.
    """
    
    # Remove duplicates first
    df = remove_duplicates(df, text_column)
    
    # Normalize text
    df["cleaned_text"] = df[text_column].apply(to_lowercase)
    df["cleaned_text"] = df["cleaned_text"].apply(remove_byte_prefix)
    df["cleaned_text"] = df["cleaned_text"].apply(normalize_quotes)
    df["cleaned_text"] = df["cleaned_text"].apply(remove_special_chars)

    # Handle mentions
    df["mentions"] = df["cleaned_text"].apply(check_mentions)
    df.loc[df["mentions"].apply(len) > 0, "cleaned_text"] = df["cleaned_text"].apply(remove_mentions)

    # Handle URLs
    df["urls"] = df["cleaned_text"].apply(check_urls)
    df.loc[df["urls"].apply(len) > 0, "cleaned_text"] = df["cleaned_text"].apply(remove_urls)

    # Handle hashtags
    df["hashtags"] = df["cleaned_text"].apply(check_hashtags)
    df.loc[df["hashtags"].apply(len) > 0, "cleaned_text"] = df["cleaned_text"].apply(remove_hashtags)

    # Handle emojis
    df["emoji"] = df["cleaned_text"].apply(check_emoji)
    df.loc[df["emoji"], "cleaned_text"] = df["cleaned_text"].apply(replace_emojis)

    # Tokenization
    df["cleaned_tokens"] = df["cleaned_text"].apply(tokenize)

    # Remove stopwords & punctuation
    df["cleaned_tokens"] = df["cleaned_tokens"].apply(remove_stopwords)
    df["cleaned_tokens"] = df["cleaned_tokens"].apply(remove_punctuation)

    # Lemmatization
    df["cleaned_tokens"] = df["cleaned_tokens"].apply(lemmatize_tokens)

    # Convert tokens back to text
    df["cleaned_text"] = df["cleaned_tokens"].apply(lambda x: " ".join(x))

    return df
### 7. Text Processing Pipeline
def text_preprocessing_pipeline(text: str) -> str:
    """Applies a series of text preprocessing steps to the input text."""
    
    # 1. Text normalization
    text = remove_byte_prefix(text)
    text = normalize_quotes(text)
    text = remove_special_chars(text)
    text = to_lowercase(text)
    
    # 2. Remove mentions, URLs, and hashtags
    text = remove_mentions(text)
    text = remove_urls(text)
    text = remove_hashtags(text)

    # 3. Replace emojis with text representation
    text = replace_emojis(text)

    # 4. Tokenization
    tokens = tokenize(text)

    # 5. Remove stopwords and punctuation
    tokens = remove_stopwords(tokens)
    tokens = remove_punctuation(tokens)

    # 6. Lemmatization
    tokens = lemmatize_tokens(tokens)

    # 7. Return cleaned text as a string
    return " ".join(tokens)