import joblib

def load_model_and_predict(model_path, vectorizer_path, text_samples):
    """
    Loads the trained model and vectorizer to predict sentiment.
    """
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
    # Load trained model (which is stored inside a dictionary)

    if loaded_model is None:
        raise ValueError("Error: Model not found in the saved file.")

    # Load vectorizer
    vectorizer = joblib.load(vectorizer_path)

    # Transform text into numerical format
    text_tfidf = vectorizer.transform(text_samples)

    # Predict sentiment
    predictions = loaded_model.predict(text_tfidf)

    return predictions
