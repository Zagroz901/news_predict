import joblib

def load_model_and_predict(model_path,
                           vectorizer_path,
                           encoder_path,
                           text_samples=None):
    """
    Loads the trained model, vectorizer, and label encoder to predict sentiment.

    Parameters:
    - model_path: Path to the trained model
    - vectorizer_path: Path to the saved TF-IDF vectorizer
    - encoder_path: Path to the saved LabelEncoder
    - text_samples: List of text samples to predict

    Returns:
    - List of predicted sentiment labels
    """

    # Load model, vectorizer, and encoder
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)

    # Transform text into numerical format
    text_tfidf = vectorizer.transform(text_samples)

    # Predict sentiment
    predictions = model.predict(text_tfidf)
    predicted_labels = label_encoder.inverse_transform(predictions)

    return predicted_labels
