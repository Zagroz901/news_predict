import joblib
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "LogisticRegression.pkl")
    vec_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
    return model_path, vec_path
