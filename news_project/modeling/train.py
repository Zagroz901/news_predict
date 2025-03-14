import pandas as pd
import scipy.sparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from loguru import logger
import joblib
import os 

def evaluate_classification(y_true, y_pred):
    """
    Evaluates classification model performance using accuracy, F1-score, precision, and recall.

    Parameters:
        y_true (array): True labels
        y_pred (array): Predicted labels

    Returns:
        dict: Dictionary containing accuracy, F1-score, precision, and recall
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
    }


# ✅ Train & Evaluate Any Classification Model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, models_df, trained_models):
    """
    Trains a classification model, evaluates its performance, and appends results to models_df.
    """
    
    model_name = type(model).__name__
    logger.info(f"Training model: {model_name}")

    try:
        # Train Model
        model.fit(X_train, y_train)
        logger.success(f"Model {model_name} trained successfully!")

        # Predictions
        y_pred = model.predict(X_test)
        logger.info("Predictions completed!")

        # Evaluate Performance
        evaluation_results = evaluate_classification(y_test, y_pred)
        logger.success(f"Evaluation Results for {model_name}: {evaluation_results}")

        # Append results to DataFrame
        new_row = {
            "Model": model_name,
            "Accuracy": evaluation_results["Accuracy"],
            "F1 Score": evaluation_results["F1 Score"],
            "Precision": evaluation_results["Precision"],
            "Recall": evaluation_results["Recall"],
        }
        models_df = pd.concat([models_df, pd.DataFrame([new_row])], ignore_index=True)

        trained_models[model_name] = model
        logger.success(f"Model {model_name} evaluation saved successfully!")

    except Exception as e:
        logger.error(f"Training failed for {model_name}: {e}")

    return models_df, trained_models



# # التصريح عن دالة إنشاء نموذج التعلم
# # مع إعطاء قيم أولية للمعاملات المترفعة
# def create_model(embed_dim = 32, hidden_unit = 16, dropout_rate = 0.2, optimizers = RMSprop, learning_rate = 0.001):
#     # التصريح عن نموذج تسلسلي
#     model = Sequential()
#     # طبقة التضمين
#     model.add(Embedding(input_dim = max_words, output_dim = embed_dim, input_length = max_len))
#     # LSTM
#     model.add(LSTM(units = hidden_unit ,dropout=dropout_rate))
#     # الطبقة الأخيرة
#     model.add(Dense(units = 3, activation = 'softmax'))
#     # يناء النموذج
#     model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizers(learning_rate = learning_rate), metrics = ['accuracy'])
#     # طباعة ملخص النموذج
#     print(model.summary())
 
#     return model

def save_best_model(model_name, trained_models, save_path="news_project/models/ml_models/"):
    """
    Saves the selected best model as a .pkl file.

    Parameters:
    - model_name (str): Name of the model to save.
    - trained_models (dict): Dictionary of trained models (keys = model names, values = model objects).
    - save_path (str): Directory where the model should be saved.

    Returns:
    - str: Path where the model was saved.
    """
    try:
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Get the trained model
        best_model = trained_models.get(model_name)

        if best_model is None:
            logger.error(f"❌ Model '{model_name}' not found in trained_models dictionary.")
            return None

        # Construct full save path
        model_save_path = os.path.join(save_path, f"{model_name}.pkl")

        # Save the model
        joblib.dump(best_model, model_save_path)
        logger.success(f"✅ Model '{model_name}' saved successfully at {model_save_path}")

        return model_save_path

    except Exception as e:
        logger.error(f"❌ Failed to save model '{model_name}': {e}")
        return None
