from news_project.modeling.predict import *
from news_project.process import *
from news_project.dataset import * 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from modeling.train import * 
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

def pipline_for_sample(model_path,vectorizer_path,text_sample):
    prediction_inforamtion  = {}
    processed_text = text_preprocessing_pipeline(text_sample)
    text_samples = [processed_text]
    predictions = load_model_and_predict(model_path, vectorizer_path, text_samples)
    if predictions[0] == 1 :
        prediction_inforamtion["output"] = predictions[0]
    else:
        prediction_inforamtion["output"] = predictions[0]
    prediction_inforamtion["text"] = text_sample
    return prediction_inforamtion

def load_data_pipline(file_path):
    data = load_data(file_path)
    return data
def clean_data_pipline(data):
    data = clean_news_data(data, text_column="news")
    return data
def labeled_target(data):
    sentiment_mapping = {"NEGATIVE": 1, "POSITIVE": 0}
    data["label"] = data["sentiment"].map(sentiment_mapping)
    return data
def select_columns(data):
    desired_columns = ['cleaned_text','cleaned_tokens','label']
    data_save = data[desired_columns]
    return data_save
def save_newdata(data):
    data.to_csv(f"../data/processed/data_V1.csv", index=False)
    path_data = "../data/processed/data_V1.csv"
    return path_data

def steps_fortraining(data):
    data_nonull = data.drop(["label"], axis=1)
    target = data_nonull["label"]
    X_train, X_test, y_train, y_test = train_test_split(data_nonull["cleaned_text"], target, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf,y_train , X_test_tfidf,y_test

def training_models(X_train,y_train,X_test,y_test,trained_model,models_df):
    # Train different classification models
    models_df, trained_models = train_and_evaluate(LogisticRegression(C=1.0, solver='saga', max_iter=500, penalty='l2', class_weight='balanced')
    , X_train, y_train, X_test, y_test, models_df, trained_model)
    models_df, trained_models = train_and_evaluate(RidgeClassifier(alpha=1.0, solver='auto')
    , X_train, y_train, X_test, y_test, models_df, trained_model)
    models_df, trained_models = train_and_evaluate(RandomForestClassifier(n_estimators=300, max_depth=50, min_samples_split=5, min_samples_leaf=2, 
                        class_weight='balanced', n_jobs=-1, random_state=42)
    , X_train, y_train, X_test, y_test, models_df, trained_model)
    models_df, trained_models = train_and_evaluate(SVC(C=10, kernel='rbf', gamma='scale', probability=True)
    , X_train, y_train, X_test, y_test, models_df, trained_model)
    models_df, trained_models = train_and_evaluate(MultinomialNB(alpha=0.1, fit_prior=True)
    , X_train, y_train, X_test, y_test, models_df, trained_model)
    models_df, trained_models = train_and_evaluate(XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=10, min_child_weight=3, 
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", use_label_encoder=False, n_jobs=-1 )
    , X_train, y_train, X_test, y_test, models_df, trained_model)
    return models_df

# def save_model():
#     save_best_model()