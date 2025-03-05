from news_project.modeling.predict import *
from news_project.process import *
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