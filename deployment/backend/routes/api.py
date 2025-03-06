from fastapi import APIRouter
from pydantic import BaseModel
from models.loading_model import * 
# from news_project.pipline import * 
# router = APIRouter()
# model_path, vec_path = load_model()

# class NewsInput(BaseModel):
#     text: str

# @router.post("/predict")
# def predict_news(data: NewsInput):
#     prediction = pipline_for_sample(model_path,vec_path,data.text)
#     return {"prediction": prediction["output"]}
