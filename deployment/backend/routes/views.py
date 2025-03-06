from fastapi import APIRouter, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from models.loading_model import load_model
from news_project.pipline import * 

router = APIRouter()
templates = Jinja2Templates(directory="deployment/backend/templates")
model_path, vec_path = load_model()

@router.post("/result")
def show_result(request: Request, newsText: str = Form(...)):
    prediction = pipline_for_sample(model_path,vec_path,newsText)
    return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction})
