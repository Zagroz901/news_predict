from fastapi import APIRouter, Request, Form, Depends
import sys
import os
from fastapi.templating import Jinja2Templates
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.loading_model import load_model
from scripts.pipline import * 
import numpy  as np
router = APIRouter()
templates = Jinja2Templates(directory="deployment/backend/templates")
model_path, vec_path = load_model()

@router.post("/result")
def predict_show_result(request: Request, newsText: str = Form(...)):
    prediction_result = pipline_for_sample(model_path, vec_path, newsText)
    output_label = prediction_result['output']
    text_entered = prediction_result['text']

    return templates.TemplateResponse("result.html", {
        "request": request,
        "output": output_label,
        "text": text_entered
    })
