from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from routes.api import router as api_router
from models.loading_model import load_model
app = FastAPI()

# Load model globally
model_path, vec_path = load_model()
app.mount("/static", StaticFiles(directory="deployment/backend/static"), name="static")
templates = Jinja2Templates(directory="deployment/backend/templates")

# Include API routes
app.include_router(api_router)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
