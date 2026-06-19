"""Entry point."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os

load_dotenv()

from api.routes import router

app = FastAPI(
    title="Pre-Ship Weight & Package Intelligence Engine",
    version="0.1.0",
)

app.include_router(router, prefix="/v1")

# Serve frontend
_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(_frontend_dir, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}
