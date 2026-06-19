"""Entry point."""

from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from api.routes import router

app = FastAPI(
    title="Pre-Ship Weight & Package Intelligence Engine",
    version="0.1.0",
)
app.include_router(router, prefix="/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
