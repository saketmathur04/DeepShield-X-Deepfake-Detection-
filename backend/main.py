import logging
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "Research"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import models, experiments, detection, analysis, reports
from backend.services.data_loader import get_docs_images_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepShield-X API", version="1.0.0")

# CORS to allow frontend at any local development port to communicate with backend at :8000
allowed_origins = [origin.strip() for origin in os.getenv(
    "DEEPSHIELD_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080,http://localhost:8081,http://127.0.0.1:8081,http://localhost:3000,http://127.0.0.1:3000"
).split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static images directory for the frontend to access
docs_images = get_docs_images_dir()
if docs_images.exists():
    app.mount("/static/images", StaticFiles(directory=str(docs_images)), name="images")
    logger.info("Mounted static images at /static/images")

# Mount static forensics video directory
forensics_dir = Path(__file__).resolve().parent / "static" / "forensics"
forensics_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/forensics", StaticFiles(directory=str(forensics_dir)), name="forensics")
logger.info("Mounted static forensics at /static/forensics")

# Include routers
app.include_router(models.router)
app.include_router(experiments.router)
app.include_router(detection.router)
app.include_router(analysis.router)
app.include_router(reports.router)

@app.get("/api/health")
def health_check():
    """Health check endpoint used by the frontend."""
    return {"status": "ok", "engine": "DeepShield-X", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting DeepShield-X API Server...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
