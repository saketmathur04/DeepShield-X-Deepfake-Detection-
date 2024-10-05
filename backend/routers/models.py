from fastapi import APIRouter, HTTPException
from backend.services.data_loader import load_models

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("")
def list_models():
    """List all models with their parameters and metrics."""
    return load_models()

@router.get("/{model_id}")
def get_model(model_id: str):
    """Get detailed info for a single model."""
    models = load_models()
    for m in models:
        if m["id"] == model_id:
            return m
    raise HTTPException(status_code=404, detail="Model not found")
