from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.dependencies import pipeline
from .api.routes.health import router as health_router
from .api.routes.live import get_live_frame_image, resolve_dataset_frame_path, router as live_router
from .api.routes.predict import router as predict_router
from .core.serialization import build_prediction_payload

def create_app() -> FastAPI:
    app = FastAPI(
        title="BEV Trajectory Backend",
        version="0.2.0",
        description="Structured FastAPI backend for CV + trajectory prediction",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(live_router, prefix="/api", tags=["live"])
    app.include_router(predict_router, prefix="/api", tags=["predict"])
    return app


app = create_app()


__all__ = [
    "app",
    "create_app",
    "pipeline",
    "build_prediction_payload",
    "resolve_dataset_frame_path",
    "get_live_frame_image",
]
