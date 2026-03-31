from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..dependencies import pipeline

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "using_fusion_model": pipeline.using_fusion_model,
        "dataset_root": str(pipeline.data_root),
        "dataset_exists": pipeline.data_root.exists(),
    }
