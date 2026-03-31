from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..dependencies import pipeline

router = APIRouter()


def resolve_dataset_frame_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (pipeline.repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    data_root = pipeline.data_root.resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Frame path is outside DataSet root.") from exc

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Frame image was not found.")

    if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported frame image file type.")

    return candidate


@router.get("/live/frames")
def list_live_frames(
    channel: str = Query(default="CAM_FRONT"),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    paths = pipeline.list_channel_image_paths(channel)
    names = [p.name for p in paths[:limit]]
    return {"channel": channel, "count": len(names), "frames": names}


@router.get("/live/frame-image")
def get_live_frame_image(path: str = Query(..., min_length=1)):
    frame_path = resolve_dataset_frame_path(path)
    media_type = mimetypes.guess_type(str(frame_path))[0] or "application/octet-stream"
    return FileResponse(path=frame_path, media_type=media_type)
