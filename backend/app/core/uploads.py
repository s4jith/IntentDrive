from __future__ import annotations

import io

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image


async def upload_to_rgb_array(upload: UploadFile) -> np.ndarray:
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{upload.filename}' is empty.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse image '{upload.filename}': {exc}") from exc

    return np.asarray(image)
