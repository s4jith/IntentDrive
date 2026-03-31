from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ...core.serialization import build_prediction_payload
from ...core.uploads import upload_to_rgb_array
from ...schemas import LiveFusionRequest, PredictionResponse
from ..dependencies import pipeline

router = APIRouter()


@router.post("/predict/two-image", response_model=PredictionResponse)
async def predict_two_image(
    image_prev: UploadFile = File(...),
    image_curr: UploadFile = File(...),
    score_threshold: float = Form(0.35),
    tracking_gate_px: float = Form(130.0),
    min_motion_px: float = Form(0.0),
    use_pose: bool = Form(False),
):
    img_prev = await upload_to_rgb_array(image_prev)
    img_curr = await upload_to_rgb_array(image_curr)

    result = pipeline.build_two_image_agents_bundle(
        img_prev=img_prev,
        img_curr=img_curr,
        score_threshold=float(score_threshold),
        tracking_gate_px=float(tracking_gate_px),
        min_motion_px=float(min_motion_px),
        use_pose=bool(use_pose),
        img_prev_name=image_prev.filename,
        img_curr_name=image_curr.filename,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return build_prediction_payload(result)


@router.post("/predict/live-fusion", response_model=PredictionResponse)
def predict_live_fusion(req: LiveFusionRequest):
    result = pipeline.build_live_agents_bundle(
        anchor_idx=req.anchor_idx,
        score_threshold=req.score_threshold,
        tracking_gate_px=req.tracking_gate_px,
        use_pose=req.use_pose,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return build_prediction_payload(result)
