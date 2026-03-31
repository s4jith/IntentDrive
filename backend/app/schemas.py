from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Point2D(BaseModel):
    x: float
    y: float


class AgentState(BaseModel):
    id: int
    type: str
    raw_label: str | None = None
    history: list[Point2D] = Field(default_factory=list)
    predictions: list[list[Point2D]] = Field(default_factory=list)
    probabilities: list[float] = Field(default_factory=list)
    is_target: bool = False


class LiveFusionRequest(BaseModel):
    anchor_idx: int = Field(default=3, ge=0)
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    tracking_gate_px: float = Field(default=130.0, ge=1.0, le=500.0)
    use_pose: bool = False


class PredictionResponse(BaseModel):
    mode: str
    target_track_id: int | None = None
    agents: list[AgentState] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    detections: dict[str, Any] | None = None
    sensors: dict[str, Any] | None = None
    scene_geometry: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")
