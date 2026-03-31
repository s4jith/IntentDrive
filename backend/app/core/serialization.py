from __future__ import annotations

from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def serialize_agents(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for agent in agents:
        serialized.append(
            {
                "id": int(agent.get("id", 0)),
                "type": str(agent.get("type", "unknown")),
                "raw_label": agent.get("raw_label"),
                "history": [
                    {"x": float(pt[0]), "y": float(pt[1])}
                    for pt in agent.get("history", [])
                ],
                "predictions": [
                    [{"x": float(pt[0]), "y": float(pt[1])} for pt in mode]
                    for mode in agent.get("predictions", [])
                ],
                "probabilities": [float(p) for p in agent.get("probabilities", [])],
                "is_target": bool(agent.get("is_target", False)),
            }
        )
    return serialized


def build_prediction_payload(result: dict[str, Any]) -> dict[str, Any]:
    core_excludes = {
        "agents",
        "target_track_id",
        "mode",
        "camera_snapshots",
        "fusion_data",
        "scene_geometry",
        "error",
    }

    payload: dict[str, Any] = {
        "mode": result.get("mode", "unknown"),
        "target_track_id": result.get("target_track_id"),
        "agents": serialize_agents(result.get("agents", [])),
        "meta": to_jsonable({k: v for k, v in result.items() if k not in core_excludes}),
    }

    snapshots = result.get("camera_snapshots")
    if snapshots:
        payload["detections"] = {
            name: {
                "frame_path": snap.get("frame_path"),
                "detections": to_jsonable(snap.get("detections", [])),
            }
            for name, snap in snapshots.items()
        }

    fusion_data = result.get("fusion_data")
    if fusion_data:
        payload["sensors"] = {
            "sample_token": fusion_data.get("sample_token"),
            "lidar_points": int(len(fusion_data.get("lidar_xy", []))),
            "radar_points": int(len(fusion_data.get("radar_xy", []))),
            "radar_channel_counts": to_jsonable(fusion_data.get("radar_channel_counts", {})),
        }

    if result.get("scene_geometry") is not None:
        payload["scene_geometry"] = to_jsonable(result.get("scene_geometry"))

    return payload
