import json
import io
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    KeypointRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
)

from inference import USING_FUSION_MODEL, predict as trajectory_predict
from sensor_fusion import load_fusion_for_cam_frame, radar_stabilize_motion

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Multi-Agent Trajectory Prediction Simulator", layout="wide")

BG_PRIMARY = "#05070f"
BG_SECONDARY = "#0b1220"
GRID_COLOR = "rgba(100, 116, 139, 0.22)"
ACCENT = "#eb6b26"
TARGET_PURPLE = "#a855f7"
VRU_GREEN = "#22c55e"
VEHICLE_YELLOW = "#facc15"
EGO_CYAN = "#22d3ee"
WHITE = "#e5e7eb"
TRAJ_MODE_COLORS = ["#22d3ee", "#a855f7", "#fb923c"]

ROAD_ASPHALT = "rgba(26, 34, 45, 0.94)"
ROAD_SHOULDER = "rgba(12, 18, 28, 0.90)"
LANE_SOLID = "rgba(226, 232, 240, 0.88)"
LANE_DASH = "rgba(203, 213, 225, 0.72)"
CENTER_DASH = "rgba(250, 204, 21, 0.82)"

CAMERA_VIEWS = [
    ("CAM_FRONT", "Front", 0.0),
    ("CAM_FRONT_LEFT", "Front-Left", 40.0),
    ("CAM_FRONT_RIGHT", "Front-Right", -40.0),
]

SYNTH_SKELETON_EDGES = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (1, 6),
    (6, 7),
    (6, 8),
]

COCO_SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

COCO_TO_LABEL = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

VRU_LABELS = {"person", "bicycle", "motorcycle"}
VEHICLE_LABELS = {"car", "bus", "truck"}


def normalize_probs(probs):
    arr = np.asarray(probs, dtype=float)
    arr = np.clip(arr, 1e-6, None)
    arr = arr / arr.sum()
    return arr.tolist()


def agent_color(agent):
    if agent.get("is_target", False):
        return TARGET_PURPLE
    if agent.get("type") == "pedestrian":
        return VRU_GREEN
    return VEHICLE_YELLOW


def coco_kind(label_name):
    if label_name in VRU_LABELS:
        return "pedestrian"
    if label_name in VEHICLE_LABELS:
        return "vehicle"
    return None


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 1e-9:
        return 0.0
    return inter / union


def pixel_to_bev(center_x, bottom_y, width, height):
    # Dynamic scaling from current frame dimensions (no hardcoded resolution assumptions).
    x_div = max(1.0, width / 80.0)
    y_div = max(1.0, height / 50.0)

    x_m = (center_x - 0.5 * width) / x_div
    y_m = (bottom_y - 0.58 * height) / y_div
    return float(x_m), float(y_m)


def fallback_canvas():
    h, w = 540, 960
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :, 0] = 10
    canvas[:, :, 1] = 14
    canvas[:, :, 2] = 28
    return canvas


@st.cache_data(show_spinner=False)
def list_channel_image_paths(channel):
    base = Path("DataSet") / "samples" / channel
    if not base.exists():
        return []
    return [str(p) for p in sorted(base.glob("*.jpg"))]


@st.cache_data(show_spinner=False)
def load_image_array(image_path):
    return np.asarray(Image.open(image_path).convert("RGB"))


def load_camera_frame(channel, frame_idx=0):
    image_paths = list_channel_image_paths(channel)
    if image_paths:
        idx = int(np.clip(frame_idx, 0, len(image_paths) - 1))
        return load_image_array(image_paths[idx]), image_paths[idx]
    return fallback_canvas(), None


@st.cache_resource(show_spinner=False)
def load_cv_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        det_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        det_model = fasterrcnn_resnet50_fpn(weights=det_weights, progress=False)
        det_model.to(device).eval()

        pose_weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        pose_model = keypointrcnn_resnet50_fpn(weights=pose_weights, progress=False)
        pose_model.to(device).eval()

        return {
            "device": device,
            "device_name": str(device),
            "det_model": det_model,
            "det_weights": det_weights,
            "pose_model": pose_model,
            "pose_weights": pose_weights,
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "device": device,
            "device_name": str(device),
        }


def detect_objects_and_pose(image_arr, models, score_threshold=0.55, use_pose=True):
    if "error" in models:
        return []

    device = models["device"]
    pil_img = Image.fromarray(image_arr)

    det_input = models["det_weights"].transforms()(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        det_out = models["det_model"](det_input)[0]

    boxes = det_out["boxes"].detach().cpu().numpy() if len(det_out["boxes"]) > 0 else np.zeros((0, 4))
    scores = det_out["scores"].detach().cpu().numpy() if len(det_out["scores"]) > 0 else np.zeros((0,))
    labels = det_out["labels"].detach().cpu().numpy() if len(det_out["labels"]) > 0 else np.zeros((0,))

    detections = []
    for i in range(len(scores)):
        score = float(scores[i])
        label_idx = int(labels[i])
        label_name = COCO_TO_LABEL.get(label_idx)

        if label_name is None or score < score_threshold:
            continue

        kind = coco_kind(label_name)
        if kind is None:
            continue

        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        detections.append(
            {
                "score": score,
                "raw_label": label_name,
                "kind": kind,
                "box": [x1, y1, x2, y2],
                "center_x": 0.5 * (x1 + x2),
                "bottom_y": y2,
                "keypoints": None,
            }
        )

    if use_pose:
        pose_input = models["pose_weights"].transforms()(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            pose_out = models["pose_model"](pose_input)[0]

        p_boxes = pose_out["boxes"].detach().cpu().numpy() if len(pose_out["boxes"]) > 0 else np.zeros((0, 4))
        p_scores = pose_out["scores"].detach().cpu().numpy() if len(pose_out["scores"]) > 0 else np.zeros((0,))
        p_labels = pose_out["labels"].detach().cpu().numpy() if len(pose_out["labels"]) > 0 else np.zeros((0,))
        p_keypoints = pose_out["keypoints"].detach().cpu().numpy() if len(pose_out["keypoints"]) > 0 else np.zeros((0, 17, 3))

        assigned = set()
        for i in range(len(p_scores)):
            if int(p_labels[i]) != 1:
                continue
            if float(p_scores[i]) < max(0.25, 0.8 * score_threshold):
                continue

            pose_box = [float(v) for v in p_boxes[i]]
            best_idx = None
            best_iou = 0.0

            for det_idx, det in enumerate(detections):
                if det_idx in assigned:
                    continue
                if det["raw_label"] != "person":
                    continue
                iou_val = iou_xyxy(det["box"], pose_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = det_idx

            if best_idx is not None and best_iou > 0.1:
                detections[best_idx]["keypoints"] = p_keypoints[i].tolist()
                assigned.add(best_idx)

    return detections


def track_front_agents(front_paths, models, score_threshold=0.55, tracking_gate_px=90.0, use_pose=True):
    tracks = {}
    next_track_id = 1
    front_final_detections = []

    for frame_idx, frame_path in enumerate(front_paths):
        frame_arr = load_image_array(frame_path)
        h, w = frame_arr.shape[:2]

        detections = detect_objects_and_pose(
            frame_arr,
            models,
            score_threshold=score_threshold,
            use_pose=use_pose,
        )
        detections.sort(key=lambda d: d["score"], reverse=True)

        matched_track_ids = set()
        frame_dets_with_ids = []

        for det in detections:
            wx, wy = pixel_to_bev(det["center_x"], det["bottom_y"], w, h)

            best_track_id = None
            best_dist = 1e9

            for tid, tr in tracks.items():
                if tr["kind"] != det["kind"]:
                    continue
                if tr["last_seen"] != frame_idx - 1:
                    continue
                if tid in matched_track_ids:
                    continue

                px_last, py_last = tr["history_pixel"][-1]
                dist = math.hypot(det["center_x"] - px_last, det["bottom_y"] - py_last)
                if dist < tracking_gate_px and dist < best_dist:
                    best_dist = dist
                    best_track_id = tid

            if best_track_id is None:
                best_track_id = next_track_id
                next_track_id += 1
                tracks[best_track_id] = {
                    "id": best_track_id,
                    "kind": det["kind"],
                    "raw_label": det["raw_label"],
                    "history_pixel": [],
                    "history_world": [],
                    "last_seen": -1,
                    "last_box": None,
                    "last_keypoints": None,
                    "misses": 0,
                }

            tr = tracks[best_track_id]
            tr["history_pixel"].append((float(det["center_x"]), float(det["bottom_y"])))
            tr["history_world"].append((float(wx), float(wy)))
            tr["last_seen"] = frame_idx
            tr["raw_label"] = det["raw_label"]
            tr["last_box"] = det["box"]
            tr["last_keypoints"] = det.get("keypoints")
            tr["misses"] = 0

            matched_track_ids.add(best_track_id)

            det = dict(det)
            det["track_id"] = best_track_id
            frame_dets_with_ids.append(det)

        # Extrapolate temporarily-lost tracks so 4-point histories can still be formed.
        for tid, tr in tracks.items():
            if tr["last_seen"] == frame_idx:
                continue
            if tr["last_seen"] < frame_idx - 1:
                continue

            if len(tr["history_pixel"]) >= 2:
                px_prev, py_prev = tr["history_pixel"][-2]
                px_last, py_last = tr["history_pixel"][-1]
                wx_prev, wy_prev = tr["history_world"][-2]
                wx_last, wy_last = tr["history_world"][-1]

                px_ex = px_last + (px_last - px_prev)
                py_ex = py_last + (py_last - py_prev)
                wx_ex = wx_last + (wx_last - wx_prev)
                wy_ex = wy_last + (wy_last - wy_prev)
            else:
                px_ex, py_ex = tr["history_pixel"][-1]
                wx_ex, wy_ex = tr["history_world"][-1]

            tr["history_pixel"].append((float(px_ex), float(py_ex)))
            tr["history_world"].append((float(wx_ex), float(wy_ex)))
            tr["last_seen"] = frame_idx
            tr["misses"] += 1

        if frame_idx == len(front_paths) - 1:
            front_final_detections = frame_dets_with_ids

    valid_tracks = []
    for tid, tr in tracks.items():
        if len(tr["history_world"]) != len(front_paths):
            continue
        if tr["misses"] > 2:
            continue

        x0, y0 = tr["history_world"][0]
        x1, y1 = tr["history_world"][-1]
        motion = math.hypot(x1 - x0, y1 - y0)
        if motion < 0.08:
            continue

        valid_tracks.append(
            {
                "id": tid,
                "kind": tr["kind"],
                "raw_label": tr["raw_label"],
                "history_pixel": [tuple(p) for p in tr["history_pixel"]],
                "history_world": [tuple(p) for p in tr["history_world"]],
                "last_box": tr["last_box"],
                "last_keypoints": tr["last_keypoints"],
            }
        )

    valid_tracks.sort(key=lambda t: t["id"])
    return valid_tracks, front_final_detections


def raw_label_to_stabilizer_type(raw_label):
    if raw_label == "person":
        return "Person"
    if raw_label == "bicycle":
        return "Bicycle"
    if raw_label == "motorcycle":
        return "Motorcycle"
    if raw_label == "bus":
        return "Bus"
    if raw_label == "truck":
        return "Truck"
    return "Car"


def build_fusion_features(history_world, fusion_data):
    if not fusion_data:
        return None

    lidar_xy = fusion_data.get("lidar_xy")
    radar_xy = fusion_data.get("radar_xy")

    if lidar_xy is None and radar_xy is None:
        return None

    feats = []
    for px, py in history_world:
        if lidar_xy is not None and len(lidar_xy) > 0:
            dl = np.hypot(lidar_xy[:, 0] - px, lidar_xy[:, 1] - py)
            lidar_cnt = int((dl < 2.0).sum())
        else:
            lidar_cnt = 0

        if radar_xy is not None and len(radar_xy) > 0:
            dr = np.hypot(radar_xy[:, 0] - px, radar_xy[:, 1] - py)
            radar_cnt = int((dr < 2.5).sum())
        else:
            radar_cnt = 0

        lidar_norm = min(80.0, float(lidar_cnt)) / 80.0
        radar_norm = min(30.0, float(radar_cnt)) / 30.0
        sensor_strength = min(1.0, (float(lidar_cnt) + 2.0 * float(radar_cnt)) / 100.0)
        feats.append([lidar_norm, radar_norm, sensor_strength])

    return feats


def stabilize_tracks_with_radar(tracks, fusion_data):
    if not tracks:
        return tracks

    packed = []
    for tr in tracks:
        hist = tr["history_world"]
        if len(hist) >= 2:
            dx = float(hist[-1][0] - hist[-2][0])
            dy = float(hist[-1][1] - hist[-2][1])
        else:
            dx = 0.0
            dy = 0.0

        packed.append(
            {
                "type": raw_label_to_stabilizer_type(tr.get("raw_label", "car")),
                "history": [tuple(p) for p in hist],
                "dx": dx,
                "dy": dy,
            }
        )

    stabilized = radar_stabilize_motion(packed, fusion_data, dt_seconds=0.5)

    updated = []
    for tr, st in zip(tracks, stabilized):
        t_copy = dict(tr)
        t_copy["history_world"] = [(float(x), float(y)) for x, y in st["history"]]
        updated.append(t_copy)

    return updated


def choose_target_track_id(tracks):
    if not tracks:
        return None

    peds = [t for t in tracks if t["kind"] == "pedestrian"]
    if peds:
        best = min(peds, key=lambda t: math.hypot(t["history_world"][-1][0], t["history_world"][-1][1]))
        return best["id"]

    return tracks[0]["id"]


def build_agents_from_tracks(tracks, fusion_data):
    if not tracks:
        return [], None, []

    tracks_work = []
    for tr in tracks:
        tracks_work.append(
            {
                "id": tr["id"],
                "kind": tr["kind"],
                "raw_label": tr["raw_label"],
                "history_pixel": [tuple(p) for p in tr["history_pixel"]],
                "history_world": [tuple(p) for p in tr["history_world"]],
                "last_box": tr.get("last_box"),
                "last_keypoints": tr.get("last_keypoints"),
            }
        )

    tracks_work = stabilize_tracks_with_radar(tracks_work, fusion_data)

    target_id = choose_target_track_id(tracks_work)
    agents = []

    for tr in tracks_work:
        neighbors = []
        for other in tracks_work:
            if other["id"] == tr["id"]:
                continue
            neighbors.append(other["history_world"])

        if len(neighbors) > 12:
            x0, y0 = tr["history_world"][-1]
            neighbors = sorted(
                neighbors,
                key=lambda nh: math.hypot(nh[-1][0] - x0, nh[-1][1] - y0),
            )[:12]

        fusion_feats = build_fusion_features(tr["history_world"], fusion_data)

        pred, probs, _ = trajectory_predict(
            tr["history_world"],
            neighbor_points_list=neighbors,
            fusion_feats=fusion_feats,
        )

        pred_np = pred.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        predictions = []
        for mode_i in range(pred_np.shape[0]):
            mode_path = [(float(p[0]), float(p[1])) for p in pred_np[mode_i]]
            predictions.append(mode_path)

        agents.append(
            {
                "id": int(tr["id"]),
                "type": "pedestrian" if tr["kind"] == "pedestrian" else "vehicle",
                "raw_label": tr["raw_label"],
                "history": [tuple(map(float, p)) for p in tr["history_world"]],
                "predictions": predictions,
                "probabilities": normalize_probs(probs_np.tolist()),
                "is_target": tr["id"] == target_id,
            }
        )

    return agents, target_id, tracks_work


def assign_track_ids_to_front_detections(detections, tracks, gate_px=90.0):
    if not detections:
        return []

    out = []
    used_ids = set()

    for det_idx, det in enumerate(detections):
        d = dict(det)
        d.setdefault("det_id", det_idx + 1)

        if d.get("track_id") is not None:
            used_ids.add(d["track_id"])
            out.append(d)
            continue

        best_id = None
        best_dist = 1e9

        for tr in tracks:
            if tr["id"] in used_ids:
                continue
            if tr["kind"] != d["kind"]:
                continue

            px, py = tr["history_pixel"][-1]
            dist = math.hypot(d["center_x"] - px, d["bottom_y"] - py)
            if dist < gate_px and dist < best_dist:
                best_dist = dist
                best_id = tr["id"]

        d["track_id"] = best_id
        if best_id is not None:
            used_ids.add(best_id)
        out.append(d)

    return out


@st.cache_data(show_spinner=False)
def build_live_agents_bundle(anchor_idx, score_threshold, tracking_gate_px, use_pose):
    front_paths = list_channel_image_paths("CAM_FRONT")
    if len(front_paths) < 4:
        return {"error": "Need at least 4 CAM_FRONT frames in DataSet/samples/CAM_FRONT."}

    if anchor_idx < 3:
        anchor_idx = 3
    if anchor_idx >= len(front_paths):
        anchor_idx = len(front_paths) - 1

    models = load_cv_models()
    if "error" in models:
        return {
            "error": f"Could not load CV models ({models['error']}).",
            "device": models.get("device_name", "unknown"),
        }

    window_paths = front_paths[anchor_idx - 3 : anchor_idx + 1]

    tracks, front_dets = track_front_agents(
        window_paths,
        models,
        score_threshold=score_threshold,
        tracking_gate_px=tracking_gate_px,
        use_pose=use_pose,
    )

    if len(tracks) == 0:
        return {"error": "No valid tracked moving agents found in selected frame window."}

    front_curr = window_paths[-1]
    fusion_data = load_fusion_for_cam_frame(Path(front_curr).name)

    agents, target_id, tracks_stable = build_agents_from_tracks(tracks, fusion_data)
    if len(agents) == 0:
        return {"error": "Tracking succeeded but trajectory prediction produced no agents."}

    snapshots = {}
    for channel, _, _ in CAMERA_VIEWS:
        ch_paths = list_channel_image_paths(channel)

        if not ch_paths:
            snapshots[channel] = {
                "image": fallback_canvas(),
                "detections": [],
                "frame_path": None,
            }
            continue

        ch_idx = min(anchor_idx, len(ch_paths) - 1)
        ch_path = ch_paths[ch_idx]
        ch_arr = load_image_array(ch_path)

        if channel == "CAM_FRONT" and Path(ch_path).name == Path(front_curr).name:
            ch_dets = [dict(d) for d in front_dets]
        else:
            ch_dets = detect_objects_and_pose(
                ch_arr,
                models,
                score_threshold=score_threshold,
                use_pose=use_pose,
            )

        for i, det in enumerate(ch_dets):
            det.setdefault("track_id", None)
            det.setdefault("det_id", i + 1)

        snapshots[channel] = {
            "image": ch_arr,
            "detections": ch_dets,
            "frame_path": ch_path,
        }

    if "CAM_FRONT" in snapshots:
        snapshots["CAM_FRONT"]["detections"] = assign_track_ids_to_front_detections(
            snapshots["CAM_FRONT"]["detections"],
            tracks_stable,
            gate_px=tracking_gate_px,
        )

    return {
        "agents": agents,
        "fusion_data": fusion_data,
        "camera_snapshots": snapshots,
        "target_track_id": target_id,
        "device": models.get("device_name", "unknown"),
        "front_anchor_path": front_curr,
        "mode": "live_fusion",
    }


def uploaded_file_to_array(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return np.asarray(Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB"))
    except Exception:
        return None


def match_two_frame_tracks(det_prev, det_curr, tracking_gate_px=90.0, min_motion_px=0.0):
    used_curr = set()
    matches = []

    det_prev = sorted(det_prev, key=lambda d: d["score"], reverse=True)
    det_curr = sorted(det_curr, key=lambda d: d["score"], reverse=True)

    for d0 in det_prev:
        best_idx = None
        best_dist = 1e9

        for j, d1 in enumerate(det_curr):
            if j in used_curr:
                continue
            if d0["kind"] != d1["kind"]:
                continue

            dist = math.hypot(d1["center_x"] - d0["center_x"], d1["bottom_y"] - d0["bottom_y"])
            if dist < tracking_gate_px and dist < best_dist:
                best_dist = dist
                best_idx = j

        if best_idx is None:
            continue

        used_curr.add(best_idx)
        d1 = det_curr[best_idx]

        matches.append((d0, d1, float(best_dist)))

    return matches


def build_two_image_agents_bundle(img_prev, img_curr, score_threshold, tracking_gate_px, min_motion_px, use_pose):
    models = load_cv_models()
    if "error" in models:
        return {
            "error": f"Could not load CV models ({models['error']}).",
            "device": models.get("device_name", "unknown"),
        }

    det_prev = detect_objects_and_pose(img_prev, models, score_threshold=score_threshold, use_pose=use_pose)
    det_curr = detect_objects_and_pose(img_curr, models, score_threshold=score_threshold, use_pose=use_pose)

    # Two-image mode focuses on VRUs (pedestrians/cyclists/motorcycles).
    det_prev_vru = [d for d in det_prev if d.get("kind") == "pedestrian"]
    det_curr_vru = [d for d in det_curr if d.get("kind") == "pedestrian"]

    for i, d in enumerate(det_prev):
        d["det_id"] = i + 1
        d["track_id"] = None
    for i, d in enumerate(det_curr):
        d["det_id"] = i + 1
        d["track_id"] = None

    if len(det_curr_vru) == 0:
        return {"error": "No pedestrian/cyclist detections found in image 2 (t0)."}

    matches = match_two_frame_tracks(
        det_prev_vru,
        det_curr_vru,
        tracking_gate_px=tracking_gate_px,
        min_motion_px=0.0,
    )

    # Backfill unmatched current VRUs so every visible VRU at t0 gets a prediction.
    matched_curr_ids = {id(m[1]) for m in matches}
    for d1 in det_curr_vru:
        if id(d1) in matched_curr_ids:
            continue

        if len(det_prev_vru) == 0:
            matches.append((None, d1, float("inf")))
            continue

        nearest_prev = min(
            det_prev_vru,
            key=lambda d0: math.hypot(d1["center_x"] - d0["center_x"], d1["bottom_y"] - d0["bottom_y"]),
        )
        dist = math.hypot(
            d1["center_x"] - nearest_prev["center_x"],
            d1["bottom_y"] - nearest_prev["bottom_y"],
        )

        # If previous frame support is weak, still include the agent with near-static history.
        if dist <= 1.5 * tracking_gate_px:
            matches.append((nearest_prev, d1, float(dist)))
        else:
            matches.append((None, d1, float("inf")))

    h0, w0 = img_prev.shape[:2]
    h1, w1 = img_curr.shape[:2]

    tracks = []
    for track_id, (d0, d1, dist_px) in enumerate(matches, start=1):
        if d0 is not None and d0.get("track_id") is None:
            d0["track_id"] = track_id
        d1["track_id"] = track_id

        if d0 is not None:
            p_prev = pixel_to_bev(d0["center_x"], d0["bottom_y"], w0, h0)
        else:
            p_prev = None
        p_curr = pixel_to_bev(d1["center_x"], d1["bottom_y"], w1, h1)

        if p_prev is None:
            vx, vy = 0.0, 0.0
            p_prev = p_curr
        else:
            vx = p_curr[0] - p_prev[0]
            vy = p_curr[1] - p_prev[1]

        # Keep the agent even if tiny displacement; just make observation history static.
        if dist_px < float(min_motion_px):
            vx, vy = 0.0, 0.0
            p_prev = p_curr

        # Reconstruct a 4-point observation history from 2 frames.
        hist = [
            (p_curr[0] - 3.0 * vx, p_curr[1] - 3.0 * vy),
            (p_curr[0] - 2.0 * vx, p_curr[1] - 2.0 * vy),
            (p_prev[0], p_prev[1]),
            (p_curr[0], p_curr[1]),
        ]

        tracks.append(
            {
                "id": track_id,
                "kind": d1["kind"],
                "raw_label": d1["raw_label"],
                "history_world": hist,
            }
        )

    # In this mode, every VRU is treated as a target for prediction display.
    target_track_id = None

    agents = []
    for tr in tracks:
        neighbors = [other["history_world"] for other in tracks if other["id"] != tr["id"]]

        pred, probs, _ = trajectory_predict(
            tr["history_world"],
            neighbor_points_list=neighbors,
            fusion_feats=None,
        )

        pred_np = pred.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        predictions = []
        for mode_i in range(pred_np.shape[0]):
            predictions.append([(float(p[0]), float(p[1])) for p in pred_np[mode_i]])

        agents.append(
            {
                "id": int(tr["id"]),
                "type": "pedestrian" if tr["kind"] == "pedestrian" else "vehicle",
                "raw_label": tr["raw_label"],
                "history": [tuple(map(float, p)) for p in tr["history_world"]],
                "predictions": predictions,
                "probabilities": normalize_probs(probs_np.tolist()),
                "is_target": True,
            }
        )

    return {
        "agents": agents,
        "target_track_id": target_track_id,
        "camera_snapshots": {
            "pair_prev": {"image": img_prev, "detections": det_prev},
            "pair_curr": {"image": img_curr, "detections": det_curr},
        },
        "device": models.get("device_name", "unknown"),
        "mode": "two_upload",
        "match_count": len(agents),
    }


def bev_to_pixel(x_m, y_m, width, height):
    x_div = max(1.0, width / 80.0)
    y_div = max(1.0, height / 50.0)

    px = x_m * x_div + 0.5 * width
    py = y_m * y_div + 0.58 * height
    return float(px), float(py)


def create_prediction_overlay_figure(image_arr, detections, agents, step, target_track_id=None, highlight_track_ids=None):
    fig = create_camera_figure_detections(
        image_arr,
        detections,
        camera_label="Prediction Output",
        target_track_id=target_track_id,
        highlight_track_ids=highlight_track_ids,
    )

    h, w = image_arr.shape[:2]

    for a in agents:
        color = agent_color(a)
        k = best_mode_idx(a)
        pred = a["predictions"][k]
        end_idx = max(1, min(step, len(pred)))
        path_world = [a["history"][-1]] + pred[:end_idx]

        px = []
        py = []
        for xw, yw in path_world:
            u, v = bev_to_pixel(xw, yw, w, h)
            px.append(u)
            py.append(v)

        # Glow trail for a cleaner, reference-style visual emphasis.
        for lw, op in [(14, 0.12), (8, 0.20), (4, 0.95)]:
            fig.add_trace(
                go.Scatter(
                    x=px,
                    y=py,
                    mode="lines",
                    line={"color": color, "width": lw, "shape": "spline", "smoothing": 1.1},
                    opacity=op,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    return fig


def remove_vru_foreground_from_scene(scene_image, scene_detections=None):
    if scene_image is None or cv2 is None:
        return scene_image

    if scene_detections is None or len(scene_detections) == 0:
        return scene_image

    h, w = scene_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for det in scene_detections:
        if det.get("kind") != "pedestrian":
            continue

        x1, y1, x2, y2 = det.get("box", [0, 0, 0, 0])
        padx = 0.08 * (x2 - x1)
        pady = 0.10 * (y2 - y1)

        xa = int(max(0, min(w - 1, x1 - padx)))
        ya = int(max(0, min(h - 1, y1 - pady)))
        xb = int(max(0, min(w - 1, x2 + padx)))
        yb = int(max(0, min(h - 1, y2 + pady)))

        if xb > xa and yb > ya:
            cv2.rectangle(mask, (xa, ya), (xb, yb), color=255, thickness=-1)

    if int(mask.sum()) == 0:
        return scene_image

    bgr = cv2.cvtColor(scene_image, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def build_pseudo_bev_background(scene_image, x_min, x_max, y_min, y_max, scene_detections=None):
    # Context BEV from a single front-view frame using inverse-perspective remap.
    if scene_image is None or cv2 is None:
        return None

    cleaned = remove_vru_foreground_from_scene(scene_image, scene_detections=scene_detections)
    h, w = cleaned.shape[:2]
    if h < 20 or w < 20:
        return None

    out_w, out_h = 1100, 820

    xs = np.linspace(x_min, x_max, out_w, dtype=np.float32)
    ys = np.linspace(y_max, y_min, out_h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)

    cx = 0.5 * w
    horizon = 0.42 * h

    depth = np.clip((yg - y_min) + 2.0, 2.0, None)

    map_x = cx + (0.95 * w) * xg / (depth + 6.0)
    map_y = horizon + (5.8 * h) / depth

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

    warped = cv2.remap(cleaned, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warped = cv2.GaussianBlur(warped, (0, 0), 0.8)
    warped = np.clip(warped.astype(np.float32) * 0.78, 0, 255).astype(np.uint8)
    return warped


def compute_reference_bounds(agents, step, show_multimodal):
    xs = [0.0]
    ys = [0.0]

    for a in agents:
        for xh, yh in a["history"]:
            xs.append(float(xh))
            ys.append(float(yh))

        k_best = best_mode_idx(a)
        best_path = a["predictions"][k_best][: max(1, min(step, len(a["predictions"][k_best])))]
        for xp, yp in best_path:
            xs.append(float(xp))
            ys.append(float(yp))

        if show_multimodal:
            for m, m_path in enumerate(a["predictions"]):
                if m == k_best:
                    continue
                m_slice = m_path[: max(1, min(step, len(m_path)))]
                for xp, yp in m_slice:
                    xs.append(float(xp))
                    ys.append(float(yp))

    x_min = min(xs) - 6.0
    x_max = max(xs) + 6.0
    y_min = min(ys) - 8.0
    y_max = max(ys) + 10.0

    min_x_span = 44.0
    min_y_span = 64.0

    x_span = x_max - x_min
    y_span = y_max - y_min

    if x_span < min_x_span:
        xc = 0.5 * (x_min + x_max)
        x_min = xc - 0.5 * min_x_span
        x_max = xc + 0.5 * min_x_span

    if y_span < min_y_span:
        yc = 0.5 * (y_min + y_max)
        y_min = yc - 0.5 * min_y_span
        y_max = yc + 0.5 * min_y_span

    return x_min, x_max, y_min, y_max


def spread_agent_markers(agents, step, tol=0.45, radius=0.55):
    positions = [position_at_step(a, step) for a in agents]
    offsets = []

    for i, (xi, yi) in enumerate(positions):
        near = []
        for j, (xj, yj) in enumerate(positions):
            if math.hypot(xi - xj, yi - yj) <= tol:
                near.append(j)

        if len(near) <= 1:
            offsets.append((0.0, 0.0))
            continue

        near_sorted = sorted(near)
        rank = near_sorted.index(i)
        ang = 2.0 * math.pi * rank / len(near_sorted)
        offsets.append((radius * math.cos(ang), radius * math.sin(ang)))

    return positions, offsets


def hex_to_rgba(hex_color, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    c = str(hex_color).lstrip("#")
    if len(c) != 6:
        return f"rgba(229,231,235,{alpha:.3f})"
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def summarize_agent_probabilities(agent):
    bins = {"Straight": 0.0, "Left": 0.0, "Right": 0.0, "Stop": 0.0}

    classifier = globals().get("classify_direction")
    for mode_idx, mode_path in enumerate(agent.get("predictions", [])):
        if mode_idx >= len(agent.get("probabilities", [])):
            continue

        if callable(classifier):
            direction = classifier(agent["history"], mode_path)
        else:
            direction = ["Straight", "Left", "Right"][mode_idx % 3]

        if direction not in bins:
            direction = "Straight"

        bins[direction] += float(agent["probabilities"][mode_idx])

    ranked = sorted(bins.items(), key=lambda kv: kv[1], reverse=True)
    top3 = ranked[:3]
    summary = ", ".join([f"{name} {prob * 100:.0f}%" for name, prob in top3])
    return summary, bins


def add_structured_road_scene(fig, x_min, x_max, y_min, y_max, add_crosswalk=True):
    road_half = float(np.clip(0.24 * (x_max - x_min), 9.5, 15.5))
    shoulder_half = road_half + 3.2

    fig.add_shape(
        type="rect",
        x0=x_min,
        y0=y_min,
        x1=x_max,
        y1=y_max,
        line={"width": 0},
        fillcolor=ROAD_SHOULDER,
        layer="below",
    )

    fig.add_shape(
        type="rect",
        x0=-shoulder_half,
        y0=y_min,
        x1=shoulder_half,
        y1=y_max,
        line={"width": 0},
        fillcolor="rgba(18, 25, 35, 0.95)",
        layer="below",
    )

    fig.add_shape(
        type="rect",
        x0=-road_half,
        y0=y_min,
        x1=road_half,
        y1=y_max,
        line={"width": 0},
        fillcolor=ROAD_ASPHALT,
        layer="below",
    )

    for x_edge in (-road_half, road_half):
        fig.add_shape(
            type="line",
            x0=x_edge,
            y0=y_min,
            x1=x_edge,
            y1=y_max,
            line={"color": LANE_SOLID, "width": 2.5},
            layer="below",
        )

    lane_w = (2.0 * road_half) / 4.0
    for lane_idx in range(1, 4):
        x_lane = -road_half + lane_idx * lane_w
        line_color = CENTER_DASH if lane_idx == 2 else LANE_DASH
        line_width = 2.4 if lane_idx == 2 else 1.8
        fig.add_shape(
            type="line",
            x0=x_lane,
            y0=y_min,
            x1=x_lane,
            y1=y_max,
            line={"color": line_color, "width": line_width, "dash": "dash"},
            layer="below",
        )

    if add_crosswalk:
        cross_y = float(np.clip(8.0, y_min + 5.5, y_max - 5.5))
        stripe_h = 0.7
        stripe_gap = 0.55
        for i in range(-4, 5):
            y0 = cross_y + i * (stripe_h + stripe_gap)
            y1 = y0 + stripe_h
            fig.add_shape(
                type="rect",
                x0=-road_half + 0.7,
                y0=y0,
                x1=road_half - 0.7,
                y1=y1,
                line={"width": 0},
                fillcolor="rgba(229, 231, 235, 0.14)",
                layer="below",
            )


def build_reference_bev_figure(agents, step, show_multimodal, scene_image=None, scene_detections=None):
    fig = go.Figure()

    x_min, x_max, y_min, y_max = compute_reference_bounds(agents, step, show_multimodal)

    bg = build_pseudo_bev_background(
        scene_image,
        x_min,
        x_max,
        y_min,
        y_max,
        scene_detections=scene_detections,
    )

    add_structured_road_scene(fig, x_min, x_max, y_min, y_max, add_crosswalk=True)

    if bg is not None:
        fig.add_layout_image(
            dict(
                source=Image.fromarray(bg),
                xref="x",
                yref="y",
                x=x_min,
                y=y_max,
                sizex=x_max - x_min,
                sizey=y_max - y_min,
                sizing="stretch",
                opacity=0.38,
                layer="below",
            )
        )

        # Dark wash to keep trajectories readable on real-scene texture.
        fig.add_shape(
            type="rect",
            x0=x_min,
            y0=y_min,
            x1=x_max,
            y1=y_max,
            line={"width": 0},
            fillcolor="rgba(4, 8, 18, 0.36)",
            layer="below",
        )

    fig.add_shape(
        type="rect",
        x0=-1.1,
        y0=-2.2,
        x1=1.1,
        y1=2.2,
        line={"color": EGO_CYAN, "width": 2.2},
        fillcolor="rgba(34,211,238,0.20)",
    )
    fig.add_annotation(
        x=0.0,
        y=4.2,
        ax=0.0,
        ay=1.2,
        showarrow=True,
        arrowhead=3,
        arrowwidth=2.8,
        arrowcolor=EGO_CYAN,
        text="",
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 10, "symbol": "circle", "color": VRU_GREEN},
            name="Pedestrian",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 10, "symbol": "square", "color": VEHICLE_YELLOW},
            name="Vehicle",
        )
    )

    positions, marker_offsets = spread_agent_markers(agents, step)
    alt_legend_added = False

    for idx, a in enumerate(agents):
        base_color = agent_color(a)
        best_idx = best_mode_idx(a)
        best_prob = float(a["probabilities"][best_idx]) if len(a["probabilities"]) > 0 else 0.0
        marker_color = hex_to_rgba(base_color, 0.48 + 0.52 * best_prob)

        cx, cy = positions[idx]
        ox, oy = marker_offsets[idx]
        curr_x = cx + ox
        curr_y = cy + oy

        summary_text, _ = summarize_agent_probabilities(a)
        hover_text = (
            f"ID {a['id']}<br>Type: {a['type'].title()}"
            f"<br>{summary_text}<br>Best path confidence: {best_prob * 100:.1f}%"
        )

        hx, hy = smooth_path(a["history"])
        fig.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="lines",
                line={"color": "rgba(226,232,240,0.55)", "width": 2.2, "dash": "dot", "shape": "spline", "smoothing": 1.0},
                hovertemplate=f"ID {a['id']} past trajectory<extra></extra>",
                name="Past trajectory" if idx == 0 else None,
                showlegend=(idx == 0),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[curr_x],
                y=[curr_y],
                mode="markers+text",
                marker={
                    "size": 11,
                    "symbol": "circle" if a.get("type") == "pedestrian" else "square",
                    "color": marker_color,
                    "line": {"color": "rgba(5,7,15,0.95)", "width": 1.2},
                },
                text=[f"ID {a['id']}"],
                textposition="top center",
                textfont={"size": 10, "color": WHITE},
                hovertemplate=f"{hover_text}<extra></extra>",
                showlegend=False,
            )
        )

        px, py = previous_position_for_velocity(a, step)
        dx, dy = cx - px, cy - py
        norm = math.hypot(dx, dy)
        if norm > 1e-3:
            vx, vy = (dx / norm) * 2.0, (dy / norm) * 2.0
            fig.add_annotation(
                x=curr_x + vx,
                y=curr_y + vy,
                ax=curr_x,
                ay=curr_y,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=base_color,
                text="",
            )

        mode_order = [best_idx, 0, 1, 2]
        mode_order = list(dict.fromkeys(mode_order))

        for rank, m in enumerate(mode_order[:3]):
            if (not show_multimodal) and rank > 0:
                continue

            mode_prob = float(a["probabilities"][m]) if m < len(a["probabilities"]) else 0.0
            mode_color = TRAJ_MODE_COLORS[m % len(TRAJ_MODE_COLORS)]

            mode_path = a["predictions"][m]
            mode_slice = mode_path[: max(1, min(step, len(mode_path)))]
            tx, ty = smooth_path([a["history"][-1]] + mode_slice)
            is_best = m == best_idx

            if is_best:
                for lw, op in [(14, 0.08), (9, 0.16)]:
                    fig.add_trace(
                        go.Scatter(
                            x=tx,
                            y=ty,
                            mode="lines",
                            line={"color": mode_color, "width": lw, "shape": "spline", "smoothing": 1.15},
                            opacity=op,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

            fig.add_trace(
                go.Scatter(
                    x=tx,
                    y=ty,
                    mode="lines",
                    line={
                        "color": mode_color,
                        "width": 4.1 if is_best else 2.1,
                        "dash": "solid" if is_best else "dash",
                        "shape": "spline",
                        "smoothing": 1.15,
                    },
                    opacity=(0.72 + 0.26 * mode_prob) if is_best else (0.36 + 0.32 * mode_prob),
                    hovertemplate=(
                        f"ID {a['id']}<br>Mode {m + 1}"
                        f"<br>Probability: {mode_prob * 100:.1f}%<extra></extra>"
                    ),
                    name=(
                        "Best path" if (is_best and idx == 0) else
                        "Alternative paths" if ((not is_best) and (not alt_legend_added)) else None
                    ),
                    showlegend=(is_best and idx == 0) or ((not is_best) and (not alt_legend_added)),
                )
            )

            if (not is_best) and (not alt_legend_added):
                alt_legend_added = True

        if a.get("is_target", False):
            fig.add_trace(
                go.Scatter(
                    x=[curr_x + 0.9],
                    y=[curr_y + 1.1],
                    mode="text",
                    text=[summary_text],
                    textfont={"size": 9, "color": "rgba(226,232,240,0.90)"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title={"text": "Main BEV Simulation", "x": 0.02, "font": {"size": 20, "color": WHITE}},
        paper_bgcolor=BG_SECONDARY,
        plot_bgcolor=BG_SECONDARY,
        legend={"orientation": "h", "y": 1.03, "x": 0.0, "font": {"color": WHITE, "size": 11}},
        margin={"l": 16, "r": 16, "t": 52, "b": 10},
        height=700,
    )
    fig.update_xaxes(
        title_text="X Lateral (m)",
        range=[x_min, x_max],
        color=WHITE,
        dtick=5,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.16)",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Y Forward (m)",
        range=[y_min, y_max],
        color=WHITE,
        dtick=5,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.16)",
        scaleanchor="x",
        scaleratio=1,
        zeroline=False,
    )

    return fig


def best_mode_idx(agent):
    probs = np.asarray(agent["probabilities"], dtype=float)
    return int(np.argmax(probs))


def position_at_step(agent, step):
    if step <= 0:
        return tuple(agent["history"][-1])

    k = best_mode_idx(agent)
    pred = agent["predictions"][k]
    idx = min(step - 1, len(pred) - 1)
    return tuple(pred[idx])


def previous_position_for_velocity(agent, step):
    if step <= 1:
        return tuple(agent["history"][-1])

    k = best_mode_idx(agent)
    pred = agent["predictions"][k]
    idx = max(0, min(step - 2, len(pred) - 1))
    return tuple(pred[idx])


def project_world_to_camera(x, y, width, height, yaw_deg):
    # Ego frame: x right, y forward.
    yaw = np.deg2rad(yaw_deg)
    side = x * np.cos(yaw) + y * np.sin(yaw)
    depth = y * np.cos(yaw) - x * np.sin(yaw)

    if depth <= 1.2:
        return None

    focal = width * 0.85
    u = width * 0.5 + (side / depth) * focal
    v = height * 0.84 - min(280.0, 460.0 / (depth + 0.6))
    return float(u), float(v), float(depth)


def build_synth_skeleton_points(u, v, box_w, box_h):
    head = (u, v - 0.38 * box_h)
    neck = (u, v - 0.28 * box_h)
    l_sh = (u - 0.22 * box_w, v - 0.22 * box_h)
    r_sh = (u + 0.22 * box_w, v - 0.22 * box_h)
    l_hand = (u - 0.34 * box_w, v - 0.03 * box_h)
    r_hand = (u + 0.34 * box_w, v - 0.03 * box_h)
    hip = (u, v - 0.02 * box_h)
    l_knee = (u - 0.14 * box_w, v + 0.30 * box_h)
    r_knee = (u + 0.14 * box_w, v + 0.30 * box_h)
    return [head, neck, l_sh, r_sh, l_hand, r_hand, hip, l_knee, r_knee]


def add_polyline_trace(fig, points, edges, color, point_size=4):
    xs = []
    ys = []
    for a, b in edges:
        if a >= len(points) or b >= len(points):
            continue
        xs.extend([points[a][0], points[b][0], None])
        ys.extend([points[a][1], points[b][1], None])

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": color, "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in points],
            y=[p[1] for p in points],
            mode="markers",
            marker={"size": point_size, "color": "#e2e8f0"},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def add_coco_pose_trace(fig, keypoints, color, conf_thresh=0.2):
    if keypoints is None:
        return
    if len(keypoints) < 17:
        return

    xs = []
    ys = []
    for a, b in COCO_SKELETON_EDGES:
        if keypoints[a][2] < conf_thresh or keypoints[b][2] < conf_thresh:
            continue
        xs.extend([keypoints[a][0], keypoints[b][0], None])
        ys.extend([keypoints[a][1], keypoints[b][1], None])

    if len(xs) > 0:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": color, "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    pts = [kp for kp in keypoints if kp[2] >= conf_thresh]
    if len(pts) > 0:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="markers",
                marker={"size": 4, "color": "#e2e8f0"},
                hoverinfo="skip",
                showlegend=False,
            )
        )


def create_camera_figure_projected(image_arr, agents, camera_label, yaw_deg, step):
    h, w = image_arr.shape[0], image_arr.shape[1]

    fig = go.Figure()
    fig.add_trace(go.Image(z=image_arr))

    for agent in agents:
        x, y = position_at_step(agent, step)
        projection = project_world_to_camera(x, y, w, h, yaw_deg)
        if projection is None:
            continue

        u, v, depth = projection
        if u < -40 or u > w + 40 or v < -40 or v > h + 40:
            continue

        is_ped = agent["type"] == "pedestrian"
        color = agent_color(agent)

        box_h = max(22.0, min(180.0, 260.0 / (depth + 0.5)))
        box_w = box_h * (0.42 if is_ped else 0.90)
        x1, y1 = u - box_w / 2, v - box_h
        x2, y2 = u + box_w / 2, v

        fig.add_shape(
            type="rect",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line={"color": color, "width": 2},
            fillcolor="rgba(0,0,0,0)",
        )

        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[max(4, y1 - 12)],
                mode="text",
                text=[f"ID {agent['id']}"],
                textfont={"size": 11, "color": color},
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if is_ped:
            kps = build_synth_skeleton_points(u, v, box_w, box_h)
            add_polyline_trace(fig, kps, SYNTH_SKELETON_EDGES, color, point_size=4)

    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title={"text": camera_label, "x": 0.02, "font": {"color": WHITE, "size": 15}},
        paper_bgcolor=BG_SECONDARY,
        plot_bgcolor=BG_SECONDARY,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        height=300,
    )
    return fig


def create_camera_figure_detections(image_arr, detections, camera_label, target_track_id=None, highlight_track_ids=None):
    h, w = image_arr.shape[0], image_arr.shape[1]

    fig = go.Figure()
    fig.add_trace(go.Image(z=image_arr))

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        kind = det.get("kind", "vehicle")
        track_id = det.get("track_id")

        if highlight_track_ids is not None and track_id is not None and track_id in highlight_track_ids:
            color = TARGET_PURPLE
        elif track_id is not None and track_id == target_track_id:
            color = TARGET_PURPLE
        elif kind == "pedestrian":
            color = VRU_GREEN
        else:
            color = VEHICLE_YELLOW

        fig.add_shape(
            type="rect",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line={"color": color, "width": 2},
            fillcolor="rgba(0,0,0,0)",
        )

        display_id = track_id if track_id is not None else f"D{det.get('det_id', i + 1)}"
        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[max(4.0, y1 - 12.0)],
                mode="text",
                text=[f"ID {display_id}"],
                textfont={"size": 11, "color": color},
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if kind == "pedestrian":
            add_coco_pose_trace(fig, det.get("keypoints"), color)

    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title={"text": camera_label, "x": 0.02, "font": {"color": WHITE, "size": 15}},
        paper_bgcolor=BG_SECONDARY,
        plot_bgcolor=BG_SECONDARY,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        height=300,
    )
    return fig


def smooth_path(points):
    return [p[0] for p in points], [p[1] for p in points]


def simulate_lidar_points(agents, step):
    rng = np.random.default_rng(1234 + step)

    bg = np.column_stack(
        [
            rng.uniform(-35, 35, 1500),
            rng.uniform(-8, 55, 1500),
        ]
    )

    clusters = []
    for a in agents:
        cx, cy = position_at_step(a, step)
        n = 110 if a["type"] == "vehicle" else 70
        spread = np.array([0.8, 0.8]) if a["type"] == "pedestrian" else np.array([1.3, 1.1])
        pts = rng.normal([cx, cy], spread, size=(n, 2))
        clusters.append(pts)

    if clusters:
        all_pts = np.vstack([bg] + clusters)
    else:
        all_pts = bg

    mask = (
        (all_pts[:, 0] > -38)
        & (all_pts[:, 0] < 38)
        & (all_pts[:, 1] > -12)
        & (all_pts[:, 1] < 58)
    )
    return all_pts[mask]


def simulate_radar_vectors(agents, step):
    vectors = []
    for a in agents:
        p_now = np.array(position_at_step(a, step), dtype=float)
        p_prev = np.array(previous_position_for_velocity(a, step), dtype=float)
        v = p_now - p_prev

        if np.linalg.norm(v) < 0.04:
            continue

        v = v / max(1e-6, np.linalg.norm(v)) * 1.6
        vectors.append((p_now[0], p_now[1], v[0], v[1], a["type"]))
    return vectors


def classify_direction(history, prediction):
    h_prev = np.array(history[-2], dtype=float)
    h_curr = np.array(history[-1], dtype=float)
    p_end = np.array(prediction[-1], dtype=float)

    heading = h_curr - h_prev
    motion = p_end - h_curr

    if np.linalg.norm(motion) < 0.7:
        return "Stop"

    if np.linalg.norm(heading) < 1e-6:
        heading = np.array([0.0, 1.0])

    heading = heading / np.linalg.norm(heading)
    motion = motion / np.linalg.norm(motion)

    cross = heading[0] * motion[1] - heading[1] * motion[0]
    dot = np.clip(np.dot(heading, motion), -1.0, 1.0)
    angle = np.degrees(np.arctan2(cross, dot))

    if abs(angle) <= 25:
        return "Straight"
    if angle > 25:
        return "Left"
    if angle < -25:
        return "Right"
    return "Stop"


def build_analytics_table(agents):
    rows = []
    direction_order = ["Straight", "Left", "Right", "Stop"]

    for a in agents:
        bins = {k: 0.0 for k in direction_order}

        for mode_idx, mode_path in enumerate(a["predictions"]):
            lbl = classify_direction(a["history"], mode_path)
            bins[lbl] += float(a["probabilities"][mode_idx])

        ranked = sorted(bins.items(), key=lambda kv: kv[1], reverse=True)
        top3 = ranked[:3]

        rows.append(
            {
                "Agent": f"ID {a['id']}",
                "Type": "Target VRU" if a.get("is_target", False) else a["type"].title(),
                "Top-1": f"{top3[0][0]} ({top3[0][1] * 100:.1f}%)",
                "Top-2": f"{top3[1][0]} ({top3[1][1] * 100:.1f}%)",
                "Top-3": f"{top3[2][0]} ({top3[2][1] * 100:.1f}%)",
            }
        )

    return pd.DataFrame(rows)


def generate_demo_agents(num_agents=8, history_steps=4, future_steps=12):
    rng = np.random.default_rng(42)
    agents = []

    ped_count = max(5, int(0.7 * num_agents))

    for i in range(num_agents):
        is_ped = i < ped_count
        a_type = "pedestrian" if is_ped else "vehicle"

        base_x = rng.uniform(-16, 16)
        base_y = rng.uniform(9, 45)

        if is_ped:
            vx = rng.uniform(-0.45, 0.45)
            vy = rng.uniform(0.15, 0.95)
        else:
            vx = rng.uniform(-0.20, 0.20)
            vy = rng.uniform(0.7, 1.6)

        history = []
        for t in range(history_steps):
            phase = t - (history_steps - 1)
            x = base_x + phase * vx + 0.06 * np.sin(0.8 * t + i)
            y = base_y + phase * vy + 0.05 * np.cos(0.5 * t + i)
            history.append((float(x), float(y)))

        probs = normalize_probs(rng.uniform(0.15, 1.0, size=3))

        predictions = []
        x0, y0 = history[-1]
        for mode in range(3):
            mode_path = []
            curve = (-0.12 + 0.12 * mode) * (1.4 if is_ped else 0.8)
            accel = 0.02 * (mode - 1)
            for s in range(1, future_steps + 1):
                x = x0 + vx * s + curve * (s ** 1.25)
                y = y0 + vy * s + accel * (s ** 1.12)
                mode_path.append((float(x), float(y)))
            predictions.append(mode_path)

        agents.append(
            {
                "id": i + 1,
                "type": a_type,
                "history": history,
                "predictions": predictions,
                "probabilities": probs,
                "is_target": (i == 0 and is_ped),
            }
        )

    return agents


def sanitize_agents(raw_agents):
    cleaned = []
    for i, a in enumerate(raw_agents):
        aid = int(a.get("id", i + 1))
        a_type = str(a.get("type", "pedestrian")).lower()
        if a_type not in ["pedestrian", "vehicle"]:
            a_type = "pedestrian"

        history = [tuple(map(float, p)) for p in a.get("history", [])]
        predictions = []
        for mode in a.get("predictions", []):
            predictions.append([tuple(map(float, p)) for p in mode])

        probs = normalize_probs(a.get("probabilities", [0.6, 0.25, 0.15]))

        if len(history) < 2 or len(predictions) < 3:
            continue

        cleaned.append(
            {
                "id": aid,
                "type": a_type,
                "history": history,
                "predictions": predictions[:3],
                "probabilities": probs[:3],
                "is_target": bool(a.get("is_target", False)),
            }
        )

    if not any(a.get("is_target", False) for a in cleaned):
        for a in cleaned:
            if a["type"] == "pedestrian":
                a["is_target"] = True
                break

    return cleaned


def build_bev_figure(
    agents,
    step,
    show_lidar,
    show_radar,
    show_multimodal,
    lidar_xy=None,
    radar_xy=None,
    radar_vel=None,
):
    fig = go.Figure()

    x_min, x_max = -36.0, 36.0
    y_min, y_max = -12.0, 58.0

    add_structured_road_scene(fig, x_min, x_max, y_min, y_max, add_crosswalk=True)

    fig.add_shape(
        type="rect",
        x0=-1.1,
        y0=-2.2,
        x1=1.1,
        y1=2.2,
        line={"color": EGO_CYAN, "width": 2.2},
        fillcolor="rgba(34,211,238,0.20)",
    )
    fig.add_annotation(
        x=0.0,
        y=4.2,
        ax=0.0,
        ay=1.2,
        arrowcolor=EGO_CYAN,
        arrowwidth=2.8,
        arrowhead=3,
        showarrow=True,
        text="",
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 10, "symbol": "circle", "color": VRU_GREEN},
            name="Pedestrian",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 10, "symbol": "square", "color": VEHICLE_YELLOW},
            name="Vehicle",
        )
    )

    if show_lidar:
        if lidar_xy is not None and len(lidar_xy) > 0:
            lidar = np.asarray(lidar_xy, dtype=float)
            mask = (
                (lidar[:, 0] > -38)
                & (lidar[:, 0] < 38)
                & (lidar[:, 1] > -12)
                & (lidar[:, 1] < 58)
            )
            lidar = lidar[mask]
        else:
            lidar = simulate_lidar_points(agents, step)

        if len(lidar) > 0:
            lidar = lidar[::6]
            fig.add_trace(
                go.Scatter(
                    x=lidar[:, 0],
                    y=lidar[:, 1],
                    mode="markers",
                    marker={"size": 3, "color": "rgba(34,211,238,0.22)"},
                    name="LiDAR",
                )
            )

    if show_radar:
        rx = []
        ry = []

        if (
            radar_xy is not None
            and radar_vel is not None
            and len(radar_xy) > 0
            and len(radar_xy) == len(radar_vel)
        ):
            radar_xy = np.asarray(radar_xy, dtype=float)
            radar_vel = np.asarray(radar_vel, dtype=float)
            stride = max(1, len(radar_xy) // 90)

            for i in range(0, len(radar_xy), stride):
                x0, y0 = radar_xy[i, 0], radar_xy[i, 1]
                vx, vy = radar_vel[i, 0], radar_vel[i, 1]
                rx.extend([x0, x0 + 0.55 * vx, None])
                ry.extend([y0, y0 + 0.55 * vy, None])
        else:
            radar_vectors = simulate_radar_vectors(agents, step)
            for x0, y0, vx, vy, _ in radar_vectors:
                rx.extend([x0, x0 + vx, None])
                ry.extend([y0, y0 + vy, None])

        if len(rx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=rx,
                    y=ry,
                    mode="lines",
                    line={"color": "rgba(250,204,21,0.75)", "width": 2},
                    name="Radar velocity",
                )
            )

    alt_legend_added = False

    for idx, a in enumerate(agents):
        base_color = agent_color(a)
        best_idx = best_mode_idx(a)
        best_prob = float(a["probabilities"][best_idx]) if len(a["probabilities"]) > 0 else 0.0
        marker_color = hex_to_rgba(base_color, 0.48 + 0.52 * best_prob)
        summary_text, _ = summarize_agent_probabilities(a)

        hx, hy = smooth_path(a["history"])
        fig.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="lines",
                line={"color": "rgba(226,232,240,0.55)", "width": 2.2, "dash": "dot", "shape": "spline", "smoothing": 1.0},
                name="Past trajectory" if idx == 0 else None,
                showlegend=(idx == 0),
                hovertemplate=f"ID {a['id']} past trajectory<extra></extra>",
            )
        )

        cx, cy = position_at_step(a, step)
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers+text",
                marker={
                    "size": 11,
                    "symbol": "circle" if a.get("type") == "pedestrian" else "square",
                    "color": marker_color,
                    "line": {"color": "#111827", "width": 1.2},
                },
                text=[f"ID {a['id']}"],
                textposition="top center",
                textfont={"size": 10, "color": WHITE},
                hovertemplate=(
                    f"ID {a['id']}<br>Type: {a['type'].title()}"
                    f"<br>{summary_text}<br>Best path confidence: {best_prob * 100:.1f}%<extra></extra>"
                ),
                showlegend=False,
            )
        )

        px, py = previous_position_for_velocity(a, step)
        dx, dy = cx - px, cy - py
        norm = np.hypot(dx, dy)
        if norm > 1e-3:
            sx, sy = (dx / norm) * 1.8, (dy / norm) * 1.8
            fig.add_annotation(x=cx + sx, y=cy + sy, ax=cx, ay=cy, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=base_color, text="")

        mode_order = [best_idx, 0, 1, 2]
        mode_order = list(dict.fromkeys(mode_order))

        for rank, m in enumerate(mode_order[:3]):
            if (not show_multimodal) and (rank > 0):
                continue

            mode_prob = float(a["probabilities"][m]) if m < len(a["probabilities"]) else 0.0
            mode_color = TRAJ_MODE_COLORS[m % len(TRAJ_MODE_COLORS)]

            mode_path = a["predictions"][m]
            end_idx = max(1, min(step, len(mode_path)))
            mode_slice = mode_path[:end_idx]
            mx, my = smooth_path([(cx, cy)] + mode_slice)

            is_best = m == best_idx

            if is_best:
                for lw, op in [(14, 0.08), (9, 0.16)]:
                    fig.add_trace(
                        go.Scatter(
                            x=mx,
                            y=my,
                            mode="lines",
                            line={"color": mode_color, "width": lw, "shape": "spline", "smoothing": 1.15},
                            opacity=op,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

            fig.add_trace(
                go.Scatter(
                    x=mx,
                    y=my,
                    mode="lines",
                    line={
                        "color": mode_color,
                        "width": 4.1 if is_best else 2.1,
                        "dash": "solid" if is_best else "dash",
                        "shape": "spline",
                        "smoothing": 1.15,
                    },
                    opacity=(0.72 + 0.26 * mode_prob) if is_best else (0.36 + 0.32 * mode_prob),
                    hovertemplate=(
                        f"ID {a['id']}<br>Mode {m + 1}"
                        f"<br>Probability: {mode_prob * 100:.1f}%<extra></extra>"
                    ),
                    name=(
                        "Best path" if (is_best and idx == 0) else
                        "Alternative paths" if ((not is_best) and (not alt_legend_added)) else None
                    ),
                    showlegend=(is_best and idx == 0) or ((not is_best) and (not alt_legend_added)),
                )
            )

            if (not is_best) and (not alt_legend_added):
                alt_legend_added = True

        if a.get("is_target", False):
            fig.add_trace(
                go.Scatter(
                    x=[cx + 0.9],
                    y=[cy + 1.1],
                    mode="text",
                    text=[summary_text],
                    textfont={"size": 9, "color": "rgba(226,232,240,0.90)"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title={"text": "Main BEV Simulation", "x": 0.02, "font": {"size": 20, "color": WHITE}},
        paper_bgcolor=BG_SECONDARY,
        plot_bgcolor=BG_SECONDARY,
        legend={"orientation": "h", "y": 1.03, "x": 0.0, "font": {"color": WHITE, "size": 11}},
        margin={"l": 16, "r": 16, "t": 52, "b": 10},
        height=700,
    )

    fig.update_xaxes(
        title_text="X Lateral (m)",
        range=[x_min, x_max],
        color=WHITE,
        dtick=5,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.16)",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Y Forward (m)",
        range=[y_min, y_max],
        color=WHITE,
        dtick=5,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.16)",
        scaleanchor="x",
        scaleratio=1,
        zeroline=False,
    )

    return fig


# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.title("Multi-Agent Trajectory Prediction Simulator (BEV)")
st.caption("Camera + LiDAR + Radar Fusion")

st.sidebar.header("Simulation Controls")

if "playing" not in st.session_state:
    st.session_state.playing = False
if "time_step" not in st.session_state:
    st.session_state.time_step = 0
if "time_step_slider" not in st.session_state:
    st.session_state.time_step_slider = 0

agent_source = st.sidebar.radio(
    "Agent Source",
    ["Two Image Upload", "Live CV + Fusion", "Synthetic Demo", "Upload JSON"],
    index=0,
)

uploaded_prev = None
uploaded_curr = None
uploaded_json = None

if agent_source == "Two Image Upload":
    uploaded_prev = st.sidebar.file_uploader("Image 1 (t-1)", type=["jpg", "jpeg", "png"], key="img_t_minus_1")
    uploaded_curr = st.sidebar.file_uploader("Image 2 (t0)", type=["jpg", "jpeg", "png"], key="img_t0")
elif agent_source == "Upload JSON":
    uploaded_json = st.sidebar.file_uploader("Upload agents JSON", type=["json"])

num_agents = st.sidebar.slider("Number of agents", min_value=5, max_value=10, value=8)

show_lidar = st.sidebar.checkbox("Show LiDAR", value=True)
show_radar = st.sidebar.checkbox("Show Radar", value=True)
show_multimodal = st.sidebar.checkbox("Show multi-modal paths", value=True)

if agent_source == "Live CV + Fusion":
    st.sidebar.caption(f"Trajectory model: {'Fusion Phase-2 checkpoint' if USING_FUSION_MODEL else 'Base checkpoint'}")

col_a, col_b = st.sidebar.columns(2)
if col_a.button("Play / Pause", use_container_width=True):
    st.session_state.playing = not st.session_state.playing
if col_b.button("Reset", use_container_width=True):
    st.session_state.playing = False
    st.session_state.time_step = 0
    st.session_state.time_step_slider = 0

step = st.sidebar.slider("Time step", min_value=0, max_value=12, value=int(st.session_state.time_step), key="time_step_slider")
st.session_state.time_step = step

# ----------------------------
# DATA INGESTION
# ----------------------------
agents = None
fusion_payload = None
camera_payload = None
target_track_id = None
live_status_msg = None

if agent_source == "Two Image Upload":
    det_threshold = st.sidebar.slider("Detection threshold", min_value=0.20, max_value=0.90, value=0.35, step=0.01)
    track_gate_px = st.sidebar.slider("Tracking gate (px)", min_value=30, max_value=220, value=130, step=5)
    min_motion_px = st.sidebar.slider("Minimum motion (px)", min_value=0, max_value=40, value=0, step=1)
    use_pose = st.sidebar.checkbox("Use Keypoint R-CNN", value=True)

    if uploaded_prev is None or uploaded_curr is None:
        st.info("Upload exactly 2 sequential images (t-1 and t0) to run prediction.")
        agents = []
    else:
        img_prev = uploaded_file_to_array(uploaded_prev)
        img_curr = uploaded_file_to_array(uploaded_curr)

        if img_prev is None or img_curr is None:
            st.warning("Could not read one of the uploaded images. Please try JPG/PNG files.")
            agents = []
        else:
            with st.spinner("Running 2-image perception and trajectory prediction..."):
                bundle = build_two_image_agents_bundle(
                    img_prev,
                    img_curr,
                    score_threshold=det_threshold,
                    tracking_gate_px=track_gate_px,
                    min_motion_px=min_motion_px,
                    use_pose=use_pose,
                )

            if "error" in bundle:
                st.warning(f"Two-image pipeline failed: {bundle['error']}")
                agents = []
                camera_payload = {
                    "mode": "two_upload",
                    "pair_prev": {"image": img_prev, "detections": []},
                    "pair_curr": {"image": img_curr, "detections": []},
                }
            else:
                agents = bundle["agents"]
                camera_payload = {"mode": "two_upload"}
                camera_payload.update(bundle.get("camera_snapshots", {}))
                target_track_id = bundle.get("target_track_id")
                live_status_msg = (
                    f"Two-image pipeline on {bundle.get('device', 'unknown')} | "
                    f"Predicted agents: {bundle.get('match_count', len(agents))}"
                )

elif agent_source == "Live CV + Fusion":
    front_paths = list_channel_image_paths("CAM_FRONT")

    if len(front_paths) < 4:
        st.warning("Live mode needs at least 4 frames in DataSet/samples/CAM_FRONT. Using synthetic data.")
        agents = generate_demo_agents(num_agents=num_agents)
    else:
        anchor_idx = st.sidebar.slider("Anchor frame index (CAM_FRONT)", min_value=3, max_value=len(front_paths) - 1, value=len(front_paths) - 1)
        det_threshold = st.sidebar.slider("Detection threshold", min_value=0.30, max_value=0.90, value=0.55, step=0.01)
        track_gate_px = st.sidebar.slider("Tracking gate (px)", min_value=40, max_value=180, value=90, step=5)
        use_pose = st.sidebar.checkbox("Use Keypoint R-CNN", value=True)

        with st.spinner("Running perception, tracking, fusion, and trajectory prediction..."):
            bundle = build_live_agents_bundle(anchor_idx, det_threshold, track_gate_px, use_pose)

        if "error" in bundle:
            st.warning(f"Live pipeline failed: {bundle['error']} Falling back to synthetic data.")
            agents = generate_demo_agents(num_agents=num_agents)
        else:
            agents = bundle["agents"]
            fusion_payload = bundle.get("fusion_data")
            camera_payload = bundle.get("camera_snapshots")
            target_track_id = bundle.get("target_track_id")
            live_status_msg = f"Live pipeline on {bundle.get('device', 'unknown')} | Tracked agents: {len(agents)}"

elif agent_source == "Upload JSON" and uploaded_json is not None:
    try:
        payload = json.load(uploaded_json)
        if isinstance(payload, dict) and "agents" in payload:
            raw_agents = payload["agents"]
        elif isinstance(payload, list):
            raw_agents = payload
        else:
            raw_agents = []

        agents = sanitize_agents(raw_agents)
        if len(agents) == 0:
            st.warning("Uploaded JSON did not contain valid agent entries. Falling back to synthetic demo data.")
            agents = generate_demo_agents(num_agents=num_agents)
    except Exception as e:
        st.warning(f"Could not parse uploaded JSON ({e}). Falling back to synthetic demo data.")
        agents = generate_demo_agents(num_agents=num_agents)

elif agent_source == "Synthetic Demo":
    agents = generate_demo_agents(num_agents=num_agents)

else:
    agents = []

if agents is None:
    agents = generate_demo_agents(num_agents=num_agents)

lidar_xy = fusion_payload.get("lidar_xy") if fusion_payload is not None else None
radar_xy = fusion_payload.get("radar_xy") if fusion_payload is not None else None
radar_vel = fusion_payload.get("radar_vel") if fusion_payload is not None else None

# ----------------------------
# TOP PANEL: MULTI-CAMERA
# ----------------------------
st.markdown("## 1. Multi-Camera View")

target_highlight_ids = {a["id"] for a in agents if a.get("is_target", False)} if len(agents) > 0 else set()

if agent_source == "Two Image Upload" and (camera_payload is None or camera_payload.get("mode") != "two_upload"):
    c1, c2, c3 = st.columns(3)
    empty = fallback_canvas()

    with c1:
        fig_prev = create_camera_figure_detections(empty, [], "Input Frame (t-1)", target_track_id=None, highlight_track_ids=None)
        st.plotly_chart(fig_prev, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig_curr = create_camera_figure_detections(empty, [], "Input Frame (t0)", target_track_id=None, highlight_track_ids=None)
        st.plotly_chart(fig_curr, use_container_width=True, config={"displayModeBar": False})

    with c3:
        fig_pred = create_camera_figure_detections(empty, [], "Prediction Output", target_track_id=None, highlight_track_ids=None)
        st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})

elif camera_payload is not None and camera_payload.get("mode") == "two_upload":
    c1, c2, c3 = st.columns(3)

    snap_prev = camera_payload.get("pair_prev", {"image": fallback_canvas(), "detections": []})
    snap_curr = camera_payload.get("pair_curr", {"image": fallback_canvas(), "detections": []})

    with c1:
        fig_prev = create_camera_figure_detections(
            snap_prev["image"],
            snap_prev["detections"],
            "Input Frame (t-1)",
            target_track_id=target_track_id,
            highlight_track_ids=target_highlight_ids,
        )
        st.plotly_chart(fig_prev, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig_curr = create_camera_figure_detections(
            snap_curr["image"],
            snap_curr["detections"],
            "Input Frame (t0)",
            target_track_id=target_track_id,
            highlight_track_ids=target_highlight_ids,
        )
        st.plotly_chart(fig_curr, use_container_width=True, config={"displayModeBar": False})

    with c3:
        fig_pred = create_prediction_overlay_figure(
            snap_curr["image"],
            snap_curr["detections"],
            agents,
            step=st.session_state.time_step,
            target_track_id=target_track_id,
            highlight_track_ids=target_highlight_ids,
        )
        st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})

else:
    cam_cols = st.columns(3)
    for i, (channel, label, yaw) in enumerate(CAMERA_VIEWS):
        with cam_cols[i]:
            if camera_payload is not None and channel in camera_payload:
                snap = camera_payload[channel]
                cam_fig = create_camera_figure_detections(
                    snap["image"],
                    snap["detections"],
                    label,
                    target_track_id=target_track_id,
                    highlight_track_ids=None,
                )
            else:
                img_arr, _ = load_camera_frame(channel, frame_idx=0)
                cam_fig = create_camera_figure_projected(img_arr, agents, label, yaw, st.session_state.time_step)

            st.plotly_chart(cam_fig, use_container_width=True, config={"displayModeBar": False})

# ----------------------------
# CENTER + SIDE PANELS
# ----------------------------
left_col, right_col = st.columns([3.6, 1.4], gap="large")

with left_col:
    if agent_source == "Two Image Upload":
        scene_ctx = None
        scene_dets = None
        if camera_payload is not None and camera_payload.get("mode") == "two_upload":
            scene_ctx = camera_payload.get("pair_curr", {}).get("image")
            scene_dets = camera_payload.get("pair_curr", {}).get("detections", [])

        bev_fig = build_reference_bev_figure(
            agents=agents,
            step=st.session_state.time_step,
            show_multimodal=show_multimodal,
            scene_image=scene_ctx,
            scene_detections=scene_dets,
        )
    else:
        bev_fig = build_bev_figure(
            agents=agents,
            step=st.session_state.time_step,
            show_lidar=show_lidar,
            show_radar=show_radar,
            show_multimodal=show_multimodal,
            lidar_xy=lidar_xy,
            radar_xy=radar_xy,
            radar_vel=radar_vel,
        )
    st.markdown("## 2. Main BEV Simulation")
    st.plotly_chart(bev_fig, use_container_width=True)

with right_col:
    st.markdown("## 3. Probability + Analytics")

    if live_status_msg:
        st.caption(live_status_msg)

    analytics_df = build_analytics_table(agents)
    st.dataframe(analytics_df, use_container_width=True, hide_index=True)

    if len(agents) == 0:
        st.info("No moving agents detected yet. Try clearer sequential frames with visible motion.")

    target_count = sum(1 for a in agents if a.get("is_target", False))
    ped_count = sum(1 for a in agents if a["type"] == "pedestrian")
    veh_count = sum(1 for a in agents if a["type"] == "vehicle")

    st.metric("Tracked Agents", len(agents))
    st.metric("VRUs", ped_count)
    st.metric("Vehicles", veh_count)
    st.metric("Target VRU", target_count)

    if fusion_payload is not None:
        st.metric("LiDAR points", int(len(lidar_xy)) if lidar_xy is not None else 0)
        st.metric("Radar points", int(len(radar_xy)) if radar_xy is not None else 0)

    st.markdown("### Legend")
    if agent_source == "Two Image Upload":
        st.markdown(
            "- Target VRU: purple\n"
            "- Other VRUs: green\n"
            "- Vehicles: yellow\n"
            "- Road model: asphalt, lane boundaries, dashed lane lines, crosswalk\n"
            "- Camera boxes/skeleton: detection + tracking\n"
            "- Trajectories: cyan/purple/orange (best = thick solid, alternatives = dashed)\n"
            "- Glow trail: best future path emphasis\n"
            "- BEV background: transformed real t0 scene with foreground cleanup"
        )
    else:
        st.markdown(
            "- Target VRU: purple\n"
            "- Other VRUs: green\n"
            "- Vehicles: yellow\n"
            "- Road model: asphalt, lane boundaries, dashed lane lines, crosswalk\n"
            "- Trajectories: cyan/purple/orange (best = thick solid, alternatives = dashed)\n"
            "- LiDAR: low-opacity cyan points\n"
            "- Radar: short yellow velocity vectors"
        )

with st.expander("Input schema expected by simulator"):
    st.code(
        """
agents = [
  {
    "id": 1,
    "type": "pedestrian",  # or "vehicle"
    "is_target": True,
    "history": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "predictions": [
      [[x, y], ...],  # mode 1
      [[x, y], ...],  # mode 2
      [[x, y], ...],  # mode 3
    ],
    "probabilities": [0.62, 0.24, 0.14]
  }
]
""",
        language="python",
    )

# ----------------------------
# PLAYBACK
# ----------------------------
if st.session_state.playing:
    time.sleep(0.15)
    nxt = (int(st.session_state.time_step) + 1) % 13
    st.session_state.time_step = nxt
    st.session_state.time_step_slider = nxt
    st.rerun()
