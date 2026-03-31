from __future__ import annotations

import base64
import io
import json
import math
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

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

REPO_ROOT = Path(__file__).resolve().parents[3]
from ..ml.inference import USING_FUSION_MODEL, predict as trajectory_predict
from ..ml.sensor_fusion import load_fusion_for_cam_frame, radar_stabilize_motion

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


@lru_cache(maxsize=1)
def _load_hd_map_indices(data_root: str, version: str) -> dict[str, Any]:
    base = Path(data_root) / version

    with open(base / "sample.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    with open(base / "sample_data.json", "r", encoding="utf-8") as f:
        sample_data = json.load(f)
    with open(base / "scene.json", "r", encoding="utf-8") as f:
        scenes = json.load(f)
    with open(base / "log.json", "r", encoding="utf-8") as f:
        logs = json.load(f)
    with open(base / "map.json", "r", encoding="utf-8") as f:
        maps = json.load(f)
    with open(base / "ego_pose.json", "r", encoding="utf-8") as f:
        ego_poses = json.load(f)

    sample_by_token = {r["token"]: r for r in samples}
    scene_by_token = {r["token"]: r for r in scenes}
    log_by_token = {r["token"]: r for r in logs}
    ego_pose_by_token = {r["token"]: r for r in ego_poses}

    sample_data_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_data_by_basename: dict[str, dict[str, Any]] = {}
    for rec in sample_data:
        sample_token = rec.get("sample_token")
        if sample_token:
            sample_data_by_sample[str(sample_token)].append(rec)

        filename = rec.get("filename")
        if filename:
            sample_data_by_basename[Path(str(filename)).name] = rec

    map_by_log_token: dict[str, dict[str, Any]] = {}
    for rec in maps:
        for log_token in rec.get("log_tokens", []):
            map_by_log_token[str(log_token)] = rec

    return {
        "sample_by_token": sample_by_token,
        "scene_by_token": scene_by_token,
        "log_by_token": log_by_token,
        "map_by_log_token": map_by_log_token,
        "sample_data_by_sample": dict(sample_data_by_sample),
        "sample_data_by_basename": sample_data_by_basename,
        "ego_pose_by_token": ego_pose_by_token,
    }


@lru_cache(maxsize=8)
def _get_map_size(map_path: str) -> tuple[int, int] | None:
    p = Path(map_path)
    if not p.exists():
        return None

    with Image.open(p) as img:
        w, h = img.size
    return int(w), int(h)


def _load_map_crop_gray(map_path: str, left: int, top: int, right: int, bottom: int) -> np.ndarray | None:
    p = Path(map_path)
    if not p.exists():
        return None

    if right <= left or bottom <= top:
        return None

    with Image.open(p) as img:
        crop = img.crop((int(left), int(top), int(right), int(bottom))).convert("L")
    return np.asarray(crop, dtype=np.uint8)


def _quat_wxyz_to_yaw(q: list[float] | tuple[float, float, float, float]) -> float:
    if len(q) != 4:
        return 0.0

    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return 0.0

    w, x, y, z = w / n, x / n, y / n, z / n
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


class TrajectoryPipeline:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self.data_root = self.repo_root / "DataSet"
        self._model_lock = threading.Lock()
        self._models: dict[str, Any] | None = None

    @property
    def using_fusion_model(self) -> bool:
        return bool(USING_FUSION_MODEL)

    @staticmethod
    def normalize_probs(probs: list[float] | np.ndarray) -> list[float]:
        arr = np.asarray(probs, dtype=float)
        arr = np.clip(arr, 1e-6, None)
        arr = arr / arr.sum()
        return arr.tolist()

    @staticmethod
    def coco_kind(label_name: str | None) -> str | None:
        if label_name in VRU_LABELS:
            return "pedestrian"
        if label_name in VEHICLE_LABELS:
            return "vehicle"
        return None

    @staticmethod
    def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
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

    @staticmethod
    def pixel_to_bev(center_x: float, bottom_y: float, width: int, height: int) -> tuple[float, float]:
        x_div = max(1.0, width / 80.0)
        y_div = max(1.0, height / 50.0)

        x_m = (center_x - 0.5 * width) / x_div
        y_m = (bottom_y - 0.58 * height) / y_div
        return float(x_m), float(y_m)

    def list_channel_image_paths(self, channel: str) -> list[Path]:
        base = self.data_root / "samples" / channel
        if not base.exists():
            return []
        return sorted(base.glob("*.jpg"))

    @staticmethod
    def load_image_array(image_path: str | Path) -> np.ndarray:
        return np.asarray(Image.open(image_path).convert("RGB"))

    @staticmethod
    def _clip_bev(x: float, y: float) -> tuple[float, float]:
        return float(np.clip(x, -40.0, 40.0)), float(np.clip(y, -14.0, 62.0))

    def _poly_px_to_bev_points(
        self,
        polygon_px: list[tuple[float, float]],
        width: int,
        height: int,
    ) -> list[dict[str, float]]:
        out = []
        for px, py in polygon_px:
            bx, by = self.pixel_to_bev(float(px), float(py), width, height)
            bx, by = self._clip_bev(bx, by)
            out.append({"x": bx, "y": by})
        return out

    def _project_detection_elements(
        self,
        detections: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        elements = []

        for det in detections:
            box = det.get("box")
            if box is None or len(box) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            cx = 0.5 * (x1 + x2)
            bx, by = self.pixel_to_bev(cx, y2, width, height)
            bx, by = self._clip_bev(bx, by)

            kind = str(det.get("kind", "vehicle"))
            box_w_px = max(1.0, x2 - x1)
            half_w = float(np.clip((box_w_px / max(1.0, width)) * 12.0, 0.25, 2.2))
            length = 0.9 if kind == "pedestrian" else 2.1

            polygon = [
                {"x": bx - half_w, "y": by - 0.25 * length},
                {"x": bx + half_w, "y": by - 0.25 * length},
                {"x": bx + half_w, "y": by + length},
                {"x": bx - half_w, "y": by + length},
            ]

            elements.append(
                {
                    "kind": kind,
                    "track_id": det.get("track_id"),
                    "score": float(det.get("score", 0.0)),
                    "polygon": polygon,
                }
            )

        return elements[:24]

    def extract_scene_geometry(
        self,
        image_arr: np.ndarray,
        detections: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        if image_arr is None:
            return None

        h, w = image_arr.shape[:2]
        if h < 20 or w < 20:
            return None

        if detections is None:
            detections = []

        roi_px = [
            (0.08 * w, h - 1),
            (0.42 * w, 0.56 * h),
            (0.58 * w, 0.56 * h),
            (0.92 * w, h - 1),
        ]

        scene = {
            "source": "camera-derived" if cv2 is not None else "heuristic-fallback",
            "quality": 0.0,
            "road_polygon": self._poly_px_to_bev_points(roi_px, w, h),
            "lane_lines": [],
            "elements": self._project_detection_elements(detections, w, h),
            "image_size": {"width": int(w), "height": int(h)},
        }

        if cv2 is None:
            scene["quality"] = 0.12
            return scene

        gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)

        roi_mask = np.zeros_like(edges)
        roi_poly = np.array([
            [
                (int(0.08 * w), h - 1),
                (int(0.42 * w), int(0.56 * h)),
                (int(0.58 * w), int(0.56 * h)),
                (int(0.92 * w), h - 1),
            ]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_poly, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(24, int(0.03 * w)),
            minLineLength=max(28, int(0.05 * w)),
            maxLineGap=max(22, int(0.03 * w)),
        )

        lane_candidates: list[tuple[float, list[dict[str, float]]]] = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = [int(v) for v in line[0]]
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                length = float(np.hypot(dx, dy))

                if length < max(24.0, 0.04 * w):
                    continue
                if abs(dy) < 8.0:
                    continue

                slope = dy / dx if abs(dx) > 1e-6 else np.sign(dy) * 1e6
                if abs(slope) < 0.35:
                    continue

                p1x, p1y = self.pixel_to_bev(float(x1), float(y1), w, h)
                p2x, p2y = self.pixel_to_bev(float(x2), float(y2), w, h)
                p1x, p1y = self._clip_bev(p1x, p1y)
                p2x, p2y = self._clip_bev(p2x, p2y)

                lane_candidates.append(
                    (
                        length,
                        [
                            {"x": p1x, "y": p1y},
                            {"x": p2x, "y": p2y},
                        ],
                    )
                )

        lane_candidates.sort(key=lambda item: item[0], reverse=True)
        scene["lane_lines"] = [item[1] for item in lane_candidates[:10]]

        edge_density = float(masked_edges.mean() / 255.0)
        lane_quality = min(1.0, len(scene["lane_lines"]) / 6.0)
        edge_quality = min(1.0, edge_density * 8.0)
        scene["quality"] = float(np.clip(0.55 * lane_quality + 0.45 * edge_quality, 0.0, 1.0))

        return scene

    def lookup_sample_token_for_filename(self, filename: str | None) -> str | None:
        if not filename:
            return None

        try:
            idx = _load_hd_map_indices(str(self.data_root), "v1.0-mini")
        except Exception:
            return None

        rec = idx["sample_data_by_basename"].get(Path(filename).name)
        if not rec:
            return None

        sample_token = rec.get("sample_token")
        if not sample_token:
            return None

        return str(sample_token)

    def _build_hd_map_layer(
        self,
        sample_token: str,
        radius_m: float = 45.0,
        out_size: int = 480,
    ) -> dict[str, Any] | None:
        try:
            idx = _load_hd_map_indices(str(self.data_root), "v1.0-mini")
        except Exception:
            return None

        sample_rec = idx["sample_by_token"].get(sample_token)
        if sample_rec is None:
            return None

        sample_data_list = idx["sample_data_by_sample"].get(sample_token, [])
        if len(sample_data_list) == 0:
            return None

        ref_rec = next(
            (r for r in sample_data_list if "LIDAR_TOP" in str(r.get("filename", ""))),
            sample_data_list[0],
        )

        ego_pose = idx["ego_pose_by_token"].get(str(ref_rec.get("ego_pose_token", "")))
        if ego_pose is None:
            return None

        scene_rec = idx["scene_by_token"].get(str(sample_rec.get("scene_token", "")))
        if scene_rec is None:
            return None

        log_token = str(scene_rec.get("log_token", ""))
        map_rec = idx["map_by_log_token"].get(log_token)
        if map_rec is None:
            return None

        map_rel = str(map_rec.get("filename", ""))
        map_path = self.data_root / map_rel
        map_size = _get_map_size(str(map_path))
        if map_size is None:
            return None
        map_w, map_h = map_size

        translation = ego_pose.get("translation", [0.0, 0.0, 0.0])
        ego_x = float(translation[0])
        ego_y = float(translation[1])
        yaw = _quat_wxyz_to_yaw(ego_pose.get("rotation", [1.0, 0.0, 0.0, 0.0]))

        # nuScenes semantic prior raster masks use 0.1m per pixel.
        ppm = 10.0
        x_right = np.linspace(-radius_m, radius_m, out_size, dtype=np.float32)
        y_forward = np.linspace(radius_m, -radius_m, out_size, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_right, y_forward)

        gx = ego_x + np.cos(yaw) * y_grid + np.sin(yaw) * x_grid
        gy = ego_y + np.sin(yaw) * y_grid - np.cos(yaw) * x_grid

        px_opts = [gx * ppm, (map_w - 1.0) - gx * ppm]
        py_opts = [gy * ppm, (map_h - 1.0) - gy * ppm]

        best_px = None
        best_py = None
        best_valid_ratio = -1.0
        for px in px_opts:
            for py in py_opts:
                valid = (px >= 0.0) & (px <= (map_w - 1.0)) & (py >= 0.0) & (py <= (map_h - 1.0))
                ratio = float(valid.mean())
                if ratio > best_valid_ratio:
                    best_valid_ratio = ratio
                    best_px = px
                    best_py = py

        if best_px is None or best_py is None or best_valid_ratio < 0.15:
            return None

        crop_left = int(max(0, math.floor(float(best_px.min())) - 2))
        crop_top = int(max(0, math.floor(float(best_py.min())) - 2))
        crop_right = int(min(map_w, math.ceil(float(best_px.max())) + 3))
        crop_bottom = int(min(map_h, math.ceil(float(best_py.max())) + 3))

        map_crop = _load_map_crop_gray(str(map_path), crop_left, crop_top, crop_right, crop_bottom)
        if map_crop is None or map_crop.size == 0:
            return None

        remap_x = best_px - float(crop_left)
        remap_y = best_py - float(crop_top)

        if cv2 is not None:
            patch = cv2.remap(
                map_crop,
                remap_x.astype(np.float32),
                remap_y.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            patch_u8 = patch.astype(np.uint8)
        else:
            crop_h, crop_w = map_crop.shape[:2]
            xi = np.clip(np.round(remap_x).astype(np.int32), 0, crop_w - 1)
            yi = np.clip(np.round(remap_y).astype(np.int32), 0, crop_h - 1)
            patch_u8 = map_crop[yi, xi]

        drivable = patch_u8 > 96
        strong = patch_u8 > 170
        if float(drivable.mean()) < 0.01:
            return None

        rgba = np.zeros((out_size, out_size, 4), dtype=np.uint8)
        rgba[drivable] = [72, 94, 114, 130]
        rgba[strong] = [170, 194, 216, 192]

        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "source": "nuscenes-semantic-prior",
            "map_token": map_rec.get("token"),
            "valid_ratio": round(best_valid_ratio, 3),
            "image_png_base64": png_b64,
            "opacity": 0.62,
            "bounds": {
                "min_x": -float(radius_m),
                "max_x": float(radius_m),
                "min_y": -float(radius_m),
                "max_y": float(radius_m),
            },
        }

    def _attach_hd_map_layer(self, scene_geometry: dict[str, Any] | None, sample_token: str | None):
        if not sample_token:
            return scene_geometry

        map_layer = self._build_hd_map_layer(sample_token)
        if map_layer is None:
            return scene_geometry

        if scene_geometry is None:
            bounds = map_layer["bounds"]
            scene_geometry = {
                "source": "hd-map",
                "quality": 0.55,
                "road_polygon": [
                    {"x": bounds["min_x"], "y": bounds["min_y"]},
                    {"x": bounds["max_x"], "y": bounds["min_y"]},
                    {"x": bounds["max_x"], "y": bounds["max_y"]},
                    {"x": bounds["min_x"], "y": bounds["max_y"]},
                ],
                "lane_lines": [],
                "elements": [],
            }
        else:
            scene_geometry = dict(scene_geometry)
            prev_source = str(scene_geometry.get("source", "")).strip()
            if "hd-map" not in prev_source:
                scene_geometry["source"] = f"{prev_source}+hd-map" if prev_source else "hd-map"
            scene_geometry["quality"] = float(np.clip(max(float(scene_geometry.get("quality", 0.0)), 0.55), 0.0, 1.0))

        scene_geometry["map_layer"] = map_layer
        return scene_geometry

    def load_cv_models(self) -> dict[str, Any]:
        if self._models is not None:
            return self._models

        with self._model_lock:
            if self._models is not None:
                return self._models

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            try:
                det_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                det_model = fasterrcnn_resnet50_fpn(weights=det_weights, progress=False)
                det_model.to(device).eval()

                pose_weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
                pose_model = keypointrcnn_resnet50_fpn(weights=pose_weights, progress=False)
                pose_model.to(device).eval()

                self._models = {
                    "device": device,
                    "device_name": str(device),
                    "det_model": det_model,
                    "det_weights": det_weights,
                    "pose_model": pose_model,
                    "pose_weights": pose_weights,
                }
            except Exception as exc:
                self._models = {
                    "error": str(exc),
                    "device": device,
                    "device_name": str(device),
                }

            return self._models

    def detect_objects_and_pose(
        self,
        image_arr: np.ndarray,
        models: dict[str, Any],
        score_threshold: float = 0.55,
        use_pose: bool = True,
    ) -> list[dict[str, Any]]:
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

        detections: list[dict[str, Any]] = []
        for i in range(len(scores)):
            score = float(scores[i])
            label_idx = int(labels[i])
            label_name = COCO_TO_LABEL.get(label_idx)

            if label_name is None or score < score_threshold:
                continue

            kind = self.coco_kind(label_name)
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
            p_keypoints = (
                pose_out["keypoints"].detach().cpu().numpy()
                if len(pose_out["keypoints"]) > 0
                else np.zeros((0, 17, 3))
            )

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

                    iou_val = self.iou_xyxy(det["box"], pose_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = det_idx

                if best_idx is not None and best_iou > 0.1:
                    detections[best_idx]["keypoints"] = p_keypoints[i].tolist()
                    assigned.add(best_idx)

        return detections

    @staticmethod
    def match_two_frame_tracks(
        det_prev: list[dict[str, Any]],
        det_curr: list[dict[str, Any]],
        tracking_gate_px: float = 90.0,
    ) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
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

    def build_two_image_agents_bundle(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        score_threshold: float,
        tracking_gate_px: float,
        min_motion_px: float,
        use_pose: bool,
        img_prev_name: str | None = None,
        img_curr_name: str | None = None,
    ) -> dict[str, Any]:
        models = self.load_cv_models()
        if "error" in models:
            return {
                "error": f"Could not load CV models ({models['error']}).",
                "device": models.get("device_name", "unknown"),
            }

        det_prev = self.detect_objects_and_pose(img_prev, models, score_threshold=score_threshold, use_pose=use_pose)
        det_curr = self.detect_objects_and_pose(img_curr, models, score_threshold=score_threshold, use_pose=use_pose)

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

        matches = self.match_two_frame_tracks(
            det_prev_vru,
            det_curr_vru,
            tracking_gate_px=tracking_gate_px,
        )

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
                p_prev = self.pixel_to_bev(d0["center_x"], d0["bottom_y"], w0, h0)
            else:
                p_prev = None

            p_curr = self.pixel_to_bev(d1["center_x"], d1["bottom_y"], w1, h1)

            if p_prev is None:
                vx, vy = 0.0, 0.0
                p_prev = p_curr
            else:
                vx = p_curr[0] - p_prev[0]
                vy = p_curr[1] - p_prev[1]

            if dist_px < float(min_motion_px):
                vx, vy = 0.0, 0.0
                p_prev = p_curr

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
                    "probabilities": self.normalize_probs(probs_np.tolist()),
                    "is_target": True,
                }
            )

        scene_geometry = self.extract_scene_geometry(img_curr, det_curr)
        sample_token = self.lookup_sample_token_for_filename(img_curr_name)
        scene_geometry = self._attach_hd_map_layer(scene_geometry, sample_token)

        return {
            "mode": "two_upload",
            "agents": agents,
            "target_track_id": None,
            "device": models.get("device_name", "unknown"),
            "match_count": len(agents),
            "scene_geometry": scene_geometry,
            "camera_snapshots": {
                "pair_prev": {"detections": det_prev},
                "pair_curr": {"detections": det_curr},
            },
        }

    def track_front_agents(
        self,
        front_paths: list[Path],
        models: dict[str, Any],
        score_threshold: float = 0.55,
        tracking_gate_px: float = 90.0,
        use_pose: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tracks: dict[int, dict[str, Any]] = {}
        next_track_id = 1
        front_final_detections: list[dict[str, Any]] = []

        for frame_idx, frame_path in enumerate(front_paths):
            frame_arr = self.load_image_array(frame_path)
            h, w = frame_arr.shape[:2]

            detections = self.detect_objects_and_pose(
                frame_arr,
                models,
                score_threshold=score_threshold,
                use_pose=use_pose,
            )
            detections.sort(key=lambda d: d["score"], reverse=True)

            matched_track_ids = set()
            frame_dets_with_ids = []

            for det in detections:
                wx, wy = self.pixel_to_bev(det["center_x"], det["bottom_y"], w, h)

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

    @staticmethod
    def raw_label_to_stabilizer_type(raw_label: str) -> str:
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

    @staticmethod
    def build_fusion_features(history_world: list[tuple[float, float]], fusion_data: dict[str, Any] | None):
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

    def stabilize_tracks_with_radar(self, tracks: list[dict[str, Any]], fusion_data: dict[str, Any] | None):
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
                    "type": self.raw_label_to_stabilizer_type(tr.get("raw_label", "car")),
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

    @staticmethod
    def choose_target_track_id(tracks: list[dict[str, Any]]) -> int | None:
        if not tracks:
            return None

        peds = [t for t in tracks if t["kind"] == "pedestrian"]
        if peds:
            best = min(peds, key=lambda t: math.hypot(t["history_world"][-1][0], t["history_world"][-1][1]))
            return best["id"]

        return tracks[0]["id"]

    def build_agents_from_tracks(self, tracks: list[dict[str, Any]], fusion_data: dict[str, Any] | None):
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

        tracks_work = self.stabilize_tracks_with_radar(tracks_work, fusion_data)

        target_id = self.choose_target_track_id(tracks_work)
        agents = []

        for tr in tracks_work:
            neighbors = [other["history_world"] for other in tracks_work if other["id"] != tr["id"]]

            if len(neighbors) > 12:
                x0, y0 = tr["history_world"][-1]
                neighbors = sorted(
                    neighbors,
                    key=lambda nh: math.hypot(nh[-1][0] - x0, nh[-1][1] - y0),
                )[:12]

            fusion_feats = self.build_fusion_features(tr["history_world"], fusion_data)

            pred, probs, _ = trajectory_predict(
                tr["history_world"],
                neighbor_points_list=neighbors,
                fusion_feats=fusion_feats,
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
                    "probabilities": self.normalize_probs(probs_np.tolist()),
                    "is_target": tr["id"] == target_id,
                }
            )

        return agents, target_id, tracks_work

    @staticmethod
    def assign_track_ids_to_front_detections(
        detections: list[dict[str, Any]],
        tracks: list[dict[str, Any]],
        gate_px: float = 90.0,
    ) -> list[dict[str, Any]]:
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

    def build_live_agents_bundle(
        self,
        anchor_idx: int,
        score_threshold: float,
        tracking_gate_px: float,
        use_pose: bool,
    ) -> dict[str, Any]:
        front_paths = self.list_channel_image_paths("CAM_FRONT")
        if len(front_paths) < 4:
            return {"error": "Need at least 4 CAM_FRONT frames in DataSet/samples/CAM_FRONT."}

        if anchor_idx < 3:
            anchor_idx = 3
        if anchor_idx >= len(front_paths):
            anchor_idx = len(front_paths) - 1

        models = self.load_cv_models()
        if "error" in models:
            return {
                "error": f"Could not load CV models ({models['error']}).",
                "device": models.get("device_name", "unknown"),
            }

        window_paths = front_paths[anchor_idx - 3 : anchor_idx + 1]

        tracks, front_dets = self.track_front_agents(
            window_paths,
            models,
            score_threshold=score_threshold,
            tracking_gate_px=tracking_gate_px,
            use_pose=use_pose,
        )

        if len(tracks) == 0:
            return {"error": "No valid tracked moving agents found in selected frame window."}

        front_curr = window_paths[-1]
        fusion_data = load_fusion_for_cam_frame(
            front_curr.name,
            data_root=str(self.data_root),
            version="v1.0-mini",
        )

        agents, target_id, tracks_stable = self.build_agents_from_tracks(tracks, fusion_data)
        if len(agents) == 0:
            return {"error": "Tracking succeeded but trajectory prediction produced no agents."}

        front_dets = self.assign_track_ids_to_front_detections(front_dets, tracks_stable, gate_px=tracking_gate_px)
        front_img = self.load_image_array(front_curr)
        scene_geometry = self.extract_scene_geometry(front_img, front_dets)
        live_sample_token = str(fusion_data.get("sample_token")) if fusion_data and fusion_data.get("sample_token") else None
        scene_geometry = self._attach_hd_map_layer(scene_geometry, live_sample_token)

        return {
            "mode": "live_fusion",
            "agents": agents,
            "target_track_id": target_id,
            "device": models.get("device_name", "unknown"),
            "front_anchor_path": str(front_curr),
            "track_count": len(agents),
            "scene_geometry": scene_geometry,
            "camera_snapshots": {
                "CAM_FRONT": {
                    "frame_path": str(front_curr),
                    "detections": front_dets,
                }
            },
            "fusion_data": fusion_data,
        }
