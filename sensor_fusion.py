import json
import os
from collections import defaultdict
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _load_sample_data_index(data_root: str, version: str):
    sample_data_path = os.path.join(data_root, version, "sample_data.json")
    with open(sample_data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    by_basename = {}
    by_sample_token = defaultdict(list)

    for rec in records:
        basename = os.path.basename(rec.get("filename", ""))
        if basename:
            by_basename[basename] = rec
        token = rec.get("sample_token")
        if token:
            by_sample_token[token].append(rec)

    return by_basename, dict(by_sample_token)


@lru_cache(maxsize=1)
def _load_calibrated_sensor_index(data_root: str, version: str):
    calib_path = os.path.join(data_root, version, "calibrated_sensor.json")
    with open(calib_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["token"]: r for r in records}


def _channel_from_filename(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[1]
    return ""


def _quat_wxyz_to_rot(q):
    # nuScenes stores quaternion as [w, x, y, z]
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)

    w, x, y, z = w / n, x / n, y / n, z / n

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _transform_points_sensor_to_ego(points_xyz: np.ndarray, calib: dict):
    if points_xyz.size == 0:
        return points_xyz

    if calib is None:
        return points_xyz

    rot = _quat_wxyz_to_rot(calib.get("rotation", [1.0, 0.0, 0.0, 0.0]))
    t = np.asarray(calib.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)

    # Row-vector form: p_ego = p_sensor * R^T + t
    return points_xyz @ rot.T + t


def _transform_vel_sensor_to_ego(vel_xy: np.ndarray, calib: dict):
    if vel_xy.size == 0:
        return vel_xy

    if calib is None:
        return vel_xy

    rot = _quat_wxyz_to_rot(calib.get("rotation", [1.0, 0.0, 0.0, 0.0]))

    v_xyz = np.zeros((vel_xy.shape[0], 3), dtype=np.float32)
    v_xyz[:, 0] = vel_xy[:, 0]
    v_xyz[:, 1] = vel_xy[:, 1]

    v_ego = v_xyz @ rot.T
    return v_ego[:, :2].astype(np.float32)


def _load_lidar_pcd_bin(file_path: str) -> np.ndarray:
    arr = np.fromfile(file_path, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # nuScenes lidar .pcd.bin is typically [x, y, z, intensity, ring_index]
    if arr.size % 5 == 0:
        pts = arr.reshape(-1, 5)[:, :3]
    elif arr.size % 4 == 0:
        pts = arr.reshape(-1, 4)[:, :3]
    else:
        usable = (arr.size // 3) * 3
        pts = arr[:usable].reshape(-1, 3)

    return pts.astype(np.float32)


def _parse_pcd_binary(file_path: str):
    # Minimal PCD parser for nuScenes radar files (DATA binary).
    with open(file_path, "rb") as f:
        raw = f.read()

    header_end = raw.find(b"DATA binary")
    if header_end == -1:
        return {}

    line_end = raw.find(b"\n", header_end)
    if line_end == -1:
        return {}

    header_blob = raw[: line_end + 1].decode("utf-8", errors="ignore")
    data_blob = raw[line_end + 1 :]

    header = {}
    for line in header_blob.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        header[key.upper()] = vals

    fields = header.get("FIELDS", [])
    sizes = [int(x) for x in header.get("SIZE", [])]
    types = header.get("TYPE", [])
    counts = [int(x) for x in header.get("COUNT", [])]
    points = int(header.get("POINTS", ["0"])[0])

    if not fields or not sizes or not types or not counts or points <= 0:
        return {}

    np_map = {
        ("F", 4): np.float32,
        ("F", 8): np.float64,
        ("I", 1): np.int8,
        ("I", 2): np.int16,
        ("I", 4): np.int32,
        ("U", 1): np.uint8,
        ("U", 2): np.uint16,
        ("U", 4): np.uint32,
    }

    dtype_parts = []
    expanded_fields = []
    for field, size, typ, cnt in zip(fields, sizes, types, counts):
        base = np_map.get((typ, size), np.float32)
        if cnt == 1:
            dtype_parts.append((field, base))
            expanded_fields.append(field)
        else:
            for i in range(cnt):
                name = f"{field}_{i}"
                dtype_parts.append((name, base))
                expanded_fields.append(name)

    point_dtype = np.dtype(dtype_parts)
    byte_need = point_dtype.itemsize * points
    if len(data_blob) < byte_need:
        return {}

    rec = np.frombuffer(data_blob[:byte_need], dtype=point_dtype, count=points)
    out = {}
    for name in expanded_fields:
        out[name] = rec[name]
    return out


def _load_radar_pcd(file_path: str):
    fields = _parse_pcd_binary(file_path)
    if not fields:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    x = fields.get("x")
    y = fields.get("y")
    z = fields.get("z")

    # Prefer compensated velocity fields when available.
    vx = fields.get("vx_comp", fields.get("vx"))
    vy = fields.get("vy_comp", fields.get("vy"))

    if x is None or y is None or z is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    if vx is None:
        vx = np.zeros_like(x)
    if vy is None:
        vy = np.zeros_like(y)

    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    vel = np.stack([vx, vy], axis=1).astype(np.float32)
    return pts, vel


def _ego_xyz_to_bev(points_xyz: np.ndarray):
    # Ego frame: +x front, +y left, +z up
    # BEV UI: +x right, +y forward
    if points_xyz.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x_bev = -points_xyz[:, 1]
    y_bev = points_xyz[:, 0]
    return np.stack([x_bev, y_bev], axis=1).astype(np.float32)


def _ego_vel_to_bev(vxy_ego: np.ndarray):
    if vxy_ego.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    vx_bev = -vxy_ego[:, 1]
    vy_bev = vxy_ego[:, 0]
    return np.stack([vx_bev, vy_bev], axis=1).astype(np.float32)


def load_fusion_for_cam_frame(cam_filename: str, data_root: str = "DataSet", version: str = "v1.0-mini"):
    by_basename, by_sample = _load_sample_data_index(data_root, version)
    calib_by_token = _load_calibrated_sensor_index(data_root, version)

    basename = os.path.basename(cam_filename)
    cam_rec = by_basename.get(basename)
    if not cam_rec:
        return None

    sample_token = cam_rec.get("sample_token")
    if not sample_token:
        return None

    related = by_sample.get(sample_token, [])

    lidar_rec = None
    radar_recs = {}
    radar_channels = [
        "RADAR_FRONT",
        "RADAR_FRONT_LEFT",
        "RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT",
        "RADAR_BACK_RIGHT",
    ]

    for rec in related:
        rel = rec.get("filename", "")
        if not rel.startswith("samples/"):
            continue

        ch = _channel_from_filename(rel)
        if ch == "LIDAR_TOP":
            lidar_rec = rec
        elif ch in radar_channels:
            radar_recs[ch] = rec

    lidar_bev = np.zeros((0, 2), dtype=np.float32)
    lidar_path = None

    if lidar_rec is not None:
        lidar_path = os.path.join(data_root, lidar_rec.get("filename", ""))
        if os.path.exists(lidar_path):
            lidar_xyz = _load_lidar_pcd_bin(lidar_path)
            lidar_calib = calib_by_token.get(lidar_rec.get("calibrated_sensor_token"))
            lidar_xyz_ego = _transform_points_sensor_to_ego(lidar_xyz, lidar_calib)
            lidar_bev = _ego_xyz_to_bev(lidar_xyz_ego)

    radar_xy_list = []
    radar_vel_list = []
    radar_paths = {}
    radar_channel_counts = {}

    for ch in radar_channels:
        rec = radar_recs.get(ch)
        if rec is None:
            continue

        p = os.path.join(data_root, rec.get("filename", ""))
        radar_paths[ch] = p
        if not os.path.exists(p):
            radar_channel_counts[ch] = 0
            continue

        radar_xyz, radar_vel_xy = _load_radar_pcd(p)
        radar_calib = calib_by_token.get(rec.get("calibrated_sensor_token"))

        radar_xyz_ego = _transform_points_sensor_to_ego(radar_xyz, radar_calib)
        radar_vel_ego = _transform_vel_sensor_to_ego(radar_vel_xy, radar_calib)

        radar_bev = _ego_xyz_to_bev(radar_xyz_ego)
        radar_vel_bev = _ego_vel_to_bev(radar_vel_ego)

        if radar_bev.size > 0:
            m_ch = (
                (radar_bev[:, 1] > -20.0)
                & (radar_bev[:, 1] < 100.0)
                & (radar_bev[:, 0] > -70.0)
                & (radar_bev[:, 0] < 70.0)
            )
            radar_bev = radar_bev[m_ch]
            radar_vel_bev = radar_vel_bev[m_ch]

        radar_channel_counts[ch] = int(radar_bev.shape[0])

        if radar_bev.size > 0:
            radar_xy_list.append(radar_bev)
            radar_vel_list.append(radar_vel_bev)

    if radar_xy_list:
        radar_bev_all = np.concatenate(radar_xy_list, axis=0).astype(np.float32)
        radar_vel_all = np.concatenate(radar_vel_list, axis=0).astype(np.float32)
    else:
        radar_bev_all = np.zeros((0, 2), dtype=np.float32)
        radar_vel_all = np.zeros((0, 2), dtype=np.float32)

    # Keep interaction region for live BEV visualization.
    if lidar_bev.size > 0:
        m = (
            (lidar_bev[:, 1] > -15.0)
            & (lidar_bev[:, 1] < 85.0)
            & (lidar_bev[:, 0] > -60.0)
            & (lidar_bev[:, 0] < 60.0)
        )
        lidar_bev = lidar_bev[m]

    if radar_bev_all.size > 0:
        m = (
            (radar_bev_all[:, 1] > -20.0)
            & (radar_bev_all[:, 1] < 100.0)
            & (radar_bev_all[:, 0] > -70.0)
            & (radar_bev_all[:, 0] < 70.0)
        )
        radar_bev_all = radar_bev_all[m]
        if radar_vel_all.shape[0] == m.shape[0]:
            radar_vel_all = radar_vel_all[m]

    return {
        "sample_token": sample_token,
        "lidar_xy": lidar_bev,
        "radar_xy": radar_bev_all,
        "radar_vel": radar_vel_all,
        "lidar_path": lidar_path,
        "radar_path": radar_paths.get("RADAR_FRONT"),
        "radar_paths": radar_paths,
        "radar_channel_counts": radar_channel_counts,
    }


def radar_stabilize_motion(tracked_agents, fusion_data, dt_seconds: float = 0.5):
    if not fusion_data:
        return tracked_agents

    radar_xy = fusion_data.get("radar_xy")
    radar_vel = fusion_data.get("radar_vel")

    if radar_xy is None or radar_vel is None or len(radar_xy) == 0:
        return tracked_agents

    stabilized = []

    for agent in tracked_agents:
        if agent.get("type") not in ["Person", "Bicycle", "Car", "Truck", "Bus", "Motorcycle"]:
            stabilized.append(agent)
            continue

        x_curr, y_curr = agent["history"][-1]
        d = np.hypot(radar_xy[:, 0] - x_curr, radar_xy[:, 1] - y_curr)
        near_idx = np.where(d < 3.0)[0]

        if near_idx.size > 0:
            rv = radar_vel[near_idx].mean(axis=0)
            radar_dx = float(rv[0] * dt_seconds)
            radar_dy = float(rv[1] * dt_seconds)

            cam_dx = float(agent.get("dx", 0.0))
            cam_dy = float(agent.get("dy", 0.0))

            fused_dx = 0.7 * cam_dx + 0.3 * radar_dx
            fused_dy = 0.7 * cam_dy + 0.3 * radar_dy

            x4, y4 = x_curr, y_curr
            h3 = (x4 - 3.0 * fused_dx, y4 - 3.0 * fused_dy)
            h2 = (x4 - 2.0 * fused_dx, y4 - 2.0 * fused_dy)
            h1 = (x4 - 1.0 * fused_dx, y4 - 1.0 * fused_dy)

            agent = dict(agent)
            agent["dx"] = fused_dx
            agent["dy"] = fused_dy
            agent["history"] = [h3, h2, h1, (x4, y4)]

        stabilized.append(agent)

    return stabilized
