import time

import torch
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)

from backend.app.ml.sensor_fusion import load_fusion_for_cam_frame
from backend.app.ml.inference import predict, USING_FUSION_MODEL


def main():
    img_path = r"DataSet/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
    img = Image.open(img_path).convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    print("using_fusion_model", USING_FUSION_MODEL)

    t0 = time.perf_counter()
    w_det = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    m_det = fasterrcnn_resnet50_fpn(weights=w_det, progress=False).to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_det = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    w_pose = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    m_pose = keypointrcnn_resnet50_fpn(weights=w_pose, progress=False).to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_pose = (time.perf_counter() - t0) * 1000

    print("load_ms_fasterrcnn", round(load_det, 2))
    print("load_ms_keypointrcnn", round(load_pose, 2))

    in_det = w_det.transforms()(img).unsqueeze(0).to(device)
    in_pose = w_pose.transforms()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = m_det(in_det)
        _ = m_pose(in_pose)
    if device.type == "cuda":
        torch.cuda.synchronize()

    n = 5

    st = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            _ = m_det(in_det)
    if device.type == "cuda":
        torch.cuda.synchronize()
    det_ms = (time.perf_counter() - st) * 1000 / n

    st = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            _ = m_pose(in_pose)
    if device.type == "cuda":
        torch.cuda.synchronize()
    pose_ms = (time.perf_counter() - st) * 1000 / n

    print("avg_ms_det_per_frame", round(det_ms, 2))
    print("avg_ms_pose_per_frame", round(pose_ms, 2))

    m = 30
    st = time.perf_counter()
    for _ in range(m):
        _ = load_fusion_for_cam_frame(
            "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
            data_root="DataSet",
        )
    fusion_ms = (time.perf_counter() - st) * 1000 / m
    print("avg_ms_fusion_lookup", round(fusion_ms, 2))

    pts = [(0, 10), (2, 10), (4, 10), (6, 10)]
    neigh = [
        [(8, 12), (8.5, 12), (9, 12), (9.5, 12)],
        [(15, 7), (15.5, 7.2), (16, 7.5), (16.4, 7.7)],
    ]
    fusion_feats = [[0.2, 0.1, 0.25], [0.25, 0.1, 0.3], [0.3, 0.12, 0.35], [0.35, 0.15, 0.4]]

    for _ in range(10):
        _ = predict(pts, neigh, fusion_feats=fusion_feats)
    if device.type == "cuda":
        torch.cuda.synchronize()

    k = 300
    st = time.perf_counter()
    for _ in range(k):
        _ = predict(pts, neigh, fusion_feats=fusion_feats)
    if device.type == "cuda":
        torch.cuda.synchronize()
    pred_ms = (time.perf_counter() - st) * 1000 / k
    print("avg_ms_transformer_predict", round(pred_ms, 4))

    approx = 2 * det_ms + pose_ms + fusion_ms + 6 * pred_ms
    fps = 1000.0 / approx if approx > 0 else 0.0
    print("approx_live_2frame_ms", round(approx, 2))
    print("approx_live_equiv_fps", round(fps, 2))


if __name__ == "__main__":
    main()
