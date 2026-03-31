import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backend.app.legacy.data_loader import (
    load_json,
    extract_pedestrian_instances,
    build_trajectories_with_sensor,
    create_windows_with_sensor,
)
from backend.app.legacy.dataset_fusion import FusionTrajectoryDataset
from backend.app.ml.model_fusion import TrajectoryTransformerFusion

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FUSION_CKPT = REPO_ROOT / "models" / "best_social_model_fusion.pth"


def collate_fn_fusion(batch):
    obs, neighbors, fusion_obs, future = zip(*batch)
    obs = torch.stack(obs)
    fusion_obs = torch.stack(fusion_obs)
    future = torch.stack(future)
    return obs, list(neighbors), fusion_obs, future


def compute_ade(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=2))


def compute_fde(pred, gt):
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=1))


def load_fusion_samples():
    sample_annotations = load_json("sample_annotation")
    instances = load_json("instance")
    categories = load_json("category")

    ped_instances = extract_pedestrian_instances(sample_annotations, instances, categories)
    trajectories = build_trajectories_with_sensor(sample_annotations, ped_instances)
    return create_windows_with_sensor(trajectories)


def evaluate_fusion(ckpt=DEFAULT_FUSION_CKPT):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Phase 2 Fusion Evaluation on {device}...")

    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path

    samples = load_fusion_samples()
    random.seed(42)
    random.shuffle(samples)
    train_size = int(0.8 * len(samples))
    val_samples = samples[train_size:]

    dataset = FusionTrajectoryDataset(val_samples, augment=False)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn_fusion)

    model = TrajectoryTransformerFusion(fusion_dim=3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    total_ade = 0.0
    total_fde = 0.0
    miss_count = 0

    cv_total_ade = 0.0
    cv_total_fde = 0.0
    cv_miss_count = 0

    total_n = 0
    miss_threshold = 2.0

    with torch.no_grad():
        for obs, neighbors, fusion_obs, future in loader:
            obs = obs.to(device)
            fusion_obs = fusion_obs.to(device)
            future = future.to(device)

            pred, goals, probs, _ = model(obs, neighbors, fusion_obs)

            gt = future.unsqueeze(1)
            err = torch.norm(pred - gt, dim=3).mean(dim=2)
            best_idx = torch.argmin(err, dim=1)
            best_pred = pred[torch.arange(pred.size(0), device=device), best_idx]

            total_ade += compute_ade(best_pred, future).item() * obs.size(0)
            total_fde += compute_fde(best_pred, future).item() * obs.size(0)

            final_disp = torch.norm(best_pred[:, -1] - future[:, -1], dim=1)
            miss_count += (final_disp > miss_threshold).sum().item()

            # Constant velocity baseline for comparison.
            vx = obs[:, 3, 2].unsqueeze(1)
            vy = obs[:, 3, 3].unsqueeze(1)
            t = torch.arange(1, 13, device=device).unsqueeze(0).float()
            x_last = obs[:, 3, 0].unsqueeze(1)
            y_last = obs[:, 3, 1].unsqueeze(1)

            cv_x = x_last + vx * t
            cv_y = y_last + vy * t
            cv_pred = torch.stack([cv_x, cv_y], dim=-1)

            cv_total_ade += compute_ade(cv_pred, future).item() * obs.size(0)
            cv_total_fde += compute_fde(cv_pred, future).item() * obs.size(0)
            cv_final = torch.norm(cv_pred[:, -1] - future[:, -1], dim=1)
            cv_miss_count += (cv_final > miss_threshold).sum().item()

            total_n += obs.size(0)

    avg_ade = total_ade / total_n
    avg_fde = total_fde / total_n
    avg_miss = 100.0 * miss_count / total_n

    cv_avg_ade = cv_total_ade / total_n
    cv_avg_fde = cv_total_fde / total_n
    cv_avg_miss = 100.0 * cv_miss_count / total_n

    print("\n========================================================")
    print("         PHASE 2 FUSION METRICS REPORT                 ")
    print("========================================================")
    print(f"Total Trajectories Evaluated: {total_n}")
    print("--------------------------------------------------------")
    print("METRIC                  | BASELINE (CV) | FUSION MODEL ")
    print("------------------------|---------------|----------------")
    print(f"minADE@3 (meters)       | {cv_avg_ade:13.2f} | {avg_ade:14.2f}")
    print(f"minFDE@3 (meters)       | {cv_avg_fde:13.2f} | {avg_fde:14.2f}")
    print(f"Miss Rate (>2.0m)       | {cv_avg_miss:12.1f}% | {avg_miss:13.1f}%")
    print("========================================================\n")


if __name__ == '__main__':
    evaluate_fusion()
