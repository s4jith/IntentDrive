import argparse
import datetime
import os
import random
from pathlib import Path

import torch
import torch.optim as optim
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


def best_of_k_loss(pred, goals, gt, probs):
    gt_traj = gt.unsqueeze(1)

    error = torch.norm(pred - gt_traj, dim=3).mean(dim=2)
    min_error, best_idx = torch.min(error, dim=1)
    traj_loss = torch.mean(min_error)

    best_goals = goals[torch.arange(goals.size(0), device=goals.device), best_idx]
    goal_loss = torch.norm(best_goals - gt[:, -1, :], dim=1).mean()

    prob_loss = torch.nn.functional.nll_loss(torch.log(probs + 1e-8), best_idx)

    diversity_loss = 0.0
    K = pred.size(1)
    if K > 1:
        reg = 0.0
        pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                dist = torch.norm(pred[:, i] - pred[:, j], dim=2).mean(dim=1)
                reg = reg + torch.exp(-dist).mean()
                pairs += 1
        diversity_loss = reg / max(1, pairs)

    return traj_loss + 0.5 * goal_loss + 0.5 * prob_loss + 0.1 * diversity_loss


def get_fusion_samples():
    sample_annotations = load_json("sample_annotation")
    instances = load_json("instance")
    categories = load_json("category")

    ped_instances = extract_pedestrian_instances(sample_annotations, instances, categories)
    trajectories = build_trajectories_with_sensor(sample_annotations, ped_instances)
    samples = create_windows_with_sensor(trajectories)

    return samples


def train_phase2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_checkpoint = Path(args.base_checkpoint)
    output_checkpoint = Path(args.output_checkpoint)

    if not base_checkpoint.is_absolute():
        base_checkpoint = REPO_ROOT / base_checkpoint
    if not output_checkpoint.is_absolute():
        output_checkpoint = REPO_ROOT / output_checkpoint

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    os.makedirs("log", exist_ok=True)
    log_filename = os.path.join(
        "log",
        f"phase2_fusion_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    def log_print(msg):
        print(msg)
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_print(f"Starting Phase 2 fusion transfer-learning on {device}...")

    samples = get_fusion_samples()
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    random.seed(42)
    random.shuffle(samples)

    train_size = int(0.8 * len(samples))
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]

    train_dataset = FusionTrajectoryDataset(train_samples, augment=True)
    val_dataset = FusionTrajectoryDataset(val_samples, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_fusion,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn_fusion,
    )

    model = TrajectoryTransformerFusion(fusion_dim=3).to(device)

    if base_checkpoint.exists():
        missing, unexpected = model.load_from_base_checkpoint(str(base_checkpoint), map_location=device)
        log_print(f"Loaded base checkpoint: {base_checkpoint}")
        log_print(f"Missing keys count: {len(missing)}")
        log_print(f"Unexpected keys count: {len(unexpected)}")
    else:
        log_print(f"Base checkpoint not found: {base_checkpoint}")

    base_params = []
    fusion_params = []
    for n, p in model.named_parameters():
        if n.startswith("fusion_embed") or n.startswith("fusion_ln"):
            fusion_params.append(p)
        else:
            base_params.append(p)

    optimizer = optim.Adam(
        [
            {"params": base_params, "lr": args.base_lr},
            {"params": fusion_params, "lr": args.fusion_lr},
        ]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4,
    )

    best_val_ade = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for obs, neighbors, fusion_obs, future in train_loader:
            obs = obs.to(device)
            fusion_obs = fusion_obs.to(device)
            future = future.to(device)

            pred, goals, probs, _ = model(obs, neighbors, fusion_obs)
            loss = best_of_k_loss(pred, goals, future, probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_ade = 0.0
        val_fde = 0.0
        batches = 0

        with torch.no_grad():
            for obs, neighbors, fusion_obs, future in val_loader:
                obs = obs.to(device)
                fusion_obs = fusion_obs.to(device)
                future = future.to(device)

                pred, goals, probs, _ = model(obs, neighbors, fusion_obs)

                gt = future.unsqueeze(1)
                err = torch.norm(pred - gt, dim=3).mean(dim=2)
                best_idx = torch.argmin(err, dim=1)
                best_pred = pred[torch.arange(pred.size(0), device=device), best_idx]

                val_ade += compute_ade(best_pred, future).item()
                val_fde += compute_fde(best_pred, future).item()
                batches += 1

        val_ade = val_ade / max(1, batches)
        val_fde = val_fde / max(1, batches)

        scheduler.step(val_ade)
        curr_lr_base = optimizer.param_groups[0]['lr']
        curr_lr_fusion = optimizer.param_groups[1]['lr']

        log_print(f"Epoch {epoch + 1}/{args.epochs}")
        log_print(f"Train Loss: {train_loss:.4f}")
        log_print(f"Val ADE: {val_ade:.4f} | Val FDE: {val_fde:.4f}")
        log_print(f"LR base={curr_lr_base:.6f} | fusion={curr_lr_fusion:.6f}")
        log_print("-" * 44)

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            patience_counter = 0
            torch.save(model.state_dict(), output_checkpoint)
            log_print(f"New best fusion model saved: {output_checkpoint}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            log_print(f"Early stopping at epoch {epoch + 1} (patience reached).")
            break

    log_print("Phase 2 fusion transfer-learning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: LiDAR/Radar Fusion Transfer-Learning")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--fusion-lr", type=float, default=8e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="Use first N samples for quick debug run. 0 = full data.")
    parser.add_argument("--base-checkpoint", type=str, default="models/best_social_model.pth")
    parser.add_argument("--output-checkpoint", type=str, default="models/best_social_model_fusion.pth")
    args = parser.parse_args()

    train_phase2(args)
