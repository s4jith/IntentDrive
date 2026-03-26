import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
import datetime

from dataset import TrajectoryDataset
from model import TrajectoryLSTM
from data_loader import (
    load_json, extract_pedestrian_instances,
    build_trajectories, create_windows
)


# ----------------------------
# CUSTOM COLLATE (IMPORTANT)
# ----------------------------
def collate_fn(batch):
    obs, neighbors, future = zip(*batch)

    obs = torch.stack(obs)
    future = torch.stack(future)

    return obs, list(neighbors), future


# ----------------------------
# LOAD DATA
# ----------------------------
def get_data():
    sample_annotations = load_json("sample_annotation")
    instances = load_json("instance")
    categories = load_json("category")

    ped_instances = extract_pedestrian_instances(
        sample_annotations, instances, categories
    )

    trajectories = build_trajectories(sample_annotations, ped_instances)
    samples = create_windows(trajectories)

    return samples


# ----------------------------
# METRICS
# ----------------------------
def compute_ade(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=2))


def compute_fde(pred, gt):
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=1))


# ----------------------------
# LOSS
# ----------------------------
def best_of_k_loss(pred, gt, probs):
    gt = gt.unsqueeze(1)

    error = torch.norm(pred - gt, dim=3).mean(dim=2)

    min_error, best_idx = torch.min(error, dim=1)

    traj_loss = torch.mean(min_error)

    prob_loss = torch.nn.functional.cross_entropy(probs, best_idx)

    # -----------------------------
    # DIVERSITY REGULARIZATION
    # -----------------------------
    diversity_loss = 0
    K = pred.size(1)
    if K > 1:
        for i in range(K):
            for j in range(i + 1, K):
                dist = torch.norm(pred[:, i] - pred[:, j], dim=2).mean(dim=1)
                diversity_loss += torch.exp(-dist).mean()
        diversity_loss /= (K * (K - 1) / 2)

    return traj_loss + 0.5 * prob_loss + 0.1 * diversity_loss


# ----------------------------
# TRAIN
# ----------------------------
def train():
    os.makedirs("log", exist_ok=True)
    log_filename = os.path.join("log", f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_filename, "a") as f:
            f.write(msg + "\n")

    log_print("Starting training...")
    samples = get_data()
    dataset = TrajectoryDataset(samples)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=64, collate_fn=collate_fn
    )

    model = TrajectoryLSTM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_ade = float("inf")

    for epoch in range(20):
        model.train()
        total_loss = 0

        for obs, neighbors, future in train_loader:
            pred, probs, _ = model(obs, neighbors)

            loss = best_of_k_loss(pred, future, probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---------------- VALIDATION ----------------
        model.eval()
        ade, fde = 0, 0

        with torch.no_grad():
            for obs, neighbors, future in val_loader:
                pred, probs, _ = model(obs, neighbors)

                gt = future.unsqueeze(1)

                error = torch.norm(pred - gt, dim=3).mean(dim=2)
                best_idx = torch.argmin(error, dim=1)

                best_pred = pred[torch.arange(pred.size(0)), best_idx]

                ade += compute_ade(best_pred, future).item()
                fde += compute_fde(best_pred, future).item()

        log_print(f"Epoch {epoch+1}")
        log_print(f"Train Loss: {total_loss:.4f}")
        log_print(f"ADE: {ade:.4f}, FDE: {fde:.4f}")
        log_print("-" * 40)

        # Save best model
        if ade < best_ade:
            log_print(f"New best model found! ADE improved from {best_ade:.4f} to {ade:.4f}")
            best_ade = ade
            torch.save(model.state_dict(), "best_social_model.pth")

    log_print("Training complete!")


if __name__ == "__main__":
    train()