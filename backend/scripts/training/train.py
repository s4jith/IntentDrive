import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
import datetime
from pathlib import Path

from backend.app.legacy.dataset import TrajectoryDataset
from backend.app.ml.model import TrajectoryTransformer
from backend.app.legacy.data_loader import (
    load_json, extract_pedestrian_instances,
    build_trajectories, create_windows
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "models"


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
def best_of_k_loss(pred, goals, gt, probs):
    gt_traj = gt.unsqueeze(1)  # (B, 1, 6, 2)
    gt_goal = gt[:, -1, :].unsqueeze(1) # (B, 1, 2)

    # Error calculation over the entire path 
    error = torch.norm(pred - gt_traj, dim=3).mean(dim=2) # (B, K)
    min_error, best_idx = torch.min(error, dim=1)

    traj_loss = torch.mean(min_error)

    # Goal Loss: force the network to explicitly predict accurate endpoints!
    best_goals = goals[torch.arange(goals.size(0)), best_idx] # (B, 2)
    goal_loss = torch.norm(best_goals - gt[:, -1, :], dim=1).mean()

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

    return traj_loss + 0.5 * goal_loss + 0.5 * prob_loss + 0.1 * diversity_loss


# ----------------------------
# TRAIN
# ----------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs("log", exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log_filename = os.path.join("log", f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    best_model_path = MODEL_DIR / "best_social_model.pth"
    
    def log_print(msg):
        print(msg)
        with open(log_filename, "a") as f:
            f.write(msg + "\n")

    import random
    log_print(f"Starting training on {device}...")
    samples = get_data()
    
    # Deterministic split as promised
    random.seed(42)
    random.shuffle(samples)

    train_size = int(0.8 * len(samples))
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]

    train_dataset = TrajectoryDataset(train_samples, augment=True)
    val_dataset = TrajectoryDataset(val_samples, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=64, collate_fn=collate_fn
    )

    model = TrajectoryTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_ade = float("inf")
    patience_counter = 0
    max_patience = 15

    for epoch in range(100): # Increased to 100 max epochs with early stopping
        model.train()
        total_loss = 0

        for obs, neighbors, future in train_loader:
            obs, future = obs.to(device), future.to(device)

            pred, goals, probs, _ = model(obs, neighbors)

            loss = best_of_k_loss(pred, goals, future, probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---------------- VALIDATION ----------------
        model.eval()
        ade, fde = 0, 0

        with torch.no_grad():
            for obs, neighbors, future in val_loader:
                obs, future = obs.to(device), future.to(device)
                
                pred, goals, probs, _ = model(obs, neighbors)
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
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Update Learning Rate
        scheduler.step(ade)
        current_lr = optimizer.param_groups[0]['lr']
        log_print(f"Current Learning Rate: {current_lr}")
        
        if patience_counter >= max_patience:
            log_print(f"Early stopping triggered! No improvement for {max_patience} epochs.")
            break

    log_print("Training complete!")


if __name__ == "__main__":
    train()