import torch
from torch.utils.data import DataLoader
from pathlib import Path
from backend.app.legacy.dataset import TrajectoryDataset
from backend.app.ml.model import TrajectoryTransformer
from backend.scripts.training.train import get_data, collate_fn, compute_ade, compute_fde
import numpy as np
import random

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CKPT = REPO_ROOT / "models" / "best_social_model.pth"

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
    print(f"Running Evaluation on {device}...")

    samples = get_data()
    
    # Use the same deterministic split as train.py to evaluate on validation set
    random.seed(42)
    random.shuffle(samples)
    train_size = int(0.8 * len(samples))
    val_samples = samples[train_size:]

    dataset = TrajectoryDataset(val_samples, augment=False)
    eval_loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)     

    # Load Model
    model = TrajectoryTransformer().to(device)
    try:
        model.load_state_dict(torch.load(BASE_CKPT, map_location=device, weights_only=True))
        print("Successfully loaded 'best_social_model.pth' from models folder")
    except Exception as e:
        print(f"Could not load model weights: {e}")
        return

    model.eval()

    total_ade = 0
    total_fde = 0
    miss_rate = 0
    
    cv_total_ade = 0
    cv_total_fde = 0
    cv_miss_rate = 0
    
    total_samples = 0

    # Miss rate threshold: if best path's endpoint is off by more than 2.0 meters
    MISS_THRESHOLD = 2.0

    print("\n--- Starting Deep Evaluation ---")
    with torch.no_grad():
        for obs, neighbors, future in eval_loader:
            obs, future = obs.to(device), future.to(device)

            # --- MODEL PREDICTION ---
            pred, goals, probs, _ = model(obs, neighbors)

            # Find the best prediction out of K=3 for each item in the batch    
            gt = future.unsqueeze(1)
            error = torch.norm(pred - gt, dim=3).mean(dim=2)
            best_idx = torch.argmin(error, dim=1)
            best_pred = pred[torch.arange(pred.size(0)), best_idx]

            # Metrics Model
            batch_ade = compute_ade(best_pred, future).item()
            batch_fde = compute_fde(best_pred, future).item()

            total_ade += batch_ade * obs.size(0)
            total_fde += batch_fde * obs.size(0)

            final_displacement = torch.norm(best_pred[:, -1] - future[:, -1], dim=1)
            misses = (final_displacement > MISS_THRESHOLD).sum().item()
            miss_rate += misses
            
            # --- CONSTANT VELOCITY BASELINE ---
            vx = obs[:, 3, 2].unsqueeze(1) # dx at last observed step
            vy = obs[:, 3, 3].unsqueeze(1) # dy at last observed step
            
            t = torch.arange(1, 13, device=device).unsqueeze(0).float() # Horizon is 12 steps
            
            x_last = obs[:, 3, 0].unsqueeze(1) # x at last step
            y_last = obs[:, 3, 1].unsqueeze(1) # y at last step
            
            cv_pred_x = x_last + vx * t
            cv_pred_y = y_last + vy * t
            cv_pred = torch.stack([cv_pred_x, cv_pred_y], dim=-1)
            
            # Metrics CV Baseline
            cv_batch_ade = compute_ade(cv_pred, future).item()
            cv_batch_fde = compute_fde(cv_pred, future).item()
            
            cv_total_ade += cv_batch_ade * obs.size(0)
            cv_total_fde += cv_batch_fde * obs.size(0)
            
            cv_final_displacement = torch.norm(cv_pred[:, -1] - future[:, -1], dim=1)
            cv_misses = (cv_final_displacement > MISS_THRESHOLD).sum().item()
            cv_miss_rate += cv_misses

            total_samples += obs.size(0)

    # Average metrics
    avg_ade = total_ade / total_samples
    avg_fde = total_fde / total_samples
    avg_miss_rate = (miss_rate / total_samples) * 100
    
    cv_avg_ade = cv_total_ade / total_samples
    cv_avg_fde = cv_total_fde / total_samples
    cv_avg_miss_rate = (cv_miss_rate / total_samples) * 100

    print("\n========================================================")
    print("           HACKATHON FINAL METRICS REPORT               ")
    print("========================================================")
    print(f"Total Trajectories Evaluated (Val Set): {total_samples}")
    print(f"Prediction Horizon:           6 Seconds (12 steps)")
    print(f"Social Context Radius:        50 Meters")
    print("--------------------------------------------------------")
    print("METRIC                  | BASELINE (CV) | OUR TRANSFORMER ")
    print("------------------------|---------------|-----------------")
    print(f"minADE@3 (meters)       | {cv_avg_ade:13.2f} | {avg_ade:15.2f}")
    print(f"minFDE@3 (meters)       | {cv_avg_fde:13.2f} | {avg_fde:15.2f}")
    print(f"Miss Rate (>2.0m)       | {cv_avg_miss_rate:12.1f}% | {avg_miss_rate:14.1f}%")
    print("========================================================\n")

if __name__ == '__main__':
    evaluate()
