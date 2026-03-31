import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import math
import numpy as np
from pathlib import Path
from backend.app.ml import model as TransformerBrain # Importing our Hackathon AI Model

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "models"
BASE_CKPT = MODEL_DIR / "best_social_model.pth"
CV_SYNC_CKPT = MODEL_DIR / "best_cv_synced_model.pth"
EXTRACTED_DATA_JSON = REPO_ROOT / "extracted_training_data.json"

print("[Step 1] Loading the Computer Vision Trajectory Data...")

class ExtractedPhysDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        self.inputs = []
        self.targets = []
        
        for item in data:
            coords = item['trajectory_pixels']
            if len(coords) == 4:
                processed_track = []
                
                # Math formatting bridging pixels to the network space
                # Convert raw pixels to 7-dimensional features: [x, y, dx, dy, speed, sin_t, cos_t]
                for i in range(4):
                    x = (coords[i][0] - 800) / 20.0
                    y = (coords[i][1] - 450) / 20.0
                    
                    if i == 0:
                        dx, dy = 0.0, 0.0
                    else:
                        prev_x = (coords[i-1][0] - 800) / 20.0
                        prev_y = (coords[i-1][1] - 450) / 20.0
                        dx = x - prev_x
                        dy = y - prev_y
                        
                    speed = math.hypot(dx, dy)
                    sin_t = dy / speed if speed > 1e-5 else 0.0
                    cos_t = dx / speed if speed > 1e-5 else 0.0
                    
                    processed_track.append([x, y, dx, dy, speed, sin_t, cos_t])
                
                self.inputs.append(processed_track)
                
                # Synthetic target creation (future 12 steps)
                t_x = processed_track[-1][0]
                t_y = processed_track[-1][1]
                v_x = processed_track[-1][2]
                v_y = processed_track[-1][3]
                
                target_fut = []
                for step in range(1, 13): 
                    target_fut.append([t_x + (v_x * step), t_y + (v_y * step)])
                
                self.targets.append(target_fut)
                
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return input track, empty neighbors [], and target future
        return self.inputs[idx], [], self.targets[idx]

def custom_collate(batch):
    obs_batch = []
    neighbors_batch = []
    future_batch = []
    for obs, neighbors, future in batch:
        obs_batch.append(obs)
        neighbors_batch.append(neighbors)
        future_batch.append(future)
    return torch.stack(obs_batch), neighbors_batch, torch.stack(future_batch)

cv_dataset = ExtractedPhysDataset(str(EXTRACTED_DATA_JSON))
cv_loader = DataLoader(cv_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

print(f"[Step 2] Prepared {len(cv_dataset)} real-world tracks for Brain Transfer.")

def fine_tune_ai_brain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[Step 3] Initializing Transformer Brain on {device.type.upper()}...")
    
    # Load our Hackathon specific Architecture
    ai_model = TransformerBrain.TrajectoryTransformer().to(device)
    
    try:
        ai_model.load_state_dict(torch.load(BASE_CKPT, map_location=device))
        print("  -> Transplanted initial knowledge from base training!")
    except Exception as e:
        print("  -> Starting fresh brain mapping (No previous weights found or mismatch).")

    optimizer = torch.optim.Adam(ai_model.parameters(), lr=0.001)
    
    print("\n[Step 4] Fine-Tuning the AI on Computer Vision Pixels -> 3D Maps")
    EPOCHS = 5 # Quick fine-tune pass
    
    ai_model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_in, batch_neighbors, batch_target in cv_loader:
            batch_in, batch_target = batch_in.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: returns traj, goals, probs, attn_weights
            traj, goals, probs, _ = ai_model(batch_in, batch_neighbors)
            
            # Simple Hackathon training logic: Just force the primary mode (k=0) to match the target
            # since CV target paths are linearly projected
            predictions = traj[:, 0, :, :] 
            
            # PyTorch Loss Function
            loss = torch.mean((predictions - batch_target) ** 2)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  | Epoch {epoch+1}/{EPOCHS} - Reality Mapping Loss: {total_loss/len(cv_loader):.4f}")
        
    print("\n[Step 5] Fine-Tuning Complete! Saving Real-World Synced Weights.")
    torch.save(ai_model.state_dict(), CV_SYNC_CKPT)
    print(" >>> Final Brain State Saved: 'best_cv_synced_model.pth' in models folder. Ready to impress the judges!")

if __name__ == '__main__':
    fine_tune_ai_brain()
