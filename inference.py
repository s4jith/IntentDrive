import os

import torch
from model import TrajectoryTransformer
from model_fusion import TrajectoryTransformerFusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# LOAD MODEL
# ----------------------------
USING_FUSION_MODEL = False

if os.path.exists("best_social_model_fusion.pth"):
    model = TrajectoryTransformerFusion(fusion_dim=3).to(device)
    try:
        model.load_state_dict(torch.load("best_social_model_fusion.pth", map_location=device))
        USING_FUSION_MODEL = True
        print("Inference: Loaded Phase 2 fusion checkpoint (best_social_model_fusion.pth).")
    except Exception as e:
        print(f"Warning: could not load fusion checkpoint ({e}). Falling back to base model.")
        model = TrajectoryTransformer().to(device)
        try:
            model.load_state_dict(torch.load("best_social_model.pth", map_location=device))
            print("Inference: Loaded base checkpoint (best_social_model.pth).")
        except Exception as e2:
            print(f"Warning: could not load base checkpoint ({e2}).")
else:
    model = TrajectoryTransformer().to(device)
    try:
        model.load_state_dict(torch.load("best_social_model.pth", map_location=device))
        print("Inference: Loaded base checkpoint (best_social_model.pth).")
    except Exception as e:
        print(f"Warning: could not load model weights ({e}), starting fresh.")

model.eval()


# ----------------------------
# PREPROCESS INPUT
# ----------------------------
def prepare_input(points):
    import math
    x3, y3 = points[3]
    window = [[x - x3, y - y3] for x, y in points]

    vel = []
    for j in range(len(window)):
        if j == 0:
            vel.append([0, 0, 0, 0, 0])
        else:
            dx = window[j][0] - window[j-1][0]
            dy = window[j][1] - window[j-1][1]
            speed = math.hypot(dx, dy)
            if speed > 1e-5:
                sin_t = dy / speed
                cos_t = dx / speed
            else:
                sin_t = 0.0
                cos_t = 0.0
            vel.append([dx, dy, speed, sin_t, cos_t])

    obs = []
    for j in range(4):
        obs.append([
            window[j][0],
            window[j][1],
            vel[j][0],
            vel[j][1],
            vel[j][2],
            vel[j][3],
            vel[j][4]
        ])

    return obs, (x3, y3)


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(points, neighbor_points_list=None, fusion_feats=None):
    if neighbor_points_list is None:
        neighbor_points_list = []
        
    obs, origin = prepare_input(points)

    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # (1,4,7)

    # Prepare neighbors exactly as the main trajectory
    import math
    x1, y1 = points[-1]
    neighbors = []
    for np_points in neighbor_points_list:
        n_window = [[x - x1, y - y1] for x, y in np_points]
        vel_n = []
        for j in range(len(n_window)):
            if j == 0:
                vel_n.append([0, 0, 0, 0, 0])
            else:
                dx = n_window[j][0] - n_window[j-1][0]
                dy = n_window[j][1] - n_window[j-1][1]
                speed = math.hypot(dx, dy)
                if speed > 1e-5:
                    sin_t = dy / speed
                    cos_t = dx / speed
                else:
                    sin_t = 0.0
                    cos_t = 0.0
                vel_n.append([dx, dy, speed, sin_t, cos_t])
        
        n_obs = []
        for j in range(4):
            n_obs.append([
                n_window[j][0], n_window[j][1],
                vel_n[j][0], vel_n[j][1], vel_n[j][2], vel_n[j][3], vel_n[j][4]
            ])
        neighbors.append(n_obs)

    neighbors_batch = [neighbors]  # batch size = 1

    with torch.no_grad():
        if USING_FUSION_MODEL:
            if fusion_feats is None:
                fusion_tensor = torch.zeros((1, 4, 3), dtype=torch.float32, device=device)
            else:
                fusion_tensor = torch.tensor(fusion_feats, dtype=torch.float32).unsqueeze(0).to(device)
            pred, goals, probs, attn_weights = model(obs, neighbors_batch, fusion_tensor)
        else:
            pred, goals, probs, attn_weights = model(obs, neighbors_batch)

    pred = pred.squeeze(0).cpu()
    probs = probs.squeeze(0).cpu()

    if attn_weights and attn_weights[0] is not None:
        attn_weights = [w.cpu() for w in attn_weights]

    # convert back to real coordinates
    x0, y0 = origin
    pred_real = pred.clone()

    pred_real[:, :, 0] += x0
    pred_real[:, :, 1] += y0

    return pred_real, probs, attn_weights


# ----------------------------
# DEMO RUN
# ----------------------------
if __name__ == "__main__":

    points = [
        (0, 0),
        (10, 0),
        (20, 0),
        (30, 0)
    ]

    pred, probs, _ = predict(points)

    print("\nInput Points:")
    print(points)

    print("\nPredicted Trajectories (Real Coordinates):")
    for i in range(pred.shape[0]):
        print(f"\nTrajectory {i+1} (prob={probs[i].item():.2f}):")
        print(pred[i])