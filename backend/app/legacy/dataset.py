import torch
from torch.utils.data import Dataset
import random
import math

def augment_data(obs, neighbors, future):
    # obs: (4, 7) tensor
    # neighbors: list of (4, 7) tensors
    # future: (12, 2) tensor
    
    # Random Scene Rotation (0-360)
    theta = random.uniform(0, 2 * math.pi)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Random X-axis reflection
    flip_x = random.choice([-1.0, 1.0])
    
    # Gaussian Coordinate Noise
    noise_std = 0.05

    def apply_transform(feat, is_obs=True):
        new_feat = feat.clone()
        for i in range(new_feat.size(0)):
            x, y = new_feat[i, 0].item(), new_feat[i, 1].item()
            
            # Apply Noise
            x += random.gauss(0, noise_std)
            y += random.gauss(0, noise_std)
            
            # Apply Flip
            x *= flip_x
            
            # Apply Rotation
            nx = x * cos_t - y * sin_t
            ny = x * sin_t + y * cos_t
            
            new_feat[i, 0] = nx
            new_feat[i, 1] = ny
            
            if is_obs:
                # Transform dx, dy
                dx, dy = new_feat[i, 2].item(), new_feat[i, 3].item()
                dx *= flip_x
                ndx = dx * cos_t - dy * sin_t
                ndy = dx * sin_t + dy * cos_t
                new_feat[i, 2] = ndx
                new_feat[i, 3] = ndy
                
                # Recompute sin_t, cos_t based on new dx, dy to be safe
                speed = math.hypot(ndx, ndy)
                if speed > 1e-5:
                    new_feat[i, 5] = ndy / speed
                    new_feat[i, 6] = ndx / speed
                else:
                    new_feat[i, 5] = 0.0
                    new_feat[i, 6] = 0.0

        return new_feat

    new_obs = apply_transform(obs, is_obs=True)
    new_future = apply_transform(future, is_obs=False)
    
    new_neighbors = []
    for n in neighbors: # n is (4, 7) tensor
        if not isinstance(n, torch.Tensor):
            n = torch.tensor(n, dtype=torch.float32)
        new_neighbors.append(apply_transform(n, is_obs=True))
        
    return new_obs, new_neighbors, new_future

class TrajectoryDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.obs = []
        self.neighbors = []
        self.future = []
        self.augment = augment

        for obs, neighbors, future in samples:
            self.obs.append(obs)
            self.neighbors.append(neighbors)
            self.future.append(future)

        # Convert to tensors
        self.obs = torch.tensor(self.obs, dtype=torch.float32)
        self.future = torch.tensor(self.future, dtype=torch.float32)
        # Neighbors remain lists of matrices, will convert in getitem or augment

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = self.obs[idx].clone()
        future = self.future[idx].clone()
        neighbors = [torch.tensor(n, dtype=torch.float32) for n in self.neighbors[idx]]
        
        if self.augment:
            obs, neighbors, future = augment_data(obs, neighbors, future)
            
        return obs, neighbors, future
