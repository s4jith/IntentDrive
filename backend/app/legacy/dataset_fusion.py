import torch
from torch.utils.data import Dataset

from .dataset import augment_data


class FusionTrajectoryDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.obs = []
        self.neighbors = []
        self.fusion = []
        self.future = []
        self.augment = augment

        for obs, neighbors, fusion_obs, future in samples:
            self.obs.append(obs)
            self.neighbors.append(neighbors)
            self.fusion.append(fusion_obs)
            self.future.append(future)

        self.obs = torch.tensor(self.obs, dtype=torch.float32)
        self.fusion = torch.tensor(self.fusion, dtype=torch.float32)
        self.future = torch.tensor(self.future, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = self.obs[idx].clone()
        fusion_obs = self.fusion[idx].clone()
        future = self.future[idx].clone()
        neighbors = [torch.tensor(n, dtype=torch.float32) for n in self.neighbors[idx]]

        if self.augment:
            obs, neighbors, future = augment_data(obs, neighbors, future)

        return obs, neighbors, fusion_obs, future
