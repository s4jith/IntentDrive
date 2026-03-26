import torch
from torch.utils.data import Dataset



class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.obs = []
        self.neighbors = []
        self.future = []

        for obs, neighbors, future in samples:
            self.obs.append(obs)
            self.neighbors.append(neighbors)
            self.future.append(future)

        # Convert only fixed-size data to tensor
        self.obs = torch.tensor(self.obs, dtype=torch.float32)
        self.future = torch.tensor(self.future, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.neighbors[idx], self.future[idx]
