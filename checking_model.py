from model import TrajectoryLSTM

model = TrajectoryLSTM()

obs, neighbors, future = dataset[0]

obs = obs.unsqueeze(0)
neighbors = [neighbors]

pred, probs = model(obs, neighbors)

print("Pred shape:", pred.shape)
print("Probs shape:", probs.shape)