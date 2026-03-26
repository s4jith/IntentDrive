from data_loader import load_json, extract_pedestrian_instances, build_trajectories, create_windows
from dataset import TrajectoryDataset

# get data
sample_annotations = load_json("sample_annotation")
instances = load_json("instance")
categories = load_json("category")

ped_instances = extract_pedestrian_instances(sample_annotations, instances, categories)
trajectories = build_trajectories(sample_annotations, ped_instances)
samples = create_windows(trajectories)

dataset = TrajectoryDataset(samples)

obs, neighbors, future = dataset[0]

print("OBS:", obs.shape)
print("NEIGHBORS:", len(neighbors))
print("FUTURE:", future.shape)