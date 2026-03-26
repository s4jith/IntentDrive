from pathlib import Path
import json

DATA_ROOT = Path("DataSet/v1.0-mini")


def load_json(name):
    with open(DATA_ROOT / f"{name}.json") as f:
        return json.load(f)


def build_lookup(table):
    return {item['token']: item for item in table}


def extract_pedestrian_instances(sample_annotations, instances, categories):
    cat_lookup = build_lookup(categories)
    inst_lookup = build_lookup(instances)

    pedestrian_instances = set()

    for ann in sample_annotations:
        inst = inst_lookup[ann['instance_token']]
        category = cat_lookup[inst['category_token']]['name']

        if "pedestrian" in category:
            pedestrian_instances.add(ann['instance_token'])

    return pedestrian_instances


def build_trajectories(sample_annotations, pedestrian_instances):
    ann_lookup = build_lookup(sample_annotations)

    visited = set()
    trajectories = []

    for ann in sample_annotations:
        if ann['token'] in visited:
            continue

        if ann['instance_token'] not in pedestrian_instances:
            continue

        current = ann
        while current['prev'] != "":
            current = ann_lookup[current['prev']]

        traj = []

        while current:
            visited.add(current['token'])

            x, y, _ = current['translation']
            traj.append([x, y])

            if current['next'] == "":
                break

            current = ann_lookup[current['next']]

        if len(traj) >= 16:  # 4 past + 12 future (6 seconds)
            trajectories.append(traj)

    return trajectories


def create_windows(trajectories):
    import math
    samples = []

    for traj in trajectories:
        # Require 16 frames: 4 history + 12 future
        for i in range(len(traj) - 15):

            # ---------------- MAIN TRAJECTORY ----------------
            window = traj[i:i+16]

            x3, y3 = window[3]
            window = [[x - x3, y - y3] for x, y in window]

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

            # Future is now 12 steps (6 seconds)
            future = window[4:16]

            # ---------------- NEIGHBORS ----------------
            neighbors = []

            for other_traj in trajectories:
                if other_traj is traj:
                    continue

                if len(other_traj) < i + 4:
                    continue

                x1, y1 = traj[i + 3] # Main trajectory center
                x2, y2 = other_traj[i + 3]

                dist = math.hypot(x1 - x2, y1 - y2)

                # Expanded Social Radius to 50 meters to account for much longer timeframe
                if dist < 50.0:  

                    n_window = other_traj[i:i+4]

                    # Center around main trajectory's last observed timestep
                    n_window = [[x - x1, y - y1] for x, y in n_window]

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
                            n_window[j][0],
                            n_window[j][1],
                            vel_n[j][0],
                            vel_n[j][1],
                            vel_n[j][2],
                            vel_n[j][3],
                            vel_n[j][4]
                        ])

                    neighbors.append(n_obs)

            samples.append((obs, neighbors, future))

    return samples


def main():
    print("Loading data...")

    sample_annotations = load_json("sample_annotation")
    instances = load_json("instance")
    categories = load_json("category")

    print("Filtering pedestrians...")
    ped_instances = extract_pedestrian_instances(
        sample_annotations, instances, categories
    )

    print("Building trajectories...")
    trajectories = build_trajectories(sample_annotations, ped_instances)

    print("Creating training samples...")
    samples = create_windows(trajectories)

    print("\n--- DEBUG ---")
    obs, neighbors, future = samples[0]

    print("Obs length:", len(obs))
    print("Future length:", len(future))
    print("Neighbors count:", len(neighbors))

    if len(neighbors) > 0:
        print("One neighbor shape:", len(neighbors[0]), len(neighbors[0][0]))

    print(f"\nTotal trajectories: {len(trajectories)}")
    print(f"Total samples: {len(samples)}")


if __name__ == "__main__":
    main()