import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import os
import glob
import math
import json

TARGET_CLASSES = {1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle'}

# Set up GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_perception_model():
    print(f"[System] Loading Pre-Trained Faster R-CNN (ResNet-50-FPN) on {device.type.upper()}...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
    model.to(device) # Move model to GPU
    model.eval()
    return model, weights

def extract_features(img_path, model, weights, score_threshold=0.7):
    image = Image.open(img_path).convert("RGB")
    preprocess = weights.transforms()
    # Move the image tensor to the GPU so the math runs on CUDA
    input_batch = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch)[0]
        
    extracted = []
    # prediction items are on GPU, so we use .item() to pull the raw number back out
    for i, box in enumerate(prediction['boxes']):
        score = prediction['scores'][i].item()
        label = prediction['labels'][i].item()
        if score > score_threshold and label in TARGET_CLASSES:
            center_x = (box[0] + box[2]).item() / 2.0
            bottom_y = box[3].item()
            extracted.append({
                'type': TARGET_CLASSES[label],
                'coord': (round(center_x, 2), round(bottom_y, 2))
            })
    return extracted

def process_dataset_into_trajectories():
    print("="*60)
    print(f"| Starting Automated Dataset Pre-Processing Pipeline |")
    print(f"| Hardware Acceleration: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} |")
    print("="*60)
    
    model, weights = load_perception_model()
    
    # Get images chronologically to simulate a video feed
    image_paths = sorted(glob.glob("DataSet/samples/CAM_FRONT/*.jpg"))
    if not image_paths:
        print("[!] No images found to process.")
        return
        
    print(f"[System] Success: Found a total of {len(image_paths)} valid image frames in the folder. Processing now...")
    
    dataset_trajectories = []
    
    # We need 4 frames of history for our AI Model (T-3, T-2, T-1, T0)
    for i in range(len(image_paths) - 3):
        frames = image_paths[i:i+4]
        frame_data = []
        
        # Output progress every 50 frames
        if i % 50 == 0:
            print(f"   -> Processing frame sequence {i}/{len(image_paths)}")
            
        for f in frames:
            objs = extract_features(f, model, weights)
            frame_data.append(objs)
            
        for obj_t0 in frame_data[0]:
            target_type = obj_t0['type']
            track_history = [obj_t0['coord']]
            valid_track = True
            
            last_coord = obj_t0['coord']
            for t in range(1, 4):
                best_dist = float('inf')
                best_coord = None
                for obj_t_next in frame_data[t]:
                    if obj_t_next['type'] == target_type:
                        dist = math.sqrt((last_coord[0] - obj_t_next['coord'][0])**2 + 
                                         (last_coord[1] - obj_t_next['coord'][1])**2)
                        if dist < 60.0 and dist < best_dist: 
                            best_dist = dist
                            best_coord = obj_t_next['coord']
                            
                if best_coord:
                    track_history.append(best_coord)
                    last_coord = best_coord
                else:
                    valid_track = False
                    break
                    
            if valid_track:
                dataset_trajectories.append({
                    "agent_type": target_type,
                    "trajectory_pixels": track_history
                })

    output_file = "extracted_training_data.json"
    with open(output_file, "w") as f:
        json.dump(dataset_trajectories, f, indent=4)
        
    print(f"\n[Success] Pipeline Complete!")
    print(f"[+] Extracted {len(dataset_trajectories)} valid moving trajectories from raw images.")
    print(f"[+] Saved AI Training payload to: {output_file}")

if __name__ == '__main__':
    try:
        process_dataset_into_trajectories()
    except Exception as e:
        print(f"Error during processing: {e}")
