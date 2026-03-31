import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw
import os
import math

# Map COCO classes to our Hackathon targets
TARGET_CLASSES = {
    1: 'Person',
    2: 'Bicycle',
    3: 'Car',
    4: 'Motorcycle'
}

def load_perception_model():
    print("[System] Loading Faster R-CNN (ResNet-50-FPN Backbone)...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
    model.eval()
    return model, weights

def extract_features(img_path, model, weights, score_threshold=0.7):
    image = Image.open(img_path).convert("RGB")
    preprocess = weights.transforms()
    input_batch = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(input_batch)[0]
        
    extracted = []
    for i, box in enumerate(prediction['boxes']):
        score = prediction['scores'][i].item()
        label = prediction['labels'][i].item()
        
        if score > score_threshold and label in TARGET_CLASSES:
            box = box.tolist()
            class_name = TARGET_CLASSES[label]
            # Get bottom-center coordinate for BEV mapping
            center_x = (box[0] + box[2]) / 2.0
            bottom_y = box[3]
            
            extracted.append({
                'type': class_name,
                'bbox': box,
                'coord': (center_x, bottom_y)
            })
    return extracted, image

def calculate_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def process_frame_sequence(frame1_path, frame2_path, model, weights):
    """
    Takes 2 sequential frames, detects objects, matches them to find movement, 
    and bridges the data to the AI Brain.
    """
    print(f"\n[Step 1] Analyzing Frame T-1: {os.path.basename(frame1_path)}")
    objs_f1, img1 = extract_features(frame1_path, model, weights)
    
    print(f"[Step 2] Analyzing Frame T0: {os.path.basename(frame2_path)}")
    objs_f2, img2 = extract_features(frame2_path, model, weights)
    
    print("\n[Step 3] Temporal Tracking (Finding Moving Cyclists/Pedestrians)")
    tracked_history = []
    
    # Simple Tracking by linking nearest objects between Frame 1 and Frame 2
    for obj2 in objs_f2:
        best_match = None
        min_dist = float('inf')
        
        for obj1 in objs_f1:
            if obj1['type'] == obj2['type']: # Must be same class
                dist = calculate_distance(obj1['coord'], obj2['coord'])
                if dist < 50.0:  # Max pixel movement threshold between 2 frames
                    min_dist = dist
                    best_match = obj1
                    
        if best_match:
            # Calculate movement vector (Velocity)
            dx = obj2['coord'][0] - best_match['coord'][0]
            dy = obj2['coord'][1] - best_match['coord'][1]
            is_moving = abs(dx) > 1.0 or abs(dy) > 1.0
            
            if is_moving and obj2['type'] in ['Person', 'Bicycle']:
                print(f" -> Spotted Moving {obj2['type']}! dx: {dx:.2f}, dy: {dy:.2f}")
                
                # Format: [(x_t-1, y_t-1), (x_t0, y_t0)] 
                # This is EXACTLY what the AI Brain needs!
                history = [best_match['coord'], obj2['coord']]
                
                tracked_history.append({
                    "type": obj2['type'],
                    "history": history
                })
                
    print(f"\n[Step 4] Handoff to AI Brain: Found {len(tracked_history)} moving VRUs.")
    return tracked_history

if __name__ == '__main__':
    # We will use two identical images to simulate the script architecture
    # In reality, this would be image_001.jpg and image_002.jpg
    import glob
    cam_front_images = glob.glob("DataSet/samples/CAM_FRONT/*.jpg")
    
    if len(cam_front_images) >= 2:
        f1 = cam_front_images[0]
        f2 = cam_front_images[1] # Next sequential frame
        
        try:
            model, weights = load_perception_model()
            vru_data_for_ai = process_frame_sequence(f1, f2, model, weights)
            
            print("\n--- FINAL JSON PAYLOAD FOR TRANSFORMER MODEL ---")
            for person in vru_data_for_ai:
                print(f"Target: {person['type']}")
                print(f"Historical Trajectory [T-1, T0]: {person['history']}")
        except Exception as e:
            print("Model not loaded, but script structure is ready.")
