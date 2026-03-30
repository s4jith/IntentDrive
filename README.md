# BEV VRU Trajectory Prediction Transformer

A High-Speed, Socially-Aware Trajectory Prediction system designed for Vulnerable Road Users (VRUs) using the nuScenes dataset. 
This project focuses on protecting pedestrians, bicycles, and motorcycles by giving our autonomous ego-vehicle extreme long-range foresight. It predicts paths **6 seconds into the future** (12 timesteps) to allow for highway-speed braking distances.

## How the AI Works

### The Data (Math over Pixels)
In standard autonomous vehicle stacks, Perception AI (vision/LIDAR) tracks objects and passes their coordinates to Prediction AI. 
To eliminate vision latency and maximize compute efficiency, our model trains purely on **kinematic mathematics**.
*   **Target Files:** We extract exact `[X, Y]` coordinates exclusively from the `v1.0-mini` dataset using `category.json`, `instance.json`, and `sample_annotation.json`.
*   **The Input:** A simple array of 4 recent `(X, Y)` spatial coordinates representing a 2-second tracking history.
*   **The Output:** 3 separate diverse mode predictions spanning 6 seconds into the future (12 coordinates per path).

### Key Technical Architecture
1. **Transformer Sequence Encoder**: We completely bypassed legacy LSTMs, building a `nn.TransformerEncoder` with custom Temporal Positional Encodings to map kinematic geometry (velocities, sine/cosine angular arcs).
2. **Social Attention Pooling**: Uses a `MultiheadAttention` mechanism. The model calculates the real-time distance of ALL other road users within a massive **50-meter radius**, applying dynamic attention weights to prevent predicting paths that crash into others.
3. **Goal-Conditioned Decoding**: The Transformer splits trajectory prediction into two tasks: first predicting the final 6-second physical endpoint (Goal), then rendering the continuous curve to reach it.
4. **Native BEV Map Render Synthesis**: The app dynamically intercepts raw image rasters from the `v1.0-mini` metadata, converting grayscale masks into RGBA transparency layers. It overlays the predicted mathematical trajectory directly onto the actual HD road layer for visual confirmation.

## How to use
**Activate Virtual Environment:**
```bash
.\venv\Scripts\Activate.ps1
```

**1. Train the Core Transformer:**
```bash
python train.py
```
*Current configuration computes trajectory losses, goal-accuracy loss, and diverse mode pushing over 50 epochs on GPU execution.*

**2. Generate Hackathon Metrics Report:**
```bash
python evaluate.py
```
*Calculates deep validation metrics required by judges, including Average Displacement Error (ADE), Final Displacement Error (FDE), and standard Miss Rate (>2.0m).*

**3. Run the Interactive Dashboard:**
```bash
streamlit run app.py
```
*Runs the custom prediction engine. Accepts custom (X,Y) coordinate points manually typed by the user, mathematically scales the tracking history, calculates social attention to nearby neighbors dynamically, and plots the scaled output directly onto the real-world dataset map patch.*