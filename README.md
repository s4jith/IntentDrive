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

## How to Use
**Activate virtual environment:**
```bash
.\venv\Scripts\Activate.ps1
```

**1. Run Backend (FastAPI):**
```bash
.\venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**2. Run Frontend (Vite + React):**
```bash
cd frontend
npm install
npm run dev
```

**3. Optional legacy training/evaluation scripts:**
```bash
python -m backend.scripts.training.train
python -m backend.scripts.evaluation.evaluate
```

Note: root-level script wrappers were removed during cleanup. Run training/evaluation/data utilities via `python -m backend.scripts...` module paths.
Checkpoint files are now centralized in the `models/` folder at repository root.

## Phase 1 Folder Structure

The repository now includes a clean split for migration work:

```text
backend/
	app/
		api/
			dependencies.py
			routes/
				health.py
				live.py
				predict.py
		core/
			serialization.py
			uploads.py
		ml/
			inference.py
			model.py
			model_fusion.py
			sensor_fusion.py
		legacy/
			dataset.py
			dataset_fusion.py
			data_loader.py
			cv_perception.py
			map_renderer.py
			visualization.py
		services/
			pipeline.py
		main.py
		schemas.py
	scripts/
		training/
		evaluation/
		data/
		tools/
		legacy/
	requirements.txt

frontend/
	src/
		App.tsx
		api/client.ts
		components/BevCanvas.tsx
		components/CameraDetectionsPanel.tsx
		types.ts
	package.json
```

## Phase 1 Run Commands

Backend (FastAPI):

```bash
.\venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend (TypeScript + Vite):

```bash
cd frontend
npm install
npm run dev
```