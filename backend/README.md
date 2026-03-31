# Backend (Phase 1)

This backend exposes your existing CV + trajectory prediction pipeline through FastAPI.

## Folder Structure

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
```

Notes:
- Runtime model and fusion logic is now under `backend/app/ml`.
- Legacy helper modules were moved under `backend/app/legacy`.
- Training, evaluation, and data scripts were moved under `backend/scripts/*`.
- Root-level `inference.py`, `model.py`, `model_fusion.py`, and `sensor_fusion.py` remain as compatibility wrappers.

## Run

From the repository root:

```powershell
.\\venv\\Scripts\\python.exe -m pip install -r backend/requirements.txt
.\\venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /api/health`
- `GET /api/live/frames?channel=CAM_FRONT&limit=200`
- `GET /api/live/frame-image?path=<dataset_frame_path>`
- `POST /api/predict/two-image` (multipart form)
- `POST /api/predict/live-fusion` (JSON body)

## Phase 2 Scene Geometry

Prediction responses now include `scene_geometry` with image-grounded BEV primitives:

- `road_polygon`: camera-derived drivable area in BEV coordinates.
- `lane_lines`: lane candidates projected into BEV.
- `elements`: projected actor footprints from detections.
- `quality`: confidence score in `[0, 1]` for extracted scene structure.

Notes:
- The backend keeps using existing model files at repository root.
- Keep running from repo root so relative data paths remain stable.
