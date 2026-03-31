# Frontend (Phase 1)

TypeScript + React dashboard for calling the FastAPI backend and visualizing BEV trajectories.

## Run

From repository root:

```powershell
Set-Location frontend
npm install
npm run dev
```

Optional API base URL override:

```powershell
$env:VITE_API_BASE_URL = "http://localhost:8000"
npm run dev
```

## Included in Phase 1

- Two-image upload flow
- Live fusion request flow
- BEV trajectory canvas for returned agents
- Backend health and frame inventory checks

## Phase 2 Additions

- Scene-grounded BEV layer rendering from backend `scene_geometry`.
- Camera-derived road polygon and lane lines in the BEV canvas.
- Projected actor elements (from detections) shown as BEV footprints.
- Camera panel overlays using backend frame snapshots and detection boxes.
- Overlay controls: confidence threshold slider, per-camera filter, and label on/off toggle.
