import { useEffect, useMemo, useState } from "react";

import { API_BASE, getHealth, getLiveFrames, predictLiveFusion, predictTwoImage } from "./api/client";
import BevCanvas from "./components/BevCanvas";
import CameraDetectionsPanel from "./components/CameraDetectionsPanel";
import type { AgentState, HealthResponse, LiveFramesResponse, LiveFusionRequest, PredictionResponse } from "./types";

type TabMode = "two-image" | "live-fusion";

const initialLiveReq: LiveFusionRequest = {
  anchor_idx: 3,
  score_threshold: 0.35,
  tracking_gate_px: 130,
  use_pose: false
};

export default function App() {
  const [mode, setMode] = useState<TabMode>("two-image");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [framesInfo, setFramesInfo] = useState<LiveFramesResponse | null>(null);

  const [prevImage, setPrevImage] = useState<File | null>(null);
  const [currImage, setCurrImage] = useState<File | null>(null);
  const [prevImagePreview, setPrevImagePreview] = useState<string | null>(null);
  const [currImagePreview, setCurrImagePreview] = useState<string | null>(null);

  const [scoreThreshold, setScoreThreshold] = useState(0.35);
  const [trackingGatePx, setTrackingGatePx] = useState(130);
  const [minMotionPx, setMinMotionPx] = useState(0);
  const [usePose, setUsePose] = useState(false);

  const [liveReq, setLiveReq] = useState<LiveFusionRequest>(initialLiveReq);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((err: Error) => setError(err.message));
    getLiveFrames("CAM_FRONT", 1000).then(setFramesInfo).catch(() => null);
  }, []);

  useEffect(() => {
    if (!prevImage) {
      setPrevImagePreview(null);
      return;
    }

    const objectUrl = URL.createObjectURL(prevImage);
    setPrevImagePreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [prevImage]);

  useEffect(() => {
    if (!currImage) {
      setCurrImagePreview(null);
      return;
    }

    const objectUrl = URL.createObjectURL(currImage);
    setCurrImagePreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [currImage]);

  const agents = result?.agents ?? [];
  const sceneQuality = result?.scene_geometry?.quality;

  const targetSummary = useMemo(() => {
    if (!result) {
      return "No prediction yet.";
    }

    const target = result.agents.find((a) => a.is_target);
    if (!target) {
      return `Returned ${result.agents.length} agents.`;
    }

    const maxProb = target.probabilities.length ? Math.max(...target.probabilities) : 0;
    return `Target ID ${target.id}, best mode confidence ${(maxProb * 100).toFixed(1)}%.`;
  }, [result]);

  async function runTwoImage() {
    if (!prevImage || !currImage) {
      setError("Please upload both images first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const payload = await predictTwoImage({
        imagePrev: prevImage,
        imageCurr: currImage,
        scoreThreshold,
        trackingGatePx,
        minMotionPx,
        usePose
      });
      setResult(payload);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function runLiveFusion() {
    setLoading(true);
    setError(null);

    try {
      const payload = await predictLiveFusion(liveReq);
      setResult(payload);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-root">
      <header className="topbar">
        <div>
          <p className="kicker">Phase 1 Migration</p>
          <h1>BEV Control Room</h1>
        </div>
        <div className="status-pills">
          <span className="pill">API: {API_BASE}</span>
          <span className={`pill ${health?.status === "ok" ? "ok" : "warn"}`}>
            Backend {health?.status === "ok" ? "healthy" : "pending"}
          </span>
          <span className="pill">Fusion Model: {health?.using_fusion_model ? "ON" : "OFF"}</span>
        </div>
      </header>

      <main className="layout">
        <section className="panel controls">
          <div className="tabs">
            <button className={mode === "two-image" ? "active" : ""} onClick={() => setMode("two-image")}>Two Image Upload</button>
            <button className={mode === "live-fusion" ? "active" : ""} onClick={() => setMode("live-fusion")}>Live CV + Fusion</button>
          </div>

          {mode === "two-image" ? (
            <div className="control-block">
              <label>
                Image t-1
                <input type="file" accept="image/png,image/jpeg" onChange={(e) => setPrevImage(e.target.files?.[0] ?? null)} />
              </label>
              <label>
                Image t0
                <input type="file" accept="image/png,image/jpeg" onChange={(e) => setCurrImage(e.target.files?.[0] ?? null)} />
              </label>

              <label>
                Detection threshold
                <input
                  type="range"
                  min={0.2}
                  max={0.9}
                  step={0.01}
                  value={scoreThreshold}
                  onChange={(e) => setScoreThreshold(Number(e.target.value))}
                />
                <span>{scoreThreshold.toFixed(2)}</span>
              </label>

              <label>
                Tracking gate (px)
                <input
                  type="number"
                  min={20}
                  max={300}
                  value={trackingGatePx}
                  onChange={(e) => setTrackingGatePx(Number(e.target.value))}
                />
              </label>

              <label>
                Minimum motion (px)
                <input
                  type="number"
                  min={0}
                  max={80}
                  value={minMotionPx}
                  onChange={(e) => setMinMotionPx(Number(e.target.value))}
                />
              </label>

              <label className="checkbox-row">
                <input type="checkbox" checked={usePose} onChange={(e) => setUsePose(e.target.checked)} />
                Use keypoint model
              </label>

              <button disabled={loading} onClick={runTwoImage} className="run-btn">
                {loading ? "Running..." : "Run Prediction"}
              </button>
            </div>
          ) : (
            <div className="control-block">
              <label>
                Anchor frame index
                <input
                  type="number"
                  min={3}
                  max={Math.max(3, (framesInfo?.count ?? 4) - 1)}
                  value={liveReq.anchor_idx}
                  onChange={(e) => setLiveReq((prev) => ({ ...prev, anchor_idx: Number(e.target.value) }))}
                />
              </label>

              <label>
                Detection threshold
                <input
                  type="range"
                  min={0.2}
                  max={0.9}
                  step={0.01}
                  value={liveReq.score_threshold}
                  onChange={(e) => setLiveReq((prev) => ({ ...prev, score_threshold: Number(e.target.value) }))}
                />
                <span>{liveReq.score_threshold.toFixed(2)}</span>
              </label>

              <label>
                Tracking gate (px)
                <input
                  type="number"
                  min={30}
                  max={300}
                  value={liveReq.tracking_gate_px}
                  onChange={(e) => setLiveReq((prev) => ({ ...prev, tracking_gate_px: Number(e.target.value) }))}
                />
              </label>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={liveReq.use_pose}
                  onChange={(e) => setLiveReq((prev) => ({ ...prev, use_pose: e.target.checked }))}
                />
                Use keypoint model
              </label>

              <p className="hint">Available CAM_FRONT frames: {framesInfo?.count ?? "unknown"}</p>

              <button disabled={loading} onClick={runLiveFusion} className="run-btn">
                {loading ? "Running..." : "Run Live Fusion"}
              </button>
            </div>
          )}

          {error && <p className="error-box">{error}</p>}
        </section>

        <section className="panel output">
          <div className="output-head">
            <h2>Bird's-Eye Forecast</h2>
            <p>{targetSummary}</p>
          </div>

          <BevCanvas agents={agents as AgentState[]} sceneGeometry={result?.scene_geometry} />

          <div className="meta-grid">
            <article>
              <h3>Prediction Summary</h3>
              <p>Mode: {result?.mode ?? "-"}</p>
              <p>Agents: {result?.agents.length ?? 0}</p>
              <p>Target Track: {result?.target_track_id ?? "n/a"}</p>
            </article>
            <article>
              <h3>Sensor Summary</h3>
              <p>LiDAR points: {result?.sensors?.lidar_points ?? "n/a"}</p>
              <p>Radar points: {result?.sensors?.radar_points ?? "n/a"}</p>
              <p>Sample token: {result?.sensors?.sample_token ?? "n/a"}</p>
              <p>Scene source: {result?.scene_geometry?.source ?? "n/a"}</p>
              <p>Scene quality: {sceneQuality !== undefined ? `${(sceneQuality * 100).toFixed(0)}%` : "n/a"}</p>
            </article>
          </div>

          <CameraDetectionsPanel
            detections={result?.detections}
            prevImagePreview={prevImagePreview}
            currImagePreview={currImagePreview}
          />
        </section>
      </main>
    </div>
  );
}
