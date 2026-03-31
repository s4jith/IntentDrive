import type {
  HealthResponse,
  LiveFramesResponse,
  LiveFusionRequest,
  PredictionResponse
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      if (typeof err?.detail === "string") {
        detail = err.detail;
      }
    } catch {
      // Keep default detail.
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  const resp = await fetch(`${API_BASE}/api/health`);
  return parseJsonOrThrow<HealthResponse>(resp);
}

export async function getLiveFrames(channel = "CAM_FRONT", limit = 200): Promise<LiveFramesResponse> {
  const url = new URL(`${API_BASE}/api/live/frames`);
  url.searchParams.set("channel", channel);
  url.searchParams.set("limit", String(limit));
  const resp = await fetch(url);
  return parseJsonOrThrow<LiveFramesResponse>(resp);
}

export async function predictTwoImage(args: {
  imagePrev: File;
  imageCurr: File;
  scoreThreshold: number;
  trackingGatePx: number;
  minMotionPx: number;
  usePose: boolean;
}): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("image_prev", args.imagePrev);
  form.append("image_curr", args.imageCurr);
  form.append("score_threshold", String(args.scoreThreshold));
  form.append("tracking_gate_px", String(args.trackingGatePx));
  form.append("min_motion_px", String(args.minMotionPx));
  form.append("use_pose", String(args.usePose));

  const resp = await fetch(`${API_BASE}/api/predict/two-image`, {
    method: "POST",
    body: form
  });

  return parseJsonOrThrow<PredictionResponse>(resp);
}

export async function predictLiveFusion(payload: LiveFusionRequest): Promise<PredictionResponse> {
  const resp = await fetch(`${API_BASE}/api/predict/live-fusion`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  return parseJsonOrThrow<PredictionResponse>(resp);
}

export { API_BASE };
