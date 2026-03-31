export type Point2D = {
  x: number;
  y: number;
};

export type AgentState = {
  id: number;
  type: "pedestrian" | "vehicle" | string;
  raw_label?: string | null;
  history: Point2D[];
  predictions: Point2D[][];
  probabilities: number[];
  is_target: boolean;
};

export type SceneElement = {
  kind: string;
  track_id?: number | null;
  score?: number;
  polygon: Point2D[];
};

export type MapLayer = {
  source?: string;
  map_token?: string;
  valid_ratio?: number;
  image_png_base64: string;
  opacity?: number;
  bounds: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
  };
};

export type SceneGeometry = {
  source?: string;
  quality: number;
  road_polygon: Point2D[];
  lane_lines: Point2D[][];
  elements?: SceneElement[];
  map_layer?: MapLayer;
  image_size?: {
    width: number;
    height: number;
  };
};

export type DetectionItem = {
  det_id?: number;
  track_id?: number | null;
  kind?: string;
  raw_label?: string;
  score?: number;
  box?: [number, number, number, number] | number[];
};

export type DetectionSnapshot = {
  frame_path?: string;
  detections: DetectionItem[];
};

export type PredictionResponse = {
  mode: string;
  target_track_id: number | null;
  agents: AgentState[];
  meta: Record<string, unknown>;
  detections?: Record<string, DetectionSnapshot>;
  scene_geometry?: SceneGeometry;
  sensors?: {
    sample_token?: string;
    lidar_points?: number;
    radar_points?: number;
    radar_channel_counts?: Record<string, number>;
  };
};

export type HealthResponse = {
  status: string;
  using_fusion_model: boolean;
  dataset_root: string;
  dataset_exists: boolean;
};

export type LiveFramesResponse = {
  channel: string;
  count: number;
  frames: string[];
};

export type LiveFusionRequest = {
  anchor_idx: number;
  score_threshold: number;
  tracking_gate_px: number;
  use_pose: boolean;
};
