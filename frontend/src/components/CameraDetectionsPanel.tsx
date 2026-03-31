import { useMemo, useState } from "react";

import { API_BASE } from "../api/client";
import type { DetectionItem, DetectionSnapshot } from "../types";

type Props = {
  detections?: Record<string, DetectionSnapshot>;
  prevImagePreview?: string | null;
  currImagePreview?: string | null;
};

type FrameCard = {
  key: string;
  title: string;
  sourceLabel: string;
  detections: DetectionItem[];
  imageSrc?: string;
};

type ImageSize = {
  width: number;
  height: number;
};

function prettyCameraName(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function detectionColor(kind?: string): string {
  if (kind === "pedestrian") {
    return "#10b981";
  }
  if (kind === "vehicle") {
    return "#38bdf8";
  }
  return "#f59e0b";
}

function normalizedScore(score: number | undefined): number {
  const value = Number(score ?? 0);
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function normalizeBox(box: DetectionItem["box"]): [number, number, number, number] | null {
  if (!Array.isArray(box) || box.length !== 4) {
    return null;
  }

  const values = box.map((value) => Number(value));
  if (values.some((value) => !Number.isFinite(value))) {
    return null;
  }

  const [x1, y1, x2, y2] = values;
  const left = Math.min(x1, x2);
  const right = Math.max(x1, x2);
  const top = Math.min(y1, y2);
  const bottom = Math.max(y1, y2);

  return [left, top, right, bottom];
}

function frameImageSource(
  key: string,
  snap: DetectionSnapshot,
  prevImagePreview?: string | null,
  currImagePreview?: string | null
): { src?: string; sourceLabel: string } {
  if (snap.frame_path) {
    return {
      src: `${API_BASE}/api/live/frame-image?path=${encodeURIComponent(snap.frame_path)}`,
      sourceLabel: "dataset frame"
    };
  }

  const lower = key.toLowerCase();
  if (lower.includes("prev") && prevImagePreview) {
    return { src: prevImagePreview, sourceLabel: "uploaded image t-1" };
  }
  if (lower.includes("curr") && currImagePreview) {
    return { src: currImagePreview, sourceLabel: "uploaded image t0" };
  }

  return { sourceLabel: "no preview source" };
}

function OverlayFrame({ card, minScore, showLabels }: { card: FrameCard; minScore: number; showLabels: boolean }) {
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  const [imageError, setImageError] = useState(false);

  const validDetections = card.detections
    .map((det, index) => ({ det, index, box: normalizeBox(det.box), score: normalizedScore(det.score) }))
    .filter((item) => item.box !== null);

  const visibleDetections = validDetections.filter((item) => item.score >= minScore);

  return (
    <article className="camera-card">
      <header>
        <h4>{card.title}</h4>
        <p>{card.sourceLabel}</p>
      </header>

      {card.imageSrc && !imageError ? (
        <div className="camera-frame-shell">
          <img
            src={card.imageSrc}
            alt={`${card.title} frame`}
            onLoad={(evt) => {
              setImageError(false);
              setImageSize({
                width: evt.currentTarget.naturalWidth,
                height: evt.currentTarget.naturalHeight
              });
            }}
            onError={() => {
              setImageError(true);
              setImageSize(null);
            }}
          />

          {imageSize && visibleDetections.length > 0 && (
            <svg
              className="camera-overlay-svg"
              viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
              preserveAspectRatio="none"
            >
              {visibleDetections.map(({ det, index, box, score }) => {
                const [x1, y1, x2, y2] = box as [number, number, number, number];
                const boxWidth = Math.max(1, x2 - x1);
                const boxHeight = Math.max(1, y2 - y1);
                const color = detectionColor(det.kind);
                const confidenceText = `${(score * 100).toFixed(0)}%`;
                const idTag = det.track_id !== undefined && det.track_id !== null ? `ID ${det.track_id}` : `D${index + 1}`;
                const label = `${idTag} ${det.kind ?? det.raw_label ?? "object"} ${confidenceText}`;

                return (
                  <g key={`${card.key}-box-${index}`}>
                    <rect
                      x={x1}
                      y={y1}
                      width={boxWidth}
                      height={boxHeight}
                      fill="none"
                      stroke={color}
                      strokeWidth={3}
                      rx={2}
                    />
                    {showLabels && (
                      <text x={x1 + 4} y={Math.max(14, y1 - 6)} fill={color} fontSize={14} fontWeight={700}>
                        {label}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          )}
        </div>
      ) : (
        <div className="camera-placeholder">
          <p>{imageError ? "Frame preview is not reachable from backend." : "No frame image available for this snapshot."}</p>
        </div>
      )}

      <footer>
        <span>Detections: {card.detections.length}</span>
        <span>Shown: {visibleDetections.length}</span>
      </footer>
    </article>
  );
}

export default function CameraDetectionsPanel({ detections, prevImagePreview, currImagePreview }: Props) {
  const [minScore, setMinScore] = useState(0.35);
  const [showLabels, setShowLabels] = useState(true);
  const [cameraFilter, setCameraFilter] = useState("all");

  const cards = useMemo<FrameCard[]>(() => {
    if (!detections) {
      return [];
    }

    return Object.entries(detections)
      .map(([key, snap]) => {
        const source = frameImageSource(key, snap, prevImagePreview, currImagePreview);
        return {
          key,
          title: prettyCameraName(key),
          sourceLabel: source.sourceLabel,
          detections: snap.detections ?? [],
          imageSrc: source.src
        };
      })
      .sort((a, b) => a.title.localeCompare(b.title));
  }, [detections, prevImagePreview, currImagePreview]);

  const filteredCards = useMemo(() => {
    if (cameraFilter === "all") {
      return cards;
    }
    return cards.filter((card) => card.key === cameraFilter);
  }, [cards, cameraFilter]);

  const shownCount = useMemo(() => {
    return filteredCards.reduce((acc, card) => {
      const count = card.detections.filter((det) => normalizedScore(det.score) >= minScore).length;
      return acc + count;
    }, 0);
  }, [filteredCards, minScore]);

  return (
    <section className="camera-panel">
      <div className="camera-panel-head">
        <h3>Camera Detection Overlays</h3>
        <p>{cards.length > 0 ? `${cards.length} camera snapshots` : "No snapshots returned yet"}</p>
      </div>

      {cards.length > 0 && (
        <div className="camera-toolbar">
          <label>
            Min confidence: {(minScore * 100).toFixed(0)}%
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={minScore}
              onChange={(evt) => setMinScore(Number(evt.target.value))}
            />
          </label>

          <label>
            Camera
            <select value={cameraFilter} onChange={(evt) => setCameraFilter(evt.target.value)}>
              <option value="all">All cameras</option>
              {cards.map((card) => (
                <option key={card.key} value={card.key}>
                  {card.title}
                </option>
              ))}
            </select>
          </label>

          <label className="camera-toggle-row">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(evt) => setShowLabels(evt.target.checked)}
            />
            Show labels
          </label>

          <p className="camera-toolbar-stats">Visible boxes: {shownCount}</p>
        </div>
      )}

      {filteredCards.length > 0 ? (
        <div className="camera-grid">
          {filteredCards.map((card) => (
            <OverlayFrame key={card.key} card={card} minScore={minScore} showLabels={showLabels} />
          ))}
        </div>
      ) : (
        <div className="camera-placeholder">
          <p>{cards.length > 0 ? "No camera matches the selected filter." : "Run a prediction to inspect camera detections overlaid on frames."}</p>
        </div>
      )}
    </section>
  );
}
