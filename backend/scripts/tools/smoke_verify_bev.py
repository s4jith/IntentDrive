from __future__ import annotations

from pathlib import Path

from backend.app.main import app
from backend.app.services.pipeline import TrajectoryPipeline


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    log_dir = repo_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []

    pipeline = TrajectoryPipeline(repo_root=repo_root)
    frames = pipeline.list_channel_image_paths("CAM_FRONT")
    report_lines.append(f"frame_count={len(frames)}")

    if len(frames) >= 4:
        bundle = pipeline.build_live_agents_bundle(
            anchor_idx=3,
            score_threshold=0.35,
            tracking_gate_px=130.0,
            use_pose=False,
        )
        scene = bundle.get("scene_geometry") if isinstance(bundle, dict) else None
        report_lines.append(f"pipeline_has_error={'error' in bundle}")
        report_lines.append(f"pipeline_agent_count={len(bundle.get('agents', [])) if isinstance(bundle, dict) else 0}")
        report_lines.append(f"pipeline_has_scene_geometry={scene is not None}")
        report_lines.append(f"pipeline_has_map_layer={bool(scene and scene.get('map_layer'))}")
        report_lines.append(f"pipeline_scene_source={scene.get('source') if scene else 'none'}")
    else:
        report_lines.append("pipeline_has_error=True")
        report_lines.append("pipeline_agent_count=0")
        report_lines.append("pipeline_has_scene_geometry=False")
        report_lines.append("pipeline_has_map_layer=False")
        report_lines.append("pipeline_scene_source=none")

    route_paths = sorted(r.path for r in app.routes if hasattr(r, "path"))
    report_lines.append(f"route_count={len(route_paths)}")
    report_lines.append(f"has_health_route={'/api/health' in route_paths}")
    report_lines.append(f"has_live_frames_route={'/api/live/frames' in route_paths}")
    report_lines.append(f"has_predict_two_image_route={'/api/predict/two-image' in route_paths}")
    report_lines.append(f"has_predict_live_fusion_route={'/api/predict/live-fusion' in route_paths}")

    report_path = log_dir / "bev_smoke_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"report_path={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
