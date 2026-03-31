from __future__ import annotations

from pathlib import Path

from ..services.pipeline import TrajectoryPipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
pipeline = TrajectoryPipeline(repo_root=REPO_ROOT)


def get_pipeline() -> TrajectoryPipeline:
    return pipeline
