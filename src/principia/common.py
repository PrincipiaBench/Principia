"""Shared SAM3 tracking and evaluation loop for the Principia eval scripts.

Every scenario script tracks the two annotated objects through each generated
video and then derives scenario-specific metrics from the tracks. This module
holds everything except the metrics: loading the reference prompts, running
SAM3, iterating over samples, writing results and reporting failures.

Expected layout::

    <root>/
    ├── <model_dir>/<scenario>/sample_<n>/video.mp4
    └── Principia/<scenario>/sample_<n>/metadata.json

Results are written to ``<model_dir>/<scenario>/sample_<n>/results.json``.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# (width, height) at which the reference points in metadata.json were annotated.
# Scenarios filmed in portrait override this.
LANDSCAPE = (1280, 720)
PORTRAIT = (720, 1280)
OBJECT_LABELS = ("object_1", "object_2")
SAMPLE_PREFIX = "sample_"

Point = tuple[float, float]
Resolution = tuple[int, int]


@dataclass
class VideoInfo:
    path: Path
    fps: float
    resolution: Resolution
    # Resolution of the reference frame the metadata points refer to.
    annotation_resolution: Resolution = LANDSCAPE


@dataclass
class Tracks:
    """Per-object, per-frame tracking output in pixels at the video's resolution."""

    centroids: list[list[Optional[Point]]]
    bottoms: list[list[Optional[int]]]


# A scenario's metric function: (reference metadata, video, tracks) -> results dict.
MetricsFn = Callable[[dict, VideoInfo, Tracks], dict]


@dataclass
class Scenario:
    name: str  # folder name under the model and reference directories
    compute_metrics: MetricsFn
    annotation_resolution: Resolution = LANDSCAPE


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def get_video_info(video_path: Path, annotation_resolution: Resolution = LANDSCAPE) -> VideoInfo:
    """Read the frame rate and (width, height) of a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(video_path, fps, (width, height), annotation_resolution)


def rescale_point(point: Point, src_res: Resolution, dst_res: Resolution) -> Point:
    """Map a pixel coordinate from one resolution to another."""
    return point[0] * dst_res[0] / src_res[0], point[1] * dst_res[1] / src_res[1]


def reference_point(metadata: dict, label: str, video: VideoInfo) -> Point:
    """Return the first positive point of an annotated label, at the video's resolution."""
    point = metadata[label]["positive_points"][0]
    return rescale_point(point, video.annotation_resolution, video.resolution)


def annotation_scale(video: VideoInfo) -> tuple[float, float]:
    """Return the (x, y) factors mapping video pixels to annotation pixels."""
    (aw, ah), (vw, vh) = video.annotation_resolution, video.resolution
    return aw / vw, ah / vh


def points_to_annotation(points: list[Optional[Point]], video: VideoInfo) -> list[Optional[Point]]:
    """Rescale a list of (possibly missing) points to annotation resolution."""
    sx, sy = annotation_scale(video)
    return [(p[0] * sx, p[1] * sy) if p is not None else None for p in points]


def ys_to_annotation(ys: list[Optional[float]], video: VideoInfo) -> list[Optional[float]]:
    """Rescale a list of (possibly missing) y-values to annotation resolution."""
    _, sy = annotation_scale(video)
    return [y * sy if y is not None else None for y in ys]


def mask_centroid(mask: torch.Tensor) -> Optional[Point]:
    """Return the (x, y) centroid of a binary mask, or None if it is empty."""
    m = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
    moments = cv2.moments(m)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def mask_bottom_y(mask: torch.Tensor) -> Optional[int]:
    """Return the largest row index covered by a binary mask, or None if it is empty."""
    m = mask.squeeze().detach().cpu().numpy()
    rows = np.flatnonzero(m.any(axis=1))
    return int(rows[-1]) if rows.size else None


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
def build_predictor(device: str = "cuda"):
    """Build the SAM3 tracker with the detector backbone attached."""
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(device=device)
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    return predictor


def load_prompts(metadata: dict, video: VideoInfo) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build SAM3 point prompts for each object, normalised to [0, 1]."""
    prompts = []
    for label in OBJECT_LABELS:
        positives = metadata[label]["positive_points"]
        negatives = metadata[label]["negative_points"]
        if not positives and not negatives:
            raise ValueError(f"No points provided for {label}")

        points = np.array(
            [rescale_point(p, video.annotation_resolution, video.resolution) for p in positives + negatives],
            dtype=np.float32,
        )
        points /= np.array(video.resolution, dtype=np.float32)
        labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
        prompts.append((points, labels))
    return prompts


def track_objects(
    predictor,
    video: VideoInfo,
    prompts: list[tuple[np.ndarray, np.ndarray]],
    max_frames: int,
) -> Tracks:
    """Prompt SAM3 on frame 0 and propagate every object through the video."""
    n_objects = len(prompts)
    tracks = Tracks([[] for _ in range(n_objects)], [[] for _ in range(n_objects)])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(video.path))
        for obj_id, (points, labels) in enumerate(prompts, start=1):
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        for _, _, _, masks, _ in predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=max_frames,
            reverse=False,
            propagate_preflight=True,
        ):
            for i in range(n_objects):
                binary = masks[i] > 0.0
                tracks.centroids[i].append(mask_centroid(binary))
                tracks.bottoms[i].append(mask_bottom_y(binary))

    return tracks


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    model_dir: Path
    reference_root: Path
    video_name: str = "video.mp4"
    output_name: str = "results.json"
    max_frames: int = 240
    skip_existing: bool = False


def list_samples(scenario_dir: Path) -> list[Path]:
    """Return the ``sample_<n>`` folders of a scenario, sorted by n."""
    samples = [
        p
        for p in scenario_dir.iterdir()
        if p.is_dir()
        and p.name.startswith(SAMPLE_PREFIX)
        and p.name[len(SAMPLE_PREFIX):].isdigit()
    ]
    return sorted(samples, key=lambda p: int(p.name[len(SAMPLE_PREFIX):]))


def find_scenario(scenario: str, config: EvalConfig) -> tuple[Path, Path, list[Path]]:
    """Return the model and reference folders of a scenario and its sample folders.

    Raises FileNotFoundError if either folder or the samples are missing.
    """
    scenario_dir = config.model_dir / scenario
    reference_dir = config.reference_root / scenario
    for path in (scenario_dir, reference_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
    samples = list_samples(scenario_dir)
    if not samples:
        raise FileNotFoundError(f"No {SAMPLE_PREFIX}<n> folders in {scenario_dir}")
    return scenario_dir, reference_dir, samples


def evaluate_sample(
    predictor,
    scenario: Scenario,
    sample_dir: Path,
    reference_dir: Path,
    config: EvalConfig,
) -> None:
    """Track one sample, compute its metrics and write its results JSON."""
    video_path = sample_dir / config.video_name
    metadata_path = reference_dir / "metadata.json"
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing reference metadata: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    video = get_video_info(video_path, scenario.annotation_resolution)
    prompts = load_prompts(metadata, video)
    tracks = track_objects(predictor, video, prompts, config.max_frames)
    results = scenario.compute_metrics(metadata, video, tracks)

    with open(sample_dir / config.output_name, "w") as f:
        json.dump(results, f, indent=4)


def evaluate_scenario(predictor, scenario: Scenario, config: EvalConfig) -> dict[str, str]:
    """Evaluate every sample of one scenario.

    Returns a mapping of failed sample names to error messages, which is also
    written to ``<model_dir>/<scenario>/failures.json``.
    """
    name = scenario.name
    scenario_dir, reference_dir, samples = find_scenario(name, config)
    logger.info("[%s] Evaluating %d samples in %s", name, len(samples), scenario_dir)

    failures: dict[str, str] = {}
    n_skipped = 0
    for sample_dir in samples:
        if config.skip_existing and (sample_dir / config.output_name).exists():
            n_skipped += 1
            continue
        try:
            evaluate_sample(predictor, scenario, sample_dir, reference_dir / sample_dir.name, config)
            logger.info("[%s] %s: done", name, sample_dir.name)
        except Exception as e:
            logger.exception("[%s] %s: failed", name, sample_dir.name)
            failures[sample_dir.name] = str(e)

    n_done = len(samples) - n_skipped - len(failures)
    logger.info(
        "[%s] %d succeeded, %d failed, %d skipped", name, n_done, len(failures), n_skipped
    )
    with open(scenario_dir / "failures.json", "w") as f:
        json.dump(failures, f, indent=4)
    return failures


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by every eval script and the master runner."""
    parser.add_argument("model_dir", type=Path, help="Folder holding the model's generated scenarios.")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Reference dataset root (default: <model_dir>/../Principia).",
    )
    parser.add_argument("--video-name", default="video.mp4", help="Video filename in each sample folder.")
    parser.add_argument("--output-name", default="results.json", help="Results filename in each sample folder.")
    parser.add_argument("--max-frames", type=int, default=240, help="Maximum number of frames to track.")
    parser.add_argument("--device", default="cuda", help="Torch device for SAM3.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples that already have a results file.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    model_dir = args.model_dir.resolve()
    reference_root = (args.reference_dir or model_dir.parent / "Principia").resolve()
    return EvalConfig(
        model_dir=model_dir,
        reference_root=reference_root,
        video_name=args.video_name,
        output_name=args.output_name,
        max_frames=args.max_frames,
        skip_existing=args.skip_existing,
    )


def run_scenario_cli(scenario: Scenario, description: str) -> None:
    """Entry point for a single-scenario eval script."""
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    parser.add_argument("--scenario", default=scenario.name, help="Scenario folder name.")
    args = parser.parse_args()
    scenario = Scenario(args.scenario, scenario.compute_metrics, scenario.annotation_resolution)
    setup_logging(args.verbose)

    config = config_from_args(args)
    try:
        find_scenario(scenario.name, config)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    predictor = build_predictor(args.device)
    failures = evaluate_scenario(predictor, scenario, config)
    if failures:
        logger.warning("Failed samples: %s", ", ".join(failures))
