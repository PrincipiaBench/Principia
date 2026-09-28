"""Spring evaluation.

Each mass hangs from a spring and is released. Measures how far each mass
falls before it first turns back upward below the ground line, alongside the
reference masses.

Usage::

    python -m principia.scenarios.spring /path/to/<model_dir>
"""

from typing import Optional

from ..common import (
    PORTRAIT,
    Point,
    Scenario,
    Tracks,
    VideoInfo,
    annotation_scale,
    points_to_annotation,
    reference_point,
    run_scenario_cli,
)


def find_lowest_point(
    centroids: list[Optional[Point]], ground_y: float
) -> tuple[Optional[float], Optional[int]]:
    """Return (vertical displacement from frame 0, frame) at the first low turning point.

    The turning point is the frame before the first one that is below the
    ground line and moving up. Frames where the mass is not tracked are
    skipped. Both values are None if the mass is missing on frame 0 or never
    turns.
    """
    if not centroids or centroids[0] is None:
        return None, None
    for i in range(1, len(centroids)):
        if centroids[i] is None or centroids[i - 1] is None:
            continue
        y, prev_y = centroids[i][1], centroids[i - 1][1]
        if y > ground_y and y - prev_y < 0:
            return prev_y - centroids[0][1], i - 1
    return None, None


def compute_metrics(metadata: dict, video: VideoInfo, tracks: Tracks) -> dict:
    ground_y = reference_point(metadata, "ground", video)[1]
    (y_disp_1, drop_1), (y_disp_2, drop_2) = (
        find_lowest_point(c, ground_y) for c in tracks.centroids
    )

    _, sy = annotation_scale(video)
    return {
        "mass1": metadata["mass_1"],
        "mass2": metadata["mass_2"],
        "y_disp_1": y_disp_1,
        "y_disp_2": y_disp_2,
        "drop_frame_1": drop_1,
        "drop_frame_2": drop_2,

        "centroid_positions_gen_1_model": tracks.centroids[0],
        "centroid_positions_gen_2_model": tracks.centroids[1],
        "ground_gt_threshold_model": ground_y,
        "centroid_positions_gen_1_global": points_to_annotation(tracks.centroids[0], video),
        "centroid_positions_gen_2_global": points_to_annotation(tracks.centroids[1], video),
        "ground_gt_threshold_global": ground_y * sy,
    }


SCENARIO = Scenario("spring", compute_metrics, PORTRAIT)


if __name__ == "__main__":
    run_scenario_cli(SCENARIO, __doc__.split("\n")[0])
