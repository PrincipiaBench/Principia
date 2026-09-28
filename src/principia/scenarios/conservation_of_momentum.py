"""Conservation-of-momentum evaluation.

Measures each object's maximum vertical displacement and scores whether the
ordering of the two displacements matches the reference: the object marked
higher should travel further, and for equal heights the lighter object should.

Usage::

    python -m principia.scenarios.conservation_of_momentum /path/to/<model_dir>
"""

from typing import Optional

from ..common import Scenario, Tracks, VideoInfo, run_scenario_cli


def score_displacements(
    relative_height: str, mass_1: float, mass_2: float, y_disp_1: float, y_disp_2: float
) -> Optional[int]:
    """Return 1 if the displacement ordering matches the reference, 0 if not.

    Returns None when the reference gives no expected ordering (equal heights
    and equal masses).
    """
    if relative_height == "high":
        return int(y_disp_1 > y_disp_2)
    if relative_height == "low":
        return int(y_disp_1 < y_disp_2)
    if relative_height == "same":
        if mass_1 > mass_2:
            return int(y_disp_1 < y_disp_2)
        if mass_1 < mass_2:
            return int(y_disp_1 > y_disp_2)
    return None


def compute_metrics(metadata: dict, video: VideoInfo, tracks: Tracks) -> dict:
    bottoms_1, bottoms_2 = tracks.bottoms
    mass_1, mass_2 = metadata["mass_1"], metadata["mass_2"]
    relative_height = metadata["left_relative_height"]

    y_disp_1 = y_disp_2 = None
    last_frame_gen_1 = last_frame_gen_2 = None
    if bottoms_1[-1] is None or bottoms_2[-1] is None:
        # An object has left the frame by the end of the video.
        gen_score = None
    elif None in bottoms_1 or None in bottoms_2:
        # An object was lost part-way through.
        gen_score = 0
    else:
        max_y_1, max_y_2 = max(bottoms_1), max(bottoms_2)
        last_frame_gen_1 = bottoms_1.index(max_y_1)
        last_frame_gen_2 = bottoms_2.index(max_y_2)
        y_disp_1 = abs(max_y_1 - bottoms_1[0])
        y_disp_2 = abs(max_y_2 - bottoms_2[0])
        gen_score = score_displacements(relative_height, mass_1, mass_2, y_disp_1, y_disp_2)

    return {
        "gen_score": gen_score,
        "mass_1": mass_1,
        "mass_2": mass_2,
        "y_disp_1": y_disp_1,
        "y_disp_2": y_disp_2,
        "left_relative_height": relative_height,
        "bottom_positions_gen_1": bottoms_1,
        "bottom_positions_gen_2": bottoms_2,
        "centroid_positions_gen_1": tracks.centroids[0],
        "centroid_positions_gen_2": tracks.centroids[1],
        "last_frame_gen_1": last_frame_gen_1,
        "last_frame_gen_2": last_frame_gen_2,
    }


SCENARIO = Scenario("conservation_of_momentum", compute_metrics)


if __name__ == "__main__":
    run_scenario_cli(SCENARIO, __doc__.split("\n")[0])
