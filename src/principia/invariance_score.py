"""Principia invariance score.

Run after the scenario scripts (or ``run_all.py``). Reads every
``<model_dir>/<scenario>/sample_<n>/results.json`` that exists, scores each
sample against its physical invariant, and prints the per-scenario scores and
their average. Samples without a results file, and scenarios with no results
at all, are skipped.

A sample only keeps its score if both objects pass the directional-consistency
check, read from a JSON file next to ``results.json`` holding
``direction_consistency_gen_1`` and ``direction_consistency_gen_2``.

Passing several model directories (e.g. one per seed of the same model)
averages each sample's score across them before the statistics are computed.

Usage::

    python -m principia.invariance_score /path/to/<model_dir>
    python -m principia.invariance_score /path/to/<model>_seed_42 /path/to/<model>_seed_123
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .common import list_samples

DIRECTIONAL_THRESHOLD = 0.8

SPRING_MASS_LOOKUP = {2: 27, 1: 8, 0: 27 / 8}

# Scored scenario -> folder holding its results. Gravity is scored from the
# restitution videos.
SCENARIO_FOLDERS = {
    "restitution": "restitution",
    "gravity": "restitution",
    "coeff_friction": "coeff_friction",
    "moment_of_inertia": "moment_of_inertia",
    "projectile": "projectile",
    "conservation_of_momentum": "conservation_of_momentum",
    "spring": "spring",
    "pendulum": "pendulum",
}

# Scenarios whose motion is horizontal; the rest use the vertical check.
HORIZONTAL_SCENARIOS = {"pendulum"}


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def compute_delta_phi(o1: float, o2: float) -> Optional[float]:
    """Symmetric relative difference between two observed ratios."""
    denom = abs(o1) + abs(o2)
    if denom < 1e-8:
        return None
    return abs(o1 - o2) / denom


def score_from_delta(delta: Optional[float]) -> float:
    return 1.0 - delta if delta is not None else 0.0


def ratio_score(o1: Optional[float], o2: Optional[float]) -> float:
    if o1 is None or o2 is None:
        return 0.0
    return score_from_delta(compute_delta_phi(o1, o2))


def compute_score(scenario: str, data: dict) -> Optional[float]:
    """Return the [0, 1] consistency score for one sample.

    Returns None when the sample has no expected outcome and should be
    ignored rather than counted as a failure.
    """
    if scenario in ("coeff_friction", "restitution", "gravity"):
        keys = {
            "coeff_friction": ("drop_frame_gen_1", "drop_frame_gen_2"),
            "restitution": ("coeff_rest_gen_1", "coeff_rest_gen_2"),
            "gravity": ("height_ratio", "time_of_flight_ratio_squared"),
        }[scenario]
        return ratio_score(data.get(keys[0]), data.get(keys[1]))

    if scenario == "moment_of_inertia":
        d1, d2 = data.get("drop_frame_gen_1"), data.get("drop_frame_gen_2")
        if not d1 or not d2:
            return 0.0
        o1 = d1 / d2 if data.get("hollow_is_left") else d2 / d1
        return ratio_score(o1, 2 / np.sqrt(3))

    if scenario == "spring":
        m1 = SPRING_MASS_LOOKUP[data.get("mass1")]
        m2 = SPRING_MASS_LOOKUP[data.get("mass2")]
        y1, y2 = data.get("y_disp_1"), data.get("y_disp_2")
        if y1 is None or y2 is None or y1 <= 0 or y2 <= 0:
            return 0.0
        return ratio_score(m1 / m2, y1 / y2)

    if scenario == "pendulum":
        l1, l2 = data.get("length1_model"), data.get("length2_model")
        t1, t2 = data.get("time_period_gen_1"), data.get("time_period_gen_2")
        if any(v is None or v <= 0 for v in (l1, l2, t1, t2)):
            return 0.0
        return ratio_score(l1 / l2, (t1 / t2) ** 2)

    if scenario in ("projectile", "conservation_of_momentum"):
        gen_score = data.get("gen_score")
        if gen_score is None:
            return None
        return 1.0 if gen_score == 1 else 0.0

    raise ValueError(f"Unhandled scenario: {scenario}")


def direction_passed(path: Path) -> Optional[bool]:
    """Return whether both objects pass the directional check, or None if the file is missing."""
    if not path.is_file():
        return None
    with open(path) as f:
        d = json.load(f)
    s1, s2 = d.get("direction_consistency_gen_1"), d.get("direction_consistency_gen_2")
    if s1 is None or s2 is None:
        return False
    return min(s1, s2) >= DIRECTIONAL_THRESHOLD


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def compute_invariance_scores(
    model_dirs: list[Path],
    output_name: str = "results.json",
    vertical_name: str = "direction_consistency_vertical_only.json",
    horizontal_name: str = "direction_consistency_horizontal_only.json",
) -> dict:
    """Score every scenario that has results and return the per-scenario and average scores.

    Returns ``{"scenarios": {scenario: {"mean", "std", "n_samples", ...}}, "average": float}``;
    scenarios without any results file are left out.
    """
    scenarios: dict[str, dict] = {}
    for scenario, folder in SCENARIO_FOLDERS.items():
        direction_name = horizontal_name if scenario in HORIZONTAL_SCENARIOS else vertical_name
        # sample name -> scores from each model dir that has a usable result.
        per_sample: dict[str, list[float]] = defaultdict(list)
        n_found = n_ignored = n_direction_fail = n_direction_missing = 0

        for model_dir in model_dirs:
            scenario_dir = model_dir / folder
            if not scenario_dir.is_dir():
                continue
            for sample_dir in list_samples(scenario_dir):
                results_path = sample_dir / output_name
                if not results_path.is_file():
                    continue
                try:
                    with open(results_path) as f:
                        data = json.load(f)
                    score = compute_score(scenario, data)
                    passed = direction_passed(sample_dir / direction_name)
                except Exception as e:
                    print(f"Error scoring {results_path}: {e}")
                    continue

                n_found += 1
                if score is None:
                    n_ignored += 1
                    continue
                if passed is None:
                    n_direction_missing += 1
                if not passed:
                    n_direction_fail += 1
                    score = 0.0
                per_sample[sample_dir.name].append(score)

        if not per_sample:
            continue
        values = [float(np.mean(v)) for v in per_sample.values()]
        scenarios[scenario] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_samples": len(values),
            "n_found": n_found,
            "n_ignored": n_ignored,
            "n_direction_fail": n_direction_fail,
            "n_direction_missing": n_direction_missing,
        }

    means = [s["mean"] for s in scenarios.values()]
    return {"scenarios": scenarios, "average": float(np.mean(means)) if means else None}


def print_scores(scores: dict) -> None:
    print("\n=== INVARIANCE SCORES ===\n")
    for scenario in SCENARIO_FOLDERS:
        s = scores["scenarios"].get(scenario)
        if s is None:
            print(f"  {scenario:26s}: skipped (no results)")
            continue
        notes = f"n_samples={s['n_samples']:4d} | ignored={s['n_ignored']} | direction_fail={s['n_direction_fail']}"
        if s["n_direction_missing"]:
            notes += f" (missing file: {s['n_direction_missing']})"
        print(f"  {scenario:26s}: mean={s['mean']:.4f} | std={s['std']:.4f} | {notes}")

    avg = scores["average"]
    n = len(scores["scenarios"])
    print(f"\n  {'AVERAGE':26s}: " + (f"{avg:.4f} over {n} scenarios" if avg is not None else "n/a"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dirs", type=Path, nargs="+",
                        help="Model output folder(s); several are averaged per sample (e.g. seeds).")
    parser.add_argument("--output-name", default="results.json", help="Results filename in each sample folder.")
    parser.add_argument("--vertical-name", default="direction_consistency_vertical_only.json",
                        help="Vertical direction-consistency filename in each sample folder.")
    parser.add_argument("--horizontal-name", default="direction_consistency_horizontal_only.json",
                        help="Horizontal direction-consistency filename (pendulum).")
    parser.add_argument("--save", type=Path, default=None, help="Optionally write the scores to this JSON file.")
    args = parser.parse_args()

    for d in args.model_dirs:
        if not d.is_dir():
            raise SystemExit(f"Directory not found: {d}")

    scores = compute_invariance_scores(
        [d.resolve() for d in args.model_dirs], args.output_name, args.vertical_name, args.horizontal_name
    )
    print_scores(scores)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(scores, f, indent=4)
        print(f"\nSaved {args.save}")


if __name__ == "__main__":
    main()
