#!/usr/bin/env python3
"""Evaluate RadianceReg-GS on pair NPZ files with evaluation-only ground truth."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from radiancereg_gs import RegistrationConfig, load_pair_npz, register, save_registration
from radiancereg_gs.core import robust_extent

STRICT = (10.0, 0.10, 10.0)
RELAXED = (15.0, 0.30, 20.0)


def errors(result, path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as archive:
        gt_scale = float(archive["gt_scale"])
        gt_rotation = np.asarray(archive["gt_rotation"], np.float64)
        gt_translation = np.asarray(archive["gt_translation"], np.float64)
        extent = robust_extent(np.asarray(archive["target_means"], np.float64))
    estimate = result.transform
    cosine = np.clip((np.trace(estimate.rotation @ gt_rotation.T) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "rre_deg": float(np.degrees(np.arccos(cosine))),
        "rte_extent": float(np.linalg.norm(estimate.translation - gt_translation) / extent),
        "rse_percent": float(100.0 * abs(estimate.scale - gt_scale) / gt_scale),
    }


def passes(metric: dict, gate: tuple[float, float, float]) -> bool:
    return bool(metric["rre_deg"] < gate[0] and metric["rte_extent"] < gate[1]
                and metric["rse_percent"] < gate[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--save-aligned", action="store_true")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    paths = sorted(args.data.glob(args.pattern))
    if not paths or (args.expected_pairs is not None and len(paths) != args.expected_pairs):
        raise SystemExit(f"found {len(paths)} pair files in {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = RegistrationConfig(proposal_iterations=args.iterations, seed=args.seed)
    rows = []
    for index, path in enumerate(paths, 1):
        started = time.perf_counter()
        pair_output = args.output / path.stem
        try:
            source, target = load_pair_npz(path)
            result = register(source, target, config)
            metric = errors(result, path)
            row = {
                "pair": path.stem,
                "status": "ok",
                "elapsed_seconds": time.perf_counter() - started,
                "estimated_scale": float(result.transform.scale),
                "estimated_rotation": result.transform.rotation.tolist(),
                "estimated_translation": result.transform.translation.tolist(),
                "reciprocal_support_matches": result.reciprocal_matches,
                "candidate_count": result.candidate_count,
                "eligible_candidates": result.eligible_candidates,
                "accepted_refinement": result.accepted_refinement,
                **metric,
            }
            row["strict_success"] = passes(row, STRICT)
            row["success"] = passes(row, RELAXED)
            if args.save_aligned:
                save_registration(result, pair_output)
        except Exception as error:
            row = {
                "pair": path.stem,
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
                "strict_success": False,
                "success": False,
            }
        pair_output.mkdir(parents=True, exist_ok=True)
        (pair_output / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(f"[{index:03d}/{len(paths):03d}] {path.stem}: {row['status']}", flush=True)
    valid = [row for row in rows if row["status"] == "ok"]
    median = lambda key: float(np.median([row[key] for row in valid])) if valid else None
    summary = {
        "method": "RadianceReg-GS",
        "attempted_pairs": len(rows),
        "valid_pairs": len(valid),
        "strict_successes": sum(row["strict_success"] for row in rows),
        "successes": sum(row["success"] for row in rows),
        "median_rre_deg": median("rre_deg"),
        "median_rte_extent": median("rte_extent"),
        "median_rse_percent": median("rse_percent"),
        "median_elapsed_seconds": median("elapsed_seconds"),
        "protocol": {
            "ground_truth_used_for_registration": False,
            "all_attempted_pairs_remain_in_denominator": True,
            "strict_gate": STRICT,
            "relaxed_gate": RELAXED,
        },
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
