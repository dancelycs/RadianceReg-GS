"""Input and output for standard 3DGS PLY files and compact NPZ maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from .core import GaussianMap, RegistrationResult

SH_C0 = 0.28209479177387814


def _quaternion_matrices(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quaternion.T
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), axis=1).reshape(-1, 3, 3)


def _load_ply(path: Path) -> GaussianMap:
    vertex = PlyData.read(path)["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {
        "x", "y", "z", "opacity",
        *(f"f_dc_{index}" for index in range(3)),
        *(f"f_rest_{index}" for index in range(45)),
        *(f"scale_{index}" for index in range(3)),
        *(f"rot_{index}" for index in range(4)),
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is missing required 3DGS properties: {missing}")
    means = np.stack([vertex[key] for key in ("x", "y", "z")], axis=1)
    dc = np.stack([vertex[f"f_dc_{index}"] for index in range(3)], axis=1)
    colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    rest = np.stack([vertex[f"f_rest_{index}"] for index in range(45)], axis=1)
    sh = rest.reshape(-1, 3, 15).transpose(0, 2, 1)
    opacities = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], np.float64)))
    scales = np.exp(np.stack([vertex[f"scale_{index}"] for index in range(3)], axis=1))
    quaternion = np.stack([vertex[f"rot_{index}"] for index in range(4)], axis=1)
    rotation = _quaternion_matrices(quaternion)
    covariances = np.einsum("nij,nj,nkj->nik", rotation, scales**2, rotation)
    return GaussianMap(means, covariances, colors, opacities, sh).validated()


def _load_npz(path: Path, prefix: str = "") -> GaussianMap:
    stem = f"{prefix}_" if prefix else ""
    with np.load(path, allow_pickle=False) as archive:
        return GaussianMap(
            archive[f"{stem}means"],
            archive[f"{stem}covariances"],
            archive[f"{stem}colors"],
            archive[f"{stem}opacities"],
            archive[f"{stem}sh"],
        ).validated()


def load_gaussian_map(path: str | Path) -> GaussianMap:
    path = Path(path)
    if path.suffix.lower() == ".ply":
        return _load_ply(path)
    if path.suffix.lower() == ".npz":
        return _load_npz(path)
    raise ValueError("supported Gaussian-map formats are .ply and .npz")


def load_pair_npz(path: str | Path) -> tuple[GaussianMap, GaussianMap]:
    path = Path(path)
    return _load_npz(path, "source"), _load_npz(path, "target")


def save_registration(result: RegistrationResult, output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    aligned = result.aligned_source
    np.savez_compressed(
        output / "aligned_source.npz",
        means=aligned.means.astype(np.float32),
        covariances=aligned.covariances.astype(np.float32),
        colors=aligned.colors.astype(np.float32),
        opacities=aligned.opacities.astype(np.float32),
        sh=aligned.sh.astype(np.float32),
    )
    payload = {
        "transform_convention": "x_target = scale * rotation @ x_source + translation",
        "estimated_scale": float(result.transform.scale),
        "estimated_rotation": result.transform.rotation.tolist(),
        "estimated_translation": result.transform.translation.tolist(),
        "estimated_matrix": result.transform.matrix.tolist(),
        "reciprocal_support_matches": result.reciprocal_matches,
        "candidate_count": result.candidate_count,
        "eligible_candidates": result.eligible_candidates,
        "selected_candidate": result.selected_candidate,
        "accepted_refinement": result.accepted_refinement,
        "local_radiance_pairs": result.local_pairs,
        "degree_scores": result.degree_scores,
        "degree_weights": result.degree_weights,
        "aligned_map": "aligned_source.npz",
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
