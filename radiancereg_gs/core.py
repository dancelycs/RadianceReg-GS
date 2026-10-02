"""Final map-only RadianceReg-GS inference implementation.

The registration path accepts only observable attributes from the two Gaussian
maps. Ground-truth transformations are intentionally absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from e3nn import o3
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

_EPS = 1e-8
_SH_BLOCKS = (slice(0, 3), slice(3, 8), slice(8, 15))


@dataclass(frozen=True)
class GaussianMap:
    means: np.ndarray
    covariances: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray

    def validated(self) -> "GaussianMap":
        means = np.asarray(self.means, dtype=np.float64)
        covariances = np.asarray(self.covariances, dtype=np.float64)
        colors = np.asarray(self.colors, dtype=np.float64)
        opacities = np.asarray(self.opacities, dtype=np.float64).reshape(-1)
        sh = np.asarray(self.sh, dtype=np.float64)
        count = len(means)
        expected = {
            "means": (count, 3),
            "covariances": (count, 3, 3),
            "colors": (count, 3),
            "opacities": (count,),
            "sh": (count, 15, 3),
        }
        arrays = {
            "means": means,
            "covariances": covariances,
            "colors": colors,
            "opacities": opacities,
            "sh": sh,
        }
        for name, array in arrays.items():
            if array.shape != expected[name]:
                raise ValueError(f"{name} must have shape {expected[name]}, got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if count < 4:
            raise ValueError("a Gaussian map must contain at least four primitives")
        return GaussianMap(means, covariances, colors, opacities, sh)


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


@dataclass(frozen=True)
class RegistrationConfig:
    neighbourhoods: tuple[int, ...] = (16, 32, 64)
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    retrieval_candidates: int = 20
    proposal_iterations: int = 5000
    maximum_hypotheses: int = 48
    inlier_fraction: float = 0.04
    support_ratio: float = 0.75
    minimum_support: int = 24
    auxiliary_support: int = 64
    observable_quantile: float = 0.35
    reassociation_neighbours: int = 8
    reassociation_iterations: int = 2
    support_retention: float = 0.8
    min_scale: float = 0.2
    max_scale: float = 5.0
    seed: int = 20260914


@dataclass(frozen=True)
class Candidate:
    transform: Sim3
    inliers: int
    confidence: float


@dataclass(frozen=True)
class RegistrationResult:
    transform: Sim3
    aligned_source: GaussianMap
    reciprocal_matches: int
    candidate_count: int
    eligible_candidates: int
    selected_candidate: int
    accepted_refinement: bool
    local_pairs: int
    degree_scores: tuple[float, ...]
    degree_weights: tuple[float, ...]


def _primitive_descriptors(gaussians: GaussianMap) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(gaussians.covariances).clip(_EPS)
    log_shape = np.log(eigenvalues)
    log_shape -= log_shape.mean(axis=1, keepdims=True)
    intensity = gaussians.colors.mean(axis=1, keepdims=True)
    return np.concatenate((log_shape, gaussians.colors - intensity, intensity), axis=1)


def distribution_descriptors(
    gaussians: GaussianMap,
    neighbourhoods: tuple[int, ...] = (16, 32, 64),
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> np.ndarray:
    """Construct the multi-scale distributional proposal descriptor."""
    gaussians = gaussians.validated()
    primitive = _primitive_descriptors(gaussians)
    tree = cKDTree(gaussians.means)
    blocks = [primitive]
    for requested in neighbourhoods:
        k = min(max(4, int(requested)), len(gaussians.means) - 1)
        indices = tree.query(gaussians.means, k=k + 1, workers=-1)[1][:, 1:]
        values = primitive[indices]
        distribution = np.quantile(values, quantiles, axis=1)
        distribution = distribution.transpose(1, 0, 2).reshape(len(values), -1)
        blocks.extend((distribution, values.mean(axis=1), values.std(axis=1)))
    return np.concatenate(blocks, axis=1)


def _pair_normalize(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate((source, target), axis=0)
    center = np.median(joined, axis=0)
    scale = np.quantile(np.abs(joined - center), 0.75, axis=0)
    scale = np.where(scale > _EPS, scale, 1.0)
    source, target = (source - center) / scale, (target - center) / scale
    source /= np.maximum(np.linalg.norm(source, axis=1, keepdims=True), _EPS)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), _EPS)
    return source, target


def reciprocal_matches(
    source: np.ndarray, target: np.ndarray, candidates: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, target = _pair_normalize(source, target)
    k = min(max(1, int(candidates)), len(target))
    distances, indices = cKDTree(target).query(source, k=k, workers=-1)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]
    reverse = cKDTree(source).query(target, workers=-1)[1]
    source_ids = np.arange(len(source), dtype=np.int64)
    keep = reverse[indices[:, 0]] == source_ids
    return source_ids[keep], indices[keep, 0].astype(np.int64), distances[keep, 0]


def robust_extent(points: np.ndarray) -> float:
    low, high = np.quantile(points, (0.05, 0.95), axis=0)
    extent = float(np.linalg.norm(high - low))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("target map has an invalid robust extent")
    return extent


def weighted_umeyama(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> Sim3:
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if len(source) < 3 or source.shape != target.shape or source.shape[1:] != (3,):
        raise ValueError("Sim(3) estimation needs at least three paired 3D points")
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, np.float64)
    weights = np.maximum(weights.reshape(-1), 0.0)
    weights /= max(float(weights.sum()), 1e-12)
    source_center = np.sum(weights[:, None] * source, axis=0)
    target_center = np.sum(weights[:, None] * target, axis=0)
    xs, xt = source - source_center, target - target_center
    u, singular, vt = np.linalg.svd((weights[:, None] * xt).T @ xs)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = float(np.sum(weights * np.sum(xs * xs, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1e-12))
    translation = target_center - scale * (rotation @ source_center)
    return Sim3(scale, rotation, translation)


def _pose_distance(first: Sim3, second: Sim3, extent: float) -> tuple[float, float, float]:
    relative = first.rotation @ second.rotation.T
    angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))
    translation = float(np.linalg.norm(first.translation - second.translation) / extent)
    scale = float(abs(np.log(first.scale / second.scale)))
    return angle, translation, scale


def _generate_hypotheses(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    extent: float,
    config: RegistrationConfig,
) -> list[Candidate]:
    order = np.argsort(-confidence)
    source, target, confidence = source[order], target[order], confidence[order]
    probability = np.maximum(confidence, _EPS)
    probability /= probability.sum()
    rng = np.random.default_rng(config.seed)
    threshold = config.inlier_fraction * extent
    hypotheses: list[Candidate] = []
    for _ in range(config.proposal_iterations):
        sample = rng.choice(len(source), 3, replace=False, p=probability)
        pose = weighted_umeyama(source[sample], target[sample])
        if not config.min_scale <= pose.scale <= config.max_scale:
            continue
        moved = pose.scale * (source @ pose.rotation.T) + pose.translation
        inliers = np.linalg.norm(moved - target, axis=1) < threshold
        if inliers.sum() < 3:
            continue
        refit = weighted_umeyama(source[inliers], target[inliers], confidence[inliers])
        hypotheses.append(Candidate(refit, int(inliers.sum()), float(confidence[inliers].sum())))
    if not hypotheses:
        raise RuntimeError("proposal correspondences produced no valid similarity")
    hypotheses.sort(key=lambda item: (item.inliers, item.confidence), reverse=True)
    selected: list[Candidate] = []
    for candidate in hypotheses:
        duplicate = any(
            angle < 2.0 and translation < 0.015 and scale < 0.02
            for angle, translation, scale in
            (_pose_distance(candidate.transform, kept.transform, extent) for kept in selected)
        )
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= config.maximum_hypotheses:
            break
    return selected


def rotate_sh(sh: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate degree-1--3 real SH coefficients in the standard 3DGS convention."""
    permutation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    transformed = np.linalg.inv(permutation) @ np.asarray(rotation) @ permutation
    matrix = torch.as_tensor(transformed, dtype=torch.float64)
    angles = o3.matrix_to_angles(matrix)
    output = torch.as_tensor(np.asarray(sh), dtype=torch.float64).clone()
    for degree, block in zip((1, 2, 3), _SH_BLOCKS):
        coefficients = output[:, block, :].permute(0, 2, 1)
        action = o3.wigner_D(degree, angles[0], -angles[1], angles[2])
        output[:, block, :] = torch.einsum("ij,ncj->nci", action, coefficients).permute(0, 2, 1)
    return output.numpy()


def _row_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first, second = first.reshape(len(first), -1), second.reshape(len(second), -1)
    numerator = np.einsum("ij,ij->i", first, second)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def _degree_score(
    source_sh: np.ndarray,
    target_sh: np.ndarray,
    source_index: np.ndarray,
    target_index: np.ndarray,
    rotation: np.ndarray,
    observable_quantile: float,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    rotated = rotate_sh(source_sh[source_index], rotation)
    scores, reliability = [], []
    for block in _SH_BLOCKS:
        first, second = rotated[:, block], target_sh[target_index, block]
        cosine = _row_cosine(first, second)
        energy = np.sqrt(
            np.linalg.norm(first.reshape(len(first), -1), axis=1)
            * np.linalg.norm(second.reshape(len(second), -1), axis=1)
        )
        cutoff = np.quantile(energy, observable_quantile)
        keep = energy >= cutoff
        score, strength = float(np.median(cosine[keep])), float(np.median(energy[keep]))
        observable = energy > 0
        if score == 0.0 and np.mean(~observable) >= observable_quantile and observable.any():
            cutoff = np.quantile(energy[observable], observable_quantile)
            keep = observable & (energy >= cutoff)
            score, strength = float(np.median(cosine[keep])), float(np.median(energy[keep]))
        scores.append(score)
        reliability.append(strength)
    weights = np.asarray(reliability, dtype=np.float64)
    weights /= max(float(weights.sum()), 1e-12)
    return float(np.dot(weights, scores)), tuple(scores), tuple(weights.tolist())


def _score_candidate(
    candidate: Candidate,
    source: GaussianMap,
    target: GaussianMap,
    extent: float,
    config: RegistrationConfig,
) -> dict:
    pose = candidate.transform
    moved = pose.scale * (source.means @ pose.rotation.T) + pose.translation
    target_tree, source_tree = cKDTree(target.means), cKDTree(moved)
    distance, target_index = target_tree.query(moved, workers=-1)
    reverse = source_tree.query(target.means, workers=-1)[1]
    source_index = np.arange(len(moved))
    keep = (reverse[target_index] == source_index) & (distance < config.inlier_fraction * extent)
    if keep.sum() < config.minimum_support:
        keep = distance < config.inlier_fraction * extent
    if keep.sum() < config.minimum_support:
        order = np.argsort(distance)[: min(config.auxiliary_support, len(distance))]
        source_index, target_index = source_index[order], target_index[order]
    else:
        source_index, target_index = source_index[keep], target_index[keep]
    score, degree_scores, degree_weights = _degree_score(
        source.sh, target.sh, source_index, target_index, pose.rotation,
        config.observable_quantile,
    )
    return {
        "spatial_pairs": int(len(source_index)),
        "radiance_score": score,
        "degree_scores": degree_scores,
        "degree_weights": degree_weights,
    }


def _local_pairs(
    pose: Sim3,
    source: GaussianMap,
    target: GaussianMap,
    extent: float,
    degree_weights: np.ndarray,
    config: RegistrationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moved = pose.scale * (source.means @ pose.rotation.T) + pose.translation
    distance, target_index = cKDTree(target.means).query(
        moved, k=min(config.reassociation_neighbours, len(target.means)), workers=-1
    )
    if distance.ndim == 1:
        distance, target_index = distance[:, None], target_index[:, None]
    rotated = rotate_sh(source.sh, pose.rotation)
    agreement = np.zeros_like(distance, dtype=np.float64)
    for weight, block in zip(degree_weights, _SH_BLOCKS):
        first = np.repeat(rotated[:, None, block, :], distance.shape[1], axis=1)
        second = target.sh[target_index][:, :, block, :]
        agreement += weight * _row_cosine(
            first.reshape(-1, block.stop - block.start, 3),
            second.reshape(-1, block.stop - block.start, 3),
        ).reshape(distance.shape)
    agreement[distance >= config.inlier_fraction * extent] = -np.inf
    source_index = np.arange(len(moved))
    rank = np.argmax(agreement, axis=1)
    chosen_target = target_index[source_index, rank]
    chosen_score = agreement[source_index, rank]
    chosen_distance = distance[source_index, rank]
    keep = np.isfinite(chosen_score) & (chosen_score > 0)
    source_index, chosen_target = source_index[keep], chosen_target[keep]
    chosen_score, chosen_distance = chosen_score[keep], chosen_distance[keep]
    if len(source_index) < config.minimum_support:
        return source_index, chosen_target, chosen_score
    order = np.lexsort((chosen_distance, -chosen_score, chosen_target))
    first = np.r_[True, chosen_target[order][1:] != chosen_target[order][:-1]]
    chosen = order[first]
    source_index, chosen_target, chosen_score = (
        source_index[chosen], chosen_target[chosen], chosen_score[chosen]
    )
    keep = chosen_score >= np.median(chosen_score)
    return source_index[keep], chosen_target[keep], chosen_score[keep]


def _reassociate(
    pose: Sim3,
    source: GaussianMap,
    target: GaussianMap,
    extent: float,
    degree_weights: np.ndarray,
    config: RegistrationConfig,
) -> tuple[Sim3, int]:
    current, pair_count = pose, 0
    for _ in range(config.reassociation_iterations):
        source_index, target_index, score = _local_pairs(
            current, source, target, extent, degree_weights, config
        )
        pair_count = len(source_index)
        if pair_count < config.minimum_support:
            break
        weights = np.maximum(score - np.median(score) + 1e-3, 1e-3)
        current = weighted_umeyama(source.means[source_index], target.means[target_index], weights)
    return current, pair_count


def transform_map(source: GaussianMap, pose: Sim3) -> GaussianMap:
    means = pose.scale * (source.means @ pose.rotation.T) + pose.translation
    covariances = pose.scale**2 * np.einsum(
        "ij,njk,lk->nil", pose.rotation, source.covariances, pose.rotation
    )
    return GaussianMap(
        means, covariances, source.colors.copy(), source.opacities.copy(),
        rotate_sh(source.sh, pose.rotation),
    )


def register(
    source: GaussianMap,
    target: GaussianMap,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    """Estimate a source-to-target Sim(3) and return the aligned Gaussian map."""
    config = config or RegistrationConfig()
    source, target = source.validated(), target.validated()
    source_descriptor = distribution_descriptors(source, config.neighbourhoods, config.quantiles)
    target_descriptor = distribution_descriptors(target, config.neighbourhoods, config.quantiles)
    source_ids, target_ids, distances = reciprocal_matches(
        source_descriptor, target_descriptor, config.retrieval_candidates
    )
    if len(source_ids) < 3:
        raise RuntimeError("fewer than three reciprocal proposal matches")
    confidence = 1.0 / np.maximum(distances, 1e-6)
    extent = robust_extent(target.means)
    candidates = _generate_hypotheses(
        source.means[source_ids], target.means[target_ids], confidence, extent, config
    )
    diagnostics = [_score_candidate(item, source, target, extent, config) for item in candidates]
    support_floor = max(3, int(np.ceil(config.support_ratio * candidates[0].inliers)))
    eligible = [index for index, item in enumerate(candidates) if item.inliers >= support_floor]
    selected_index = max(eligible, key=lambda index: diagnostics[index]["radiance_score"])
    initial = candidates[selected_index].transform
    initial_diagnostics = diagnostics[selected_index]
    degree_weights = np.asarray(initial_diagnostics["degree_weights"])
    refined, local_count = _reassociate(initial, source, target, extent, degree_weights, config)
    refined_diagnostics = _score_candidate(
        Candidate(refined, local_count, float(local_count)), source, target, extent, config
    )
    accepted = bool(
        local_count >= config.support_retention * initial_diagnostics["spatial_pairs"]
        and refined_diagnostics["radiance_score"] > initial_diagnostics["radiance_score"]
    )
    pose = refined if accepted else initial
    return RegistrationResult(
        transform=pose,
        aligned_source=transform_map(source, pose),
        reciprocal_matches=len(source_ids),
        candidate_count=len(candidates),
        eligible_candidates=len(eligible),
        selected_candidate=selected_index,
        accepted_refinement=accepted,
        local_pairs=local_count,
        degree_scores=tuple(initial_diagnostics["degree_scores"]),
        degree_weights=tuple(initial_diagnostics["degree_weights"]),
    )
