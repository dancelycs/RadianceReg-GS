#!/usr/bin/env python3
"""Create a deterministic pair that exercises geometry and degree-1--3 radiance."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from radiancereg_gs import GaussianMap, Sim3, transform_map


def main() -> None:
    rng = np.random.default_rng(7)
    count = 240
    means = rng.normal(size=(count, 3)) * np.array([2.0, 1.1, 0.6])
    axes = np.exp(rng.normal(-2.2, 0.35, size=(count, 3)))
    frames = Rotation.random(count, random_state=rng).as_matrix()
    covariances = np.einsum("nij,nj,nkj->nik", frames, axes**2, frames)
    source = GaussianMap(
        means,
        covariances,
        rng.uniform(0.05, 0.95, size=(count, 3)),
        rng.uniform(0.3, 1.0, size=count),
        rng.normal(scale=0.12, size=(count, 15, 3)),
    )
    truth = Sim3(
        1.35,
        Rotation.from_rotvec([0.35, -0.50, 0.25]).as_matrix(),
        np.array([1.2, -0.7, 0.45]),
    )
    target = transform_map(source, truth)
    output = Path(__file__).with_name("synthetic_pair.npz")
    np.savez_compressed(
        output,
        source_means=source.means,
        source_covariances=source.covariances,
        source_colors=source.colors,
        source_opacities=source.opacities,
        source_sh=source.sh,
        target_means=target.means,
        target_covariances=target.covariances,
        target_colors=target.colors,
        target_opacities=target.opacities,
        target_sh=target.sh,
        gt_scale=truth.scale,
        gt_rotation=truth.rotation,
        gt_translation=truth.translation,
    )
    print(output)


if __name__ == "__main__":
    main()
