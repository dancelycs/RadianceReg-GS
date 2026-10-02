import numpy as np
from scipy.spatial.transform import Rotation

from radiancereg_gs import (
    GaussianMap,
    RegistrationConfig,
    Sim3,
    register,
    rotate_sh,
    transform_map,
)


def random_map(seed=5, count=180):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(count, 3)) * np.array([2.0, 1.2, 0.7])
    axes = np.exp(rng.normal(-2.1, 0.4, size=(count, 3)))
    frames = Rotation.random(count, random_state=rng).as_matrix()
    covariance = np.einsum("nij,nj,nkj->nik", frames, axes**2, frames)
    return GaussianMap(
        means,
        covariance,
        rng.uniform(0.05, 0.95, size=(count, 3)),
        rng.uniform(0.3, 1.0, size=count),
        rng.normal(scale=0.15, size=(count, 15, 3)),
    )


def test_sh_action_composes():
    source = random_map(count=32)
    first = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    second = Rotation.from_rotvec([-0.1, 0.4, 0.2]).as_matrix()
    sequential = rotate_sh(rotate_sh(source.sh, first), second)
    composed = rotate_sh(source.sh, second @ first)
    np.testing.assert_allclose(sequential, composed, atol=2e-5, rtol=2e-5)


def test_end_to_end_recovers_synthetic_transform():
    source = random_map(count=190)
    truth = Sim3(
        1.25,
        Rotation.from_rotvec([0.3, -0.45, 0.2]).as_matrix(),
        np.array([0.8, -0.5, 0.35]),
    )
    target = transform_map(source, truth)
    result = register(
        source,
        target,
        RegistrationConfig(
            proposal_iterations=300,
            maximum_hypotheses=16,
            minimum_support=12,
            seed=9,
        ),
    )
    np.testing.assert_allclose(result.transform.scale, truth.scale, atol=1e-7)
    np.testing.assert_allclose(result.transform.rotation, truth.rotation, atol=1e-7)
    np.testing.assert_allclose(result.transform.translation, truth.translation, atol=1e-7)
