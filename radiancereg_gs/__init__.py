"""Public inference API for RadianceReg-GS."""

from .core import (
    GaussianMap,
    RegistrationConfig,
    RegistrationResult,
    Sim3,
    register,
    rotate_sh,
    transform_map,
)
from .io import load_gaussian_map, load_pair_npz, save_registration

__all__ = [
    "GaussianMap",
    "RegistrationConfig",
    "RegistrationResult",
    "Sim3",
    "load_gaussian_map",
    "load_pair_npz",
    "register",
    "rotate_sh",
    "save_registration",
    "transform_map",
]
