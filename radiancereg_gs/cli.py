"""Command-line interface for RadianceReg-GS."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import RegistrationConfig, register
from .io import load_gaussian_map, load_pair_npz, save_registration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Globally register two independently reconstructed 3D Gaussian maps"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pair", type=Path, help="NPZ containing source_* and target_* arrays")
    inputs.add_argument("--source", type=Path, help="source .ply or .npz Gaussian map")
    parser.add_argument("--target", type=Path, help="target .ply or .npz; required with --source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--hypotheses", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pair:
        source, target = load_pair_npz(args.pair)
    else:
        if args.target is None:
            raise SystemExit("--target is required when --source is used")
        source = load_gaussian_map(args.source)
        target = load_gaussian_map(args.target)
    config = RegistrationConfig(
        proposal_iterations=args.iterations,
        maximum_hypotheses=args.hypotheses,
        seed=args.seed,
    )
    result = register(source, target, config)
    save_registration(result, args.output)
    print(args.output / "result.json")


if __name__ == "__main__":
    main()
