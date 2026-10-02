# RadianceReg-GS

<p align="center">
  <a href="https://dancelycs.github.io/RadianceReg-GS/"><strong>🌐 Complete Interactive Results Gallery</strong></a>
</p>

<p align="center">
  View all ScanNet-GSReg, Objaverse, and ETH3D registration cases directly.
</p>

**RadianceReg-GS: Global Registration of 3D Gaussian Maps via Support-Constrained Spherical-Harmonic Radiance**.

RadianceReg-GS registers two independently reconstructed 3D Gaussian Splatting maps without original images, camera parameters, semantic annotations, or an initial pose. Geometry and degree-zero color generate a diverse Sim(3) candidate bank. Each candidate then rotates the source degree-1-3 spherical-harmonic coefficients, establishes spatial support, and receives an observable-energy-weighted radiance score. The selected pose constrains local radiance reassociation before coherent transport of centers, covariances, and directional radiance.


## Installation

Python 3.10 or newer is recommended. The implementation is CPU compatible; a CUDA-enabled PyTorch installation can accelerate tensor operations but is not required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[evaluation,test]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

The spherical-harmonic action is implemented with `e3nn`; install the PyTorch build appropriate for your platform before the editable installation if a specific CUDA version is required.

## Register two Gaussian maps

For two standard degree-3 3DGS PLY files:

```bash
radiancereg-gs \
  --source path/to/source/point_cloud.ply \
  --target path/to/target/point_cloud.ply \
  --output outputs/example
```

For a compact pair NPZ:

```bash
radiancereg-gs --pair path/to/pair.npz --output outputs/example
```

The output contains:

- `result.json`: estimated scale, rotation, translation, candidate diagnostics, and refinement status;
- `aligned_source.npz`: aligned centers, covariances, degree-zero colors, opacities, and rotated degree-1--3 SH coefficients.

The source-to-target convention is

```text
x_target = scale * rotation @ x_source + translation
```

### NPZ schema

A map NPZ stores `means [N,3]`, `covariances [N,3,3]`, `colors [N,3]`, `opacities [N]`, and `sh [N,15,3]`. A pair file uses the same keys with `source_` and `target_` prefixes. Evaluation pair files may additionally contain `gt_scale`, `gt_rotation`, and `gt_translation`; the evaluation script opens these fields only after inference returns a pose.

Create and register a deterministic example with:

```bash
python examples/make_synthetic_pair.py
radiancereg-gs --pair examples/synthetic_pair.npz --output outputs/synthetic
```

## Browse all released results

### [Open the complete interactive result gallery →](https://dancelycs.github.io/RadianceReg-GS/)

The gallery contains every evaluated ScanNet-GSReg, Objaverse, and ETH3D case, together with per-method RRE, RTE, RSE, runtime, success status, and available alignment visualizations. For offline inspection, clone the repository and open [`results_gallery/index.html`](results_gallery/index.html); no local server is required.

## Reproduce tables, statistics, and plots

```bash
python reproduce.py
```

The command reads the released per-case estimates under `data/` and recreates:

- dataset-level CSV and Markdown tables;
- cross-dataset recall and threshold-robustness figures;
- Wilson confidence intervals and paired exact McNemar tests with Holm correction;
- a validation report covering per-method case counts and gallery assets.

Outputs are written to `reproduced/`. The stored predictions, rather than manuscript values, are the source of every reproduced statistic. Invalid estimates remain in the recall denominator, while medians use valid estimates only.

The released predictions reproduce the following headline counts:

| Dataset | Cases | RadianceReg-GS strict/relaxed | Strongest public strict/relaxed result |
|---|---:|---:|---:|
| ScanNet-GSReg | 24 | 10/17 | GeoTransformer and BUFFER-X: 0/1 |
| Objaverse | 13 | 12/12 | GeoTransformer: 1/1 |
| ETH3D | 19 | 16/17 | CoFiNet: 11/15; BUFFER-X: 9/15 |

Strict success requires RRE/RTE/RSE below `10 deg / 0.10 / 10%`; relaxed success uses `15 deg / 0.30 / 20%`. Every attempted case remains in the recall denominator.

## Evaluate prepared benchmark pairs

```bash
python scripts/evaluate_pairs.py \
  --data /path/to/pair_npz_directory \
  --output outputs/benchmark \
  --save-aligned
```

Raw public datasets and third-party model weights are not redistributed. Prepared pairs must follow the NPZ schema above and retain native degree-1--3 SH coefficients. Dataset-specific acquisition and licensing remain governed by ScanNet-GSReg, Objaverse/GSA, and ETH3D.

## Repository layout

```text
radiancereg_gs/             # final end-to-end method and I/O
examples/                   # deterministic quick-start example
scripts/                    # benchmark evaluation entry point
tests/                      # SH-action and end-to-end regression tests
data/                       # released per-case predictions and provenance
results_gallery/            # complete static quantitative/qualitative website
reproduce.py                # paper table, statistics, and plot reproduction
```

## License

The source code is released under the MIT License. Released JSON predictions and derived visualizations are provided for research verification. Public datasets and third-party methods remain subject to their original licenses.
