#!/usr/bin/env python3
"""Recreate paper-facing tables, statistics, and compact plots from released results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GALLERY = ROOT / "results_gallery"
OUTPUT = ROOT / "reproduced"
STRICT = (10.0, 0.10, 10.0)
RELAXED = (15.0, 0.30, 20.0)

LABELS = {
    "icp": "ICP",
    "fpfh_ransac": "FPFH+RANSAC",
    "fcgf": "FCGF",
    "predator": "PREDATOR",
    "cofinet": "CoFiNet",
    "geotransformer": "GeoTransformer",
    "splatreg": "SplatReg",
    "parenet": "PARE-Net",
    "cast": "CAST",
    "bufferx": "BUFFER-X",
    "randpcr": "RandPCR",
    "refreshreg": "RefreshReg",
    "radiancereg_gs": "RadianceReg-GS (ours)",
}
ORDER = tuple(LABELS)


def read_json(path: Path):
    return json.loads(path.read_text())


def canonical_rows(dataset: str, key: str) -> list[dict]:
    payload = read_json(DATA / dataset / f"{key}.json")
    timing = {}
    timing_path = DATA / dataset / f"{key}_timing.json"
    if timing_path.exists():
        timing = {row["pair"]: row["elapsed_seconds"] for row in read_json(timing_path)["rows"]}
    rows = []
    for raw in payload["rows"]:
        if key == "radiancereg_gs":
            metric = raw["output"]
            row = {
                "pair": raw["pair"],
                "status": "ok",
                "elapsed_seconds": timing.get(raw["pair"]),
                **metric,
            }
        else:
            row = dict(raw)
            row.setdefault("status", "ok" if row.get("rre_deg") is not None else "failed")
        rows.append(row)
    return rows


def passes(row: dict, gate=RELAXED, multiplier=1.0) -> bool:
    return bool(
        row.get("status") == "ok"
        and row.get("rre_deg") is not None
        and float(row["rre_deg"]) < gate[0] * multiplier
        and float(row["rte_extent"]) < gate[1] * multiplier
        and float(row["rse_percent"]) < gate[2] * multiplier
    )


def summarize(dataset: str, key: str, rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "ok" and row.get("rre_deg") is not None]

    def median(field: str):
        values = [float(row[field]) for row in valid if row.get(field) is not None]
        return float(np.median(values)) if values else None

    return {
        "dataset": dataset,
        "key": key,
        "method": LABELS[key],
        "attempted": len(rows),
        "valid": len(valid),
        "strict": sum(passes(row, STRICT) for row in rows),
        "relaxed": sum(passes(row, RELAXED) for row in rows),
        "rre": median("rre_deg"),
        "rte": median("rte_extent"),
        "rse": median("rse_percent"),
        "time": median("elapsed_seconds"),
    }


def available_methods(dataset: str) -> list[str]:
    return [key for key in ORDER if (DATA / dataset / f"{key}.json").exists()]


def write_table(dataset: str, summaries: list[dict]) -> None:
    table_dir = OUTPUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fields = ("method", "attempted", "valid", "strict", "relaxed", "rre", "rte", "rse", "time")
    with (table_dir / f"{dataset}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "| Method | Valid | S/R | Median RRE | Median RTE | Median RSE | Median time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        time = "--" if row["time"] is None else f"{row['time']:.3f}"
        lines.append(
            f"| {row['method']} | {row['valid']}/{row['attempted']} | "
            f"{row['strict']}/{row['relaxed']} | {row['rre']:.2f} | {row['rte']:.3f} | "
            f"{row['rse']:.2f} | {time} |"
        )
    (table_dir / f"{dataset}.md").write_text("\n".join(lines) + "\n")


def wilson(successes: int, total: int, z=1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / denominator
    return [center - radius, center + radius]


def exact_mcnemar(ours: list[dict], other: list[dict], gate) -> dict:
    ours_map = {row["pair"]: passes(row, gate) for row in ours}
    other_map = {row["pair"]: passes(row, gate) for row in other}
    common = sorted(set(ours_map) & set(other_map))
    wins = sum(ours_map[pair] and not other_map[pair] for pair in common)
    losses = sum(other_map[pair] and not ours_map[pair] for pair in common)
    discordant = wins + losses
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
        probability = min(1.0, 2 * tail / 2**discordant)
    else:
        probability = 1.0
    return {"wins": wins, "losses": losses, "p_value": probability}


def statistics(groups: dict[str, dict[str, list[dict]]]) -> dict:
    output = {"wilson_95pct": {}, "paired_exact_mcnemar": {}}
    tests = []
    for dataset, methods in groups.items():
        ours = methods["radiancereg_gs"]
        output["wilson_95pct"][dataset] = {
            name: wilson(sum(passes(row, gate) for row in ours), len(ours))
            for name, gate in (("strict", STRICT), ("relaxed", RELAXED))
        }
        output["paired_exact_mcnemar"][dataset] = {}
        for key, rows in methods.items():
            if key == "radiancereg_gs":
                continue
            result = {}
            for name, gate in (("strict", STRICT), ("relaxed", RELAXED)):
                test = exact_mcnemar(ours, rows, gate)
                result[name] = test
                tests.append((dataset, key, name, test["p_value"]))
            output["paired_exact_mcnemar"][dataset][key] = result
    ranked = sorted(enumerate(tests), key=lambda item: item[1][3])
    adjusted = [0.0] * len(tests)
    running = 0.0
    for rank, (original, test) in enumerate(ranked):
        running = max(running, min(1.0, (len(tests) - rank) * test[3]))
        adjusted[original] = running
    for test, value in zip(tests, adjusted):
        dataset, key, name, _ = test
        output["paired_exact_mcnemar"][dataset][key][name]["holm_p_value"] = value
    return output


def plot_cross_dataset(summaries: dict[str, list[dict]]) -> None:
    selected = ("geotransformer", "splatreg", "cofinet", "bufferx", "radiancereg_gs")
    datasets = ("scannet", "objaverse", "eth3d")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x = np.arange(len(datasets))
    width = 0.15
    for offset, key in enumerate(selected):
        strict, relaxed = [], []
        for dataset in datasets:
            row = next((item for item in summaries[dataset] if item["key"] == key), None)
            strict.append(np.nan if row is None else 100 * row["strict"] / row["attempted"])
            relaxed.append(np.nan if row is None else 100 * row["relaxed"] / row["attempted"])
        axes[0].bar(x + (offset - 2) * width, strict, width, label=LABELS[key])
        axes[1].bar(x + (offset - 2) * width, relaxed, width, label=LABELS[key])
    for axis, title in zip(axes, ("Strict recall", "Relaxed recall")):
        axis.set_xticks(x, ("ScanNet-GSReg", "Objaverse", "ETH3D"))
        axis.set_ylabel("Recall (%)")
        axis.set_title(title)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "cross_dataset_recall.png", dpi=240)
    fig.savefig(figure_dir / "cross_dataset_recall.pdf")
    plt.close(fig)


def plot_thresholds(groups: dict[str, dict[str, list[dict]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    multipliers = np.linspace(0.5, 2.0, 31)
    keys = ("radiancereg_gs", "geotransformer", "splatreg", "cofinet", "bufferx")
    for axis, (dataset, methods) in zip(axes, groups.items()):
        for key in keys:
            if key not in methods:
                continue
            values = [100 * np.mean([passes(row, STRICT, value) for row in methods[key]])
                      for value in multipliers]
            axis.plot(multipliers, values, linewidth=2, label=LABELS[key])
        axis.axvline(1.0, color="#777777", linestyle="--", linewidth=1)
        axis.set_title({"scannet": "ScanNet-GSReg", "objaverse": "Objaverse", "eth3d": "ETH3D"}[dataset])
        axis.set_xlabel("Threshold multiplier")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Joint recall (%)")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "threshold_robustness.png", dpi=240)
    fig.savefig(figure_dir / "threshold_robustness.pdf")
    plt.close(fig)


def validate(groups: dict[str, dict[str, list[dict]]]) -> dict:
    manifest = read_json(GALLERY / "manifest.json")
    missing = [record["image"] for record in manifest["records"]
               if not (GALLERY / record["image"]).is_file()]
    expected = {
        dataset: sum(record["dataset"] == dataset for record in manifest["records"])
        for dataset in manifest["datasets"]
    }
    group_report = {}
    for dataset, methods in groups.items():
        counts = {key: len(rows) for key, rows in methods.items()}
        group_report[dataset] = {
            "expected": expected[dataset],
            "method_counts": counts,
            "complete": all(count == expected[dataset] for count in counts.values()),
        }
    return {
        "datasets": group_report,
        "gallery": {
            "records": len(manifest["records"]),
            "missing_images": missing,
            "complete": not missing,
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups, summaries = {}, {}
    for dataset in ("scannet", "objaverse", "eth3d"):
        groups[dataset] = {
            key: canonical_rows(dataset, key) for key in available_methods(dataset)
        }
        summaries[dataset] = [
            summarize(dataset, key, groups[dataset][key]) for key in available_methods(dataset)
        ]
        write_table(dataset, summaries[dataset])
    (OUTPUT / "statistics.json").write_text(json.dumps(statistics(groups), indent=2) + "\n")
    plot_cross_dataset(summaries)
    plot_thresholds(groups)
    report = validate(groups)
    (OUTPUT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(item["complete"] for item in report["datasets"].values()) or not report["gallery"]["complete"]:
        raise SystemExit("release validation failed; inspect reproduced/validation_report.json")
    print("Reproduction complete and validated.")
    print(f"Tables and plots: {OUTPUT}")
    print(f"Interactive gallery: {GALLERY / 'index.html'}")


if __name__ == "__main__":
    main()
