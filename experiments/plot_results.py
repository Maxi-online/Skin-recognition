"""Publication figures from a completed run; no model selection or refitting."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "gabor_only": "Gabor", "glcm_only": "GLCM", "lbp": "LBP",
    "hybrid": "Hybrid, local ROIs", "hybrid_global": "Hybrid, whole face",
    "global": "Hybrid, whole face", "gabor_glcm": "Gabor + GLCM",
    "no_wpca": "Hybrid without weighted PCA", "no_multifold": "Hybrid without PCA/ICA",
}


def plot_run(run_dir):
    run_dir = Path(run_dir).resolve()
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if results["status"] != "completed":
        raise ValueError("Run must be completed")
    methods = results["methods"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, (method, result) in enumerate(methods.items()):
        label, color = LABELS.get(method, method), colors(index)
        with (run_dir / method / "roc_det.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        far = np.asarray([float(row["false_accept_rate"]) for row in rows])
        tar = np.asarray([float(row["true_accept_rate"]) for row in rows])
        axes[0].plot(far, tar, label=label, color=color)
        cmc = result["test_cmc"]
        axes[1].plot(np.arange(1, len(cmc)+1), cmc, marker="o", markersize=3, label=label, color=color)
        lo, hi = result["test_intervals"]["intervals"]["eer"]
        axes[2].hlines(index, 100*lo, 100*hi, color=color, linewidth=2)
        axes[2].plot(100*result["test"]["eer"], index, "o", color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=.8)
    axes[0].set(xlabel="False accept rate", ylabel="True accept rate", title="A. Test ROC", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set(xlabel="Identity rank", ylabel="Identification rate", title="B. Test CMC", ylim=(0, 1))
    axes[2].set_yticks(range(len(methods)), [LABELS.get(method, method) for method in methods])
    axes[2].set(xlabel="Equal error rate (%)", title="C. EER with identity-bootstrap 95% intervals")
    axes[2].invert_yaxis()
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(run_dir / "comparison.png", dpi=220)
    figure.savefig(run_dir / "comparison.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    plot_run(parser.parse_args().run_dir)

