#!/usr/bin/env python
"""Simple grid visualization of inclusion domains from batch simulations."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from pathlib import Path

NORD_BG = "#2e3440"
NORD_ACCENT = "#88c0d0"  # Nord frost (cyan)


def main():
    batch_dir = Path("/data/2d-simulations")
    params_dir = batch_dir / "parameters"
    sims_dir = batch_dir / "simulations"

    # Load batch config as fallback for old parameter files
    with open(batch_dir / "batch_config.json") as f:
        config = json.load(f)

    # Get simulation IDs in same order as plot_sensor_grid (sorted sim_* dirs with sensor_data.npy)
    sim_dirs = sorted(sims_dir.glob("sim_*"))
    sim_ids = [d.name for d in sim_dirs if (d / "sensor_data.npy").exists()]

    # Load parameters for each simulation
    params = []
    for sim_id in sim_ids:
        param_file = params_dir / f"{sim_id}.json"
        with open(param_file) as f:
            p = json.load(f)
        # Use per-file values if available, otherwise fall back to batch config
        p.setdefault("inclusion_size", config["inclusion_size"])
        p.setdefault("domain_size", config["domain_size"])
        params.append(p)

    n = len(params)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.set_facecolor(NORD_BG)
    axes = np.atleast_2d(axes).flatten()

    for i, p in enumerate(params):
        ax = axes[i]
        ax.set_facecolor("white")

        domain_size = p["domain_size"]
        inclusion_size = p["inclusion_size"]

        ax.set_xlim(0, domain_size)
        ax.set_ylim(0, domain_size)
        ax.set_aspect("equal")

        cx, cy = p["center_x"], p["center_y"]
        inc_type = p["inclusion_type"]

        if inc_type == "circle":
            # inclusion_size is diameter (matches side length of square/triangle)
            patch = Circle(
                (cx, cy), inclusion_size / 2, fill=False,
                edgecolor="black", linewidth=2
            )
        elif inc_type == "square":
            half = inclusion_size / 2
            patch = Rectangle(
                (cx - half, cy - half), inclusion_size, inclusion_size,
                fill=False, edgecolor="black", linewidth=2
            )
        elif inc_type == "triangle":
            side = inclusion_size
            height = (np.sqrt(3) / 2) * side
            vertices = [
                (cx - side / 2, cy - height / 3),
                (cx + side / 2, cy - height / 3),
                (cx, cy + 2 * height / 3),
            ]
            patch = Polygon(
                vertices, closed=True, fill=False,
                edgecolor="black", linewidth=2
            )

        ax.add_patch(patch)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.grid(True, color="#e0e0e0", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused axes
    for i in range(len(params), len(axes)):
        axes[i].axis("off")
        axes[i].set_facecolor(NORD_BG)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
    plt.savefig(batch_dir / "inclusion_grid.png", dpi=300, facecolor=NORD_BG)
    plt.show()


if __name__ == "__main__":
    main()
