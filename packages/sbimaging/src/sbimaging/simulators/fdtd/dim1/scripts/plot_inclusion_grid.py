#!/usr/bin/env python
"""Simple grid visualization of inclusion domains from 1D batch simulations."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

NORD_BG = "#2e3440"
NORD_FG = "#eceff4"
NORD_BLUE = "#5e81ac"
NORD_CYAN = "#88c0d0"


def main():
    batch_dir = Path("/data/1d-simulations")
    params_dir = batch_dir / "parameters"
    sims_dir = batch_dir / "simulations"

    # Load batch config
    with open(batch_dir / "batch_config.json") as f:
        config = json.load(f)

    # Get simulation IDs (sorted sim_* dirs with sensor_data.npy)
    sim_dirs = sorted(sims_dir.glob("sim_*"))
    sim_ids = [d.name for d in sim_dirs if (d / "sensor_data.npy").exists()]

    # Load parameters for each simulation
    params = []
    for sim_id in sim_ids:
        param_file = params_dir / f"{sim_id}.json"
        with open(param_file) as f:
            p = json.load(f)
        params.append(p)

    n = len(params)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.set_facecolor(NORD_BG)
    axes = np.atleast_2d(axes).flatten()

    # Find min/max density for alpha scaling
    densities = [p["inclusion_density"] for p in params]
    min_density = config.get("min_inclusion_density", min(densities))
    max_density = config.get("max_inclusion_density", max(densities))

    for i, p in enumerate(params):
        ax = axes[i]
        ax.set_facecolor(NORD_BG)

        domain_size = p["domain_size"]
        inclusion_center = p["inclusion_center"]
        inclusion_size = p["inclusion_size"]
        inclusion_density = p["inclusion_density"]

        # Scale alpha based on density (higher density = more opaque)
        alpha = 0.3 + 0.5 * (inclusion_density - min_density) / (max_density - min_density)

        tube_height = 0.15
        tube_bottom = -tube_height / 2
        tube_top = tube_height / 2

        ax.set_xlim(0, domain_size)
        ax.set_ylim(-0.5, 0.5)  # Square plot with tube centered

        # Draw tube interior (white background)
        tube_bg = Rectangle(
            (0, tube_bottom), domain_size, tube_height,
            facecolor="white", edgecolor="none"
        )
        ax.add_patch(tube_bg)

        # Draw tube walls (top and bottom lines) in light color
        ax.axhline(tube_top, color=NORD_FG, linewidth=1.5)
        ax.axhline(tube_bottom, color=NORD_FG, linewidth=1.5)

        # Draw inclusion as a rectangle inside the tube
        x_min = inclusion_center - inclusion_size / 2
        rect = Rectangle(
            (x_min, tube_bottom), inclusion_size, tube_height,
            facecolor=NORD_BLUE, alpha=alpha,
            edgecolor=NORD_CYAN, linewidth=1.5
        )
        ax.add_patch(rect)

        ax.axis("off")

    # Hide unused axes
    for i in range(len(params), len(axes)):
        axes[i].axis("off")
        axes[i].set_facecolor(NORD_BG)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0.1)
    plt.savefig(batch_dir / "inclusion_grid.png", dpi=300, facecolor=NORD_BG)
    plt.show()


if __name__ == "__main__":
    main()
