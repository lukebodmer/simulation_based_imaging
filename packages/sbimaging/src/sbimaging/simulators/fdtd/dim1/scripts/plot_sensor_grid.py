#!/usr/bin/env python
"""Simple grid visualization of sensor data from 1D batch simulations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

NORD_BG = "#2e3440"
NORD_FG = "#eceff4"
NORD_RED = "#bf616a"
NORD_CYAN = "#88c0d0"


def main():
    batch_dir = Path("/data/1d-simulations")
    sims_dir = batch_dir / "simulations"

    # Load all sensor data
    sim_dirs = sorted(sims_dir.glob("sim_*"))
    data = []
    for sim_dir in sim_dirs:
        sensor_file = sim_dir / "sensor_data.npy"
        if sensor_file.exists():
            data.append(np.load(sensor_file))

    from matplotlib.gridspec import GridSpec

    n = len(data)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(3 * cols, 2.5 * rows))
    fig.set_facecolor(NORD_BG)

    # Create outer grid for simulation cells with larger spacing
    outer_grid = GridSpec(rows, cols, figure=fig, wspace=0.15, hspace=0.25,
                          left=0.02, right=0.98, top=0.98, bottom=0.02)

    # Find global y limits across all data
    ymax = max(np.abs(d).max() for d in data) * 1.1

    for i, d in enumerate(data):
        row = i // cols
        col = i % cols

        # Create inner grid for the two sensors with smaller spacing
        inner_grid = outer_grid[row, col].subgridspec(2, 1, hspace=0.05)

        ax_top = fig.add_subplot(inner_grid[0])
        ax_bottom = fig.add_subplot(inner_grid[1])

        for ax in [ax_top, ax_bottom]:
            ax.set_facecolor(NORD_BG)

        # d has shape (num_sensors, num_timesteps) - typically (2, N)
        num_timesteps = d.shape[1]
        t = np.arange(num_timesteps)

        # Plot left sensor (red) on top
        ax_top.plot(t, d[0], color=NORD_RED, linewidth=0.8)
        ax_top.set_xlim(0, num_timesteps)
        ax_top.set_ylim(-ymax, ymax)

        # Plot right sensor (cyan) on bottom
        ax_bottom.plot(t, d[1], color=NORD_CYAN, linewidth=0.8)
        ax_bottom.set_xlim(0, num_timesteps)
        ax_bottom.set_ylim(-ymax, ymax)

        # Style axes
        for ax in [ax_top, ax_bottom]:
            ax.tick_params(labelbottom=False, labelleft=False, length=0)
            for spine in ax.spines.values():
                spine.set_color(NORD_FG)
                spine.set_linewidth(0.5)

    plt.savefig(batch_dir / "sensor_grid.png", dpi=300, facecolor=NORD_BG)
    plt.show()


if __name__ == "__main__":
    main()
