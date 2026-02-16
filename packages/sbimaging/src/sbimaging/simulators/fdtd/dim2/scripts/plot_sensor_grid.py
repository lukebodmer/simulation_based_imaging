#!/usr/bin/env python
"""Simple grid visualization of sensor data from batch simulations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    batch_dir = Path("/data/2d-simulations/simulations")

    # Load all sensor data
    sim_dirs = sorted(batch_dir.glob("sim_*"))
    data = []
    for sim_dir in sim_dirs:
        sensor_file = sim_dir / "sensor_data.npy"
        if sensor_file.exists():
            data.append(np.load(sensor_file))

    n = len(data)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    nord_bg = "#2e3440"  # Nord polar night (darkest)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.set_facecolor(nord_bg)
    axes = np.atleast_2d(axes).flatten()

    vmax = max(np.abs(d).max() for d in data)

    for i, d in enumerate(data):
        axes[i].imshow(d, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax, interpolation="nearest")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].axis("off")

    # Hide unused axes
    for i in range(len(data), len(axes)):
        axes[i].axis("off")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0.05)
    plt.savefig("/data/2d-simulations/sensor_grid.png", dpi=300, facecolor=nord_bg)
    plt.show()


if __name__ == "__main__":
    main()
