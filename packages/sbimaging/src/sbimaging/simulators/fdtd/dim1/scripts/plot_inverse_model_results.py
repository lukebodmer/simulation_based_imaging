#!/usr/bin/env python
"""Grid visualization comparing ground truth inclusions with neural network predictions."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

from sbimaging.inverse_models.dim1 import DataLoader1D, NeuralNetwork1D
from sbimaging.inverse_models.dim1.data import params_to_density_profile

NORD_BG = "#434c5e"
NORD_FG = "#eceff4"
NORD_BLUE = "#5e81ac"
NORD_CYAN = "#88c0d0"
GRID_SIZE = 100


def sigmoid_alpha(density: float, midpoint: float = 1.5, steepness: float = 4.0) -> float:
    """Compute opacity using a sigmoid function.

    Args:
        density: Density value (1.0 = background, higher = inclusion).
        midpoint: Density value where alpha = 0.5.
        steepness: How sharply the sigmoid transitions.

    Returns:
        Alpha value in [0, 1].
    """
    # Sigmoid centered at midpoint
    x = (density - midpoint) * steepness
    alpha = 1.0 / (1.0 + np.exp(-x))
    return float(alpha)


def draw_density_profile(ax, profile, domain_size, grayscale=False):
    """Draw density profile as a colored horizontal bar."""
    tube_height = 0.3
    tube_bottom = -tube_height / 2

    ax.set_xlim(0, domain_size)
    ax.set_ylim(-0.25, 0.25)

    # Draw tube interior (white background with black outline)
    tube_bg = Rectangle(
        (0, tube_bottom), domain_size, tube_height,
        facecolor="white", edgecolor="black", linewidth=1.5
    )
    ax.add_patch(tube_bg)

    # Draw density values as colored segments
    x = np.linspace(0, domain_size, len(profile), endpoint=False)
    dx = domain_size / len(profile)

    for i, density in enumerate(profile):
        alpha = sigmoid_alpha(density, midpoint=2.3, steepness=10.0)
        if alpha > 0.05:  # Only draw if visible
            color = "#555555" if grayscale else NORD_BLUE
            rect = Rectangle(
                (x[i], tube_bottom), dx, tube_height,
                facecolor=color, alpha=alpha,
                edgecolor="none"
            )
            ax.add_patch(rect)


    ax.set_facecolor(NORD_BG)
    ax.axis("off")


def main():
    batch_dir = Path("/data/1d-simulations")
    model_path = Path("models/1d_inverse_model.pkl")
    params_dir = batch_dir / "parameters"
    sims_dir = batch_dir / "simulations"

    # Load model
    print("Loading model...")
    model = NeuralNetwork1D()
    model.load(model_path)

    # Get test indices from the model
    test_ids = model.test_indices
    if not test_ids:
        print("No test indices found in model. Using first 25 simulations.")
        sim_dirs = sorted(sims_dir.glob("sim_*"))
        test_ids = [d.name for d in sim_dirs if (d / "sensor_data.npy").exists()][:25]

    print(f"Found {len(test_ids)} test samples")

    # Load batch config
    with open(batch_dir / "batch_config.json") as f:
        config = json.load(f)

    background_density = config.get("background_density", 1.0)

    # Prepare data loader for preprocessing parameters
    loader = DataLoader1D(batch_dir, grid_size=GRID_SIZE, trim_timesteps=500, downsample_factor=4)

    # Load test data and make predictions
    test_params = []
    predictions = []
    ground_truths = []

    for sim_id in test_ids:
        # Load parameters
        param_file = params_dir / f"{sim_id}.json"
        with open(param_file) as f:
            p = json.load(f)
        test_params.append(p)

        # Load and preprocess sensor data
        sensor_file = sims_dir / sim_id / "sensor_data.npy"
        sensor_data = loader._load_sensor_data(sensor_file)
        X = sensor_data.reshape(1, -1)

        # Predict density profile
        pred_profile = model.predict(X)[0]
        predictions.append(pred_profile)

        # Ground truth profile
        true_profile = params_to_density_profile(
            inclusion_center=p["inclusion_center"],
            inclusion_size=p["inclusion_size"],
            inclusion_density=p["inclusion_density"],
            domain_size=p["domain_size"],
            grid_size=GRID_SIZE,
            background_density=background_density,
        )
        ground_truths.append(true_profile)

    # Create plots with 12 samples per page (4 rows x 3 cols, each with ground truth above prediction)
    from matplotlib.gridspec import GridSpec

    samples_per_page = 12
    n_pages = (len(test_ids) + samples_per_page - 1) // samples_per_page

    for page in range(n_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(test_ids))
        page_count = end_idx - start_idx

        # 4 rows x 3 cols layout, with ground truth and prediction stacked vertically
        rows = 4
        cols = 3

        fig = plt.figure(figsize=(6 * cols, 2.8 * rows))
        fig.set_facecolor(NORD_BG)

        # Create outer grid for sample cells with larger spacing
        outer_grid = GridSpec(rows, cols, figure=fig, wspace=0.1, hspace=0.15,
                              left=0.01, right=0.99, top=0.98, bottom=0.02)

        for i in range(page_count):
            idx = start_idx + i
            row = i // cols
            col = i % cols

            p = test_params[idx]
            domain_size = p["domain_size"]

            # Create inner grid for truth/prediction pair with minimal spacing
            inner_grid = outer_grid[row, col].subgridspec(2, 1, hspace=0.0)

            # Ground truth (top)
            ax_true = fig.add_subplot(inner_grid[0])
            draw_density_profile(ax_true, ground_truths[idx], domain_size)

            # Prediction (bottom, grayscale)
            ax_pred = fig.add_subplot(inner_grid[1])
            draw_density_profile(ax_pred, predictions[idx], domain_size, grayscale=True)

        output_path = batch_dir / f"inverse_model_results_{page + 1:02d}.png"
        plt.savefig(output_path, dpi=300, facecolor=NORD_BG)
        print(f"Saved to {output_path}")
        plt.close(fig)

    print(f"Generated {n_pages} plots")


if __name__ == "__main__":
    main()
