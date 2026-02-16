#!/usr/bin/env python
"""Grid visualization comparing ground truth inclusions with neural network predictions."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from pathlib import Path

from sbimaging.inverse_models.dim2 import (
    DataLoader2D,
    KSpace2D,
    NeuralNetwork2D,
    kspace_to_image,
    create_inclusion_image,
)

NORD_BG = "#2e3440"
GRID_SIZE = 64


def draw_inclusion_outline(ax, params):
    """Draw inclusion outline on axes."""
    cx, cy = params["center_x"], params["center_y"]
    inc_type = params["inclusion_type"]
    inclusion_size = params["inclusion_size"]

    if inc_type == "circle":
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


def setup_ax(ax):
    """Configure axes appearance."""
    ax.set_facecolor("white")
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.grid(True, color="#e0e0e0", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    batch_dir = Path("/data/2d-simulations")
    model_path = Path("models/2d_inverse_model.pkl")
    params_dir = batch_dir / "parameters"
    sims_dir = batch_dir / "simulations"

    # Load model
    print("Loading model...")
    model = NeuralNetwork2D()
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

    # Prepare data loader for preprocessing parameters
    loader = DataLoader2D(batch_dir, grid_size=GRID_SIZE)

    # Load test data and make predictions
    test_params = []
    predictions = []
    ground_truths = []

    for sim_id in test_ids:
        # Load parameters
        param_file = params_dir / f"{sim_id}.json"
        with open(param_file) as f:
            p = json.load(f)
        p.setdefault("inclusion_size", config["inclusion_size"])
        p.setdefault("domain_size", config["domain_size"])
        test_params.append(p)

        # Load and preprocess sensor data
        sensor_file = sims_dir / sim_id / "sensor_data.npy"
        sensor_data = loader._load_sensor_data(sensor_file)
        X = sensor_data.reshape(1, -1)

        # Predict k-space
        y_pred = model.predict(X)[0]
        kspace_pred = KSpace2D.from_flat(y_pred, GRID_SIZE)
        pred_image = kspace_to_image(kspace_pred)
        predictions.append(pred_image)

        # Ground truth image
        true_image = create_inclusion_image(
            inclusion_type=p["inclusion_type"],
            center_x=p["center_x"],
            center_y=p["center_y"],
            inclusion_size=p["inclusion_size"],
            domain_size=p["domain_size"],
            grid_size=GRID_SIZE,
        )
        ground_truths.append(true_image)

    # Create plots with 8 samples per page (4 rows x 2 cols, each with ground truth + prediction)
    samples_per_page = 8
    n_pages = (len(test_ids) + samples_per_page - 1) // samples_per_page

    for page in range(n_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(test_ids))
        page_count = end_idx - start_idx

        # 4 rows x 2 cols layout, with ground truth and prediction side by side
        rows = 4
        cols = 2

        fig, axes = plt.subplots(rows, cols * 2, figsize=(3 * cols * 2, 3 * rows))
        fig.set_facecolor(NORD_BG)
        axes = np.atleast_2d(axes)

        for i in range(page_count):
            idx = start_idx + i
            row = i // cols
            col = i % cols

            p = test_params[idx]
            domain_size = p["domain_size"]

            # Ground truth (left)
            ax_true = axes[row, col * 2]
            setup_ax(ax_true)
            ax_true.set_xlim(0, domain_size)
            ax_true.set_ylim(0, domain_size)
            ax_true.set_aspect("equal")
            draw_inclusion_outline(ax_true, p)

            # Prediction (right)
            ax_pred = axes[row, col * 2 + 1]
            setup_ax(ax_pred)
            ax_pred.imshow(
                predictions[idx],
                extent=[0, domain_size, 0, domain_size],
                origin="lower",
                cmap="gray_r",
                vmin=0,
                vmax=1,
            )
            ax_pred.set_xlim(0, domain_size)
            ax_pred.set_ylim(0, domain_size)
            ax_pred.set_aspect("equal")

        # Hide unused axes
        for i in range(page_count, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col * 2].axis("off")
            axes[row, col * 2].set_facecolor(NORD_BG)
            axes[row, col * 2 + 1].axis("off")
            axes[row, col * 2 + 1].set_facecolor(NORD_BG)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
        output_path = batch_dir / f"inverse_model_results_{page + 1:02d}.png"
        plt.savefig(output_path, dpi=300, facecolor=NORD_BG)
        print(f"Saved to {output_path}")
        plt.close(fig)

    print(f"Generated {n_pages} plots")


if __name__ == "__main__":
    main()
