#!/usr/bin/env python
"""Generate a single figure with selected inverse model comparison plots.

Creates a publication-ready figure with user-specified test samples arranged
in a customizable grid layout.
"""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from sbimaging.inverse_models.dim1 import (
    DataLoader1D,
    GaussianProcess1D,
    NeuralNetwork1D,
    params_to_density_profile,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Publication-ready color scheme
COLOR_GROUND_TRUTH = "#000000"  # Black for ground truth
COLOR_GP = "#0072B2"  # Blue for GP (colorblind-safe)
COLOR_NN = "#D55E00"  # Orange for NN (colorblind-safe)
COLOR_CI = "#0072B2"  # Same as GP for CI fill

GRID_SIZE = 100


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            # Font settings
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            # Figure settings
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            # Line settings
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.3,
            "lines.linewidth": 1.0,
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#cccccc",
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.4,
            "legend.borderpad": 0.3,
            "legend.labelspacing": 0.3,
            # LaTeX-compatible output
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )


def plot_density_comparison(
    ax: "Axes",
    x: np.ndarray,
    true_profile: np.ndarray,
    gp_mean: np.ndarray | None = None,
    gp_lower: np.ndarray | None = None,
    gp_upper: np.ndarray | None = None,
    nn_pred: np.ndarray | None = None,
    confidence_label: str = "90% CI",
) -> None:
    """Plot ground truth vs GP and NN predictions."""
    # Plot GP prediction with confidence interval first (so it's behind)
    if gp_mean is not None:
        if gp_lower is not None and gp_upper is not None:
            ax.fill_between(
                x,
                gp_lower,
                gp_upper,
                alpha=0.2,
                color=COLOR_CI,
                label=confidence_label,
                linewidth=0,
            )
        ax.plot(
            x,
            gp_mean,
            color=COLOR_GP,
            linewidth=1.5,
            label="GP",
        )

    # Plot NN prediction
    if nn_pred is not None:
        ax.plot(
            x,
            nn_pred,
            color=COLOR_NN,
            linewidth=1.5,
            label="NN",
        )

    # Plot ground truth last (on top, dashed)
    ax.plot(
        x,
        true_profile,
        color=COLOR_GROUND_TRUTH,
        linewidth=1.5,
        linestyle="--",
        label="Ground Truth",
    )

    ax.set_xlabel("Position (m)")
    ax.set_ylabel(r"Density (kg/m$^3$)")
    ax.legend(loc="upper right")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single figure with selected inverse model plots"
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="/data/1d-simulations",
        help="Path to batch simulation directory",
    )
    parser.add_argument(
        "--gp-model-path",
        type=str,
        default="models/1d_gp_inverse_model.pkl",
        help="Path to trained GP model",
    )
    parser.add_argument(
        "--nn-model-path",
        type=str,
        default="models/1d_inverse_model.pkl",
        help="Path to trained NN model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: batch_dir/selected_comparisons.pdf)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        help="Density profile resolution",
    )
    parser.add_argument(
        "--trim-timesteps",
        type=int,
        default=500,
        help="Initial timesteps to trim from sensor data",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=4,
        help="Downsample factor for sensor data",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
        help="Confidence level for GP credible intervals",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="1,5,9,10,12,15,22,23",
        help="Comma-separated list of 1-indexed sample numbers to include",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of rows in the figure",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Number of columns in the figure",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "eps", "svg"],
        help="Output format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats",
    )

    args = parser.parse_args()

    # Parse sample indices (convert from 1-indexed to 0-indexed)
    sample_indices = [int(s.strip()) - 1 for s in args.samples.split(",")]
    print(f"Selected samples (1-indexed): {[i + 1 for i in sample_indices]}")

    # Setup publication style
    setup_publication_style()

    batch_dir = Path(args.batch_dir)
    gp_model_path = Path(args.gp_model_path)
    nn_model_path = Path(args.nn_model_path)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = batch_dir / f"selected_comparisons.{args.format}"

    params_dir = batch_dir / "parameters"
    sims_dir = batch_dir / "simulations"

    # Load models
    gp_model = None
    nn_model = None

    if gp_model_path.exists():
        print(f"Loading GP model from {gp_model_path}...")
        gp_model = GaussianProcess1D()
        gp_model.load(gp_model_path)
    else:
        print(f"GP model not found at {gp_model_path}")

    if nn_model_path.exists():
        print(f"Loading NN model from {nn_model_path}...")
        nn_model = NeuralNetwork1D()
        nn_model.load(nn_model_path)
    else:
        print(f"NN model not found at {nn_model_path}")

    if gp_model is None and nn_model is None:
        print("Error: No models found.")
        return

    # Get test IDs
    if gp_model is not None and gp_model.test_indices:
        test_ids = gp_model.test_indices
    elif nn_model is not None and nn_model.test_indices:
        test_ids = nn_model.test_indices
    else:
        print("No test indices found in models.")
        return

    print(f"Total test samples available: {len(test_ids)}")

    # Validate indices
    for idx in sample_indices:
        if idx < 0 or idx >= len(test_ids):
            print(f"Error: Sample {idx + 1} is out of range (1-{len(test_ids)})")
            return

    # Load batch config
    with open(batch_dir / "batch_config.json") as f:
        config = json.load(f)
    background_density = config.get("background_density", 1.0)

    # Prepare data loader
    loader = DataLoader1D(
        batch_dir,
        grid_size=args.grid_size,
        trim_timesteps=args.trim_timesteps,
        downsample_factor=args.downsample_factor,
    )

    confidence_label = f"{int(args.confidence * 100)}% CI"

    # Create figure
    fig, axes = plt.subplots(
        args.rows, args.cols, figsize=(7, 2.5 * args.rows)
    )
    axes_flat = np.asarray(axes).flatten()

    for plot_idx, sample_idx in enumerate(sample_indices):
        if plot_idx >= len(axes_flat):
            print(f"Warning: More samples than plot slots, skipping remaining")
            break

        sim_id = test_ids[sample_idx]
        print(f"  Plot {plot_idx + 1}: Sample {sample_idx + 1} ({sim_id})")

        # Load parameters
        param_file = params_dir / f"{sim_id}.json"
        with open(param_file) as f:
            p = json.load(f)

        # Load and preprocess sensor data
        sensor_file = sims_dir / sim_id / "sensor_data.npy"
        sensor_data = loader._load_sensor_data(sensor_file)
        X = sensor_data.reshape(1, -1)

        # GP prediction
        gp_pred = None
        if gp_model is not None:
            pred_result = gp_model.predict_with_uncertainty(X, confidence=args.confidence)
            gp_pred = {
                "mean": pred_result["mean"][0],
                "lower": pred_result["lower"][0],
                "upper": pred_result["upper"][0],
            }

        # NN prediction
        nn_pred = None
        if nn_model is not None:
            nn_pred = nn_model.predict(X)[0]

        # Ground truth
        true_profile = params_to_density_profile(
            inclusion_center=p["inclusion_center"],
            inclusion_size=p["inclusion_size"],
            inclusion_density=p["inclusion_density"],
            domain_size=p["domain_size"],
            grid_size=args.grid_size,
            background_density=background_density,
        )

        x = np.linspace(0, p["domain_size"], args.grid_size)

        plot_density_comparison(
            axes_flat[plot_idx],
            x,
            true_profile,
            gp_mean=gp_pred["mean"] if gp_pred else None,
            gp_lower=gp_pred["lower"] if gp_pred else None,
            gp_upper=gp_pred["upper"] if gp_pred else None,
            nn_pred=nn_pred,
            confidence_label=confidence_label,
        )

    # Hide unused subplots
    for i in range(len(sample_indices), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
