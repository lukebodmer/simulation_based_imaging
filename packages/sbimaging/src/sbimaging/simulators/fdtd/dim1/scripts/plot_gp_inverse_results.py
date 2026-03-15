#!/usr/bin/env python
"""Quantitative visualization comparing GP and NN inverse model predictions.

Plots density profiles as line graphs comparing ground truth with both
Gaussian Process and Neural Network predictions. GP predictions include
credible intervals to show prediction uncertainty.

Designed for publication in academic papers/dissertations.
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
COLOR_SCATTER = "#009E73"  # Green for scatter (colorblind-safe)

GRID_SIZE = 100


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            # Font settings - keep small
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
            # Legend - compact
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
    """Plot ground truth vs GP and NN predictions.

    Args:
        ax: Matplotlib axes to plot on.
        x: Position array.
        true_profile: Ground truth density values.
        gp_mean: GP predicted mean density.
        gp_lower: Lower bound of GP credible interval.
        gp_upper: Upper bound of GP credible interval.
        nn_pred: Neural network predicted density.
        confidence_label: Label for confidence interval.
    """
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


def compute_metrics(
    true_profile: np.ndarray,
    pred_mean: np.ndarray,
    pred_lower: np.ndarray | None = None,
    pred_upper: np.ndarray | None = None,
) -> dict:
    """Compute quantitative metrics for prediction quality.

    Args:
        true_profile: Ground truth density values.
        pred_mean: Predicted mean density.
        pred_lower: Lower bound of credible interval (optional).
        pred_upper: Upper bound of credible interval (optional).

    Returns:
        Dictionary with MSE, MAE, and optionally coverage and interval width.
    """
    mse = float(np.mean((pred_mean - true_profile) ** 2))
    mae = float(np.mean(np.abs(pred_mean - true_profile)))

    result = {"mse": mse, "mae": mae}

    if pred_lower is not None and pred_upper is not None:
        in_interval = (true_profile >= pred_lower) & (true_profile <= pred_upper)
        result["coverage"] = float(np.mean(in_interval))
        result["interval_width"] = float(np.mean(pred_upper - pred_lower))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Plot GP and NN inverse model results with confidence intervals"
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: batch_dir)",
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
        help="Confidence level for GP credible intervals (default 0.90)",
    )
    parser.add_argument(
        "--samples-per-page",
        type=int,
        default=6,
        help="Number of samples per output page",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "eps", "svg"],
        help="Output format (default: pdf for LaTeX)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats (default: 300)",
    )

    args = parser.parse_args()

    # Setup publication style
    setup_publication_style()

    batch_dir = Path(args.batch_dir)
    gp_model_path = Path(args.gp_model_path)
    nn_model_path = Path(args.nn_model_path)
    output_dir = Path(args.output_dir) if args.output_dir else batch_dir

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
        print(f"GP model not found at {gp_model_path}, skipping GP predictions")

    if nn_model_path.exists():
        print(f"Loading NN model from {nn_model_path}...")
        nn_model = NeuralNetwork1D()
        nn_model.load(nn_model_path)
    else:
        print(f"NN model not found at {nn_model_path}, skipping NN predictions")

    if gp_model is None and nn_model is None:
        print("Error: No models found. Please provide at least one model.")
        return

    # Determine test IDs from whichever model is available
    if gp_model is not None and gp_model.test_indices:
        test_ids = gp_model.test_indices
    elif nn_model is not None and nn_model.test_indices:
        test_ids = nn_model.test_indices
    else:
        print("No test indices found in models. Using first 25 simulations.")
        sim_dirs = sorted(sims_dir.glob("sim_*"))
        test_ids = [d.name for d in sim_dirs if (d / "sensor_data.npy").exists()][:25]

    print(f"Found {len(test_ids)} test samples")

    # Load batch config
    with open(batch_dir / "batch_config.json") as f:
        config = json.load(f)

    background_density = config.get("background_density", 1.0)

    # Prepare data loader for preprocessing
    loader = DataLoader1D(
        batch_dir,
        grid_size=args.grid_size,
        trim_timesteps=args.trim_timesteps,
        downsample_factor=args.downsample_factor,
    )

    # Load test data and make predictions
    test_params = []
    gp_predictions = []
    nn_predictions = []
    ground_truths = []
    gp_metrics = []
    nn_metrics = []

    confidence_label = f"{int(args.confidence * 100)}% CI"

    for sim_id in test_ids:
        param_file = params_dir / f"{sim_id}.json"
        with open(param_file) as f:
            p = json.load(f)
        test_params.append(p)

        # Load and preprocess sensor data
        sensor_file = sims_dir / sim_id / "sensor_data.npy"
        sensor_data = loader._load_sensor_data(sensor_file)
        X = sensor_data.reshape(1, -1)

        # GP prediction with uncertainty
        if gp_model is not None:
            pred_result = gp_model.predict_with_uncertainty(
                X, confidence=args.confidence
            )
            gp_predictions.append(
                {
                    "mean": pred_result["mean"][0],
                    "lower": pred_result["lower"][0],
                    "upper": pred_result["upper"][0],
                    "std": pred_result["std"][0],
                }
            )
        else:
            gp_predictions.append(None)

        # NN prediction
        if nn_model is not None:
            nn_pred = nn_model.predict(X)[0]
            nn_predictions.append(nn_pred)
        else:
            nn_predictions.append(None)

        # Ground truth profile
        true_profile = params_to_density_profile(
            inclusion_center=p["inclusion_center"],
            inclusion_size=p["inclusion_size"],
            inclusion_density=p["inclusion_density"],
            domain_size=p["domain_size"],
            grid_size=args.grid_size,
            background_density=background_density,
        )
        ground_truths.append(true_profile)

        # Compute metrics
        if gp_predictions[-1] is not None:
            gp_m = compute_metrics(
                true_profile,
                gp_predictions[-1]["mean"],
                gp_predictions[-1]["lower"],
                gp_predictions[-1]["upper"],
            )
            gp_metrics.append(gp_m)
        else:
            gp_metrics.append(None)

        if nn_predictions[-1] is not None:
            nn_m = compute_metrics(true_profile, nn_predictions[-1])
            nn_metrics.append(nn_m)
        else:
            nn_metrics.append(None)

    # Print aggregate metrics
    print(f"\nAggregate Metrics ({len(test_ids)} test samples):")

    avg_gp_mse = avg_gp_mae = avg_gp_coverage = avg_gp_width = 0.0
    avg_nn_mse = avg_nn_mae = 0.0

    if gp_model is not None:
        valid_gp = [m for m in gp_metrics if m is not None]
        avg_gp_mse = np.mean([m["mse"] for m in valid_gp])
        avg_gp_mae = np.mean([m["mae"] for m in valid_gp])
        avg_gp_coverage = np.mean([m["coverage"] for m in valid_gp])
        avg_gp_width = np.mean([m["interval_width"] for m in valid_gp])
        print("\n  Gaussian Process:")
        print(f"    Mean MSE:          {avg_gp_mse:.6e}")
        print(f"    Mean MAE:          {avg_gp_mae:.6e}")
        print(f"    Coverage ({confidence_label}): {avg_gp_coverage:.2%}")
        print(f"    Mean CI Width:     {avg_gp_width:.4f}")

    if nn_model is not None:
        valid_nn = [m for m in nn_metrics if m is not None]
        avg_nn_mse = np.mean([m["mse"] for m in valid_nn])
        avg_nn_mae = np.mean([m["mae"] for m in valid_nn])
        print("\n  Neural Network:")
        print(f"    Mean MSE:          {avg_nn_mse:.6e}")
        print(f"    Mean MAE:          {avg_nn_mae:.6e}")

    # Create comparison plots
    samples_per_page = args.samples_per_page
    n_pages = (len(test_ids) + samples_per_page - 1) // samples_per_page

    cols = 3
    rows = (samples_per_page + cols - 1) // cols

    for page in range(n_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(test_ids))
        page_count = end_idx - start_idx

        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))

        if rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        for i in range(page_count):
            idx = start_idx + i
            p = test_params[idx]
            gp_pred = gp_predictions[idx]
            nn_pred = nn_predictions[idx]
            true_prof = ground_truths[idx]

            x = np.linspace(0, p["domain_size"], args.grid_size)

            plot_density_comparison(
                axes_flat[i],
                x,
                true_prof,
                gp_mean=gp_pred["mean"] if gp_pred else None,
                gp_lower=gp_pred["lower"] if gp_pred else None,
                gp_upper=gp_pred["upper"] if gp_pred else None,
                nn_pred=nn_pred,
                confidence_label=confidence_label,
            )

        for i in range(page_count, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()

        output_path = (
            output_dir / f"inverse_model_comparison_{page + 1:02d}.{args.format}"
        )
        plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {output_path}")
        plt.close(fig)

    # Create summary statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # MSE comparison histogram
    ax = axes[0, 0]
    if gp_model is not None:
        gp_mse_values = [m["mse"] for m in gp_metrics if m is not None]
        ax.hist(
            gp_mse_values,
            bins=15,
            color=COLOR_GP,
            alpha=0.6,
            label=f"GP (mean: {avg_gp_mse:.2e})",
            edgecolor="white",
            linewidth=0.5,
        )
    if nn_model is not None:
        nn_mse_values = [m["mse"] for m in nn_metrics if m is not None]
        ax.hist(
            nn_mse_values,
            bins=15,
            color=COLOR_NN,
            alpha=0.6,
            label=f"NN (mean: {avg_nn_mse:.2e})",
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.set_title("MSE Distribution")
    ax.legend()

    # MAE comparison histogram
    ax = axes[0, 1]
    if gp_model is not None:
        gp_mae_values = [m["mae"] for m in gp_metrics if m is not None]
        ax.hist(
            gp_mae_values,
            bins=15,
            color=COLOR_GP,
            alpha=0.6,
            label=f"GP (mean: {avg_gp_mae:.2e})",
            edgecolor="white",
            linewidth=0.5,
        )
    if nn_model is not None:
        nn_mae_values = [m["mae"] for m in nn_metrics if m is not None]
        ax.hist(
            nn_mae_values,
            bins=15,
            color=COLOR_NN,
            alpha=0.6,
            label=f"NN (mean: {avg_nn_mae:.2e})",
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xlabel("MAE")
    ax.set_ylabel("Count")
    ax.set_title("MAE Distribution")
    ax.legend()

    # GP Coverage histogram
    ax = axes[1, 0]
    if gp_model is not None:
        coverage_values = [m["coverage"] for m in gp_metrics if m is not None]
        ax.hist(
            coverage_values,
            bins=15,
            color=COLOR_GP,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(
            args.confidence,
            color=COLOR_NN,
            linestyle="--",
            linewidth=1.5,
            label=f"Target: {args.confidence:.0%}",
        )
        ax.axvline(
            avg_gp_coverage,
            color=COLOR_GROUND_TRUTH,
            linestyle="-",
            linewidth=1.5,
            label=f"Mean: {avg_gp_coverage:.0%}",
        )
        ax.legend()
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Count")
    ax.set_title(f"GP {confidence_label} Coverage")

    # MSE scatter: GP vs NN
    ax = axes[1, 1]
    if gp_model is not None and nn_model is not None:
        gp_mse_scatter = []
        nn_mse_scatter = []
        for gp_m, nn_m in zip(gp_metrics, nn_metrics, strict=True):
            if gp_m is not None and nn_m is not None:
                gp_mse_scatter.append(gp_m["mse"])
                nn_mse_scatter.append(nn_m["mse"])

        ax.scatter(
            gp_mse_scatter,
            nn_mse_scatter,
            c=COLOR_SCATTER,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
            s=30,
        )

        # Add diagonal line (equal performance)
        max_mse = max(max(gp_mse_scatter), max(nn_mse_scatter))
        ax.plot(
            [0, max_mse],
            [0, max_mse],
            color="#888888",
            linestyle="--",
            linewidth=1,
            label="Equal performance",
        )

        ax.set_xlabel("GP MSE")
        ax.set_ylabel("NN MSE")
        ax.set_title("GP vs NN (below line = GP better)")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Need both models\nfor comparison",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_title("GP vs NN Comparison")

    plt.tight_layout()
    summary_path = output_dir / f"inverse_model_summary.{args.format}"
    plt.savefig(summary_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved summary to {summary_path}")
    plt.close(fig)

    print(f"\nGenerated {n_pages} comparison plots and 1 summary plot")


if __name__ == "__main__":
    main()
