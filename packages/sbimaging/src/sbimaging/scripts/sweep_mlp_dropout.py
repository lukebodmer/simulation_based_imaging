#!/usr/bin/env python
"""Hyperparameter sweep for MLP dropout.

This script trains multiple MLP models with different dropout
configurations and produces comparison charts for the dissertation.

Usage:
    python -m sbimaging.scripts.sweep_mlp_dropout --batch <batch_name>

Example:
    python -m sbimaging.scripts.sweep_mlp_dropout --batch 3d_cubes_500
"""

import argparse
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tomli

from sbimaging.inverse_models.nn.network import NeuralNetworkModel
from sbimaging.logging import configure_logging, get_logger

DEFAULT_DATA_DIR = Path("/data/simulations")

logger = get_logger(__name__)


@dataclass
class SweepConfig:
    """Configuration for the hyperparameter sweep."""

    # Data preprocessing (fixed)
    trim_timesteps: int = 45
    downsample_factor: int = 4
    voxel_grid_size: int = 32
    use_kspace: bool = False

    # Training parameters (fixed)
    epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 0.0001
    test_fraction: float = 0.1
    early_stopping: bool = True
    early_stopping_patience: int = 50

    # MLP parameters (fixed based on previous sweep)
    hidden_layers: list[int] = field(
        default_factory=lambda: [8192, 4096, 2048, 1024]
    )

    # Sweep parameter
    dropout_options: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5]
    )


@dataclass
class SweepResult:
    """Result from a single training run."""

    dropout: float
    train_loss: float
    test_loss: float
    voxel_error: float
    epochs_completed: int
    training_time_seconds: float
    model_name: str


def process_sensor_data(
    sensor_file: Path,
    trim_timesteps: int,
    downsample_factor: int,
) -> np.ndarray:
    """Process sensor data with specified preprocessing."""
    with open(sensor_file, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "pressure" in data:
        sensor_data = data["pressure"]
    else:
        sensor_data = data

    if hasattr(sensor_data, "get"):
        sensor_data = sensor_data.get()

    sensor_data = np.asarray(sensor_data)

    if sensor_data.ndim == 2:
        sensor_data = sensor_data[:, trim_timesteps:]
        if downsample_factor > 1:
            sensor_data = sensor_data[:, ::downsample_factor]

    return sensor_data.flatten().astype(np.float32)


def process_config_to_voxels(config_file: Path, grid_size: int) -> np.ndarray:
    """Convert config to voxel representation (no k-space)."""
    with open(config_file, "rb") as f:
        config = tomli.load(f)

    mesh_cfg = config.get("mesh", {})
    material_cfg = config.get("material", {})

    cube_centers = mesh_cfg.get("cube_centers", [])
    cube_widths = mesh_cfg.get("cube_widths", [])
    density = material_cfg.get("inclusion_density", 2.0)

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    x = np.linspace(0, 1, grid_size, endpoint=False)
    y = np.linspace(0, 1, grid_size, endpoint=False)
    z = np.linspace(0, 1, grid_size, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    for center, width in zip(cube_centers, cube_widths, strict=False):
        if len(center) >= 3:
            cx, cy, cz = center[:3]
            hw = width / 2
            mask = (
                (np.abs(X - cx) <= hw)
                & (np.abs(Y - cy) <= hw)
                & (np.abs(Z - cz) <= hw)
            )
            grid[mask] = density

    return grid.flatten().astype(np.float32)


def generate_training_data(
    batch_dir: Path,
    config: SweepConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], int | None]:
    """Generate or load training data for the batch.

    Returns:
        Tuple of (X, y, sample_ids, num_sensors)
    """
    sims_dir = batch_dir / "simulations"

    X_list = []
    y_list = []
    sample_ids = []
    num_sensors = None

    for sim_dir in sorted(sims_dir.iterdir()):
        if not sim_dir.is_dir():
            continue

        sensor_file = sim_dir / "sensor_data.pkl"
        config_file = sim_dir / "config.toml"

        if not sensor_file.exists() or not config_file.exists():
            continue

        try:
            x = process_sensor_data(
                sensor_file,
                config.trim_timesteps,
                config.downsample_factor,
            )
            y = process_config_to_voxels(config_file, config.voxel_grid_size)

            # Infer num_sensors from first sample
            if num_sensors is None:
                with open(sensor_file, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "pressure" in data:
                    sensor_data = data["pressure"]
                else:
                    sensor_data = data
                if hasattr(sensor_data, "get"):
                    sensor_data = sensor_data.get()
                sensor_data = np.asarray(sensor_data)
                if sensor_data.ndim == 2:
                    num_sensors = sensor_data.shape[0]

            X_list.append(x)
            y_list.append(y)
            sample_ids.append(sim_dir.name)

        except Exception as e:
            logger.warning(f"Failed to process {sim_dir.name}: {e}")

    if not X_list:
        raise RuntimeError(f"No valid training data found in {sims_dir}")

    logger.info(f"Loaded {len(X_list)} samples")
    return np.stack(X_list), np.stack(y_list), sample_ids, num_sensors


def compute_voxel_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    grid_size: int,
) -> float:
    """Compute sum of absolute voxel errors, averaged across samples."""
    errors = []

    for i in range(len(predictions)):
        pred = predictions[i].reshape((grid_size, grid_size, grid_size))
        target = targets[i].reshape((grid_size, grid_size, grid_size))
        sample_error = np.sum(np.abs(pred - target))
        errors.append(sample_error)

    return float(np.mean(errors))


def train_single_config(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    dropout: float,
    config: SweepConfig,
    models_dir: Path,
) -> SweepResult:
    """Train a single model configuration and return results."""
    dropout_str = f"{dropout:.1f}".replace(".", "p")
    model_name = f"mlp_sweep_dropout_{dropout_str}"

    logger.info(f"Training {model_name} with dropout={dropout}")

    model = NeuralNetworkModel(
        name=model_name,
        architecture="mlp",
        mlp_hidden_layers=config.hidden_layers,
        mlp_dropout=dropout,
    )

    start_time = time.time()

    metrics = model.train(
        X,
        y,
        test_fraction=config.test_fraction,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        sample_ids=sample_ids,
        early_stopping=config.early_stopping,
        early_stopping_patience=config.early_stopping_patience,
    )

    training_time = time.time() - start_time

    # Compute voxel error on test set
    test_indices = [sample_ids.index(tid) for tid in model.test_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    predictions = model.predict(X_test)
    voxel_error = compute_voxel_error(predictions, y_test, config.voxel_grid_size)

    # Save the model
    model_data = {
        "model": model,
        "model_type": "nn_mlp",
        "model_name": model_name,
        "test_hashes": model.test_indices,
        "metrics": {
            "train_loss": metrics["train_loss"],
            "test_loss": metrics["test_loss"],
            "avg_voxel_error": voxel_error,
        },
        "training_config": {
            "epochs": config.epochs,
            "epochs_completed": metrics.get("epochs_completed", config.epochs),
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "test_fraction": config.test_fraction,
            "training_duration_seconds": training_time,
            "early_stopping": config.early_stopping,
            "early_stopping_patience": config.early_stopping_patience,
            "stopped_early": metrics.get("stopped_early", False),
        },
        "architecture_config": {
            "mlp_hidden_layers": config.hidden_layers,
            "mlp_dropout": dropout,
        },
        "output_config": {
            "voxel_grid_size": config.voxel_grid_size,
            "use_kspace": config.use_kspace,
        },
        "preprocessing_config": {
            "trim_timesteps": config.trim_timesteps,
            "downsample_factor": config.downsample_factor,
        },
    }

    model_path = models_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(
        f"  train_loss={metrics['train_loss']:.6e}, "
        f"test_loss={metrics['test_loss']:.6e}, "
        f"voxel_error={voxel_error:.2f}, "
        f"time={training_time:.1f}s"
    )

    return SweepResult(
        dropout=dropout,
        train_loss=metrics["train_loss"],
        test_loss=metrics["test_loss"],
        voxel_error=voxel_error,
        epochs_completed=int(metrics.get("epochs_completed", config.epochs)),
        training_time_seconds=training_time,
        model_name=model_name,
    )


def generate_charts(
    results: list[SweepResult],
    output_dir: Path,
    sweep_name: str,
) -> None:
    """Generate comparison charts from sweep results."""
    # Prepare data
    labels = [f"{r.dropout:.1f}" for r in results]
    voxel_errors = [r.voxel_error for r in results]
    test_losses = [r.test_loss for r in results]
    train_losses = [r.train_loss for r in results]

    x_pos = np.arange(len(labels))

    # Chart 1: Voxel Error comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_pos, voxel_errors, color="steelblue", edgecolor="black")
    ax.set_xlabel("Dropout Rate", fontsize=12)
    ax.set_ylabel("Voxel Error (Sum of Absolute Errors)", fontsize=12)
    ax.set_title("MLP: Voxel Error vs Dropout Rate", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)

    # Add value labels on bars
    for bar, val in zip(bars, voxel_errors, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(voxel_errors) * 0.01,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_dir / f"{sweep_name}_voxel_error.png", dpi=150)
    fig.savefig(output_dir / f"{sweep_name}_voxel_error.pdf")
    plt.close(fig)

    # Chart 2: Train vs Test Loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    ax.bar(
        x_pos - width / 2, train_losses, width, label="Train Loss", color="royalblue"
    )
    ax.bar(
        x_pos + width / 2, test_losses, width, label="Test Loss", color="coral"
    )
    ax.set_xlabel("Dropout Rate", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title("MLP: Training vs Test Loss", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(output_dir / f"{sweep_name}_losses.png", dpi=150)
    fig.savefig(output_dir / f"{sweep_name}_losses.pdf")
    plt.close(fig)

    logger.info(f"Charts saved to {output_dir}")


def run_sweep(batch_name: str, config: SweepConfig) -> None:
    """Run the full hyperparameter sweep."""
    batch_dir = DEFAULT_DATA_DIR / batch_name
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    models_dir = batch_dir / "inverse_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    sweep_output_dir = batch_dir / "sweep_results"
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting dropout sweep for batch: {batch_name}")
    logger.info("Fixed parameters:")
    logger.info(f"  hidden_layers: {config.hidden_layers}")
    logger.info(f"  trim_timesteps: {config.trim_timesteps}")
    logger.info(f"  downsample_factor: {config.downsample_factor}")
    logger.info(f"  voxel_grid_size: {config.voxel_grid_size}")
    logger.info(f"  epochs: {config.epochs}")
    logger.info(f"Sweep options: {config.dropout_options}")

    # Load training data
    logger.info("Loading training data...")
    X, y, sample_ids, num_sensors = generate_training_data(batch_dir, config)
    if num_sensors is None:
        raise RuntimeError("Could not determine number of sensors from data")
    logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")
    logger.info(f"Number of sensors: {num_sensors}")

    # Run sweep
    results: list[SweepResult] = []
    total_start = time.time()

    for dropout in config.dropout_options:
        result = train_single_config(
            X, y, sample_ids, dropout, config, models_dir
        )
        results.append(result)

    total_time = time.time() - total_start
    logger.info(f"Sweep completed in {total_time:.1f}s")

    # Save results summary
    summary = {
        "batch_name": batch_name,
        "sweep_parameter": "dropout",
        "fixed_config": {
            "hidden_layers": config.hidden_layers,
            "trim_timesteps": config.trim_timesteps,
            "downsample_factor": config.downsample_factor,
            "voxel_grid_size": config.voxel_grid_size,
            "use_kspace": config.use_kspace,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        },
        "results": [
            {
                "dropout": r.dropout,
                "train_loss": r.train_loss,
                "test_loss": r.test_loss,
                "voxel_error": r.voxel_error,
                "epochs_completed": r.epochs_completed,
                "training_time_seconds": r.training_time_seconds,
                "model_name": r.model_name,
            }
            for r in results
        ],
        "total_time_seconds": total_time,
        "num_samples": len(sample_ids),
        "num_sensors": num_sensors,
        "input_dim": X.shape[1],
        "output_dim": y.shape[1],
    }

    sweep_name = "sweep_mlp_dropout"
    summary_path = sweep_output_dir / f"{sweep_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Generate charts
    generate_charts(results, sweep_output_dir, sweep_name)

    # Print summary table
    print("\n" + "=" * 80)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Dropout':<12} {'Train Loss':<12} {'Test Loss':<12} {'Voxel Error':<12}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.dropout:<12.1f} {r.train_loss:<12.6e} "
            f"{r.test_loss:<12.6e} {r.voxel_error:<12.1f}"
        )
    print("=" * 80)

    # Find best configuration
    best = min(results, key=lambda r: r.voxel_error)
    print(f"\nBest configuration: dropout={best.dropout}")
    print(f"  Voxel Error: {best.voxel_error:.1f}")
    print(f"  Test Loss: {best.test_loss:.6e}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for MLP dropout"
    )
    parser.add_argument(
        "--batch",
        required=True,
        help="Name of the simulation batch to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs (default: 500)",
    )
    parser.add_argument(
        "--trim",
        type=int,
        default=45,
        help="Timesteps to trim from start (default: 45)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Downsample factor (default: 4)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=32,
        help="Voxel grid size (default: 32)",
    )

    args = parser.parse_args()

    configure_logging(log_to_file=True)

    config = SweepConfig(
        trim_timesteps=args.trim,
        downsample_factor=args.downsample,
        voxel_grid_size=args.grid_size,
        epochs=args.epochs,
    )

    run_sweep(args.batch, config)


if __name__ == "__main__":
    main()
