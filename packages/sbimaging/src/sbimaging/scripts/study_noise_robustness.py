#!/usr/bin/env python
"""Noise robustness study with k-fold cross-validation.

This script trains 2D CNN models at different noise levels using k-fold
cross-validation to produce robust performance estimates for the dissertation.

Usage:
    python -m sbimaging.scripts.study_noise_robustness --batch <batch_name>

Example:
    python -m sbimaging.scripts.study_noise_robustness --batch multi_cube_1000_p2
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
from sklearn.model_selection import KFold

from sbimaging.inverse_models.nn.network import NeuralNetworkModel
from sbimaging.logging import configure_logging, get_logger

DEFAULT_DATA_DIR = Path("/data/simulations")

logger = get_logger(__name__)


@dataclass
class StudyConfig:
    """Configuration for the noise robustness study."""

    # Data preprocessing (fixed)
    trim_timesteps: int = 45
    downsample_factor: int = 4
    voxel_grid_size: int = 32
    use_kspace: bool = False

    # Training parameters (fixed)
    epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 0.0001
    early_stopping: bool = True
    early_stopping_patience: int = 50

    # Optimized 2D CNN parameters
    conv_channels: list[int] = field(default_factory=lambda: [64, 128])
    pool_size: tuple[int, int] = (12, 24)
    regressor_hidden: int = 512
    dropout: float = 0.5
    kernel_size: tuple[int, int] = (3, 5)
    stride: tuple[int, int] = (1, 3)
    use_residual: bool = True

    # K-fold settings
    n_folds: int = 5
    random_seed: int = 42

    # Noise levels to test (as fractions of global peak)
    noise_levels: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10]
    )


@dataclass
class FoldResult:
    """Result from a single fold."""

    fold: int
    train_loss: float
    test_loss: float
    voxel_error: float
    epochs_completed: int
    training_time_seconds: float


@dataclass
class NoiseResult:
    """Aggregated results for a noise level."""

    noise_level: float
    fold_results: list[FoldResult]
    mean_train_loss: float
    std_train_loss: float
    mean_test_loss: float
    std_test_loss: float
    mean_voxel_error: float
    std_voxel_error: float


def process_sensor_data(
    sensor_file: Path,
    trim_timesteps: int,
    downsample_factor: int,
    noise_level: float,
    global_max: float,
) -> np.ndarray:
    """Process sensor data with specified preprocessing and noise."""
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

        # Add noise if specified
        if noise_level > 0 and global_max > 0:
            noise_std = noise_level * global_max
            noise = np.random.normal(0, noise_std, sensor_data.shape)
            sensor_data = sensor_data + noise

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


def compute_global_max(
    batch_dir: Path,
    trim_timesteps: int,
    downsample_factor: int,
) -> float:
    """Compute global max across all sensor data for noise scaling."""
    sims_dir = batch_dir / "simulations"
    global_max = 0.0

    for sim_dir in sorted(sims_dir.iterdir()):
        if not sim_dir.is_dir():
            continue

        sensor_file = sim_dir / "sensor_data.pkl"
        if not sensor_file.exists():
            continue

        try:
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

            sample_max = float(np.abs(sensor_data).max())
            global_max = max(global_max, sample_max)

        except Exception as e:
            logger.warning(f"Failed to process {sim_dir.name}: {e}")

    return global_max


def generate_training_data(
    batch_dir: Path,
    config: StudyConfig,
    noise_level: float,
    global_max: float,
) -> tuple[np.ndarray, np.ndarray, list[str], int | None]:
    """Generate training data for the batch with specified noise level.

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
                noise_level,
                global_max,
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

    logger.info(f"Loaded {len(X_list)} samples with noise_level={noise_level:.0%}")
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


def train_fold(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    num_sensors: int,
    fold: int,
    noise_level: float,
    config: StudyConfig,
    models_dir: Path,
) -> FoldResult:
    """Train a single fold and return results."""
    noise_pct = int(noise_level * 100)
    model_name = f"cnn2d_noise_{noise_pct:02d}pct_fold_{fold}"

    logger.info(f"Training {model_name}")

    # Split data according to k-fold indices
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_ids = [sample_ids[i] for i in train_idx]
    test_ids = [sample_ids[i] for i in test_idx]

    model = NeuralNetworkModel(
        name=model_name,
        architecture="cnn2d",
        cnn_conv_channels=config.conv_channels,
        cnn_regressor_hidden=config.regressor_hidden,
        cnn_dropout=config.dropout,
        cnn_use_residual=config.use_residual,
        cnn2d_num_sensors=num_sensors,
        cnn2d_pool_size=config.pool_size,
        cnn2d_kernel_size=config.kernel_size,
        cnn2d_stride=config.stride,
    )

    start_time = time.time()

    # Train on training fold, use small fraction for internal validation
    # The real test is on the held-out fold
    metrics = model.train(
        X_train,
        y_train,
        test_fraction=0.1,  # Internal validation split
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        sample_ids=train_ids,
        early_stopping=config.early_stopping,
        early_stopping_patience=config.early_stopping_patience,
    )

    training_time = time.time() - start_time

    # Compute test loss and voxel error on held-out fold
    predictions = model.predict(X_test)
    test_loss = float(np.mean((predictions - y_test) ** 2))
    voxel_error = compute_voxel_error(predictions, y_test, config.voxel_grid_size)

    # Save the model
    model_data = {
        "model": model,
        "model_type": "nn_cnn2d",
        "model_name": model_name,
        "fold": fold,
        "noise_level": noise_level,
        "test_hashes": test_ids,
        "metrics": {
            "train_loss": metrics["train_loss"],
            "test_loss": test_loss,  # Loss on held-out fold
            "avg_voxel_error": voxel_error,
        },
        "training_config": {
            "epochs": config.epochs,
            "epochs_completed": metrics.get("epochs_completed", config.epochs),
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "training_duration_seconds": training_time,
            "early_stopping": config.early_stopping,
            "early_stopping_patience": config.early_stopping_patience,
            "stopped_early": metrics.get("stopped_early", False),
        },
        "architecture_config": {
            "cnn_conv_channels": config.conv_channels,
            "cnn_regressor_hidden": config.regressor_hidden,
            "cnn_dropout": config.dropout,
            "cnn_use_residual": config.use_residual,
            "cnn2d_pool_size": config.pool_size,
            "cnn2d_kernel_size": config.kernel_size,
            "cnn2d_stride": config.stride,
        },
        "output_config": {
            "voxel_grid_size": config.voxel_grid_size,
            "use_kspace": config.use_kspace,
        },
        "preprocessing_config": {
            "trim_timesteps": config.trim_timesteps,
            "downsample_factor": config.downsample_factor,
            "noise_level": noise_level,
            "num_sensors": num_sensors,
        },
    }

    model_path = models_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(
        f"  Fold {fold}: train_loss={metrics['train_loss']:.6e}, "
        f"test_loss={test_loss:.6e}, "
        f"voxel_error={voxel_error:.2f}"
    )

    return FoldResult(
        fold=fold,
        train_loss=metrics["train_loss"],
        test_loss=test_loss,
        voxel_error=voxel_error,
        epochs_completed=int(metrics.get("epochs_completed", config.epochs)),
        training_time_seconds=training_time,
    )


def run_kfold_for_noise_level(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    num_sensors: int,
    noise_level: float,
    config: StudyConfig,
    models_dir: Path,
) -> NoiseResult:
    """Run k-fold cross-validation for a single noise level."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {config.n_folds}-fold CV for noise_level={noise_level:.0%}")
    logger.info(f"{'='*60}")

    kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)

    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        result = train_fold(
            X, y, sample_ids,
            train_idx, test_idx,
            num_sensors, fold, noise_level,
            config, models_dir,
        )
        fold_results.append(result)

    # Compute statistics
    train_losses = [r.train_loss for r in fold_results]
    test_losses = [r.test_loss for r in fold_results]
    voxel_errors = [r.voxel_error for r in fold_results]

    return NoiseResult(
        noise_level=noise_level,
        fold_results=fold_results,
        mean_train_loss=float(np.mean(train_losses)),
        std_train_loss=float(np.std(train_losses)),
        mean_test_loss=float(np.mean(test_losses)),
        std_test_loss=float(np.std(test_losses)),
        mean_voxel_error=float(np.mean(voxel_errors)),
        std_voxel_error=float(np.std(voxel_errors)),
    )


def generate_charts(
    results: list[NoiseResult],
    output_dir: Path,
    study_name: str,
) -> None:
    """Generate comparison charts from study results."""
    noise_levels = [r.noise_level * 100 for r in results]  # Convert to percentage
    mean_voxel_errors = [r.mean_voxel_error for r in results]
    std_voxel_errors = [r.std_voxel_error for r in results]
    mean_test_losses = [r.mean_test_loss for r in results]
    std_test_losses = [r.std_test_loss for r in results]

    x_pos = np.arange(len(noise_levels))
    labels = [f"{n:.0f}%" for n in noise_levels]

    # Chart 1: Voxel Error with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x_pos, mean_voxel_errors,
        yerr=std_voxel_errors,
        color="steelblue", edgecolor="black",
        capsize=5,
    )
    ax.set_xlabel("Noise Level (% of Peak)", fontsize=12)
    ax.set_ylabel("Voxel Error (Sum of Absolute Errors)", fontsize=12)
    ax.set_title("2D CNN: Voxel Error vs Noise Level (5-Fold CV)", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)

    # Add value labels on bars
    for bar, mean, std in zip(bars, mean_voxel_errors, std_voxel_errors, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + max(mean_voxel_errors) * 0.02,
            f"{mean:.1f}\n±{std:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_voxel_error.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_voxel_error.pdf")
    plt.close(fig)

    # Chart 2: Test Loss with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        x_pos, mean_test_losses,
        yerr=std_test_losses,
        color="coral", edgecolor="black",
        capsize=5,
    )
    ax.set_xlabel("Noise Level (% of Peak)", fontsize=12)
    ax.set_ylabel("Test Loss (MSE)", fontsize=12)
    ax.set_title("2D CNN: Test Loss vs Noise Level (5-Fold CV)", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_test_loss.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_test_loss.pdf")
    plt.close(fig)

    # Chart 3: Line plot showing degradation
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "steelblue"
    ax1.set_xlabel("Noise Level (% of Peak)", fontsize=12)
    ax1.set_ylabel("Voxel Error", fontsize=12, color=color1)
    ax1.errorbar(
        noise_levels, mean_voxel_errors, yerr=std_voxel_errors,
        marker="o", color=color1, linewidth=2, markersize=8,
        capsize=5, label="Voxel Error",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.set_ylabel("Test Loss (MSE)", fontsize=12, color=color2)
    ax2.errorbar(
        noise_levels, mean_test_losses, yerr=std_test_losses,
        marker="s", color=color2, linewidth=2, markersize=8,
        capsize=5, label="Test Loss",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("2D CNN: Performance Degradation with Noise (5-Fold CV)", fontsize=14)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_degradation.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_degradation.pdf")
    plt.close(fig)

    logger.info(f"Charts saved to {output_dir}")


def run_study(batch_name: str, config: StudyConfig) -> None:
    """Run the full noise robustness study."""
    batch_dir = DEFAULT_DATA_DIR / batch_name
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    models_dir = batch_dir / "inverse_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    study_output_dir = batch_dir / "study_results"
    study_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting noise robustness study for batch: {batch_name}")
    logger.info(f"Noise levels: {[f'{n:.0%}' for n in config.noise_levels]}")
    logger.info(f"K-folds: {config.n_folds}")
    logger.info("Optimized 2D CNN parameters:")
    logger.info(f"  conv_channels: {config.conv_channels}")
    logger.info(f"  pool_size: {config.pool_size}")
    logger.info(f"  regressor_hidden: {config.regressor_hidden}")
    logger.info(f"  dropout: {config.dropout}")
    logger.info(f"  kernel_size: {config.kernel_size}")
    logger.info(f"  stride: {config.stride}")

    # Compute global max for noise scaling (before adding any noise)
    logger.info("Computing global max for noise scaling...")
    global_max = compute_global_max(
        batch_dir, config.trim_timesteps, config.downsample_factor
    )
    logger.info(f"Global max: {global_max:.6f}")

    # Run study for each noise level
    all_results: list[NoiseResult] = []
    total_start = time.time()

    for noise_level in config.noise_levels:
        # Generate data with this noise level
        X, y, sample_ids, num_sensors = generate_training_data(
            batch_dir, config, noise_level, global_max
        )
        if num_sensors is None:
            raise RuntimeError("Could not determine number of sensors from data")

        result = run_kfold_for_noise_level(
            X, y, sample_ids, num_sensors, noise_level, config, models_dir
        )
        all_results.append(result)

    total_time = time.time() - total_start
    logger.info(f"\nStudy completed in {total_time:.1f}s")

    # Save results summary
    summary = {
        "batch_name": batch_name,
        "study_type": "noise_robustness",
        "n_folds": config.n_folds,
        "random_seed": config.random_seed,
        "global_max": global_max,
        "config": {
            "trim_timesteps": config.trim_timesteps,
            "downsample_factor": config.downsample_factor,
            "voxel_grid_size": config.voxel_grid_size,
            "conv_channels": config.conv_channels,
            "pool_size": list(config.pool_size),
            "regressor_hidden": config.regressor_hidden,
            "dropout": config.dropout,
            "kernel_size": list(config.kernel_size),
            "stride": list(config.stride),
            "use_residual": config.use_residual,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        },
        "results": [
            {
                "noise_level": r.noise_level,
                "mean_train_loss": r.mean_train_loss,
                "std_train_loss": r.std_train_loss,
                "mean_test_loss": r.mean_test_loss,
                "std_test_loss": r.std_test_loss,
                "mean_voxel_error": r.mean_voxel_error,
                "std_voxel_error": r.std_voxel_error,
                "folds": [
                    {
                        "fold": f.fold,
                        "train_loss": f.train_loss,
                        "test_loss": f.test_loss,
                        "voxel_error": f.voxel_error,
                        "epochs_completed": f.epochs_completed,
                        "training_time_seconds": f.training_time_seconds,
                    }
                    for f in r.fold_results
                ],
            }
            for r in all_results
        ],
        "total_time_seconds": total_time,
    }

    study_name = "study_noise_robustness"
    summary_path = study_output_dir / f"{study_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Generate charts
    generate_charts(all_results, study_output_dir, study_name)

    # Print summary table
    print("\n" + "=" * 80)
    print("NOISE ROBUSTNESS STUDY RESULTS (5-Fold CV)")
    print("=" * 80)
    print(f"{'Noise Level':<12} {'Train Loss':<20} {'Test Loss':<20} {'Voxel Error':<20}")
    print("-" * 80)
    for r in all_results:
        noise_str = f"{r.noise_level:.0%}"
        train_str = f"{r.mean_train_loss:.4e} ± {r.std_train_loss:.4e}"
        test_str = f"{r.mean_test_loss:.4e} ± {r.std_test_loss:.4e}"
        voxel_str = f"{r.mean_voxel_error:.1f} ± {r.std_voxel_error:.1f}"
        print(f"{noise_str:<12} {train_str:<20} {test_str:<20} {voxel_str:<20}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Noise robustness study with k-fold cross-validation"
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
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    configure_logging(log_to_file=True)

    config = StudyConfig(
        epochs=args.epochs,
        n_folds=args.folds,
        random_seed=args.seed,
    )

    run_study(args.batch, config)


if __name__ == "__main__":
    main()
