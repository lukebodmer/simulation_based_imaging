#!/usr/bin/env python
"""Sensor sparsity study with k-fold cross-validation.

This script trains 2D CNN models with different sensor configurations using k-fold
cross-validation to understand minimum sensor requirements and optimal placement.

Usage:
    python -m sbimaging.scripts.study_sensor_sparsity --batch <batch_name>

Example:
    python -m sbimaging.scripts.study_sensor_sparsity --batch multi_cube_1000_p2
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


# Sensor configurations from the manual study
SENSOR_CONFIGS = {
    "144_full": {
        "name": "Full (144)",
        "description": "All sensors active",
        "active": list(range(144)),
    },
    "120_uniform": {
        "name": "Uniform 83% (120)",
        "description": "Uniform sparse (4 removed per face)",
        "active": [
            0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 55, 57, 58, 59, 60, 61, 62, 64, 66, 67, 68, 69, 70, 71,
            72, 73, 74, 75, 76, 77, 79, 81, 82, 83, 84, 85, 86, 88, 90, 91, 92, 93, 94, 95,
            96, 97, 98, 99, 100, 101, 103, 105, 106, 107, 108, 109, 110, 112, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 127, 129, 130, 131, 132, 133, 134, 136, 138, 139, 140, 141, 142, 143,
        ],
    },
    "72_checkerboard": {
        "name": "Checkerboard 50% (72)",
        "description": "Checkerboard pattern",
        "active": [
            0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23,
            24, 26, 28, 30, 32, 34, 37, 39, 41, 43, 45, 47,
            48, 50, 52, 54, 56, 58, 61, 63, 65, 67, 69, 71,
            72, 74, 76, 78, 80, 82, 85, 87, 89, 91, 93, 95,
            96, 98, 100, 102, 104, 106, 109, 111, 113, 115, 117, 119,
            120, 122, 124, 126, 128, 130, 133, 135, 137, 139, 141, 143,
        ],
    },
    "48_sparse": {
        "name": "Sparse 33% (48)",
        "description": "Sparse uniform pattern",
        "active": [
            2, 6, 8, 10, 13, 15, 17, 21,
            26, 30, 32, 34, 37, 39, 41, 45,
            50, 54, 56, 58, 61, 63, 65, 69,
            74, 78, 80, 82, 85, 87, 89, 93,
            98, 102, 104, 106, 109, 111, 113, 117,
            122, 126, 128, 130, 133, 135, 137, 141,
        ],
    },
    "24_center": {
        "name": "Center 17% (24)",
        "description": "Center of faces (4 per face)",
        "active": [
            6, 8, 15, 17,
            30, 32, 39, 41,
            54, 56, 63, 65,
            78, 80, 87, 89,
            102, 104, 111, 113,
            126, 128, 135, 137,
        ],
    },
}


@dataclass
class StudyConfig:
    """Configuration for the sensor sparsity study."""

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

    # Sensor configurations to test
    sensor_configs: list[str] = field(
        default_factory=lambda: list(SENSOR_CONFIGS.keys())
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
class SensorConfigResult:
    """Aggregated results for a sensor configuration."""

    config_key: str
    config_name: str
    num_sensors: int
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
    active_sensors: list[int],
) -> np.ndarray:
    """Process sensor data with specified preprocessing and sensor selection."""
    with open(sensor_file, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "pressure" in data:
        sensor_data = data["pressure"]
    else:
        sensor_data = data

    if hasattr(sensor_data, "get") and callable(sensor_data.get):
        sensor_data = sensor_data.get()  # type: ignore[union-attr]

    sensor_data = np.asarray(sensor_data)

    if sensor_data.ndim == 2:
        # Select only active sensors
        sensor_data = sensor_data[active_sensors, :]

        # Apply temporal preprocessing
        sensor_data = sensor_data[:, trim_timesteps:]
        if downsample_factor > 1:
            sensor_data = sensor_data[:, ::downsample_factor]

    return sensor_data.flatten().astype(np.float32)


def process_config_to_voxels(config_file: Path, grid_size: int) -> np.ndarray:
    """Convert config to voxel representation."""
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
    config: StudyConfig,
    active_sensors: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate training data for the batch with specified sensor configuration.

    Returns:
        Tuple of (X, y, sample_ids)
    """
    sims_dir = batch_dir / "simulations"

    X_list = []
    y_list = []
    sample_ids = []

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
                active_sensors,
            )
            y = process_config_to_voxels(config_file, config.voxel_grid_size)

            X_list.append(x)
            y_list.append(y)
            sample_ids.append(sim_dir.name)

        except Exception as e:
            logger.warning(f"Failed to process {sim_dir.name}: {e}")

    if not X_list:
        raise RuntimeError(f"No valid training data found in {sims_dir}")

    X = np.stack(X_list)
    y = np.stack(y_list)

    # Verify input dimensions match expected sensor count
    expected_features_per_sensor = X.shape[1] // len(active_sensors)
    logger.info(f"Loaded {len(X_list)} samples with {len(active_sensors)} sensors")
    logger.info(f"Input shape: {X.shape}, features per sensor: {expected_features_per_sensor}")
    logger.info(f"First active sensor indices: {active_sensors[:5]}...")

    return X, y, sample_ids


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
    config_key: str,
    config: StudyConfig,
    models_dir: Path,
) -> FoldResult:
    """Train a single fold and return results."""
    model_name = f"cnn2d_sensors_{config_key}_fold_{fold}"

    logger.info(f"Training {model_name}")

    # Split data according to k-fold indices
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_ids = [sample_ids[i] for i in train_idx]
    test_ids = [sample_ids[i] for i in test_idx]

    # Verify input dimensions
    input_dim = X_train.shape[1]
    expected_timesteps = input_dim // num_sensors
    logger.info(
        f"  Input dim: {input_dim}, num_sensors: {num_sensors}, "
        f"timesteps: {expected_timesteps}"
    )

    if input_dim % num_sensors != 0:
        raise ValueError(
            f"Input dimension {input_dim} not divisible by num_sensors {num_sensors}"
        )

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

    metrics = model.train(
        X_train,
        y_train,
        test_fraction=0.1,
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
    sensor_cfg = SENSOR_CONFIGS[config_key]
    model_data = {
        "model": model,
        "model_type": "nn_cnn2d",
        "model_name": model_name,
        "fold": fold,
        "sensor_config": {
            "key": config_key,
            "name": sensor_cfg["name"],
            "description": sensor_cfg["description"],
            "active_sensors": sensor_cfg["active"],
            "num_sensors": len(sensor_cfg["active"]),
        },
        "test_hashes": test_ids,
        "metrics": {
            "train_loss": metrics["train_loss"],
            "test_loss": test_loss,
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


def run_kfold_for_sensor_config(
    batch_dir: Path,
    config_key: str,
    config: StudyConfig,
    models_dir: Path,
) -> SensorConfigResult:
    """Run k-fold cross-validation for a single sensor configuration."""
    sensor_cfg = SENSOR_CONFIGS[config_key]
    active_sensors = sensor_cfg["active"]
    num_sensors = len(active_sensors)

    logger.info(f"\n{'='*60}")
    logger.info(f"Running {config.n_folds}-fold CV for {sensor_cfg['name']}")
    logger.info(f"Active sensors: {num_sensors}/144")
    logger.info(f"First 10 active indices: {active_sensors[:10]}")
    logger.info(f"Last 10 active indices: {active_sensors[-10:]}")
    logger.info(f"{'='*60}")

    # Generate training data with this sensor configuration
    X, y, sample_ids = generate_training_data(batch_dir, config, active_sensors)

    kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)

    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        result = train_fold(
            X, y, sample_ids,
            train_idx, test_idx,
            num_sensors, fold, config_key,
            config, models_dir,
        )
        fold_results.append(result)

    # Compute statistics
    train_losses = [r.train_loss for r in fold_results]
    test_losses = [r.test_loss for r in fold_results]
    voxel_errors = [r.voxel_error for r in fold_results]

    return SensorConfigResult(
        config_key=config_key,
        config_name=sensor_cfg["name"],
        num_sensors=num_sensors,
        fold_results=fold_results,
        mean_train_loss=float(np.mean(train_losses)),
        std_train_loss=float(np.std(train_losses)),
        mean_test_loss=float(np.mean(test_losses)),
        std_test_loss=float(np.std(test_losses)),
        mean_voxel_error=float(np.mean(voxel_errors)),
        std_voxel_error=float(np.std(voxel_errors)),
    )


def generate_charts(
    results: list[SensorConfigResult],
    output_dir: Path,
    study_name: str,
) -> None:
    """Generate comparison charts from study results."""
    # Sort by number of sensors (descending)
    results = sorted(results, key=lambda r: r.num_sensors, reverse=True)

    num_sensors = [r.num_sensors for r in results]
    labels = [r.config_name for r in results]
    mean_voxel_errors = [r.mean_voxel_error for r in results]
    std_voxel_errors = [r.std_voxel_error for r in results]
    mean_test_losses = [r.mean_test_loss for r in results]
    std_test_losses = [r.std_test_loss for r in results]

    x_pos = np.arange(len(results))

    # Chart 1: Voxel Error bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        x_pos, mean_voxel_errors,
        yerr=std_voxel_errors,
        color="steelblue", edgecolor="black",
        capsize=5,
    )
    ax.set_xlabel("Sensor Configuration", fontsize=12)
    ax.set_ylabel("Voxel Error (Sum of Absolute Errors)", fontsize=12)
    ax.set_title("2D CNN: Voxel Error vs Sensor Configuration (5-Fold CV)", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")

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

    # Chart 2: Line plot - Voxel Error vs Number of Sensors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        num_sensors, mean_voxel_errors, yerr=std_voxel_errors,
        marker="o", color="steelblue", linewidth=2, markersize=10,
        capsize=5, label="Voxel Error",
    )
    ax.set_xlabel("Number of Active Sensors", fontsize=12)
    ax.set_ylabel("Voxel Error (Sum of Absolute Errors)", fontsize=12)
    ax.set_title("2D CNN: Reconstruction Quality vs Sensor Count (5-Fold CV)", fontsize=14)

    # Add percentage labels
    for x, y, pct in zip(num_sensors, mean_voxel_errors, [n/144*100 for n in num_sensors], strict=True):
        ax.annotate(
            f"{pct:.0f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=9,
        )

    ax.invert_xaxis()  # More sensors on left
    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_sensors_vs_error.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_sensors_vs_error.pdf")
    plt.close(fig)

    # Chart 3: Dual axis - Voxel Error and Test Loss
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "steelblue"
    ax1.set_xlabel("Number of Active Sensors", fontsize=12)
    ax1.set_ylabel("Voxel Error", fontsize=12, color=color1)
    ax1.errorbar(
        num_sensors, mean_voxel_errors, yerr=std_voxel_errors,
        marker="o", color=color1, linewidth=2, markersize=8,
        capsize=5, label="Voxel Error",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.set_ylabel("Test Loss (MSE)", fontsize=12, color=color2)
    ax2.errorbar(
        num_sensors, mean_test_losses, yerr=std_test_losses,
        marker="s", color=color2, linewidth=2, markersize=8,
        capsize=5, label="Test Loss",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("2D CNN: Performance vs Sensor Count (5-Fold CV)", fontsize=14)
    ax1.invert_xaxis()

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_dual_axis.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_dual_axis.pdf")
    plt.close(fig)

    logger.info(f"Charts saved to {output_dir}")


def run_study(batch_name: str, config: StudyConfig) -> None:
    """Run the full sensor sparsity study."""
    batch_dir = DEFAULT_DATA_DIR / batch_name
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    models_dir = batch_dir / "inverse_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    study_output_dir = batch_dir / "study_results"
    study_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting sensor sparsity study for batch: {batch_name}")
    logger.info(f"Sensor configurations: {config.sensor_configs}")
    logger.info(f"K-folds: {config.n_folds}")
    logger.info("Optimized 2D CNN parameters:")
    logger.info(f"  conv_channels: {config.conv_channels}")
    logger.info(f"  pool_size: {config.pool_size}")
    logger.info(f"  regressor_hidden: {config.regressor_hidden}")
    logger.info(f"  dropout: {config.dropout}")
    logger.info(f"  kernel_size: {config.kernel_size}")
    logger.info(f"  stride: {config.stride}")

    # Run study for each sensor configuration
    all_results: list[SensorConfigResult] = []
    total_start = time.time()

    for config_key in config.sensor_configs:
        if config_key not in SENSOR_CONFIGS:
            logger.warning(f"Unknown sensor config: {config_key}, skipping")
            continue

        result = run_kfold_for_sensor_config(
            batch_dir, config_key, config, models_dir
        )
        all_results.append(result)

    total_time = time.time() - total_start
    logger.info(f"\nStudy completed in {total_time:.1f}s")

    # Save results summary
    summary = {
        "batch_name": batch_name,
        "study_type": "sensor_sparsity",
        "n_folds": config.n_folds,
        "random_seed": config.random_seed,
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
        "sensor_configs": {
            key: {
                "name": cfg["name"],
                "description": cfg["description"],
                "active_sensors": cfg["active"],
                "num_sensors": len(cfg["active"]),
            }
            for key, cfg in SENSOR_CONFIGS.items()
        },
        "results": [
            {
                "config_key": r.config_key,
                "config_name": r.config_name,
                "num_sensors": r.num_sensors,
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

    study_name = "study_sensor_sparsity"
    summary_path = study_output_dir / f"{study_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Generate charts
    generate_charts(all_results, study_output_dir, study_name)

    # Print summary table
    print("\n" + "=" * 90)
    print("SENSOR SPARSITY STUDY RESULTS (5-Fold CV)")
    print("=" * 90)
    print(f"{'Config':<25} {'Sensors':<10} {'Train Loss':<20} {'Test Loss':<20} {'Voxel Error':<20}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: x.num_sensors, reverse=True):
        train_str = f"{r.mean_train_loss:.4e} ± {r.std_train_loss:.4e}"
        test_str = f"{r.mean_test_loss:.4e} ± {r.std_test_loss:.4e}"
        voxel_str = f"{r.mean_voxel_error:.1f} ± {r.std_voxel_error:.1f}"
        print(f"{r.config_name:<25} {r.num_sensors:<10} {train_str:<20} {test_str:<20} {voxel_str:<20}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Sensor sparsity study with k-fold cross-validation"
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
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific sensor configs to test (default: all)",
    )

    args = parser.parse_args()

    configure_logging(log_to_file=True)

    sensor_configs = args.configs if args.configs else list(SENSOR_CONFIGS.keys())

    config = StudyConfig(
        epochs=args.epochs,
        n_folds=args.folds,
        random_seed=args.seed,
        sensor_configs=sensor_configs,
    )

    run_study(args.batch, config)


if __name__ == "__main__":
    main()
