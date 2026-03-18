#!/usr/bin/env python
"""K-space vs Voxel representation study with k-fold cross-validation.

This script trains 2D CNN models using either voxel grid or k-space output
representations using k-fold cross-validation to determine which representation
produces better results.

Usage:
    python -m sbimaging.scripts.study_kspace_vs_voxel --batch <batch_name>

Example:
    python -m sbimaging.scripts.study_kspace_vs_voxel --batch multi_cube_1000_p2
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
    """Configuration for the k-space vs voxel study."""

    # Data preprocessing (fixed)
    trim_timesteps: int = 45
    downsample_factor: int = 4
    voxel_grid_size: int = 32

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
    use_residual: bool = False

    # K-fold settings
    n_folds: int = 5
    random_seed: int = 42

    # No noise for this study
    noise_level: float = 0.0


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
class RepresentationResult:
    """Aggregated results for a representation type."""

    representation: str  # "voxel" or "kspace"
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


def process_config_to_kspace(config_file: Path, grid_size: int) -> np.ndarray:
    """Convert config to k-space representation.

    Creates the voxel grid and then applies 3D FFT, returning the
    real and imaginary parts concatenated (preserving phase information).
    """
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

    # Apply 3D FFT and store real + imaginary (preserving phase)
    kspace = np.fft.fftn(grid)
    kspace = np.fft.fftshift(kspace)
    real_part = np.real(kspace).astype(np.float32)
    imag_part = np.imag(kspace).astype(np.float32)

    return np.concatenate([real_part.ravel(), imag_part.ravel()])


def kspace_to_voxels(kspace_flat: np.ndarray, grid_size: int) -> np.ndarray:
    """Convert k-space prediction back to voxel space for error computation.

    K-space data is stored as [real_flat, imag_flat] concatenated.
    """
    n_coeffs = grid_size ** 3
    real_part = kspace_flat[:n_coeffs].reshape((grid_size, grid_size, grid_size))
    imag_part = kspace_flat[n_coeffs:].reshape((grid_size, grid_size, grid_size))

    # Reconstruct complex k-space
    kspace = real_part + 1j * imag_part

    # Inverse shift and FFT
    kspace_unshifted = np.fft.ifftshift(kspace)
    voxels = np.fft.ifftn(kspace_unshifted)

    # Take real part
    voxels = np.real(voxels)

    return voxels.flatten().astype(np.float32)


def generate_training_data(
    batch_dir: Path,
    config: StudyConfig,
    use_kspace: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], int | None]:
    """Generate training data for the batch.

    Args:
        batch_dir: Path to simulation batch directory.
        config: Study configuration.
        use_kspace: If True, use k-space representation for targets.

    Returns:
        Tuple of (X, y, sample_ids, num_sensors)
    """
    sims_dir = batch_dir / "simulations"

    X_list = []
    y_list = []
    sample_ids = []
    num_sensors = None

    process_target = process_config_to_kspace if use_kspace else process_config_to_voxels

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
            y = process_target(config_file, config.voxel_grid_size)

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

    rep_name = "k-space" if use_kspace else "voxel"
    logger.info(f"Loaded {len(X_list)} samples with {rep_name} representation")
    return np.stack(X_list), np.stack(y_list), sample_ids, num_sensors


def compute_voxel_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    grid_size: int,
    use_kspace: bool,
) -> float:
    """Compute sum of absolute voxel errors, averaged across samples.

    If predictions are in k-space, convert them to voxel space first.
    """
    errors = []

    for i in range(len(predictions)):
        pred = predictions[i]
        target = targets[i]

        if use_kspace:
            # Convert both back to voxel space for fair comparison
            pred_voxels = kspace_to_voxels(pred, grid_size)
            target_voxels = kspace_to_voxels(target, grid_size)
        else:
            pred_voxels = pred.reshape((grid_size, grid_size, grid_size))
            target_voxels = target.reshape((grid_size, grid_size, grid_size))

        sample_error = np.sum(np.abs(pred_voxels - target_voxels))
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
    use_kspace: bool,
    config: StudyConfig,
    models_dir: Path,
) -> FoldResult:
    """Train a single fold and return results."""
    rep_name = "kspace" if use_kspace else "voxel"
    model_name = f"cnn2d_{rep_name}_fold_{fold}"

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
    voxel_error = compute_voxel_error(
        predictions, y_test, config.voxel_grid_size, use_kspace
    )

    # Save the model
    model_data = {
        "model": model,
        "model_type": "nn_cnn2d",
        "model_name": model_name,
        "fold": fold,
        "use_kspace": use_kspace,
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
            "use_kspace": use_kspace,
        },
        "preprocessing_config": {
            "trim_timesteps": config.trim_timesteps,
            "downsample_factor": config.downsample_factor,
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


def run_kfold_for_representation(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    num_sensors: int,
    use_kspace: bool,
    config: StudyConfig,
    models_dir: Path,
) -> RepresentationResult:
    """Run k-fold cross-validation for a single representation type."""
    rep_name = "k-space" if use_kspace else "voxel"
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {config.n_folds}-fold CV for {rep_name} representation")
    logger.info(f"{'='*60}")

    kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)

    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        result = train_fold(
            X, y, sample_ids,
            train_idx, test_idx,
            num_sensors, fold, use_kspace,
            config, models_dir,
        )
        fold_results.append(result)

    # Compute statistics
    train_losses = [r.train_loss for r in fold_results]
    test_losses = [r.test_loss for r in fold_results]
    voxel_errors = [r.voxel_error for r in fold_results]

    return RepresentationResult(
        representation=rep_name,
        fold_results=fold_results,
        mean_train_loss=float(np.mean(train_losses)),
        std_train_loss=float(np.std(train_losses)),
        mean_test_loss=float(np.mean(test_losses)),
        std_test_loss=float(np.std(test_losses)),
        mean_voxel_error=float(np.mean(voxel_errors)),
        std_voxel_error=float(np.std(voxel_errors)),
    )


def generate_charts(
    results: list[RepresentationResult],
    output_dir: Path,
    study_name: str,
) -> None:
    """Generate comparison charts from study results."""
    representations = [r.representation for r in results]
    mean_voxel_errors = [r.mean_voxel_error for r in results]
    std_voxel_errors = [r.std_voxel_error for r in results]
    mean_test_losses = [r.mean_test_loss for r in results]
    std_test_losses = [r.std_test_loss for r in results]

    x_pos = np.arange(len(representations))
    colors = ["steelblue", "coral"]

    # Chart 1: Voxel Error comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        x_pos, mean_voxel_errors,
        yerr=std_voxel_errors,
        color=colors, edgecolor="black",
        capsize=8,
    )
    ax.set_xlabel("Output Representation", fontsize=12)
    ax.set_ylabel("Voxel Error (Sum of Absolute Errors)", fontsize=12)
    ax.set_title("2D CNN: Voxel Error by Output Representation (5-Fold CV)", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.title() for r in representations], fontsize=11)

    # Add value labels on bars
    for bar, mean, std in zip(bars, mean_voxel_errors, std_voxel_errors, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + max(mean_voxel_errors) * 0.02,
            f"{mean:.1f}\n±{std:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_voxel_error.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_voxel_error.pdf")
    plt.close(fig)

    # Chart 2: Test Loss comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        x_pos, mean_test_losses,
        yerr=std_test_losses,
        color=colors, edgecolor="black",
        capsize=8,
    )
    ax.set_xlabel("Output Representation", fontsize=12)
    ax.set_ylabel("Test Loss (MSE)", fontsize=12)
    ax.set_title("2D CNN: Test Loss by Output Representation (5-Fold CV)", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.title() for r in representations], fontsize=11)

    # Add value labels on bars
    for bar, mean, std in zip(bars, mean_test_losses, std_test_losses, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + max(mean_test_losses) * 0.02,
            f"{mean:.2e}\n±{std:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_test_loss.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_test_loss.pdf")
    plt.close(fig)

    # Chart 3: Combined comparison (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Voxel error
    bars1 = ax1.bar(
        x_pos, mean_voxel_errors,
        yerr=std_voxel_errors,
        color=colors, edgecolor="black",
        capsize=8,
    )
    ax1.set_xlabel("Output Representation", fontsize=11)
    ax1.set_ylabel("Voxel Error", fontsize=11)
    ax1.set_title("Voxel Error", fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([r.title() for r in representations], fontsize=10)

    # Test loss
    bars2 = ax2.bar(
        x_pos, mean_test_losses,
        yerr=std_test_losses,
        color=colors, edgecolor="black",
        capsize=8,
    )
    ax2.set_xlabel("Output Representation", fontsize=11)
    ax2.set_ylabel("Test Loss (MSE)", fontsize=11)
    ax2.set_title("Test Loss", fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r.title() for r in representations], fontsize=10)

    fig.suptitle("K-Space vs Voxel Representation Comparison (5-Fold CV)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_comparison.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_comparison.pdf")
    plt.close(fig)

    logger.info(f"Charts saved to {output_dir}")


def run_study(batch_name: str, config: StudyConfig) -> None:
    """Run the full k-space vs voxel study."""
    batch_dir = DEFAULT_DATA_DIR / batch_name
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    models_dir = batch_dir / "inverse_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    study_output_dir = batch_dir / "study_results"
    study_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting k-space vs voxel study for batch: {batch_name}")
    logger.info(f"K-folds: {config.n_folds}")
    logger.info("Optimized 2D CNN parameters:")
    logger.info(f"  conv_channels: {config.conv_channels}")
    logger.info(f"  pool_size: {config.pool_size}")
    logger.info(f"  regressor_hidden: {config.regressor_hidden}")
    logger.info(f"  dropout: {config.dropout}")
    logger.info(f"  kernel_size: {config.kernel_size}")
    logger.info(f"  stride: {config.stride}")

    all_results: list[RepresentationResult] = []
    total_start = time.time()

    # Run study for each representation type
    for use_kspace in [False, True]:
        rep_name = "k-space" if use_kspace else "voxel"
        logger.info(f"\nGenerating {rep_name} training data...")

        X, y, sample_ids, num_sensors = generate_training_data(
            batch_dir, config, use_kspace
        )
        if num_sensors is None:
            raise RuntimeError("Could not determine number of sensors from data")

        result = run_kfold_for_representation(
            X, y, sample_ids, num_sensors, use_kspace, config, models_dir
        )
        all_results.append(result)

    total_time = time.time() - total_start
    logger.info(f"\nStudy completed in {total_time:.1f}s")

    # Save results summary
    summary = {
        "batch_name": batch_name,
        "study_type": "kspace_vs_voxel",
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
        "results": [
            {
                "representation": r.representation,
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

    study_name = "study_kspace_vs_voxel"
    summary_path = study_output_dir / f"{study_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Generate charts
    generate_charts(all_results, study_output_dir, study_name)

    # Print summary table
    print("\n" + "=" * 80)
    print("K-SPACE VS VOXEL STUDY RESULTS (5-Fold CV)")
    print("=" * 80)
    print(f"{'Representation':<15} {'Train Loss':<22} {'Test Loss':<22} {'Voxel Error':<20}")
    print("-" * 80)
    for r in all_results:
        train_str = f"{r.mean_train_loss:.4e} ± {r.std_train_loss:.4e}"
        test_str = f"{r.mean_test_loss:.4e} ± {r.std_test_loss:.4e}"
        voxel_str = f"{r.mean_voxel_error:.1f} ± {r.std_voxel_error:.1f}"
        print(f"{r.representation.title():<15} {train_str:<22} {test_str:<22} {voxel_str:<20}")
    print("=" * 80)

    # Print recommendation
    voxel_result = next(r for r in all_results if r.representation == "voxel")
    kspace_result = next(r for r in all_results if r.representation == "k-space")

    if voxel_result.mean_voxel_error < kspace_result.mean_voxel_error:
        winner = "Voxel"
        improvement = (
            (kspace_result.mean_voxel_error - voxel_result.mean_voxel_error)
            / kspace_result.mean_voxel_error
            * 100
        )
    else:
        winner = "K-space"
        improvement = (
            (voxel_result.mean_voxel_error - kspace_result.mean_voxel_error)
            / voxel_result.mean_voxel_error
            * 100
        )

    print(f"\nRecommendation: {winner} representation ({improvement:.1f}% better voxel error)")


def main():
    parser = argparse.ArgumentParser(
        description="K-space vs Voxel representation study with k-fold cross-validation"
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
