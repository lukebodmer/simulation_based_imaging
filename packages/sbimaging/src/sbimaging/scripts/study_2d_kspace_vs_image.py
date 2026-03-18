#!/usr/bin/env python
"""K-space vs Image representation study for 2D inverse problems.

This script trains 2D neural network models using either image grid or k-space
output representations using k-fold cross-validation to determine which
representation produces better results.

Usage:
    python -m sbimaging.scripts.study_2d_kspace_vs_image

Example:
    python -m sbimaging.scripts.study_2d_kspace_vs_image --batch-dir /data/2d-simulations
"""

import argparse
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from sbimaging.inverse_models.dim2 import (
    DataLoader2D,
    KSpace2D,
    NeuralNetwork2D,
    create_inclusion_image,
    kspace_to_image,
)
from sbimaging.logging import configure_logging, get_logger

DEFAULT_BATCH_DIR = Path("/data/2d-simulations")

logger = get_logger(__name__)


@dataclass
class StudyConfig:
    """Configuration for the 2D k-space vs image study."""

    # Data preprocessing
    grid_size: int = 64
    trim_timesteps: int = 50
    downsample_factor: int = 2

    # Training parameters
    architecture: str = "mlp"  # "mlp" or "cnn"
    epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 0.0001

    # K-fold settings
    n_folds: int = 5
    random_seed: int = 42


@dataclass
class FoldResult:
    """Result from a single fold."""

    fold: int
    train_loss: float
    test_loss: float
    pixel_error: float
    epochs_completed: int
    training_time_seconds: float


@dataclass
class RepresentationResult:
    """Aggregated results for a representation type."""

    representation: str  # "image" or "kspace"
    fold_results: list[FoldResult]
    mean_train_loss: float
    std_train_loss: float
    mean_test_loss: float
    std_test_loss: float
    mean_pixel_error: float
    std_pixel_error: float


def load_sensor_data(
    sim_dir: Path,
    trim_timesteps: int,
    downsample_factor: int,
) -> np.ndarray:
    """Load and preprocess sensor data."""
    sensor_file = sim_dir / "sensor_data.npy"
    data = np.load(sensor_file)

    if trim_timesteps > 0:
        data = data[:, trim_timesteps:]
    if downsample_factor > 1:
        data = data[:, ::downsample_factor]

    return data.ravel().astype(np.float32)


def load_image_target(
    param_file: Path,
    grid_size: int,
) -> np.ndarray:
    """Load parameters and create image representation."""
    with open(param_file) as f:
        params = json.load(f)

    image = create_inclusion_image(
        inclusion_type=params["inclusion_type"],
        center_x=params["center_x"],
        center_y=params["center_y"],
        inclusion_size=params["inclusion_size"],
        domain_size=params["domain_size"],
        grid_size=grid_size,
        background_value=0.0,
        inclusion_value=1.0,
    )

    return image.flatten().astype(np.float32)


def load_kspace_target(
    param_file: Path,
    grid_size: int,
) -> np.ndarray:
    """Load parameters and create k-space representation (real + imag)."""
    with open(param_file) as f:
        params = json.load(f)

    image = create_inclusion_image(
        inclusion_type=params["inclusion_type"],
        center_x=params["center_x"],
        center_y=params["center_y"],
        inclusion_size=params["inclusion_size"],
        domain_size=params["domain_size"],
        grid_size=grid_size,
        background_value=0.0,
        inclusion_value=1.0,
    )

    # Compute k-space (real + imaginary, not just magnitude)
    kspace = np.fft.fftshift(np.fft.fft2(image))
    real = np.real(kspace).astype(np.float32)
    imag = np.imag(kspace).astype(np.float32)

    return np.concatenate([real.ravel(), imag.ravel()])


def kspace_to_image_array(kspace_flat: np.ndarray, grid_size: int) -> np.ndarray:
    """Convert k-space prediction back to image for error computation."""
    n_coeffs = grid_size * grid_size
    real = kspace_flat[:n_coeffs].reshape(grid_size, grid_size)
    imag = kspace_flat[n_coeffs:].reshape(grid_size, grid_size)

    complex_kspace = real + 1j * imag
    image = np.fft.ifft2(np.fft.ifftshift(complex_kspace))

    return np.real(image)


def generate_training_data(
    batch_dir: Path,
    config: StudyConfig,
    use_kspace: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """Generate training data for the batch.

    Args:
        batch_dir: Path to simulation batch directory.
        config: Study configuration.
        use_kspace: If True, use k-space representation for targets.

    Returns:
        Tuple of (X, y, sample_ids, params_list)
    """
    sims_dir = batch_dir / "simulations"
    params_dir = batch_dir / "parameters"

    X_list = []
    y_list = []
    sample_ids = []
    params_list = []

    target_loader = load_kspace_target if use_kspace else load_image_target

    for sim_dir in sorted(sims_dir.glob("sim_*")):
        sensor_file = sim_dir / "sensor_data.npy"
        if not sensor_file.exists():
            continue

        sim_id = sim_dir.name
        param_file = params_dir / f"{sim_id}.json"

        if not param_file.exists():
            continue

        try:
            x = load_sensor_data(
                sim_dir,
                config.trim_timesteps,
                config.downsample_factor,
            )
            y = target_loader(param_file, config.grid_size)

            with open(param_file) as f:
                params = json.load(f)

            X_list.append(x)
            y_list.append(y)
            sample_ids.append(sim_id)
            params_list.append(params)

        except Exception as e:
            logger.warning(f"Failed to process {sim_id}: {e}")

    if not X_list:
        raise RuntimeError(f"No valid training data found in {sims_dir}")

    rep_name = "k-space" if use_kspace else "image"
    logger.info(f"Loaded {len(X_list)} samples with {rep_name} representation")
    return np.stack(X_list), np.stack(y_list), sample_ids, params_list


def compute_pixel_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    grid_size: int,
    use_kspace: bool,
) -> float:
    """Compute sum of absolute pixel errors, averaged across samples.

    If predictions are in k-space, convert them to image space first.
    """
    errors = []

    for i in range(len(predictions)):
        pred = predictions[i]
        target = targets[i]

        if use_kspace:
            # Convert both back to image space for fair comparison
            pred_image = kspace_to_image_array(pred, grid_size)
            target_image = kspace_to_image_array(target, grid_size)
        else:
            pred_image = pred.reshape((grid_size, grid_size))
            target_image = target.reshape((grid_size, grid_size))

        sample_error = np.sum(np.abs(pred_image - target_image))
        errors.append(sample_error)

    return float(np.mean(errors))


def train_fold(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    use_kspace: bool,
    config: StudyConfig,
    models_dir: Path,
) -> FoldResult:
    """Train a single fold and return results."""
    rep_name = "kspace" if use_kspace else "image"
    model_name = f"nn2d_{rep_name}_fold_{fold}"

    logger.info(f"Training {model_name}")

    # Split data according to k-fold indices
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_ids = [sample_ids[i] for i in train_idx]
    test_ids = [sample_ids[i] for i in test_idx]

    model = NeuralNetwork2D(
        name=model_name,
        architecture=config.architecture,
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
    )

    training_time = time.time() - start_time

    # Compute test loss and pixel error on held-out fold
    predictions = model.predict(X_test)
    test_loss = float(np.mean((predictions - y_test) ** 2))
    pixel_error = compute_pixel_error(
        predictions, y_test, config.grid_size, use_kspace
    )

    # Save the model
    model_data = {
        "model": model,
        "model_name": model_name,
        "fold": fold,
        "use_kspace": use_kspace,
        "test_ids": test_ids,
        "metrics": {
            "train_loss": metrics["train_loss"],
            "test_loss": test_loss,
            "pixel_error": pixel_error,
        },
        "config": {
            "architecture": config.architecture,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "grid_size": config.grid_size,
        },
    }

    model_path = models_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(
        f"  Fold {fold}: train_loss={metrics['train_loss']:.6e}, "
        f"test_loss={test_loss:.6e}, "
        f"pixel_error={pixel_error:.2f}"
    )

    return FoldResult(
        fold=fold,
        train_loss=metrics["train_loss"],
        test_loss=test_loss,
        pixel_error=pixel_error,
        epochs_completed=config.epochs,
        training_time_seconds=training_time,
    )


def run_kfold_for_representation(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    use_kspace: bool,
    config: StudyConfig,
    models_dir: Path,
) -> RepresentationResult:
    """Run k-fold cross-validation for a single representation type."""
    rep_name = "k-space" if use_kspace else "image"
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {config.n_folds}-fold CV for {rep_name} representation")
    logger.info(f"{'='*60}")

    kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)

    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        result = train_fold(
            X, y, sample_ids,
            train_idx, test_idx,
            fold, use_kspace,
            config, models_dir,
        )
        fold_results.append(result)

    # Compute statistics
    train_losses = [r.train_loss for r in fold_results]
    test_losses = [r.test_loss for r in fold_results]
    pixel_errors = [r.pixel_error for r in fold_results]

    return RepresentationResult(
        representation=rep_name,
        fold_results=fold_results,
        mean_train_loss=float(np.mean(train_losses)),
        std_train_loss=float(np.std(train_losses)),
        mean_test_loss=float(np.mean(test_losses)),
        std_test_loss=float(np.std(test_losses)),
        mean_pixel_error=float(np.mean(pixel_errors)),
        std_pixel_error=float(np.std(pixel_errors)),
    )


def generate_charts(
    results: list[RepresentationResult],
    output_dir: Path,
    study_name: str,
) -> None:
    """Generate comparison charts from study results."""
    representations = [r.representation for r in results]
    mean_pixel_errors = [r.mean_pixel_error for r in results]
    std_pixel_errors = [r.std_pixel_error for r in results]
    mean_test_losses = [r.mean_test_loss for r in results]
    std_test_losses = [r.std_test_loss for r in results]

    x_pos = np.arange(len(representations))
    colors = ["steelblue", "coral"]

    # Chart 1: Pixel Error comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        x_pos, mean_pixel_errors,
        yerr=std_pixel_errors,
        color=colors, edgecolor="black",
        capsize=8,
    )
    ax.set_xlabel("Output Representation", fontsize=12)
    ax.set_ylabel("Pixel Error (Sum of Absolute Errors)", fontsize=12)
    ax.set_title("2D NN: Pixel Error by Output Representation (5-Fold CV)", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.title() for r in representations], fontsize=11)

    # Add value labels on bars
    for bar, mean, std in zip(bars, mean_pixel_errors, std_pixel_errors, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + max(mean_pixel_errors) * 0.02,
            f"{mean:.1f}\n±{std:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_pixel_error.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_pixel_error.pdf")
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
    ax.set_title("2D NN: Test Loss by Output Representation (5-Fold CV)", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.title() for r in representations], fontsize=11)

    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_test_loss.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_test_loss.pdf")
    plt.close(fig)

    # Chart 3: Combined comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(
        x_pos, mean_pixel_errors,
        yerr=std_pixel_errors,
        color=colors, edgecolor="black",
        capsize=8,
    )
    ax1.set_xlabel("Output Representation", fontsize=11)
    ax1.set_ylabel("Pixel Error", fontsize=11)
    ax1.set_title("Pixel Error", fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([r.title() for r in representations], fontsize=10)

    ax2.bar(
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

    fig.suptitle("2D: K-Space vs Image Representation Comparison (5-Fold CV)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / f"{study_name}_comparison.png", dpi=150)
    fig.savefig(output_dir / f"{study_name}_comparison.pdf")
    plt.close(fig)

    logger.info(f"Charts saved to {output_dir}")


def run_study(batch_dir: Path, config: StudyConfig) -> None:
    """Run the full k-space vs image study for 2D."""
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    models_dir = batch_dir / "inverse_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    study_output_dir = batch_dir / "study_results"
    study_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting 2D k-space vs image study for: {batch_dir}")
    logger.info(f"Architecture: {config.architecture}")
    logger.info(f"Grid size: {config.grid_size}")
    logger.info(f"K-folds: {config.n_folds}")

    all_results: list[RepresentationResult] = []
    total_start = time.time()

    # Run study for each representation type
    for use_kspace in [False, True]:
        rep_name = "k-space" if use_kspace else "image"
        logger.info(f"\nGenerating {rep_name} training data...")

        X, y, sample_ids, _ = generate_training_data(
            batch_dir, config, use_kspace
        )

        result = run_kfold_for_representation(
            X, y, sample_ids, use_kspace, config, models_dir
        )
        all_results.append(result)

    total_time = time.time() - total_start
    logger.info(f"\nStudy completed in {total_time:.1f}s")

    # Save results summary
    summary = {
        "batch_dir": str(batch_dir),
        "study_type": "2d_kspace_vs_image",
        "n_folds": config.n_folds,
        "random_seed": config.random_seed,
        "config": {
            "architecture": config.architecture,
            "grid_size": config.grid_size,
            "trim_timesteps": config.trim_timesteps,
            "downsample_factor": config.downsample_factor,
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
                "mean_pixel_error": r.mean_pixel_error,
                "std_pixel_error": r.std_pixel_error,
                "folds": [
                    {
                        "fold": f.fold,
                        "train_loss": f.train_loss,
                        "test_loss": f.test_loss,
                        "pixel_error": f.pixel_error,
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

    study_name = "study_2d_kspace_vs_image"
    summary_path = study_output_dir / f"{study_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Generate charts
    generate_charts(all_results, study_output_dir, study_name)

    # Print summary table
    print("\n" + "=" * 80)
    print("2D K-SPACE VS IMAGE STUDY RESULTS (5-Fold CV)")
    print("=" * 80)
    print(f"{'Representation':<15} {'Train Loss':<22} {'Test Loss':<22} {'Pixel Error':<20}")
    print("-" * 80)
    for r in all_results:
        train_str = f"{r.mean_train_loss:.4e} ± {r.std_train_loss:.4e}"
        test_str = f"{r.mean_test_loss:.4e} ± {r.std_test_loss:.4e}"
        pixel_str = f"{r.mean_pixel_error:.1f} ± {r.std_pixel_error:.1f}"
        print(f"{r.representation.title():<15} {train_str:<22} {test_str:<22} {pixel_str:<20}")
    print("=" * 80)

    # Print recommendation
    image_result = next(r for r in all_results if r.representation == "image")
    kspace_result = next(r for r in all_results if r.representation == "k-space")

    if image_result.mean_pixel_error < kspace_result.mean_pixel_error:
        winner = "Image"
        improvement = (
            (kspace_result.mean_pixel_error - image_result.mean_pixel_error)
            / kspace_result.mean_pixel_error
            * 100
        )
    else:
        winner = "K-space"
        improvement = (
            (image_result.mean_pixel_error - kspace_result.mean_pixel_error)
            / image_result.mean_pixel_error
            * 100
        )

    print(f"\nRecommendation: {winner} representation ({improvement:.1f}% better pixel error)")


def main():
    parser = argparse.ArgumentParser(
        description="2D K-space vs Image representation study with k-fold cross-validation"
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default=str(DEFAULT_BATCH_DIR),
        help="Path to 2D simulation batch directory",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mlp",
        choices=["mlp", "cnn"],
        help="Network architecture (default: mlp)",
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
        "--grid-size",
        type=int,
        default=64,
        help="Grid size for image/k-space (default: 64)",
    )

    args = parser.parse_args()

    configure_logging(log_to_file=True)

    config = StudyConfig(
        architecture=args.architecture,
        epochs=args.epochs,
        n_folds=args.folds,
        random_seed=args.seed,
        grid_size=args.grid_size,
    )

    run_study(Path(args.batch_dir), config)


if __name__ == "__main__":
    main()
