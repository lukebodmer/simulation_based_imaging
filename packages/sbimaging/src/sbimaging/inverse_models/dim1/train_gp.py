"""Training script for 1D Gaussian Process inverse models.

Provides a simple interface for training Gaussian Process models on 1D FDTD
simulation data, mapping sensor measurements to density profiles with
uncertainty quantification.
"""

import argparse
from pathlib import Path

import numpy as np

from sbimaging.inverse_models.base import train_test_split_by_index
from sbimaging.inverse_models.dim1.data import DataLoader1D
from sbimaging.inverse_models.dim1.gp import GaussianProcess1D
from sbimaging.logging import get_logger


def add_gaussian_noise(
    X: np.ndarray,
    noise_percent: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise to sensor data.

    Args:
        X: Input sensor data array (n_samples, n_features).
        noise_percent: Noise level as percentage of peak amplitude (0-100).
        rng: Random number generator for reproducibility.

    Returns:
        Noisy sensor data with same shape as input.
    """
    if noise_percent <= 0:
        return X

    if rng is None:
        rng = np.random.default_rng()

    peak_amplitude = np.abs(X).max()
    noise_std = (noise_percent / 100.0) * peak_amplitude
    noise = rng.normal(0, noise_std, size=X.shape).astype(X.dtype)
    return X + noise


def train_1d_gp_inverse_model(
    batch_dir: Path | str,
    output_path: Path | str,
    grid_size: int = 100,
    test_fraction: float = 0.1,
    trim_timesteps: int = 500,
    downsample_factor: int = 4,
    noise_percent: float = 0.0,
    noise_seed: int | None = None,
    isotropic: bool = True,
    nugget_est: bool = True,
    num_initial_values: int = 10,
) -> dict:
    """Train a 1D Gaussian Process inverse model on FDTD simulation data.

    Args:
        batch_dir: Root directory of the batch simulation.
        output_path: Path to save the trained model.
        grid_size: Number of points in density profile output.
        test_fraction: Fraction of data for testing.
        trim_timesteps: Initial timesteps to remove from sensor data.
        downsample_factor: Factor to downsample sensor timesteps.
        noise_percent: Gaussian noise level as percentage of peak amplitude.
        noise_seed: Random seed for noise generation.
        isotropic: Use isotropic covariance function.
        nugget_est: Estimate nugget parameter.
        num_initial_values: Number of initial values for optimization.

    Returns:
        Dictionary with training results and metrics.
    """
    logger = get_logger(__name__)

    logger.info(f"Loading data from {batch_dir}")
    loader = DataLoader1D(
        batch_dir=batch_dir,
        grid_size=grid_size,
        trim_timesteps=trim_timesteps,
        downsample_factor=downsample_factor,
    )
    X, y, sample_ids = loader.load()

    logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")

    if noise_percent > 0:
        rng = np.random.default_rng(noise_seed)
        X = add_gaussian_noise(X, noise_percent, rng)
        logger.info(f"Added {noise_percent}% Gaussian noise (seed={noise_seed})")

    model = GaussianProcess1D(
        name="1d_gp_inverse",
        isotropic=isotropic,
        nugget_est=nugget_est,
        num_initial_values=num_initial_values,
    )

    logger.info("Training Gaussian Process model...")
    results = model.train(
        X=X,
        y=y,
        test_fraction=test_fraction,
        sample_ids=sample_ids,
    )

    _, X_test, _, y_test, train_ids, test_ids = train_test_split_by_index(
        X, y, sample_ids, test_fraction
    )

    if len(X_test) > 0:
        pred_result = model.predict_with_uncertainty(X_test, confidence=0.90)
        pred_mean = pred_result["mean"]
        pred_lower = pred_result["lower"]
        pred_upper = pred_result["upper"]

        mse = np.mean((pred_mean - y_test) ** 2)
        mae = np.mean(np.abs(pred_mean - y_test))

        in_interval = (y_test >= pred_lower) & (y_test <= pred_upper)
        coverage = np.mean(in_interval)

        logger.info(f"Test MSE: {mse:.6e}")
        logger.info(f"Test MAE: {mae:.6e}")
        logger.info(f"90% CI Coverage: {coverage:.2%}")

        results["test_mse"] = float(mse)
        results["test_mae"] = float(mae)
        results["coverage_90"] = float(coverage)

    output_path = Path(output_path)
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")

    return {
        "results": results,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "grid_size": grid_size,
        "noise_percent": noise_percent,
        "noise_seed": noise_seed,
    }


if __name__ == "__main__":
    from sbimaging import configure_logging

    parser = argparse.ArgumentParser(description="Train 1D GP inverse model")
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="/data/1d-simulations",
        help="Path to batch simulation directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/1d_gp_inverse_model.pkl",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Density profile resolution",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--noise-percent",
        type=float,
        default=0.0,
        help="Gaussian noise level as percentage of peak amplitude (0-100)",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Random seed for noise generation",
    )
    parser.add_argument(
        "--anisotropic",
        action="store_true",
        help="Use anisotropic covariance (default: isotropic). Only use with "
        "low-dimensional inputs; will fail with high-dimensional sensor data.",
    )
    parser.add_argument(
        "--no-nugget",
        action="store_true",
        help="Do not estimate nugget parameter",
    )
    parser.add_argument(
        "--num-initial-values",
        type=int,
        default=10,
        help="Number of initial values for optimization",
    )

    args = parser.parse_args()

    configure_logging()

    train_1d_gp_inverse_model(
        batch_dir=args.batch_dir,
        output_path=args.output,
        grid_size=args.grid_size,
        test_fraction=args.test_fraction,
        noise_percent=args.noise_percent,
        noise_seed=args.noise_seed,
        isotropic=not args.anisotropic,
        nugget_est=not args.no_nugget,
        num_initial_values=args.num_initial_values,
    )
