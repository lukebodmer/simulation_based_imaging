"""Training script for 2D inverse models.

Provides a simple interface for training neural networks on 2D FDTD
simulation data, mapping sensor measurements to k-space coefficients.
"""

from pathlib import Path

import numpy as np

from sbimaging.inverse_models.base import train_test_split_by_index
from sbimaging.inverse_models.dim2.data import DataLoader2D
from sbimaging.inverse_models.dim2.kspace import KSpace2D, kspace_to_image
from sbimaging.inverse_models.dim2.network import NeuralNetwork2D
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
            For example, 5.0 means noise std = 5% of the peak absolute value.
        rng: Random number generator for reproducibility.

    Returns:
        Noisy sensor data with same shape as input.
    """
    if noise_percent <= 0:
        return X

    if rng is None:
        rng = np.random.default_rng()

    # Calculate peak amplitude across all samples
    peak_amplitude = np.abs(X).max()

    # Noise standard deviation as fraction of peak
    noise_std = (noise_percent / 100.0) * peak_amplitude

    # Generate and add noise
    noise = rng.normal(0, noise_std, size=X.shape).astype(X.dtype)
    return X + noise


def train_2d_inverse_model(
    batch_dir: Path | str,
    output_path: Path | str,
    architecture: str = "mlp",
    grid_size: int = 64,
    epochs: int = 500,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    test_fraction: float = 0.1,
    trim_timesteps: int = 50,
    downsample_factor: int = 2,
    noise_percent: float = 0.0,
    noise_seed: int | None = None,
) -> dict:
    """Train a 2D inverse model on FDTD simulation data.

    Args:
        batch_dir: Root directory of the batch simulation.
        output_path: Path to save the trained model.
        architecture: Network architecture ("mlp" or "cnn").
        grid_size: Resolution for k-space representation.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate.
        test_fraction: Fraction of data for testing.
        trim_timesteps: Initial timesteps to remove from sensor data.
        downsample_factor: Factor to downsample sensor timesteps.
        noise_percent: Gaussian noise level as percentage of peak amplitude (0-100).
            Added to sensor data before training to test model robustness.
        noise_seed: Random seed for noise generation (for reproducibility).

    Returns:
        Dictionary with training results and metrics.
    """
    logger = get_logger(__name__)

    # Load data
    logger.info(f"Loading data from {batch_dir}")
    loader = DataLoader2D(
        batch_dir=batch_dir,
        grid_size=grid_size,
        trim_timesteps=trim_timesteps,
        downsample_factor=downsample_factor,
    )
    X, y, sample_ids = loader.load()

    logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")

    # Add noise if requested
    if noise_percent > 0:
        rng = np.random.default_rng(noise_seed)
        X = add_gaussian_noise(X, noise_percent, rng)
        logger.info(f"Added {noise_percent}% Gaussian noise (seed={noise_seed})")

    # Create and train model
    model = NeuralNetwork2D(
        name=f"2d_inverse_{architecture}",
        architecture=architecture,
    )

    logger.info(f"Training {architecture} model for {epochs} epochs")
    results = model.train(
        X=X,
        y=y,
        test_fraction=test_fraction,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sample_ids=sample_ids,
    )

    # Evaluate on test set
    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split_by_index(
        X, y, sample_ids, test_fraction
    )
    metrics = model.evaluate(X_test, y_test)

    logger.info(f"Test metrics: MSE={metrics['mse']:.6e}, R²={metrics['r2']:.4f}")

    # Save model
    output_path = Path(output_path)
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")

    return {
        "train_loss": results["train_loss"],
        "test_loss": results["test_loss"],
        "metrics": metrics,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "grid_size": grid_size,
        "noise_percent": noise_percent,
        "noise_seed": noise_seed,
    }


def predict_and_visualize(
    model_path: Path | str,
    batch_dir: Path | str,
    sim_id: str,
    grid_size: int = 64,
    trim_timesteps: int = 50,
    downsample_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a model and predict k-space for a simulation.

    Args:
        model_path: Path to saved model.
        batch_dir: Root directory of batch simulation.
        sim_id: Simulation ID to predict.
        grid_size: K-space grid resolution.
        trim_timesteps: Initial timesteps to remove.
        downsample_factor: Downsample factor for sensor data.

    Returns:
        Tuple of (predicted_image, ground_truth_image).
    """
    import json

    batch_dir = Path(batch_dir)

    # Load model
    model = NeuralNetwork2D()
    model.load(Path(model_path))

    # Load sensor data
    sensor_path = batch_dir / "simulations" / sim_id / "sensor_data.npy"
    sensor_data = np.load(sensor_path)

    if trim_timesteps > 0:
        sensor_data = sensor_data[:, trim_timesteps:]
    if downsample_factor > 1:
        sensor_data = sensor_data[:, ::downsample_factor]

    X = sensor_data.ravel().astype(np.float32).reshape(1, -1)

    # Predict k-space
    y_pred = model.predict(X)[0]
    kspace_pred = KSpace2D.from_flat(y_pred, grid_size)
    pred_image = kspace_to_image(kspace_pred)

    # Load ground truth
    param_path = batch_dir / "parameters" / f"{sim_id}.json"
    with open(param_path) as f:
        params = json.load(f)

    from sbimaging.inverse_models.dim2.kspace import inclusion_to_kspace

    kspace_true = inclusion_to_kspace(
        inclusion_type=params["inclusion_type"],
        center_x=params["center_x"],
        center_y=params["center_y"],
        inclusion_size=params["inclusion_size"],
        domain_size=params["domain_size"],
        grid_size=grid_size,
    )
    true_image = kspace_to_image(kspace_true)

    return pred_image, true_image


if __name__ == "__main__":
    import argparse

    from sbimaging import configure_logging

    parser = argparse.ArgumentParser(description="Train 2D inverse model")
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="/data/2d-simulations",
        help="Path to batch simulation directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/2d_inverse_model.pkl",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mlp",
        choices=["mlp", "cnn"],
        help="Network architecture",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="K-space grid resolution",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
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
        help="Random seed for noise generation (for reproducibility)",
    )

    args = parser.parse_args()

    configure_logging()

    train_2d_inverse_model(
        batch_dir=args.batch_dir,
        output_path=args.output,
        architecture=args.architecture,
        grid_size=args.grid_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        noise_percent=args.noise_percent,
        noise_seed=args.noise_seed,
    )
