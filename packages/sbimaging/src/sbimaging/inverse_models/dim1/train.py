"""Training script for 1D inverse models.

Provides a simple interface for training neural networks on 1D FDTD
simulation data, mapping sensor measurements to density profiles.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sbimaging.inverse_models.base import train_test_split_by_index
from sbimaging.inverse_models.dim1.data import DataLoader1D, params_to_density_profile
from sbimaging.inverse_models.dim1.network import NeuralNetwork1D
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


def train_1d_inverse_model(
    batch_dir: Path | str,
    output_path: Path | str,
    grid_size: int = 100,
    epochs: int = 500,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    test_fraction: float = 0.1,
    trim_timesteps: int = 500,
    downsample_factor: int = 4,
    noise_percent: float = 0.0,
    noise_seed: int | None = None,
    large: bool = False,
) -> dict:
    """Train a 1D inverse model on FDTD simulation data.

    Args:
        batch_dir: Root directory of the batch simulation.
        output_path: Path to save the trained model.
        grid_size: Number of points in density profile output.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate.
        test_fraction: Fraction of data for testing.
        trim_timesteps: Initial timesteps to remove from sensor data.
        downsample_factor: Factor to downsample sensor timesteps.
        noise_percent: Gaussian noise level as percentage of peak amplitude (0-100).
            Added to sensor data before training to test model robustness.
        noise_seed: Random seed for noise generation (for reproducibility).
        large: Use larger network architecture with more capacity and dropout.

    Returns:
        Dictionary with training results and metrics.
    """
    logger = get_logger(__name__)

    # Load data
    logger.info(f"Loading data from {batch_dir}")
    loader = DataLoader1D(
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
    model = NeuralNetwork1D(name="1d_inverse_mlp", large=large)
    if large:
        logger.info("Using large network architecture (512→256→128 with 0.4 dropout)")

    # Track losses for plotting
    train_losses = []
    test_losses = []

    def progress_callback(epoch, total_epochs, train_loss, test_loss):
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    logger.info(f"Training MLP model for {epochs} epochs")
    results = model.train(
        X=X,
        y=y,
        test_fraction=test_fraction,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sample_ids=sample_ids,
        progress_callback=progress_callback,
    )

    # Plot and save loss curves
    output_path = Path(output_path)
    plot_path = output_path.parent / "1d_training_loss.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss", alpha=0.8)
    ax.plot(test_losses, label="Test Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("1D Inverse Model Training")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Loss curve saved to {plot_path}")

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
    grid_size: int = 100,
    trim_timesteps: int = 500,
    downsample_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a model and predict density profile for a simulation.

    Args:
        model_path: Path to saved model.
        batch_dir: Root directory of batch simulation.
        sim_id: Simulation ID to predict.
        grid_size: Density profile resolution.
        trim_timesteps: Initial timesteps to remove.
        downsample_factor: Downsample factor for sensor data.

    Returns:
        Tuple of (predicted_profile, ground_truth_profile).
    """
    import json

    batch_dir = Path(batch_dir)

    # Load batch config for background density
    config_path = batch_dir / "batch_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        background_density = config.get("background_density", 1.0)
    else:
        background_density = 1.0

    # Load model
    model = NeuralNetwork1D()
    model.load(Path(model_path))

    # Load sensor data
    sensor_path = batch_dir / "simulations" / sim_id / "sensor_data.npy"
    sensor_data = np.load(sensor_path)

    if trim_timesteps > 0:
        sensor_data = sensor_data[:, trim_timesteps:]
    if downsample_factor > 1:
        sensor_data = sensor_data[:, ::downsample_factor]

    X = sensor_data.ravel().astype(np.float32).reshape(1, -1)

    # Predict density profile
    pred_profile = model.predict(X)[0]

    # Load ground truth
    param_path = batch_dir / "parameters" / f"{sim_id}.json"
    with open(param_path) as f:
        params = json.load(f)

    true_profile = params_to_density_profile(
        inclusion_center=params["inclusion_center"],
        inclusion_size=params["inclusion_size"],
        inclusion_density=params["inclusion_density"],
        domain_size=params["domain_size"],
        grid_size=grid_size,
        background_density=background_density,
    )

    return pred_profile, true_profile


if __name__ == "__main__":
    import argparse

    from sbimaging import configure_logging

    parser = argparse.ArgumentParser(description="Train 1D inverse model")
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="/data/1d-simulations",
        help="Path to batch simulation directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/1d_inverse_model.pkl",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Density profile resolution",
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
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
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
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use larger network architecture (512→256→128 with higher dropout)",
    )

    args = parser.parse_args()

    configure_logging()

    train_1d_inverse_model(
        batch_dir=args.batch_dir,
        output_path=args.output,
        grid_size=args.grid_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        noise_percent=args.noise_percent,
        noise_seed=args.noise_seed,
        large=args.large,
    )
