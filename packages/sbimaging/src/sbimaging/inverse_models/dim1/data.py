"""Data loading and preparation for 1D inverse models.

Handles loading sensor data and converting inclusion parameters
to density profiles for training neural networks.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from sbimaging.logging import get_logger


def dynamic_compress(signal: np.ndarray, threshold: float = 0.1, ratio: float = 4.0) -> np.ndarray:
    """Apply dynamic range compression to a signal.

    Similar to audio compression - reduces the dynamic range by attenuating
    samples that exceed the threshold.

    Args:
        signal: Input signal array.
        threshold: Amplitude threshold (0-1 of max) above which compression applies.
        ratio: Compression ratio (e.g., 4:1 means 4dB input becomes 1dB output above threshold).

    Returns:
        Compressed signal with reduced dynamic range.
    """
    # Normalize to [-1, 1] range
    max_val = np.abs(signal).max()
    if max_val == 0:
        return signal
    normalized = signal / max_val

    # Apply soft-knee compression
    abs_signal = np.abs(normalized)
    thresh = threshold

    # Above threshold: compress
    above_mask = abs_signal > thresh
    if above_mask.any():
        # How much above threshold
        excess = abs_signal[above_mask] - thresh
        # Compressed excess (ratio:1 compression)
        compressed_excess = excess / ratio
        # New amplitude = threshold + compressed excess
        new_amplitude = thresh + compressed_excess
        # Apply to signal preserving sign
        normalized[above_mask] = np.sign(normalized[above_mask]) * new_amplitude

    # Normalize again to use full range
    compressed_max = np.abs(normalized).max()
    if compressed_max > 0:
        normalized = normalized / compressed_max

    return normalized


def params_to_density_profile(
    inclusion_center: float,
    inclusion_size: float,
    inclusion_density: float,
    domain_size: float,
    grid_size: int = 100,
    background_density: float = 1.0,
) -> np.ndarray:
    """Convert inclusion parameters to a 1D density profile.

    Args:
        inclusion_center: Center position of the inclusion.
        inclusion_size: Width of the inclusion.
        inclusion_density: Density value inside the inclusion.
        domain_size: Total length of the domain.
        grid_size: Number of points in the output profile.
        background_density: Density value outside the inclusion.

    Returns:
        1D array of density values at each grid point.
    """
    # Create position array
    x = np.linspace(0, domain_size, grid_size, endpoint=False)
    x += domain_size / (2 * grid_size)  # Center of each cell

    # Initialize with background density
    profile = np.full(grid_size, background_density, dtype=np.float32)

    # Set inclusion region
    x_min = inclusion_center - inclusion_size / 2
    x_max = inclusion_center + inclusion_size / 2

    inclusion_mask = (x >= x_min) & (x <= x_max)
    profile[inclusion_mask] = inclusion_density

    return profile


class DataLoader1D:
    """Loads and prepares training data from 1D FDTD batch simulations.

    Scans simulation directories for sensor data and parameters,
    converting them to input/output pairs for neural network training.

    Attributes:
        batch_dir: Root directory of the batch simulation.
        grid_size: Resolution of the density profile output.
    """

    _debug_plot_done = False  # Class variable to only plot once

    def __init__(
        self,
        batch_dir: Path | str,
        grid_size: int = 100,
        trim_timesteps: int = 500,
        downsample_factor: int = 4,
    ):
        """Initialize data loader.

        Args:
            batch_dir: Root directory containing batch simulation data.
            grid_size: Number of points in the density profile output.
            trim_timesteps: Number of initial timesteps to remove (transient).
            downsample_factor: Factor to downsample timesteps.
        """
        self.batch_dir = Path(batch_dir)
        self.grid_size = grid_size
        self.trim_timesteps = trim_timesteps
        self.downsample_factor = downsample_factor
        self._logger = get_logger(__name__)

        self.parameters_dir = self.batch_dir / "parameters"
        self.simulations_dir = self.batch_dir / "simulations"

        # Load batch config for background density
        config_path = self.batch_dir / "batch_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)
        else:
            self._config = {"background_density": 1.0}

    def load(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load all available training data.

        Returns:
            Tuple of (X, y, sample_ids) where:
            - X: Sensor data inputs (n_samples, n_features)
            - y: Density profile outputs (n_samples, grid_size)
            - sample_ids: List of simulation IDs
        """
        X_list = []
        y_list = []
        sample_ids = []

        sim_dirs = sorted(self.simulations_dir.glob("sim_*"))

        for sim_dir in sim_dirs:
            sensor_file = sim_dir / "sensor_data.npy"
            if not sensor_file.exists():
                continue

            sim_id = sim_dir.name
            param_file = self.parameters_dir / f"{sim_id}.json"

            if not param_file.exists():
                self._logger.warning(f"Missing parameters for {sim_id}")
                continue

            try:
                x = self._load_sensor_data(sensor_file)
                y = self._load_density_target(param_file)

                X_list.append(x)
                y_list.append(y)
                sample_ids.append(sim_id)

            except Exception as e:
                self._logger.warning(f"Failed to load {sim_id}: {e}")

        if not X_list:
            raise RuntimeError("No valid training data found")

        self._logger.info(f"Loaded {len(X_list)} samples")
        return np.stack(X_list), np.stack(y_list), sample_ids

    def _load_sensor_data(self, path: Path) -> np.ndarray:
        """Load and preprocess sensor data.

        Args:
            path: Path to sensor_data.npy file.

        Returns:
            Flattened, preprocessed sensor data.
        """
        data = np.load(path)

        # Trim initial transient
        if self.trim_timesteps > 0:
            data = data[:, self.trim_timesteps:]

        # Downsample in time
        if self.downsample_factor > 1:
            data = data[:, :: self.downsample_factor]

        # Apply dynamic compression to each sensor channel separately
        # Debug plot for first sample only
        if not DataLoader1D._debug_plot_done:
            self._plot_compression_debug(data.copy())
            DataLoader1D._debug_plot_done = True

        for i in range(data.shape[0]):
            data[i] = dynamic_compress(data[i], threshold=0.1, ratio=4.0)

        return data.ravel().astype(np.float32)

    def _plot_compression_debug(self, data_before: np.ndarray) -> None:
        """Plot before/after compression for debugging."""
        import matplotlib.pyplot as plt

        data_after = data_before.copy()
        for i in range(data_after.shape[0]):
            data_after[i] = dynamic_compress(data_after[i], threshold=0.1, ratio=4.0)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Left sensor
        axes[0, 0].plot(data_before[0], label="Before", alpha=0.8)
        axes[0, 0].set_title("Left Sensor - Before Compression")
        axes[0, 0].set_xlabel("Timestep")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(data_after[0], label="After", alpha=0.8, color="orange")
        axes[0, 1].set_title("Left Sensor - After Compression")
        axes[0, 1].set_xlabel("Timestep")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].grid(True, alpha=0.3)

        # Right sensor
        axes[1, 0].plot(data_before[1], label="Before", alpha=0.8)
        axes[1, 0].set_title("Right Sensor - Before Compression")
        axes[1, 0].set_xlabel("Timestep")
        axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(data_after[1], label="After", alpha=0.8, color="orange")
        axes[1, 1].set_title("Right Sensor - After Compression")
        axes[1, 1].set_xlabel("Timestep")
        axes[1, 1].set_ylabel("Amplitude")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("compression_debug.png", dpi=150)
        plt.close()
        self._logger.info("Saved compression debug plot to compression_debug.png")

    def _load_density_target(self, path: Path) -> np.ndarray:
        """Load parameters and convert to density profile.

        Args:
            path: Path to parameter JSON file.

        Returns:
            Density profile array.
        """
        with open(path) as f:
            params = json.load(f)

        profile = params_to_density_profile(
            inclusion_center=params["inclusion_center"],
            inclusion_size=params["inclusion_size"],
            inclusion_density=params["inclusion_density"],
            domain_size=params["domain_size"],
            grid_size=self.grid_size,
            background_density=self._config.get("background_density", 1.0),
        )

        return profile


def prepare_training_data(
    batch_dir: Path | str,
    output_dir: Path | str | None = None,
    grid_size: int = 100,
    trim_timesteps: int = 500,
    downsample_factor: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare training data from a 1D FDTD batch simulation.

    Convenience function that loads data and optionally saves
    model_input.pkl and model_output.pkl files.

    Args:
        batch_dir: Root directory of the batch simulation.
        output_dir: If provided, save prepared data to this directory.
        grid_size: Number of points in density profile output.
        trim_timesteps: Number of initial timesteps to remove.
        downsample_factor: Factor to downsample timesteps.

    Returns:
        Tuple of (X, y, sample_ids).
    """
    loader = DataLoader1D(
        batch_dir=batch_dir,
        grid_size=grid_size,
        trim_timesteps=trim_timesteps,
        downsample_factor=downsample_factor,
    )

    X, y, sample_ids = loader.load()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "model_input.pkl", "wb") as f:
            pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_dir / "model_output.pkl", "wb") as f:
            pickle.dump(y, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_dir / "sample_ids.json", "w") as f:
            json.dump(sample_ids, f)

        logger = get_logger(__name__)
        logger.info(f"Saved training data to {output_dir}")

    return X, y, sample_ids
