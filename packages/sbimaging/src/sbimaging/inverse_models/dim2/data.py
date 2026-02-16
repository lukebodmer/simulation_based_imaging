"""Data loading and preparation for 2D inverse models.

Handles loading sensor data and converting inclusion parameters
to k-space representation for training neural networks.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from sbimaging.inverse_models.dim2.kspace import inclusion_to_kspace
from sbimaging.logging import get_logger


class DataLoader2D:
    """Loads and prepares training data from 2D FDTD batch simulations.

    Scans simulation directories for sensor data and parameters,
    converting them to input/output pairs for neural network training.

    Attributes:
        batch_dir: Root directory of the batch simulation.
        grid_size: Resolution of the k-space grid.
    """

    def __init__(
        self,
        batch_dir: Path | str,
        grid_size: int = 64,
        trim_timesteps: int = 50,
        downsample_factor: int = 2,
    ):
        """Initialize data loader.

        Args:
            batch_dir: Root directory containing batch simulation data.
            grid_size: Resolution for k-space representation.
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

    def load(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load all available training data.

        Returns:
            Tuple of (X, y, sample_ids) where:
            - X: Sensor data inputs (n_samples, n_features)
            - y: K-space outputs (n_samples, grid_size^2 * 2)
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
                y = self._load_kspace_target(param_file)

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

        return data.ravel().astype(np.float32)

    def _load_kspace_target(self, path: Path) -> np.ndarray:
        """Load parameters and convert to k-space.

        Args:
            path: Path to parameter JSON file.

        Returns:
            Flattened k-space representation.
        """
        with open(path) as f:
            params = json.load(f)

        kspace = inclusion_to_kspace(
            inclusion_type=params["inclusion_type"],
            center_x=params["center_x"],
            center_y=params["center_y"],
            inclusion_size=params["inclusion_size"],
            domain_size=params["domain_size"],
            grid_size=self.grid_size,
        )

        return kspace.to_flat()


def prepare_training_data(
    batch_dir: Path | str,
    output_dir: Path | str | None = None,
    grid_size: int = 64,
    trim_timesteps: int = 50,
    downsample_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare training data from a 2D FDTD batch simulation.

    Convenience function that loads data and optionally saves
    model_input.pkl and model_output.pkl files.

    Args:
        batch_dir: Root directory of the batch simulation.
        output_dir: If provided, save prepared data to this directory.
        grid_size: Resolution for k-space representation.
        trim_timesteps: Number of initial timesteps to remove.
        downsample_factor: Factor to downsample timesteps.

    Returns:
        Tuple of (X, y, sample_ids).
    """
    loader = DataLoader2D(
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
