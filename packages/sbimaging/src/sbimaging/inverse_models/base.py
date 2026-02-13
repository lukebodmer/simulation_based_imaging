"""Base class for inverse models.

Provides common interface and utilities for all inverse models
used in simulation-based imaging.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from sbimaging.logging import get_logger


class InverseModel(ABC):
    """Abstract base class for inverse models.

    Defines the common interface for training, prediction, and
    persistence of inverse models.

    Attributes:
        name: Model name for identification.
        train_indices: Indices of training samples.
        test_indices: Indices of test samples.
    """

    def __init__(self, name: str = "inverse_model"):
        """Initialize inverse model.

        Args:
            name: Model name for identification.
        """
        self.name = name
        self.train_indices: list[str] = []
        self.test_indices: list[str] = []
        self._logger = get_logger(__name__)

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_fraction: float = 0.1,
    ) -> None:
        """Train the inverse model.

        Args:
            X: Input features with shape (n_samples, n_features).
            y: Target outputs with shape (n_samples, n_outputs).
            test_fraction: Fraction of data to hold out for testing.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for new inputs.

        Args:
            X: Input features with shape (n_samples, n_features).

        Returns:
            Predicted outputs with shape (n_samples, n_outputs).
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to file.

        Args:
            path: Path to save model.
        """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from file.

        Args:
            path: Path to load model from.
        """

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model on test data.

        Args:
            X_test: Test inputs.
            y_test: Test targets.

        Returns:
            Dictionary with evaluation metrics.
        """
        y_pred = self.predict(X_test)

        mse = np.mean((y_pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_test))

        y_var = np.var(y_test)
        r2 = 1 - mse / y_var if y_var > 0 else 0.0

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }


class DataLoader:
    """Loads training data from simulation outputs.

    Scans simulation directories for model_input.pkl and model_output.pkl
    files and assembles training datasets.
    """

    def __init__(self, simulations_dir: Path):
        """Initialize data loader.

        Args:
            simulations_dir: Directory containing simulation outputs.
        """
        self.simulations_dir = Path(simulations_dir)
        self._logger = get_logger(__name__)

    def load(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load all available training data.

        Returns:
            Tuple of (X, y, sample_ids) where:
            - X: Input features (n_samples, n_features)
            - y: Target outputs (n_samples, n_outputs)
            - sample_ids: List of simulation directory names
        """
        X_list = []
        y_list = []
        sample_ids = []

        for sim_dir in sorted(self.simulations_dir.iterdir()):
            if not sim_dir.is_dir():
                continue

            input_path = sim_dir / "model_input.pkl"
            output_path = sim_dir / "model_output.pkl"

            if not (input_path.exists() and output_path.exists()):
                continue

            try:
                x, y = self._load_sample(input_path, output_path)
                X_list.append(x)
                y_list.append(y)
                sample_ids.append(sim_dir.name)
            except Exception as e:
                self._logger.warning(f"Failed to load {sim_dir.name}: {e}")

        if not X_list:
            raise RuntimeError("No valid training data found")

        self._logger.info(f"Loaded {len(X_list)} samples")
        return np.stack(X_list), np.stack(y_list), sample_ids

    def _load_sample(
        self, input_path: Path, output_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a single sample."""
        with open(input_path, "rb") as f:
            x = pickle.load(f)
        with open(output_path, "rb") as f:
            y = pickle.load(f)

        if hasattr(x, "get"):
            x = x.get()
        if hasattr(y, "get"):
            y = y.get()

        return np.ravel(x).astype(np.float32), np.ravel(y).astype(np.float32)


def train_test_split_by_index(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Split data into training and test sets.

    Args:
        X: Input features.
        y: Target outputs.
        sample_ids: Sample identifiers.
        test_fraction: Fraction for test set.
        seed: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, train_ids, test_ids).
    """
    rng = np.random.default_rng(seed)
    n = len(sample_ids)
    indices = rng.permutation(n)

    n_test = max(1, int(n * test_fraction))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    train_ids = [sample_ids[i] for i in train_idx]
    test_ids = [sample_ids[i] for i in test_idx]

    return X_train, X_test, y_train, y_test, train_ids, test_ids
