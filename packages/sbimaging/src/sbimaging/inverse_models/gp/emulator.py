"""Gaussian Process emulator using PyRobustGaSP.

Wraps PyRobustGaSP for Parallel Partial Gaussian Process emulation
suitable for high-dimensional inverse problems.
"""

import pickle
from pathlib import Path

import numpy as np

from sbimaging.inverse_models.base import InverseModel, train_test_split_by_index
from sbimaging.inverse_models.gp.PyRobustGaSP import PyRobustGaSP
from sbimaging.logging import get_logger


class GaussianProcessModel(InverseModel):
    """Gaussian Process inverse model using Parallel Partial Emulation.

    Uses PyRobustGaSP for efficient GP emulation of high-dimensional
    outputs through parallel independent GPs.

    Attributes:
        isotropic: Whether to use isotropic covariance.
        nugget_est: Whether to estimate nugget parameter.
    """

    def __init__(
        self,
        name: str = "gaussian_process",
        isotropic: bool = True,
        nugget_est: bool = True,
        num_initial_values: int = 10,
    ):
        """Initialize Gaussian Process model.

        Args:
            name: Model name.
            isotropic: Use isotropic covariance function.
            nugget_est: Estimate nugget parameter.
            num_initial_values: Number of initial values for optimization.
        """
        super().__init__(name)
        self.isotropic = isotropic
        self.nugget_est = nugget_est
        self.num_initial_values = num_initial_values

        self._rgasp = PyRobustGaSP()
        self._model = None
        self._mask: np.ndarray | None = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_fraction: float = 0.1,
        sample_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Train the Gaussian Process emulator.

        Args:
            X: Input features (n_samples, n_features).
            y: Target outputs (n_samples, n_outputs).
            test_fraction: Fraction for test set.
            sample_ids: Optional sample identifiers.

        Returns:
            Dictionary with training metrics.
        """
        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(X))]

        X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split_by_index(
            X, y, sample_ids, test_fraction
        )

        self.train_indices = train_ids
        self.test_indices = test_ids

        y_train, self._mask = self._remove_constant_columns(y_train)
        y_test_filtered, _ = self._remove_constant_columns(y_test, ref_mask=self._mask)

        self._logger.info(
            f"Training GP with {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} inputs, {y_train.shape[1]} outputs"
        )

        task = self._rgasp.create_task(
            X_train,
            y_train,
            isotropic=self.isotropic,
            num_initial_values=self.num_initial_values,
            nugget_est=self.nugget_est,
        )

        self._model = self._rgasp.train_ppgasp(task)

        self._logger.info("GP training complete")

        if len(X_test) > 0:
            pred = self._rgasp.predict_ppgasp(self._model, X_test)["mean"]
            mse = np.mean((pred - y_test_filtered) ** 2)
            return {"test_mse": float(mse)}

        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for inputs.

        Args:
            X: Input features (n_samples, n_features).

        Returns:
            Predicted outputs (n_samples, n_outputs).
        """
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        pred = self._rgasp.predict_ppgasp(self._model, X)["mean"]

        if self._mask is not None:
            full_pred = np.zeros((pred.shape[0], len(self._mask)))
            full_pred[:, self._mask] = pred
            return full_pred

        return pred

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty quantification.

        Args:
            X: Input features (n_samples, n_features).

        Returns:
            Tuple of (mean, variance) arrays.
        """
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        result = self._rgasp.predict_ppgasp(self._model, X)
        mean = result["mean"]
        var = result.get("var", np.zeros_like(mean))

        if self._mask is not None:
            full_mean = np.zeros((mean.shape[0], len(self._mask)))
            full_var = np.zeros((var.shape[0], len(self._mask)))
            full_mean[:, self._mask] = mean
            full_var[:, self._mask] = var
            return full_mean, full_var

        return mean, var

    def save(self, path: Path) -> None:
        """Save model to file.

        Args:
            path: Path to save model.
        """
        if self._model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self._model,
            "model_type": "gaussian_process",
            "mask": self._mask,
            "isotropic": self.isotropic,
            "nugget_est": self.nugget_est,
            "num_initial_values": self.num_initial_values,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._logger.info(f"GP model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model from file.

        Args:
            path: Path to load model from.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self._mask = data.get("mask")
        self.isotropic = data.get("isotropic", True)
        self.nugget_est = data.get("nugget_est", True)
        self.num_initial_values = data.get("num_initial_values", 10)
        self.train_indices = data.get("train_indices", [])
        self.test_indices = data.get("test_indices", [])

        self._logger.info(f"GP model loaded from {path}")

    def _remove_constant_columns(
        self,
        y: np.ndarray,
        ref_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove constant columns from output matrix.

        Args:
            y: Output matrix.
            ref_mask: Reference mask to apply (for test data).

        Returns:
            Tuple of (filtered_y, mask).
        """
        if ref_mask is None:
            stds = np.std(y, axis=0)
            mask = stds > 1e-12
            dropped = np.sum(~mask)
            if dropped > 0:
                self._logger.info(
                    f"Dropped {dropped} constant columns out of {y.shape[1]}"
                )
        else:
            mask = ref_mask

        return y[:, mask], mask
