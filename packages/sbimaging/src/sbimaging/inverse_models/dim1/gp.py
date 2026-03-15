"""Gaussian Process models for 1D inverse problems.

Wraps PyRobustGaSP for learning mappings from 1D FDTD sensor data
to density profiles with uncertainty quantification.
"""

import pickle
from pathlib import Path

import numpy as np
import scipy.stats

from sbimaging.inverse_models.base import InverseModel, train_test_split_by_index
from sbimaging.inverse_models.gp.PyRobustGaSP import PyRobustGaSP
from sbimaging.logging import get_logger


class GaussianProcess1D(InverseModel):
    """Gaussian Process inverse model for 1D FDTD simulations.

    Uses Parallel Partial Gaussian Process emulation (PPGaSP) for
    learning inverse mappings from sensor data to density profiles.
    Provides uncertainty quantification via predictive distributions.

    Attributes:
        isotropic: Whether to use isotropic covariance.
        nugget_est: Whether to estimate nugget parameter.
        num_initial_values: Number of starting points for optimization.
    """

    def __init__(
        self,
        name: str = "gaussian_process_1d",
        isotropic: bool = True,
        nugget_est: bool = True,
        num_initial_values: int = 10,
    ):
        """Initialize Gaussian Process model.

        Args:
            name: Model name.
            isotropic: Use isotropic covariance function. Default True, which is
                appropriate for high-dimensional inputs (flattened sensor data).
                Anisotropic only makes sense with low-dimensional structured inputs.
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
        self._constant_values: np.ndarray | None = None  # Values for constant columns
        self._logger = get_logger(__name__)

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

        X_train, X_test, y_train, y_test, train_ids, test_ids = (
            train_test_split_by_index(X, y, sample_ids, test_fraction)
        )

        self.train_indices = train_ids
        self.test_indices = test_ids

        y_train, self._mask, self._constant_values = self._remove_constant_columns(
            y_train
        )
        y_test_filtered, _, _ = self._remove_constant_columns(
            y_test, ref_mask=self._mask
        )

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
            full_pred = self._expand_to_full_output(pred)
            return full_pred

        return pred

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        confidence: float = 0.90,
    ) -> dict[str, np.ndarray]:
        """Predict with uncertainty quantification.

        Uses the posterior predictive distribution to compute
        credible intervals at the specified confidence level.

        Args:
            X: Input features (n_samples, n_features).
            confidence: Confidence level for interval (default 0.90 for 90%).

        Returns:
            Dictionary with:
                - mean: Predicted mean (n_samples, n_outputs)
                - std: Standard deviation (n_samples, n_outputs)
                - lower: Lower bound of credible interval
                - upper: Upper bound of credible interval
        """
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        result = self._rgasp.predict_ppgasp(self._model, X)
        mean = result["mean"]
        std = result["sd"]

        # Compute credible interval using t-distribution
        # degrees of freedom from model
        df = self._model["num_obs"] - self._model["q"]
        alpha = 1 - confidence
        t_crit = scipy.stats.t.ppf(1 - alpha / 2, df)

        lower = mean - t_crit * std
        upper = mean + t_crit * std

        # Expand to full output dimension if mask was applied
        if self._mask is not None:
            full_mean = self._expand_to_full_output(mean)
            full_std = self._expand_to_full_output(std, fill_value=0.0)
            full_lower = self._expand_to_full_output(lower)
            full_upper = self._expand_to_full_output(upper)

            return {
                "mean": full_mean,
                "std": full_std,
                "lower": full_lower,
                "upper": full_upper,
            }

        return {
            "mean": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
        }

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
            "model_type": "gaussian_process_1d",
            "mask": self._mask,
            "constant_values": self._constant_values,
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
        self._constant_values = data.get("constant_values")
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove constant columns from output matrix.

        Args:
            y: Output matrix.
            ref_mask: Reference mask to apply (for test data).

        Returns:
            Tuple of (filtered_y, mask, constant_values).
            constant_values contains the value from the first row for each column.
        """
        if ref_mask is None:
            stds = np.std(y, axis=0)
            mask = stds > 1e-12
            dropped = np.sum(~mask)
            if dropped > 0:
                self._logger.info(
                    f"Dropped {dropped} constant columns out of {y.shape[1]}"
                )
            # Store the constant values (from first row, since they're constant)
            constant_values = y[0, :].copy()
        else:
            mask = ref_mask
            constant_values = (
                self._constant_values
                if self._constant_values is not None
                else y[0, :].copy()
            )

        return y[:, mask], mask, constant_values

    def _expand_to_full_output(
        self,
        reduced: np.ndarray,
        fill_value: float | None = None,
    ) -> np.ndarray:
        """Expand reduced prediction back to full output dimension.

        Args:
            reduced: Reduced prediction array (n_samples, n_kept_columns).
            fill_value: Value to fill constant columns with. If None, uses
                the stored constant values from training data.

        Returns:
            Full prediction array (n_samples, n_total_columns).
        """
        if self._mask is None:
            return reduced

        n_samples = reduced.shape[0]
        n_total = len(self._mask)
        full = np.zeros((n_samples, n_total))

        # Fill constant columns with their stored values
        if fill_value is not None:
            full[:, ~self._mask] = fill_value
        elif self._constant_values is not None:
            full[:, ~self._mask] = self._constant_values[~self._mask]

        # Fill varying columns with predictions
        full[:, self._mask] = reduced

        return full
