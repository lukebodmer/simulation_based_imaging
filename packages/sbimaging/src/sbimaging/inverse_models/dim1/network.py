"""Neural network models for 1D inverse problems.

Implements MLP architecture for learning mappings from
1D FDTD sensor data to density profiles.
"""

import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sbimaging.inverse_models.base import InverseModel, train_test_split_by_index
from sbimaging.logging import get_logger


class MLPRegressor1D(nn.Module):
    """MLP architecture for 1D inverse problems.

    Simplified architecture suitable for ~500 training samples.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InclusionWeightedMSELoss(nn.Module):
    """MSE loss that weights inclusion regions more heavily.

    Since most of the density profile is background (1.0), this loss
    increases the weight on regions where the target differs from background,
    helping the network focus on predicting inclusion location and properties.
    """

    def __init__(self, background_density: float = 1.0, inclusion_weight: float = 1.0):
        """Initialize weighted loss.

        Args:
            background_density: The background density value.
            inclusion_weight: Weight multiplier for inclusion regions.
        """
        super().__init__()
        self.background_density = background_density
        self.inclusion_weight = inclusion_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss.

        Args:
            pred: Predicted density profiles (batch, grid_size).
            target: Ground truth density profiles (batch, grid_size).

        Returns:
            Weighted MSE loss scalar.
        """
        # Compute per-element squared error
        squared_error = (pred - target) ** 2

        # Create weight mask: higher weight where target differs from background
        # Weight = 1 for background, inclusion_weight for inclusion
        is_inclusion = (target - self.background_density).abs() > 0.1
        weights = torch.ones_like(target)
        weights[is_inclusion] = self.inclusion_weight

        # Weighted mean
        weighted_loss = (squared_error * weights).sum() / weights.sum()

        return weighted_loss


class NeuralNetwork1D(InverseModel):
    """Neural network inverse model for 1D FDTD simulations.

    Learns inverse mappings from sensor data to density profiles.
    """

    def __init__(
        self,
        name: str = "neural_network_1d",
    ):
        """Initialize neural network model.

        Args:
            name: Model name.
        """
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None
        self._logger = get_logger(__name__)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_fraction: float = 0.1,
        epochs: int = 500,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        sample_ids: list[str] | None = None,
        progress_callback: Callable[[int, int, float, float], None] | None = None,
    ) -> dict[str, float]:
        """Train the neural network.

        Args:
            X: Input features (n_samples, n_features).
            y: Target outputs (n_samples, n_outputs).
            test_fraction: Fraction for test set.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Initial learning rate.
            weight_decay: Weight decay for regularization.
            sample_ids: Optional sample identifiers.
            progress_callback: Optional callback for progress updates.
                Signature: (epoch, total_epochs, train_loss, test_loss) -> None

        Returns:
            Dictionary with final training and test loss.
        """
        self._input_dim = X.shape[1]
        self._output_dim = y.shape[1]

        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(X))]

        X_train, X_test, y_train, y_test, train_ids, test_ids = (
            train_test_split_by_index(X, y, sample_ids, test_fraction)
        )

        self.train_indices = train_ids
        self.test_indices = test_ids

        self._model = self._create_model()
        self._model.to(self.device)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Use ReduceLROnPlateau for adaptive learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=50,
            min_lr=1e-6,
        )

        criterion = InclusionWeightedMSELoss(background_density=1.0, inclusion_weight=10.0)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        self._logger.info(f"Training MLP model for {epochs} epochs")
        self._logger.info(f"Device: {self.device}, AMP: {use_amp}")

        train_loss = 0.0
        test_loss = 0.0

        for epoch in range(epochs):
            train_loss = self._train_epoch(
                train_loader, optimizer, criterion, scaler, use_amp
            )
            test_loss = self._evaluate_epoch(test_loader, criterion, use_amp)

            # Step scheduler based on test loss
            scheduler.step(test_loss)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                lr = optimizer.param_groups[0]["lr"]
                self._logger.info(
                    f"Epoch {epoch + 1:03d}/{epochs} | "
                    f"Train: {train_loss:.6e} | Test: {test_loss:.6e} | LR: {lr:.2e}"
                )

            if progress_callback is not None:
                progress_callback(epoch + 1, epochs, train_loss, test_loss)

        return {"train_loss": train_loss, "test_loss": test_loss}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for inputs.

        Args:
            X: Input features (n_samples, n_features).

        Returns:
            Predicted outputs (n_samples, n_outputs).
        """
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        self._model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self._model(X_tensor).cpu().numpy()

        return pred

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
            "model_state_dict": self._model.state_dict(),
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._logger.info(f"Model saved to {path}")

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

        self._input_dim = data["input_dim"]
        self._output_dim = data["output_dim"]
        self.train_indices = data.get("train_indices", [])
        self.test_indices = data.get("test_indices", [])

        self._model = self._create_model()
        self._model.load_state_dict(data["model_state_dict"])
        self._model.to(self.device)

        self._logger.info(f"Model loaded from {path}")

    def _create_model(self) -> nn.Module:
        """Create network."""
        if self._input_dim is None or self._output_dim is None:
            raise RuntimeError("Input/output dimensions not set")

        return MLPRegressor1D(self._input_dim, self._output_dim)

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scaler: torch.amp.GradScaler,
        use_amp: bool,
    ) -> float:
        """Train for one epoch."""
        self._model.train()
        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)

        return total_loss / n_samples

    def _evaluate_epoch(
        self, loader: DataLoader, criterion: nn.Module, use_amp: bool
    ) -> float:
        """Evaluate on a dataset."""
        self._model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                n_samples += X_batch.size(0)

        return total_loss / n_samples
