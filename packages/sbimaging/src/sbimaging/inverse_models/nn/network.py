"""Neural network model for inverse problems.

Implements CNN and MLP architectures for learning mappings from
sensor data to material properties or k-space coefficients.
"""

import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sbimaging.inverse_models.base import InverseModel, train_test_split_by_index


class ResidualBlock1D(nn.Module):
    """Residual block for 1D convolutional networks."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class ResidualBlock2D(nn.Module):
    """Residual block for 2D convolutional networks."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class Conv2DRegressor(nn.Module):
    """2D CNN architecture that preserves sensor × time structure.

    Instead of flattening sensor data to a 1D vector, this architecture
    treats the input as a 2D grid (num_sensors × timesteps) and applies
    2D convolutions to capture both spatial correlations between sensors
    and temporal patterns.

    The architecture supports asymmetric kernels and strides to handle
    the different nature of the sensor (spatial) and time (temporal) dimensions.

    Attributes:
        num_sensors: Number of sensors (height dimension).
        timesteps: Number of timesteps per sensor (width dimension).
        conv_channels: List of channel sizes for conv layers.
        pool_size: Output size of adaptive average pooling (height, width).
        regressor_hidden: Hidden layer size in the regressor MLP.
        dropout: Dropout probability in the regressor.
        kernel_size: Kernel size as (H, W) tuple. Defaults to (3, 11) for
            narrow sensor receptive field and wide temporal receptive field.
        stride: Stride as (H, W) tuple. Defaults to (1, 2) to preserve
            sensor resolution while compressing time.
    """

    def __init__(
        self,
        num_sensors: int,
        timesteps: int,
        output_dim: int,
        conv_channels: list[int] | None = None,
        pool_size: tuple[int, int] = (12, 8),
        regressor_hidden: int = 512,
        dropout: float = 0.2,
        kernel_size: tuple[int, int] = (3, 11),
        stride: tuple[int, int] = (1, 2),
        use_residual: bool = True,
    ):
        """Initialize 2D CNN regressor.

        Args:
            num_sensors: Number of sensors (height dimension).
            timesteps: Number of timesteps (width dimension).
            output_dim: Output dimension.
            conv_channels: List of channel sizes for conv layers. Defaults to [32, 64].
            pool_size: Output size of adaptive average pooling (H, W). Defaults to (12, 8).
            regressor_hidden: Hidden layer size in the regressor MLP. Defaults to 512.
            dropout: Dropout probability (0-1). Defaults to 0.2.
            kernel_size: Convolution kernel size (H, W). Defaults to (3, 11).
                First layer uses 2x this size for larger receptive field.
            stride: Convolution stride (H, W). Defaults to (1, 2) to preserve
                sensor spatial resolution while downsampling time.
            use_residual: Whether to include residual blocks after each conv layer.
                Defaults to True.
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64]

        self.num_sensors = num_sensors
        self.timesteps = timesteps
        self.conv_channels = conv_channels
        self.pool_size = pool_size
        self.regressor_hidden = regressor_hidden
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_residual = use_residual

        # Build feature extractor with 2D convolutions
        feature_layers = []
        in_channels = 1

        for i, out_channels in enumerate(conv_channels):
            # First layer uses larger kernel for bigger receptive field
            if i == 0:
                k_h = kernel_size[0] * 2 + 1  # e.g., 3 -> 7
                k_w = kernel_size[1]  # keep time kernel same
            else:
                k_h, k_w = kernel_size

            # Padding to maintain spatial dimensions (before stride)
            pad_h = k_h // 2
            pad_w = k_w // 2

            feature_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(k_h, k_w),
                        stride=stride,
                        padding=(pad_h, pad_w),
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            if use_residual:
                feature_layers.append(
                    ResidualBlock2D(out_channels, kernel_size=min(k_h, k_w))
                )
            in_channels = out_channels

        feature_layers.append(nn.AdaptiveAvgPool2d(pool_size))
        self.feature_extractor = nn.Sequential(*feature_layers)

        self.flat_dim = conv_channels[-1] * pool_size[0] * pool_size[1]

        self.regressor = nn.Sequential(
            nn.Linear(self.flat_dim, regressor_hidden),
            nn.LayerNorm(regressor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_sensors * timesteps) - flattened input
        # Reshape to (batch, 1, num_sensors, timesteps)
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.num_sensors, self.timesteps)

        feats = self.feature_extractor(x)
        feats = feats.flatten(start_dim=1)
        return self.regressor(feats)


class ConvRegressor(nn.Module):
    """Configurable CNN + MLP architecture for regression.

    Attributes:
        conv_channels: List of channel sizes for conv layers.
        pool_size: Output size of adaptive average pooling.
        regressor_hidden: Hidden layer size in the regressor MLP.
        dropout: Dropout probability in the regressor.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        conv_channels: list[int] | None = None,
        pool_size: int = 16,
        regressor_hidden: int = 512,
        dropout: float = 0.2,
        use_residual: bool = True,
    ):
        """Initialize CNN regressor.

        Args:
            input_dim: Input feature dimension (not used directly, kept for API consistency).
            output_dim: Output dimension.
            conv_channels: List of channel sizes for conv layers. Defaults to [32, 64].
            pool_size: Output size of adaptive average pooling. Defaults to 16.
            regressor_hidden: Hidden layer size in the regressor MLP. Defaults to 512.
            dropout: Dropout probability (0-1). Defaults to 0.2.
            use_residual: Whether to include residual blocks after each conv layer.
                Defaults to True.
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64]

        self.conv_channels = conv_channels
        self.pool_size = pool_size
        self.regressor_hidden = regressor_hidden
        self.dropout = dropout
        self.use_residual = use_residual

        # Build feature extractor dynamically
        feature_layers = []
        in_channels = 1

        for i, out_channels in enumerate(conv_channels):
            # First layer uses larger kernel
            kernel_size = 7 if i == 0 else 5
            padding = kernel_size // 2
            feature_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                ]
            )
            if use_residual:
                feature_layers.append(ResidualBlock1D(out_channels))
            in_channels = out_channels

        feature_layers.append(nn.AdaptiveAvgPool1d(pool_size))
        self.feature_extractor = nn.Sequential(*feature_layers)

        self.flat_dim = conv_channels[-1] * pool_size

        self.regressor = nn.Sequential(
            nn.Linear(self.flat_dim, regressor_hidden),
            nn.LayerNorm(regressor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feats = self.feature_extractor(x)
        feats = feats.flatten(start_dim=1)
        return self.regressor(feats)


class MLPRegressor(nn.Module):
    """Configurable MLP architecture for regression.

    Attributes:
        hidden_layers: List of hidden layer sizes.
        dropout: Dropout probability applied after each hidden layer (except last).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        """Initialize MLP regressor.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output dimension.
            hidden_layers: List of hidden layer sizes. Defaults to [4096, 2048, 1024].
            dropout: Dropout probability (0-1). Applied after each hidden layer
                except the last one. Defaults to 0.2.
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [4096, 2048, 1024]

        self.hidden_layers = hidden_layers
        self.dropout = dropout

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            # Apply dropout to all but the last hidden layer
            if i < len(hidden_layers) - 1 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralNetworkModel(InverseModel):
    """Neural network inverse model.

    Supports CNN, CNN2D, and MLP architectures for learning inverse
    mappings from sensor data.

    Attributes:
        architecture: Network architecture ("cnn", "cnn2d", or "mlp").
        device: Torch device for computation.
        mlp_hidden_layers: Hidden layer sizes for MLP architecture.
        mlp_dropout: Dropout probability for MLP architecture.
        cnn_conv_channels: Conv channel sizes for CNN architecture.
        cnn_pool_size: Adaptive pool output size for CNN.
        cnn_regressor_hidden: Hidden layer size in CNN regressor.
        cnn_dropout: Dropout probability for CNN architecture.
        cnn_use_residual: Whether to use residual blocks in CNN architectures.
        cnn2d_num_sensors: Number of sensors for 2D CNN input shape.
        cnn2d_pool_size: Adaptive pool output size (H, W) for 2D CNN.
        cnn2d_kernel_size: Kernel size (H, W) for 2D CNN convolutions.
        cnn2d_stride: Stride (H, W) for 2D CNN convolutions.
    """

    def __init__(
        self,
        name: str = "neural_network",
        architecture: str = "cnn",
        mlp_hidden_layers: list[int] | None = None,
        mlp_dropout: float = 0.2,
        cnn_conv_channels: list[int] | None = None,
        cnn_pool_size: int = 16,
        cnn_regressor_hidden: int = 512,
        cnn_dropout: float = 0.2,
        cnn_use_residual: bool = True,
        cnn2d_num_sensors: int | None = None,
        cnn2d_pool_size: tuple[int, int] = (12, 8),
        cnn2d_kernel_size: tuple[int, int] = (3, 11),
        cnn2d_stride: tuple[int, int] = (1, 2),
    ):
        """Initialize neural network model.

        Args:
            name: Model name.
            architecture: Network architecture ("cnn", "cnn2d", or "mlp").
            mlp_hidden_layers: Hidden layer sizes for MLP. Defaults to [4096, 2048, 1024].
            mlp_dropout: Dropout probability for MLP (0-1). Defaults to 0.2.
            cnn_conv_channels: Conv channel sizes for CNN/CNN2D. Defaults to [32, 64].
            cnn_pool_size: Adaptive pool output size for 1D CNN. Defaults to 16.
            cnn_regressor_hidden: Hidden layer size in CNN regressor. Defaults to 512.
            cnn_dropout: Dropout probability for CNN (0-1). Defaults to 0.2.
            cnn_use_residual: Whether to use residual blocks after conv layers.
                Defaults to True.
            cnn2d_num_sensors: Number of sensors for 2D CNN. Required for cnn2d architecture.
            cnn2d_pool_size: Adaptive pool output size (H, W) for 2D CNN. Defaults to (12, 8).
            cnn2d_kernel_size: Kernel size (H, W) for 2D CNN. Defaults to (3, 11).
                Asymmetric kernel: small in sensor dim, large in time dim.
            cnn2d_stride: Stride (H, W) for 2D CNN. Defaults to (1, 2).
                Asymmetric stride: preserve sensor resolution, downsample time.
        """
        super().__init__(name)
        self.architecture = architecture
        # MLP config
        self.mlp_hidden_layers = mlp_hidden_layers
        self.mlp_dropout = mlp_dropout
        # CNN config (shared between cnn and cnn2d)
        self.cnn_conv_channels = cnn_conv_channels
        self.cnn_pool_size = cnn_pool_size
        self.cnn_regressor_hidden = cnn_regressor_hidden
        self.cnn_dropout = cnn_dropout
        self.cnn_use_residual = cnn_use_residual
        # CNN2D specific config
        self.cnn2d_num_sensors = cnn2d_num_sensors
        self.cnn2d_pool_size = cnn2d_pool_size
        self.cnn2d_kernel_size = cnn2d_kernel_size
        self.cnn2d_stride = cnn2d_stride

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_fraction: float = 0.1,
        epochs: int = 500,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        sample_ids: list[str] | None = None,
        progress_callback: Callable[[int, int, float, float], None] | None = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 50,
        early_stopping_min_delta: float = 0.0,
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
            early_stopping: Whether to use early stopping based on test loss.
            early_stopping_patience: Epochs to wait for improvement before stopping.
            early_stopping_min_delta: Minimum change to qualify as improvement.

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

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        self._logger.info(f"Training {self.architecture} model for {epochs} epochs")
        if early_stopping:
            self._logger.info(
                f"Early stopping enabled: patience={early_stopping_patience}, "
                f"min_delta={early_stopping_min_delta}"
            )

        # Early stopping state
        best_test_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0
        stopped_early = False

        for epoch in range(epochs):
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, criterion, scaler
            )
            test_loss = self._evaluate_epoch(test_loader, criterion)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                lr = scheduler.get_last_lr()[0]
                self._logger.info(
                    f"Epoch {epoch + 1:03d}/{epochs} | "
                    f"Train: {train_loss:.6e} | Test: {test_loss:.6e} | LR: {lr:.2e}"
                )

            if progress_callback is not None:
                progress_callback(epoch + 1, epochs, train_loss, test_loss)

            # Early stopping check
            if early_stopping:
                if test_loss < best_test_loss - early_stopping_min_delta:
                    best_test_loss = test_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self._model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    self._logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best test loss: {best_test_loss:.6e}"
                    )
                    stopped_early = True
                    break

        # Restore best model if early stopping was used
        if early_stopping and best_model_state is not None:
            self._model.load_state_dict(best_model_state)
            self._logger.info(
                f"Restored best model with test loss: {best_test_loss:.6e}"
            )
            test_loss = best_test_loss

        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "stopped_early": stopped_early,
            "epochs_completed": epoch + 1,
        }

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
            "architecture": self.architecture,
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

        self.architecture = data["architecture"]
        self._input_dim = data["input_dim"]
        self._output_dim = data["output_dim"]
        self.train_indices = data.get("train_indices", [])
        self.test_indices = data.get("test_indices", [])

        self._model = self._create_model()
        self._model.load_state_dict(data["model_state_dict"])
        self._model.to(self.device)

        self._logger.info(f"Model loaded from {path}")

    def _create_model(self) -> nn.Module:
        """Create network based on architecture setting."""
        assert self._input_dim is not None
        assert self._output_dim is not None

        if self.architecture == "cnn":
            return ConvRegressor(
                self._input_dim,
                self._output_dim,
                conv_channels=self.cnn_conv_channels,
                pool_size=self.cnn_pool_size,
                regressor_hidden=self.cnn_regressor_hidden,
                dropout=self.cnn_dropout,
                use_residual=self.cnn_use_residual,
            )
        elif self.architecture == "cnn2d":
            if self.cnn2d_num_sensors is None:
                raise ValueError("cnn2d_num_sensors must be set for cnn2d architecture")
            timesteps = self._input_dim // self.cnn2d_num_sensors
            return Conv2DRegressor(
                num_sensors=self.cnn2d_num_sensors,
                timesteps=timesteps,
                output_dim=self._output_dim,
                conv_channels=self.cnn_conv_channels,
                pool_size=self.cnn2d_pool_size,
                regressor_hidden=self.cnn_regressor_hidden,
                dropout=self.cnn_dropout,
                kernel_size=self.cnn2d_kernel_size,
                stride=self.cnn2d_stride,
                use_residual=self.cnn_use_residual,
            )
        elif self.architecture == "mlp":
            return MLPRegressor(
                self._input_dim,
                self._output_dim,
                hidden_layers=self.mlp_hidden_layers,
                dropout=self.mlp_dropout,
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        scaler: torch.cuda.amp.GradScaler,
    ) -> float:
        """Train for one epoch."""
        self._model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)

            # GradScaler may skip optimizer.step() if gradients contain inf/nan
            # Track scale before step to detect if step was skipped
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # Only step scheduler if optimizer actually stepped (scale didn't change due to inf/nan)
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            total_loss += loss.item() * X_batch.size(0)

        return total_loss / len(loader.dataset)

    def _evaluate_epoch(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Evaluate on a dataset."""
        self._model.eval()
        total_loss = 0.0

        with (
            torch.no_grad(),
            torch.cuda.amp.autocast(enabled=self.device.type == "cuda"),
        ):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)

        return total_loss / len(loader.dataset)
