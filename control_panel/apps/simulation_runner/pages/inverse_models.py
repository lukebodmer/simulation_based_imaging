"""Inverse Models page for training and testing inverse models."""

import asyncio
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.widgets import html
from trame.widgets import matplotlib as mpl_widgets
from trame.widgets import vuetify3 as v3

from sbimaging.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_DIR = Path("/data/simulations")

# PyVista offscreen rendering for trame
pv.OFF_SCREEN = True

# Available colormaps for visualization
COLORMAP_OPTIONS = [
    {"title": "Viridis", "value": "viridis"},
    {"title": "Plasma", "value": "plasma"},
    {"title": "Inferno", "value": "inferno"},
    {"title": "Magma", "value": "magma"},
    {"title": "Cividis", "value": "cividis"},
    {"title": "Turbo", "value": "turbo"},
    {"title": "Hot", "value": "hot"},
    {"title": "Cool", "value": "cool"},
    {"title": "Gray", "value": "gray"},
    {"title": "Seismic (diverging)", "value": "seismic"},
    {"title": "Coolwarm (diverging)", "value": "coolwarm"},
    {"title": "RdBu (diverging)", "value": "RdBu_r"},
]

# Opacity presets for volume rendering
OPACITY_PRESETS = [
    {"title": "Sigmoid", "value": "sigmoid"},
    {"title": "Linear", "value": "linear"},
    {"title": "Geom", "value": "geom"},
    {"title": "Geom (reversed)", "value": "geom_r"},
]

MODEL_TYPES = [
    {"title": "Neural Network (CNN)", "value": "nn_cnn"},
    {"title": "Neural Network (MLP)", "value": "nn_mlp"},
    {"title": "Gaussian Process", "value": "gp"},
]


class InverseModelsPage:
    """Page for training and testing inverse models."""

    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

        # PyVista plotters for comparison
        self.plotter_ground_truth = pv.Plotter()
        self.plotter_prediction = pv.Plotter()

        self._setup_state()

    def _setup_state(self):
        """Initialize state variables for this page."""
        # Tab state
        self.state.inv_active_tab = "train"

        # Batch selection (for training)
        self.state.inv_available_batches = []
        self.state.inv_selected_batch = ""

        # Training configuration
        self.state.inv_model_type = "nn_cnn"
        self.state.inv_model_type_items = MODEL_TYPES
        self.state.inv_model_name = ""  # Custom model name
        self.state.inv_test_fraction = 0.1
        self.state.inv_epochs = 500
        self.state.inv_batch_size = 16
        self.state.inv_learning_rate = 0.0001
        self.state.inv_kspace_grid_size = 64  # K-space resolution (grid_size^3 voxels)
        self.state.inv_trim_timesteps = 45  # Skip this many initial timesteps
        self.state.inv_downsample_factor = 2  # Downsample sensor data by this factor

        # MLP architecture configuration
        self.state.inv_mlp_hidden_layers = "4096, 2048, 1024"  # Comma-separated sizes
        self.state.inv_mlp_dropout = 0.2

        # CNN architecture configuration
        self.state.inv_cnn_conv_channels = "32, 64"  # Comma-separated channel sizes
        self.state.inv_cnn_pool_size = 16
        self.state.inv_cnn_regressor_hidden = 512
        self.state.inv_cnn_dropout = 0.2

        # Training options
        self.state.inv_early_stopping = False
        self.state.inv_early_stopping_patience = 50
        self.state.inv_kfold_validation = False
        self.state.inv_kfold_k = 5

        # Dynamic compression settings
        self.state.inv_use_dynamic_compression = False
        self.state.inv_compression_threshold = 0.1
        self.state.inv_compression_ratio = 4.0

        # Normalization settings
        self.state.inv_use_normalization = False

        # Noise settings
        self.state.inv_use_noise = False
        self.state.inv_noise_level = 0.05  # 5% of global peak by default

        # Training state
        self.state.inv_is_training = False
        self.state.inv_training_progress = 0
        self.state.inv_training_message = ""
        self.state.inv_training_log = []

        # Loss history for plotting (used in train and view tabs)
        # For regular training: single lists
        # For K-fold: list of lists (one per fold)
        self._train_losses = []
        self._test_losses = []
        self._loss_epochs = []
        self._kfold_train_losses = []  # List of lists, one per fold
        self._kfold_test_losses = []
        self._kfold_epochs = []
        self._current_fold = 0
        self.loss_chart_widget = None
        self.view_loss_chart_widget = None

        # Compression preview
        self.compression_preview_widget = None
        self._sample_sensor_data = None  # Current sensor channel for preview
        self._all_sensor_data = []  # List of (sim_name, sensor_data) tuples
        self._current_sensor_index = 0  # Current index in _all_sensor_data
        self._global_sensor_max = 0.0  # Global max across all sensors in first sim
        self.state.inv_compression_preview_label = ""  # Label showing current signal
        self.state.inv_raw_timesteps = 0  # Raw timesteps per sensor before downsampling
        self.state.inv_num_sensors = 0  # Number of sensors
        self.state.inv_downsample_hint = "Select batch to see timesteps"
        self.state.inv_input_size = 0  # Total input vector size
        self.state.inv_output_size = 0  # Total output vector size

        # View tab state - model info display
        self.state.inv_view_available_batches = []
        self.state.inv_view_selected_batch = ""
        self.state.inv_view_available_models = []
        self.state.inv_view_selected_model = ""
        self.state.inv_view_model_info = {}

        # Test tab state
        self.state.inv_test_available_batches = []
        self.state.inv_test_selected_batch = ""
        self.state.inv_test_available_models = []
        self.state.inv_test_selected_model = ""
        # K-fold fold selection for Test tab
        self.state.inv_test_is_kfold = False
        self.state.inv_test_available_folds = []
        self.state.inv_test_selected_fold = 0  # 0-indexed fold number
        self.state.inv_is_testing = False
        self.state.inv_test_progress = 0
        self.state.inv_test_message = ""
        self.state.inv_test_simulations = []
        self.state.inv_selected_test_sim = ""
        self.state.inv_test_metrics = {}

        # Visualization settings for comparison view (test tab)
        self.state.inv_colormap_options = COLORMAP_OPTIONS
        self.state.inv_opacity_presets = OPACITY_PRESETS
        self.state.inv_selected_colormap = "viridis"
        self.state.inv_selected_opacity = "sigmoid"
        self.state.inv_clim_min = 0.0
        self.state.inv_clim_max = 1.0
        self.state.inv_auto_clim = True

        # Shared color limits (computed from ground truth)
        self._shared_clim = None
        self._ground_truth_data = None
        self._prediction_data = None

        # Refresh batch list for all tabs
        self._refresh_batch_list()

        # Register state change handlers
        # Train tab handlers (compression preview)
        self.state.change("inv_selected_batch")(self._on_train_batch_selected)
        self.state.change("inv_use_dynamic_compression")(
            self._on_compression_setting_changed
        )
        self.state.change("inv_compression_threshold")(
            self._on_compression_setting_changed
        )
        self.state.change("inv_compression_ratio")(self._on_compression_setting_changed)
        self.state.change("inv_use_normalization")(self._on_compression_setting_changed)
        self.state.change("inv_use_noise")(self._on_compression_setting_changed)
        self.state.change("inv_noise_level")(self._on_compression_setting_changed)
        self.state.change("inv_trim_timesteps")(self._on_input_output_changed)
        self.state.change("inv_downsample_factor")(self._on_input_output_changed)
        self.state.change("inv_kspace_grid_size")(self._on_input_output_changed)
        # View tab handlers
        self.state.change("inv_view_selected_batch")(self._on_view_batch_selected)
        self.state.change("inv_view_selected_model")(self._on_view_model_selected)
        # Test tab handlers
        self.state.change("inv_test_selected_batch")(self._on_test_batch_selected)
        self.state.change("inv_test_selected_model")(self._on_test_model_selected)
        self.state.change("inv_selected_test_sim")(self._on_test_sim_selected)
        self.state.change("inv_selected_colormap")(self._on_viz_setting_changed)
        self.state.change("inv_selected_opacity")(self._on_viz_setting_changed)
        self.state.change("inv_clim_min")(self._on_viz_setting_changed)
        self.state.change("inv_clim_max")(self._on_viz_setting_changed)
        self.state.change("inv_auto_clim")(self._on_viz_setting_changed)

    def _refresh_batch_list(self):
        """Scan for available simulation batches for all tabs."""
        batches = []
        if DEFAULT_DATA_DIR.exists():
            for item in sorted(DEFAULT_DATA_DIR.iterdir()):
                if item.is_dir():
                    sims_dir = item / "simulations"
                    if sims_dir.exists() and any(sims_dir.iterdir()):
                        batches.append({"title": item.name, "value": item.name})

        # Update all tabs
        self.state.inv_available_batches = batches
        self.state.inv_view_available_batches = batches
        self.state.inv_test_available_batches = batches
        logger.info(f"Found {len(batches)} batches for inverse modeling")

    def _get_models_for_batch(self, batch_name: str) -> list[dict]:
        """Get list of available trained models for a batch."""
        models_dir = DEFAULT_DATA_DIR / batch_name / "inverse_models"
        models = []

        if models_dir.exists():
            for model_file in sorted(models_dir.glob("*.pkl")):
                models.append({"title": model_file.stem, "value": model_file.stem})

        return models

    # --- Train tab handlers ---

    def _on_train_batch_selected(self, inv_selected_batch, **kwargs):
        """Handle batch selection in train tab - load sensor data for preview."""
        self._sample_sensor_data = None
        self._all_sensor_data = []
        self._current_sensor_index = 0
        self._global_sensor_max = 0.0
        self.state.inv_raw_timesteps = 0
        self.state.inv_num_sensors = 0
        self.state.inv_downsample_hint = "Select batch to see timesteps"
        self.state.inv_input_size = 0
        self.state.inv_output_size = 0

        if not inv_selected_batch:
            self.state.inv_compression_preview_label = ""
            self._update_compression_preview()
            return

        # Load sensor data from simulations in batch
        batch_dir = DEFAULT_DATA_DIR / inv_selected_batch / "simulations"
        if not batch_dir.exists():
            return

        # Load sensor data from 1 simulation, all channels
        max_sims = 1
        sim_count = 0

        for sim_dir in sorted(batch_dir.iterdir()):
            if not sim_dir.is_dir():
                continue
            if sim_count >= max_sims:
                break

            sensor_file = sim_dir / "sensor_data.pkl"
            if sensor_file.exists():
                try:
                    with open(sensor_file, "rb") as f:
                        data = pickle.load(f)

                    if isinstance(data, dict) and "pressure" in data:
                        sensor_data = data["pressure"]
                    else:
                        sensor_data = data

                    # Convert from CuPy if needed
                    if hasattr(sensor_data, "get"):
                        sensor_data = sensor_data.get()

                    sensor_data = np.asarray(sensor_data)

                    # Store raw timesteps and num sensors before any processing
                    if sensor_data.ndim == 2:
                        self.state.inv_num_sensors = sensor_data.shape[0]
                        self.state.inv_raw_timesteps = sensor_data.shape[1]
                        self._update_input_output_sizes()

                    # Trim and downsample like in training
                    trim = int(self.state.inv_trim_timesteps)
                    downsample = int(self.state.inv_downsample_factor)
                    if sensor_data.ndim == 2:
                        sensor_data = sensor_data[:, trim:]
                        if downsample > 1:
                            sensor_data = sensor_data[:, ::downsample]

                        # Add each channel as a separate entry
                        for ch in range(sensor_data.shape[0]):
                            label = f"{sim_dir.name[:8]}... ch{ch}"
                            self._all_sensor_data.append((label, sensor_data[ch]))
                    else:
                        label = f"{sim_dir.name[:8]}..."
                        self._all_sensor_data.append((label, sensor_data))

                    sim_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to load sensor data from {sim_dir.name}: {e}"
                    )

        if self._all_sensor_data:
            # Compute global max across all sensors from this simulation
            all_maxes = [np.abs(data).max() for _, data in self._all_sensor_data]
            self._global_sensor_max = max(all_maxes) if all_maxes else 0.0

            self._current_sensor_index = 0
            label, data = self._all_sensor_data[0]
            self._sample_sensor_data = data
            self.state.inv_compression_preview_label = (
                f"{label} (1/{len(self._all_sensor_data)})"
            )
            logger.info(
                f"Loaded {len(self._all_sensor_data)} sensor signals for preview, "
                f"global max: {self._global_sensor_max:.4g}"
            )

        self._update_compression_preview()

    def _on_compression_setting_changed(self, **kwargs):
        """Handle changes to compression settings - update preview."""
        self._update_compression_preview()

    def _on_input_output_changed(self, **kwargs):
        """Handle changes to input/output parameters."""
        self._update_input_output_sizes()

    def _update_input_output_sizes(self):
        """Update the input/output size calculations and hints."""
        raw = self.state.inv_raw_timesteps
        num_sensors = self.state.inv_num_sensors

        if raw > 0 and num_sensors > 0:
            trim = int(self.state.inv_trim_timesteps)
            factor = int(self.state.inv_downsample_factor)
            trimmed = max(0, raw - trim)
            # Factor of 0 or 1 means no downsampling
            timesteps_per_sensor = trimmed // factor if factor > 1 else trimmed
            self.state.inv_downsample_hint = (
                f"{timesteps_per_sensor}/{trimmed} timesteps per sensor"
            )

            # Calculate input size: num_sensors * timesteps_per_sensor
            self.state.inv_input_size = num_sensors * timesteps_per_sensor
        else:
            self.state.inv_downsample_hint = "Select batch to see timesteps"
            self.state.inv_input_size = 0

        # Calculate output size: grid_size^3 * 2 (real + imaginary)
        try:
            grid_size = int(self.state.inv_kspace_grid_size)
            self.state.inv_output_size = grid_size**3 * 2
        except (ValueError, TypeError):
            self.state.inv_output_size = 0

    def _prev_sensor_signal(self):
        """Navigate to previous sensor signal."""
        if not self._all_sensor_data:
            return
        self._current_sensor_index = (self._current_sensor_index - 1) % len(
            self._all_sensor_data
        )
        label, data = self._all_sensor_data[self._current_sensor_index]
        self._sample_sensor_data = data
        self.state.inv_compression_preview_label = (
            f"{label} ({self._current_sensor_index + 1}/{len(self._all_sensor_data)})"
        )
        self._update_compression_preview()

    def _next_sensor_signal(self):
        """Navigate to next sensor signal."""
        if not self._all_sensor_data:
            return
        self._current_sensor_index = (self._current_sensor_index + 1) % len(
            self._all_sensor_data
        )
        label, data = self._all_sensor_data[self._current_sensor_index]
        self._sample_sensor_data = data
        self.state.inv_compression_preview_label = (
            f"{label} ({self._current_sensor_index + 1}/{len(self._all_sensor_data)})"
        )
        self._update_compression_preview()

    def _update_compression_preview(self):
        """Update the compression/normalization/noise preview chart."""
        if self.compression_preview_widget is None:
            return

        fig, ax = plt.subplots(figsize=(6, 2.5))

        use_compression = self.state.inv_use_dynamic_compression
        use_normalization = self.state.inv_use_normalization
        use_noise = self.state.inv_use_noise

        any_processing = use_compression or use_normalization or use_noise

        if self._sample_sensor_data is not None and any_processing:
            signal = self._sample_sensor_data
            time_axis = np.arange(len(signal))

            # Always plot original
            ax.plot(
                time_axis,
                signal,
                "b-",
                alpha=0.7,
                linewidth=0.8,
                label="Original",
            )

            # Use global max for threshold calculation (same as training)
            global_max = self._global_sensor_max

            # Build processed signal step by step
            processed = signal.copy()
            title_parts = []

            # Apply compression if enabled
            if use_compression:
                threshold = float(self.state.inv_compression_threshold)
                ratio = float(self.state.inv_compression_ratio)
                processed = self._compress_signal_for_preview(
                    processed, threshold, ratio
                )
                title_parts.append(f"Comp(t={threshold}, r={ratio}:1)")

                # Draw threshold lines (based on global max)
                thresh_val = threshold * global_max if global_max > 0 else threshold
                ax.plot(
                    time_axis,
                    np.full_like(time_axis, thresh_val, dtype=float),
                    color="orange",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=0.8,
                )
                ax.plot(
                    time_axis,
                    np.full_like(time_axis, -thresh_val, dtype=float),
                    color="orange",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=0.8,
                )

            # Apply normalization if enabled
            if use_normalization:
                processed = self._normalize_signal_for_preview(processed)
                title_parts.append("Norm")

            # Apply noise if enabled
            if use_noise:
                noise_level = float(self.state.inv_noise_level)
                processed = self._add_noise_for_preview(processed, noise_level)
                title_parts.append(f"Noise({noise_level:.0%})")

            # Plot the processed signal
            ax.plot(
                time_axis,
                processed,
                "r-",
                alpha=0.7,
                linewidth=0.8,
                label="Processed",
            )

            ax.set_title(" + ".join(title_parts), fontsize=9)
            ax.legend(loc="upper right", fontsize=7)
            ax.set_xlabel("Time Step", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)

            # Fix y-axis to global max for consistent view across signals
            if global_max > 0:
                margin = global_max * 0.1
                # Extend y-limits if noise is enabled (can exceed original range)
                if use_noise:
                    noise_level = float(self.state.inv_noise_level)
                    margin = global_max * (0.1 + noise_level)
                ax.set_ylim(-global_max - margin, global_max + margin)

        elif self._sample_sensor_data is not None:
            # Show raw original signal when no processing enabled
            signal = self._sample_sensor_data
            time_axis = np.arange(len(signal))
            ax.plot(time_axis, signal, "b-", alpha=0.7, linewidth=0.8, label="Original")
            ax.legend(loc="upper right", fontsize=7)
            ax.set_xlabel("Time Step", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.set_title("Sensor Signal (no preprocessing)", fontsize=9)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)

            # Fix y-axis to global max for consistent view across signals
            if self._global_sensor_max > 0:
                margin = self._global_sensor_max * 0.1
                ax.set_ylim(
                    -self._global_sensor_max - margin, self._global_sensor_max + margin
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Select a batch to preview preprocessing",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        plt.tight_layout()
        self.compression_preview_widget.update(fig)
        plt.close(fig)

    def _normalize_signal_for_preview(self, signal: np.ndarray) -> np.ndarray:
        """Normalize a single signal to [-1, 1] range for preview.

        Args:
            signal: 1D signal array.

        Returns:
            Normalized signal with values in [-1, 1].
        """
        max_val = np.abs(signal).max()
        if max_val == 0:
            return signal.copy()
        return signal / max_val

    def _compress_signal_for_preview(
        self, signal: np.ndarray, threshold: float, ratio: float
    ) -> np.ndarray:
        """Apply dynamic compression to a single signal for preview.

        Uses the global max across all sensors in the simulation (stored in
        self._global_sensor_max) to compute the threshold, matching the
        behavior of the actual training compression.

        Args:
            signal: 1D signal array.
            threshold: Fraction (0-1) of global max above which to compress.
            ratio: Compression ratio (e.g., 4 means 4:1 compression).

        Returns:
            Compressed signal (same scale as input, peaks reduced).
        """
        # Use global max from simulation, not this signal's max
        global_max = self._global_sensor_max
        if global_max == 0:
            return signal.copy()

        # Compute absolute threshold from fraction of global max
        abs_threshold = threshold * global_max

        # Apply compression
        result = signal.copy()
        abs_signal = np.abs(result)

        # Above threshold: compress
        above_mask = abs_signal > abs_threshold
        if above_mask.any():
            excess = abs_signal[above_mask] - abs_threshold
            compressed_excess = excess / ratio
            new_amplitude = abs_threshold + compressed_excess
            result[above_mask] = np.sign(result[above_mask]) * new_amplitude

        return result

    def _add_noise_for_preview(
        self, signal: np.ndarray, noise_level: float
    ) -> np.ndarray:
        """Add Gaussian noise to a signal for preview.

        The noise amplitude is computed as a fraction of the global peak
        across all sensors (self._global_sensor_max), matching the behavior
        of the actual training noise application.

        Args:
            signal: 1D signal array.
            noise_level: Fraction (0-1) of global peak to use as noise std.

        Returns:
            Signal with added Gaussian noise.
        """
        if noise_level <= 0:
            return signal.copy()

        # Use global max from simulation
        global_max = self._global_sensor_max
        if global_max == 0:
            return signal.copy()

        # Compute noise standard deviation as fraction of global peak
        noise_std = noise_level * global_max

        # Generate and add noise
        noise = np.random.normal(0, noise_std, len(signal))
        return signal + noise

    # --- View tab handlers ---

    def _on_view_batch_selected(self, inv_view_selected_batch, **kwargs):
        """Handle batch selection in view tab."""
        self.state.inv_view_available_models = []
        self.state.inv_view_selected_model = ""
        self.state.inv_view_model_info = {}

        if not inv_view_selected_batch:
            return

        models = self._get_models_for_batch(inv_view_selected_batch)
        self.state.inv_view_available_models = models
        logger.info(
            f"View tab: Found {len(models)} models for {inv_view_selected_batch}"
        )

    def _on_view_model_selected(self, inv_view_selected_model, **kwargs):
        """Handle model selection in view tab - load and display model info."""
        self.state.inv_view_model_info = {}

        # Clear both regular and K-fold history
        self._loss_epochs = []
        self._train_losses = []
        self._test_losses = []
        self._kfold_epochs = []
        self._kfold_train_losses = []
        self._kfold_test_losses = []

        if not inv_view_selected_model or not self.state.inv_view_selected_batch:
            self._update_view_loss_chart()
            return

        model_path = (
            DEFAULT_DATA_DIR
            / self.state.inv_view_selected_batch
            / "inverse_models"
            / f"{inv_view_selected_model}.pkl"
        )

        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)

                # Extract model info
                model_info = self._extract_model_info(data)
                model_info["batch_name"] = self.state.inv_view_selected_batch
                self.state.inv_view_model_info = model_info

                # Load and display training history
                history = data.get("training_history", {})
                if history:
                    # Check if this is K-fold history
                    if "kfold_epochs" in history:
                        self._kfold_epochs = history.get("kfold_epochs", [])
                        self._kfold_train_losses = history.get("kfold_train_losses", [])
                        self._kfold_test_losses = history.get("kfold_test_losses", [])
                        self._update_view_loss_chart()
                        k = len(self._kfold_epochs)
                        total_epochs = sum(len(e) for e in self._kfold_epochs)
                        logger.info(
                            f"View tab: Loaded K-fold history with {k} folds, "
                            f"{total_epochs} total epochs"
                        )
                    else:
                        # Regular training history
                        self._loss_epochs = history.get("epochs", [])
                        self._train_losses = history.get("train_loss", [])
                        self._test_losses = history.get("test_loss", [])
                        self._update_view_loss_chart()
                        logger.info(
                            f"View tab: Loaded training history with "
                            f"{len(self._loss_epochs)} epochs"
                        )

            except Exception as e:
                logger.error(f"Failed to load model info: {e}")

    def _extract_model_info(self, data: dict) -> dict:
        """Extract model architecture and training info from saved data."""
        model = data.get("model")
        models = data.get("models")  # K-fold: list of models
        model_type = data.get("model_type", "unknown")
        model_name = data.get("model_name")
        metrics = data.get("metrics", {})
        test_hashes = data.get("test_hashes", [])
        history = data.get("training_history", {})
        training_config = data.get("training_config", {})
        is_kfold = data.get("kfold", False)
        k = data.get("k", 0)

        # Calculate num_epochs for K-fold
        if "kfold_epochs" in history:
            total_epochs = sum(len(e) for e in history.get("kfold_epochs", []))
            num_epochs = total_epochs
        else:
            num_epochs = len(history.get("epochs", []))

        info = {
            "model_type": model_type,
            "model_name": model_name,
            "num_test_samples": len(test_hashes),
            "final_train_loss": metrics.get("train_loss", "N/A"),
            "final_test_loss": metrics.get("test_loss", "N/A"),
            "num_epochs": num_epochs,
            "is_kfold": is_kfold,
            "k": k,
        }

        # Add K-fold specific info
        if is_kfold:
            fold_metrics = metrics.get("fold_metrics", [])
            if fold_metrics:
                # Per-fold metrics for display
                info["fold_details"] = []
                for i, m in enumerate(fold_metrics):
                    fold_info = {
                        "fold": i + 1,
                        "train_loss": m.get("train_loss", 0),
                        "test_loss": m.get("test_loss", 0),
                    }
                    # Add epochs completed if available (for early stopping)
                    if "epochs_completed" in m:
                        fold_info["epochs_completed"] = m["epochs_completed"]
                    if "stopped_early" in m:
                        fold_info["stopped_early"] = m["stopped_early"]
                    info["fold_details"].append(fold_info)

                # Calculate statistics across folds
                train_losses = [m.get("train_loss", 0) for m in fold_metrics]
                test_losses = [m.get("test_loss", 0) for m in fold_metrics]
                info["fold_train_mean"] = np.mean(train_losses)
                info["fold_train_std"] = np.std(train_losses)
                info["fold_test_mean"] = np.mean(test_losses)
                info["fold_test_std"] = np.std(test_losses)
                info["fold_test_min"] = np.min(test_losses)
                info["fold_test_max"] = np.max(test_losses)

        # Add training config if available
        if training_config:
            info["training_epochs"] = training_config.get("epochs", "N/A")
            info["batch_size"] = training_config.get("batch_size", "N/A")
            info["learning_rate"] = training_config.get("learning_rate", "N/A")
            info["test_fraction"] = training_config.get("test_fraction", "N/A")

            # Add training duration if available
            duration_seconds = training_config.get("training_duration_seconds")
            if duration_seconds is not None:
                info["training_duration"] = self._format_duration(duration_seconds)
                info["training_duration_seconds"] = duration_seconds

            # Add early stopping info
            info["early_stopping"] = training_config.get("early_stopping", False)
            if info["early_stopping"]:
                info["early_stopping_patience"] = training_config.get(
                    "early_stopping_patience", 50
                )
                info["stopped_early"] = training_config.get("stopped_early", False)
                epochs_completed = training_config.get("epochs_completed")
                if epochs_completed is not None:
                    info["epochs_completed"] = epochs_completed

        # Add architecture config if available
        architecture_config = data.get("architecture_config", {})
        if architecture_config:
            # MLP config
            mlp_hidden_layers = architecture_config.get("mlp_hidden_layers")
            if mlp_hidden_layers:
                info["mlp_hidden_layers"] = mlp_hidden_layers
                info["mlp_hidden_layers_str"] = " -> ".join(
                    str(x) for x in mlp_hidden_layers
                )
            mlp_dropout = architecture_config.get("mlp_dropout")
            if mlp_dropout is not None:
                info["mlp_dropout"] = mlp_dropout

            # CNN config
            cnn_conv_channels = architecture_config.get("cnn_conv_channels")
            if cnn_conv_channels:
                info["cnn_conv_channels"] = cnn_conv_channels
                info["cnn_conv_channels_str"] = " -> ".join(
                    str(x) for x in cnn_conv_channels
                )
            cnn_pool_size = architecture_config.get("cnn_pool_size")
            if cnn_pool_size is not None:
                info["cnn_pool_size"] = cnn_pool_size
            cnn_regressor_hidden = architecture_config.get("cnn_regressor_hidden")
            if cnn_regressor_hidden is not None:
                info["cnn_regressor_hidden"] = cnn_regressor_hidden
            cnn_dropout = architecture_config.get("cnn_dropout")
            if cnn_dropout is not None:
                info["cnn_dropout"] = cnn_dropout

        # Add k-space config if available
        kspace_config = data.get("kspace_config", {})
        if kspace_config:
            grid_size = kspace_config.get("grid_size")
            if grid_size:
                info["kspace_grid_size"] = grid_size
                info["kspace_total_coeffs"] = f"{grid_size**3 * 2:,}"  # real + imag

        # Add preprocessing config if available
        preprocessing_config = data.get("preprocessing_config", {})
        if preprocessing_config:
            trim_timesteps = preprocessing_config.get("trim_timesteps")
            if trim_timesteps is not None:
                info["trim_timesteps"] = trim_timesteps
            downsample_factor = preprocessing_config.get("downsample_factor")
            if downsample_factor is not None:
                info["downsample_factor"] = downsample_factor
            info["use_dynamic_compression"] = preprocessing_config.get(
                "use_dynamic_compression", False
            )
            if info["use_dynamic_compression"]:
                info["compression_threshold"] = preprocessing_config.get(
                    "compression_threshold", 0.1
                )
                info["compression_ratio"] = preprocessing_config.get(
                    "compression_ratio", 4.0
                )
            info["use_normalization"] = preprocessing_config.get(
                "use_normalization", False
            )

        # Extract architecture info from the model object
        # For K-fold models, use the first model in the list
        model_to_inspect = model
        if model_to_inspect is None and models is not None and len(models) > 0:
            model_to_inspect = models[0]

        if (
            model_to_inspect is not None
            and hasattr(model_to_inspect, "_model")
            and model_to_inspect._model is not None
        ):
            nn_model = model_to_inspect._model
            info["architecture"] = (
                model_to_inspect.architecture
                if hasattr(model_to_inspect, "architecture")
                else "unknown"
            )

            # Count parameters
            total_params = sum(p.numel() for p in nn_model.parameters())
            trainable_params = sum(
                p.numel() for p in nn_model.parameters() if p.requires_grad
            )
            info["total_parameters"] = f"{total_params:,}"
            info["trainable_parameters"] = f"{trainable_params:,}"

            # Get layer info
            layers = []
            for name, module in nn_model.named_modules():
                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    layers.append(
                        f"{name}: Linear({module.in_features} → {module.out_features})"
                    )
                elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                    layers.append(
                        f"{name}: Conv({module.in_channels} → {module.out_channels})"
                    )
            info["layers"] = layers[:10]  # Limit to first 10 layers

            # Get input/output dimensions
            if hasattr(model_to_inspect, "_input_dim"):
                info["input_dim"] = model_to_inspect._input_dim
            if hasattr(model_to_inspect, "_output_dim"):
                info["output_dim"] = model_to_inspect._output_dim

        return info

    # --- Test tab handlers ---

    def _on_test_batch_selected(self, inv_test_selected_batch, **kwargs):
        """Handle batch selection in test tab."""
        self.state.inv_test_available_models = []
        self.state.inv_test_selected_model = ""
        self.state.inv_test_simulations = []
        self.state.inv_selected_test_sim = ""

        if not inv_test_selected_batch:
            return

        models = self._get_models_for_batch(inv_test_selected_batch)
        self.state.inv_test_available_models = models
        logger.info(
            f"Test tab: Found {len(models)} models for {inv_test_selected_batch}"
        )

    def _on_test_model_selected(self, inv_test_selected_model, **kwargs):
        """Handle model selection in test tab - load test simulations."""
        self.state.inv_test_simulations = []
        self.state.inv_selected_test_sim = ""
        self.state.inv_test_metrics = {}
        self.state.inv_test_is_kfold = False
        self.state.inv_test_available_folds = []
        self.state.inv_test_selected_fold = 0

        if not inv_test_selected_model or not self.state.inv_test_selected_batch:
            return

        model_path = (
            DEFAULT_DATA_DIR
            / self.state.inv_test_selected_batch
            / "inverse_models"
            / f"{inv_test_selected_model}.pkl"
        )

        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)

                # Check if this is a K-fold model and populate fold options
                is_kfold = data.get("kfold", False)
                if is_kfold:
                    models = data.get("models", [])
                    metrics = data.get("metrics", {})
                    fold_metrics = metrics.get("fold_metrics", [])

                    self.state.inv_test_is_kfold = True
                    # Create fold options with test loss info
                    fold_options = []
                    for i in range(len(models)):
                        test_loss = (
                            fold_metrics[i].get("test_loss", 0)
                            if i < len(fold_metrics)
                            else 0
                        )
                        fold_options.append(
                            {
                                "title": f"Fold {i + 1} (loss: {test_loss:.2f})",
                                "value": i,
                            }
                        )
                    self.state.inv_test_available_folds = fold_options
                    self.state.inv_test_selected_fold = 0
                    logger.info(
                        f"Test tab: K-fold model with {len(models)} folds detected"
                    )

                test_hashes = data.get("test_hashes", [])
                self.state.inv_test_simulations = [
                    {"title": h[:12] + "...", "value": h} for h in test_hashes
                ]
                logger.info(f"Test tab: Model has {len(test_hashes)} test simulations")

            except Exception as e:
                logger.error(f"Failed to load model for testing: {e}")

    def _on_test_sim_selected(self, inv_selected_test_sim, **kwargs):
        """Handle test simulation selection - show comparison."""
        if not inv_selected_test_sim:
            return

        self._update_comparison_view(inv_selected_test_sim)

    def _on_viz_setting_changed(self, **kwargs):
        """Handle changes to visualization settings - re-render both views."""
        if self._ground_truth_data is not None or self._prediction_data is not None:
            self._rerender_comparison_views()

    def _update_comparison_view(self, sim_hash: str):
        """Update the ground truth vs prediction comparison view."""
        batch_name = self.state.inv_test_selected_batch
        if not batch_name:
            return

        batch_dir = DEFAULT_DATA_DIR / batch_name
        model_name = self.state.inv_test_selected_model

        # Load ground truth k-space from model_output.pkl
        sim_dir = batch_dir / "simulations" / sim_hash
        output_file = sim_dir / "model_output.pkl"

        # Load prediction from model-specific subfolder
        pred_file = batch_dir / "predictions" / model_name / f"{sim_hash}.pkl"

        try:
            # Load ground truth
            self._ground_truth_data = None
            self._prediction_data = None
            self._shared_clim = None

            if output_file.exists():
                with open(output_file, "rb") as f:
                    self._ground_truth_data = pickle.load(f)

            # Load prediction
            if pred_file.exists():
                with open(pred_file, "rb") as f:
                    self._prediction_data = pickle.load(f)

            # Compute shared color limits from ground truth
            if self._ground_truth_data is not None:
                voxel_data = self._kspace_to_voxels(self._ground_truth_data)
                if voxel_data is not None:
                    data_min = float(np.min(voxel_data))
                    data_max = float(np.max(voxel_data))
                    self._shared_clim = (data_min, data_max)

                    # Update state with auto clim values
                    if self.state.inv_auto_clim:
                        self.state.inv_clim_min = data_min
                        self.state.inv_clim_max = data_max

            # Render both views with shared limits
            self._rerender_comparison_views()

        except Exception as e:
            logger.error(f"Failed to update comparison view: {e}")

    def sync_views(self):
        """Sync prediction plotter camera to ground truth plotter."""
        if self.plotter_ground_truth and self.plotter_prediction:
            self.plotter_prediction.camera_position = (
                self.plotter_ground_truth.camera_position
            )
            self.plotter_prediction.render()
            logger.info("Synchronized prediction camera to ground truth.")

    def _rerender_comparison_views(self):
        """Re-render both comparison views with current settings."""
        # Determine clim to use
        if self.state.inv_auto_clim and self._shared_clim is not None:
            clim = self._shared_clim
        else:
            clim = (float(self.state.inv_clim_min), float(self.state.inv_clim_max))

        colormap = self.state.inv_selected_colormap
        opacity = self.state.inv_selected_opacity

        # Render ground truth
        if self._ground_truth_data is not None:
            self._render_kspace_volume(
                self.plotter_ground_truth,
                self._ground_truth_data,
                "Ground Truth",
                colormap=colormap,
                opacity=opacity,
                clim=clim,
            )

        # Render prediction with same limits
        if self._prediction_data is not None:
            self._render_kspace_volume(
                self.plotter_prediction,
                self._prediction_data,
                "Prediction",
                colormap=colormap,
                opacity=opacity,
                clim=clim,
            )

        # Update views
        if hasattr(self.ctrl, "view_update_ground_truth"):
            self.ctrl.view_update_ground_truth()
        if hasattr(self.ctrl, "view_update_prediction"):
            self.ctrl.view_update_prediction()

    def _kspace_to_voxels(self, data: np.ndarray) -> np.ndarray | None:
        """Convert k-space data to real-space voxels."""
        try:
            # Convert from flat real/imag to complex 3D
            if data.ndim == 1:
                n_half = len(data) // 2
                cube_root = round(n_half ** (1 / 3))

                if cube_root**3 == n_half:
                    real_part = data[:n_half].reshape((cube_root,) * 3)
                    imag_part = data[n_half:].reshape((cube_root,) * 3)
                    kspace = real_part + 1j * imag_part
                else:
                    logger.warning(f"Cannot reshape {n_half} to cubic grid")
                    return None
            else:
                kspace = data

            # Convert to real-space image via inverse FFT
            voxel_data = np.fft.ifftn(np.fft.ifftshift(kspace))
            voxel_data = np.real(voxel_data)
            return voxel_data

        except Exception as e:
            logger.error(f"Failed to convert k-space to voxels: {e}")
            return None

    def _render_kspace_volume(
        self,
        plotter: pv.Plotter,
        data: np.ndarray,
        title: str,
        colormap: str = "viridis",
        opacity: str = "sigmoid",
        clim: tuple[float, float] | None = None,
    ):
        """Render k-space data as a volume."""
        plotter.clear()

        try:
            voxel_data = self._kspace_to_voxels(data)
            if voxel_data is None:
                return

            # Create PyVista grid
            grid = pv.ImageData()
            grid.dimensions = voxel_data.shape
            grid.spacing = tuple(1.0 / s for s in voxel_data.shape)
            grid.origin = (0, 0, 0)
            grid["values"] = voxel_data.flatten(order="F")

            plotter.add_volume(
                grid,
                scalars="values",
                opacity=opacity,
                cmap=colormap,
                clim=clim,
            )
            plotter.show_grid()
            plotter.reset_camera()
            plotter.render()

        except Exception as e:
            logger.error(f"Failed to render volume: {e}")

    def _log(self, message: str):
        """Add message to training log."""
        self.state.inv_training_log = [*self.state.inv_training_log, message]
        logger.info(message)

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.0f}s"

    def _update_loss_chart(self):
        """Update the loss chart with current training history."""
        if self.loss_chart_widget is None:
            return

        fig, ax = plt.subplots(figsize=(6, 3))

        # Check if we have K-fold training data
        if len(self._kfold_epochs) > 0 and len(self._kfold_train_losses) > 0:
            # K-fold: plot all folds with different colors
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i) for i in range(10)]
            for fold_idx in range(len(self._kfold_train_losses)):
                if len(self._kfold_epochs[fold_idx]) > 0:
                    color = colors[fold_idx % len(colors)]
                    # Train loss: solid line
                    ax.semilogy(
                        self._kfold_epochs[fold_idx],
                        self._kfold_train_losses[fold_idx],
                        "-",
                        color=color,
                        linewidth=1.0,
                        alpha=0.7,
                        label=f"Fold {fold_idx + 1} Train" if fold_idx == 0 else None,
                    )
                    # Test loss: dashed line
                    ax.semilogy(
                        self._kfold_epochs[fold_idx],
                        self._kfold_test_losses[fold_idx],
                        "--",
                        color=color,
                        linewidth=1.0,
                        alpha=0.7,
                        label=f"Fold {fold_idx + 1} Test" if fold_idx == 0 else None,
                    )
            # Add legend showing line style meaning
            ax.plot([], [], "k-", linewidth=1.0, label="Train (solid)")
            ax.plot([], [], "k--", linewidth=1.0, label="Test (dashed)")
            ax.legend(loc="upper right", fontsize=7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log scale)")
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"K-Fold Training Progress (Fold {self._current_fold + 1}/{len(self._kfold_train_losses)})"
            )
        elif len(self._loss_epochs) > 0:
            # Regular training
            ax.semilogy(
                self._loss_epochs,
                self._train_losses,
                "b-",
                linewidth=1.5,
                label="Train Loss",
            )
            ax.semilogy(
                self._loss_epochs,
                self._test_losses,
                "r-",
                linewidth=1.5,
                label="Test Loss",
            )
            ax.legend(loc="upper right")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log scale)")
            ax.grid(True, alpha=0.3)
            ax.set_title("Training Progress")
        else:
            ax.set_title("Training Progress")

        plt.tight_layout()

        self.loss_chart_widget.update(fig)
        plt.close(fig)

    def _update_view_loss_chart(self):
        """Update the loss chart in the view tab."""
        if self.view_loss_chart_widget is None:
            return

        fig, ax = plt.subplots(figsize=(8, 4))

        # Check if we have K-fold training history
        if len(self._kfold_epochs) > 0 and len(self._kfold_train_losses) > 0:
            # K-fold: plot all folds with different colors
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i) for i in range(10)]
            for fold_idx in range(len(self._kfold_train_losses)):
                if len(self._kfold_epochs[fold_idx]) > 0:
                    color = colors[fold_idx % len(colors)]
                    # Train loss: solid line
                    ax.semilogy(
                        self._kfold_epochs[fold_idx],
                        self._kfold_train_losses[fold_idx],
                        "-",
                        color=color,
                        linewidth=1.0,
                        alpha=0.7,
                        label=f"Fold {fold_idx + 1} Train",
                    )
                    # Test loss: dashed line
                    ax.semilogy(
                        self._kfold_epochs[fold_idx],
                        self._kfold_test_losses[fold_idx],
                        "--",
                        color=color,
                        linewidth=1.0,
                        alpha=0.7,
                        label=f"Fold {fold_idx + 1} Test",
                    )
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log scale)")
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"K-Fold Training History ({len(self._kfold_train_losses)} folds)"
            )
        elif len(self._loss_epochs) > 0:
            ax.semilogy(
                self._loss_epochs,
                self._train_losses,
                "b-",
                linewidth=1.5,
                label="Train Loss",
            )
            ax.semilogy(
                self._loss_epochs,
                self._test_losses,
                "r-",
                linewidth=1.5,
                label="Test Loss",
            )
            ax.legend(loc="upper right")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log scale)")
            ax.grid(True, alpha=0.3)
            ax.set_title("Training History")
        else:
            ax.set_title("Training History")

        plt.tight_layout()

        self.view_loss_chart_widget.update(fig)
        plt.close(fig)

    def _start_training(self):
        """Start training the inverse model."""
        asyncio.create_task(self._train_model_async())

    async def _train_model_async(self):
        """Train the inverse model asynchronously."""
        batch_name = self.state.inv_selected_batch
        if not batch_name:
            self._log("Error: No batch selected")
            return

        self.state.inv_is_training = True
        self.state.inv_training_progress = 0
        self.state.inv_training_log = []
        self.state.inv_training_message = "Preparing data..."

        # Clear loss history for new training run
        self._loss_epochs = []
        self._train_losses = []
        self._test_losses = []
        self._kfold_train_losses = []
        self._kfold_test_losses = []
        self._kfold_epochs = []
        self._current_fold = 0
        self._training_start_time = None
        self._training_duration_seconds = None
        self._update_loss_chart()

        batch_dir = DEFAULT_DATA_DIR / batch_name
        sims_dir = batch_dir / "simulations"
        models_dir = batch_dir / "inverse_models"
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._log(f"Training inverse model for batch: {batch_name}")
            self._log(f"Model type: {self.state.inv_model_type}")

            # Check if training data exists
            await asyncio.sleep(0)
            self.state.inv_training_message = "Checking for training data..."

            has_training_data = self._check_training_data(sims_dir)

            if not has_training_data:
                self._log("Training data not found. Generating...")
                self.state.inv_training_message = "Generating training data..."
                await asyncio.sleep(0)

                # Generate training data
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: self._generate_training_data(batch_dir)
                )

            # Load training data
            self.state.inv_training_message = "Loading training data..."
            self.state.inv_training_progress = 10
            await asyncio.sleep(0)

            loop = asyncio.get_event_loop()
            X, y, sample_ids = await loop.run_in_executor(
                None, lambda: self._load_training_data(sims_dir)
            )

            self._log(f"Loaded {len(sample_ids)} samples")
            self._log(f"Input shape: {X.shape}, Output shape: {y.shape}")

            model_type = self.state.inv_model_type
            use_kfold = bool(self.state.inv_kfold_validation)

            # Train the model with timing
            self._training_start_time = time.time()

            if use_kfold:
                # K-fold cross validation
                k = int(self.state.inv_kfold_k)
                self._log(f"Running {k}-fold cross validation")

                result = await self._train_kfold(X, y, sample_ids, model_type, k, loop)
                models, all_predictions, fold_metrics, all_test_hashes = result

                self._training_duration_seconds = (
                    time.time() - self._training_start_time
                )
                duration_str = self._format_duration(self._training_duration_seconds)
                self._log(f"K-fold training completed in {duration_str}")

                # Aggregate metrics
                avg_test_loss = np.mean([m["test_loss"] for m in fold_metrics])
                metrics = {
                    "train_loss": np.mean([m["train_loss"] for m in fold_metrics]),
                    "test_loss": avg_test_loss,
                    "fold_metrics": fold_metrics,
                    "kfold": True,
                    "k": k,
                }

                # Save K-fold model (contains all K models and predictions)
                self.state.inv_training_message = "Saving models..."
                self.state.inv_training_progress = 95
                await asyncio.sleep(0)

                custom_name = (
                    self.state.inv_model_name.strip()
                    if self.state.inv_model_name
                    else ""
                )
                if custom_name:
                    model_name = custom_name
                else:
                    model_name = f"{model_type}_{k}fold_{len(sample_ids)}samples"
                model_path = models_dir / f"{model_name}.pkl"

                await loop.run_in_executor(
                    None,
                    lambda: self._save_kfold_model(
                        models, all_predictions, all_test_hashes, model_path, metrics
                    ),
                )

                self._log(f"K-fold models saved to {model_path}")
                self._log(f"Average test loss: {avg_test_loss:.6f}")

            else:
                # Regular training
                test_fraction = float(self.state.inv_test_fraction)

                # Train model
                self.state.inv_training_message = "Training model..."
                self.state.inv_training_progress = 20
                await asyncio.sleep(0)

                # Create progress callback
                server = self.server
                page = self

                def progress_callback(
                    epoch: int, total_epochs: int, train_loss: float, test_loss: float
                ):
                    progress = 20 + int((epoch / total_epochs) * 70)

                    def update():
                        page._loss_epochs.append(epoch)
                        page._train_losses.append(train_loss)
                        page._test_losses.append(test_loss)

                        self.state.inv_training_progress = progress
                        self.state.inv_training_message = (
                            f"Epoch {epoch}/{total_epochs} | "
                            f"Train: {train_loss:.6f} | Test: {test_loss:.6f}"
                        )

                        page._update_loss_chart()
                        server.state.flush()

                    loop.call_soon_threadsafe(update)

                model, test_hashes, metrics = await loop.run_in_executor(
                    None,
                    lambda: self._train_model(
                        X,
                        y,
                        sample_ids,
                        model_type,
                        test_fraction,
                        progress_callback,
                    ),
                )

                self._training_duration_seconds = (
                    time.time() - self._training_start_time
                )
                duration_str = self._format_duration(self._training_duration_seconds)
                self._log(f"Training completed in {duration_str}")

                # Save the model
                self.state.inv_training_message = "Saving model..."
                self.state.inv_training_progress = 95
                await asyncio.sleep(0)

                custom_name = (
                    self.state.inv_model_name.strip()
                    if self.state.inv_model_name
                    else ""
                )
                if custom_name:
                    model_name = custom_name
                else:
                    model_name = f"{model_type}_{len(sample_ids)}samples"
                model_path = models_dir / f"{model_name}.pkl"

                await loop.run_in_executor(
                    None,
                    lambda: self._save_model(model, model_path, test_hashes, metrics),
                )

                self._log(f"Model saved to {model_path}")
                self._log(f"Test metrics: {metrics}")

            self.state.inv_training_progress = 100
            self.state.inv_training_message = "Training complete!"

            # Refresh model lists in view and test tabs
            models = self._get_models_for_batch(batch_name)
            self.state.inv_view_available_models = models
            self.state.inv_test_available_models = models

        except Exception as e:
            self._log(f"Error during training: {e}")
            logger.exception("Training failed")
            self.state.inv_training_message = f"Error: {e}"

        finally:
            self.state.inv_is_training = False

    def _get_current_preprocessing_params(self) -> dict:
        """Get current preprocessing parameters from UI state."""
        return {
            "trim_timesteps": int(self.state.inv_trim_timesteps),
            "downsample_factor": int(self.state.inv_downsample_factor),
            "use_compression": bool(self.state.inv_use_dynamic_compression),
            "compression_threshold": float(self.state.inv_compression_threshold),
            "compression_ratio": float(self.state.inv_compression_ratio),
            "use_normalization": bool(self.state.inv_use_normalization),
            "use_noise": bool(self.state.inv_use_noise),
            "noise_level": float(self.state.inv_noise_level),
            "kspace_grid_size": int(self.state.inv_kspace_grid_size),
        }

    def _check_training_data(self, sims_dir: Path) -> bool:
        """Check if training data exists with matching preprocessing parameters."""
        if not sims_dir.exists():
            return False

        current_params = self._get_current_preprocessing_params()

        for sim_dir in sims_dir.iterdir():
            if not sim_dir.is_dir():
                continue

            input_file = sim_dir / "model_input.pkl"
            output_file = sim_dir / "model_output.pkl"
            params_file = sim_dir / "preprocessing_params.pkl"

            if not (input_file.exists() and output_file.exists()):
                continue

            # Check if preprocessing parameters match
            if params_file.exists():
                try:
                    with open(params_file, "rb") as f:
                        cached_params = pickle.load(f)
                    if cached_params == current_params:
                        return True
                    else:
                        # Parameters don't match - need to regenerate
                        logger.info(
                            "Preprocessing params changed, will regenerate training data"
                        )
                        return False
                except Exception:
                    # Can't read params file - regenerate to be safe
                    return False
            else:
                # No params file - old format, regenerate
                logger.info(
                    "No preprocessing params found, will regenerate training data"
                )
                return False

        return False

    def _generate_training_data(self, batch_dir: Path):
        """Generate model_input.pkl and model_output.pkl for all simulations."""
        sims_dir = batch_dir / "simulations"

        # Get current preprocessing parameters
        current_params = self._get_current_preprocessing_params()

        for sim_dir in sims_dir.iterdir():
            if not sim_dir.is_dir():
                continue

            # Check for required files
            sensor_file = sim_dir / "sensor_data.pkl"
            config_file = sim_dir / "config.toml"

            if not sensor_file.exists():
                logger.warning(f"Missing sensor_data.pkl in {sim_dir.name}")
                continue

            if not config_file.exists():
                logger.warning(f"Missing config.toml in {sim_dir.name}")
                continue

            try:
                # Generate model input (processed sensor data)
                input_data = self._process_sensor_data(
                    sensor_file,
                    trim_timesteps=current_params["trim_timesteps"],
                    downsample_factor=current_params["downsample_factor"],
                    use_compression=current_params["use_compression"],
                    compression_threshold=current_params["compression_threshold"],
                    compression_ratio=current_params["compression_ratio"],
                    use_normalization=current_params["use_normalization"],
                    use_noise=current_params["use_noise"],
                    noise_level=current_params["noise_level"],
                )
                with open(sim_dir / "model_input.pkl", "wb") as f:
                    pickle.dump(input_data, f)

                # Generate model output (k-space from config)
                output_data = self._process_config_to_kspace(
                    config_file, grid_size=current_params["kspace_grid_size"]
                )
                with open(sim_dir / "model_output.pkl", "wb") as f:
                    pickle.dump(output_data, f)

                # Save preprocessing parameters for cache validation
                with open(sim_dir / "preprocessing_params.pkl", "wb") as f:
                    pickle.dump(current_params, f)

            except Exception as e:
                logger.error(f"Failed to process {sim_dir.name}: {e}")

    def _process_sensor_data(
        self,
        sensor_file: Path,
        trim_timesteps: int = 51,
        downsample_factor: int = 2,
        use_compression: bool = False,
        compression_threshold: float = 0.1,
        compression_ratio: float = 4.0,
        use_normalization: bool = False,
        use_noise: bool = False,
        noise_level: float = 0.05,
    ) -> np.ndarray:
        """Process sensor data for model input.

        Args:
            sensor_file: Path to the sensor data pickle file.
            trim_timesteps: Number of initial timesteps to skip.
            downsample_factor: Factor to downsample time series (1 = no downsampling).
            use_compression: Whether to apply dynamic range compression.
            compression_threshold: Amplitude threshold (0-1) for compression.
            compression_ratio: Compression ratio (e.g., 4 means 4:1).
            use_normalization: Whether to normalize each channel to [-1, 1].
            use_noise: Whether to add Gaussian noise to the signal.
            noise_level: Fraction (0-1) of global peak to use as noise std.

        Returns:
            Flattened sensor data array.
        """
        with open(sensor_file, "rb") as f:
            data = pickle.load(f)

        # Handle different data formats
        if isinstance(data, dict) and "pressure" in data:
            sensor_data = data["pressure"]
        else:
            sensor_data = data

        # Convert from CuPy if needed
        if hasattr(sensor_data, "get"):
            sensor_data = sensor_data.get()

        sensor_data = np.asarray(sensor_data)

        # Trim initial transient and downsample
        if sensor_data.ndim == 2:
            sensor_data = sensor_data[:, trim_timesteps:]
            if downsample_factor > 1:
                sensor_data = sensor_data[:, ::downsample_factor]

        # Apply dynamic compression if enabled
        if use_compression and sensor_data.ndim == 2:
            sensor_data = self._apply_dynamic_compression(
                sensor_data, compression_threshold, compression_ratio
            )

        # Apply normalization if enabled
        if use_normalization and sensor_data.ndim == 2:
            sensor_data = self._apply_normalization(sensor_data)

        # Apply noise if enabled
        if use_noise and sensor_data.ndim == 2:
            sensor_data = self._apply_noise(sensor_data, noise_level)

        return sensor_data.flatten().astype(np.float32)

    def _apply_dynamic_compression(
        self,
        sensor_data: np.ndarray,
        threshold: float = 0.1,
        ratio: float = 4.0,
    ) -> np.ndarray:
        """Apply dynamic range compression to sensor data.

        Similar to audio compression - reduces the dynamic range by attenuating
        samples that exceed the threshold. The threshold is computed as a fraction
        of the GLOBAL maximum across all channels, so early high-amplitude signals
        get compressed while later quieter signals remain relatively unchanged.

        Args:
            sensor_data: 2D array (n_sensors, n_timesteps).
            threshold: Fraction (0-1) of global max above which compression applies.
            ratio: Compression ratio (e.g., 4 means 4:1 compression above threshold).

        Returns:
            Compressed sensor data with reduced dynamic range.
        """
        # Find global max across all channels
        global_max = np.abs(sensor_data).max()
        if global_max == 0:
            return sensor_data.copy()

        # Compute absolute threshold value from fraction of global max
        abs_threshold = threshold * global_max

        # Apply compression
        compressed = sensor_data.copy()
        abs_data = np.abs(compressed)

        # Find samples above threshold
        above_mask = abs_data > abs_threshold

        if above_mask.any():
            # Compress: new_value = threshold + (excess / ratio)
            excess = abs_data[above_mask] - abs_threshold
            compressed_excess = excess / ratio
            new_amplitude = abs_threshold + compressed_excess
            compressed[above_mask] = np.sign(compressed[above_mask]) * new_amplitude

        return compressed

    def _apply_normalization(self, sensor_data: np.ndarray) -> np.ndarray:
        """Normalize each sensor channel to [-1, 1] range.

        Normalizes each channel independently based on its own maximum value,
        so all channels end up in the same amplitude range.

        Args:
            sensor_data: 2D array (n_sensors, n_timesteps).

        Returns:
            Normalized sensor data with each channel in [-1, 1].
        """
        normalized = sensor_data.copy()
        for i in range(normalized.shape[0]):
            max_val = np.abs(normalized[i]).max()
            if max_val > 0:
                normalized[i] = normalized[i] / max_val
        return normalized

    def _apply_noise(
        self,
        sensor_data: np.ndarray,
        noise_level: float = 0.05,
    ) -> np.ndarray:
        """Add Gaussian noise to sensor data.

        The noise amplitude is computed as a fraction of the GLOBAL peak
        across all sensors, so the noise level is consistent relative to
        the signal strength.

        Args:
            sensor_data: 2D array (n_sensors, n_timesteps).
            noise_level: Fraction (0-1) of global peak to use as noise std.

        Returns:
            Sensor data with added Gaussian noise.
        """
        if noise_level <= 0:
            return sensor_data.copy()

        # Find global max across all channels
        global_max = np.abs(sensor_data).max()
        if global_max == 0:
            return sensor_data.copy()

        # Compute noise standard deviation as fraction of global peak
        noise_std = noise_level * global_max

        # Generate and add noise
        noise = np.random.normal(0, noise_std, sensor_data.shape)
        return sensor_data + noise

    def _process_config_to_kspace(
        self, config_file: Path, grid_size: int | None = None
    ):
        """Convert config parameters to k-space representation."""
        import tomli

        if grid_size is None:
            grid_size = int(self.state.inv_kspace_grid_size)

        with open(config_file, "rb") as f:
            config = tomli.load(f)

        # Extract inclusion parameters
        mesh_cfg = config.get("mesh", {})
        material_cfg = config.get("material", {})

        cube_centers = mesh_cfg.get("cube_centers", [])
        cube_widths = mesh_cfg.get("cube_widths", [])
        density = material_cfg.get("inclusion_density", 2.0)

        # Create voxel grid
        grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

        x = np.linspace(0, 1, grid_size, endpoint=False)
        y = np.linspace(0, 1, grid_size, endpoint=False)
        z = np.linspace(0, 1, grid_size, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        for center, width in zip(cube_centers, cube_widths, strict=False):
            if len(center) >= 3:
                cx, cy, cz = center[:3]
                hw = width / 2
                mask = (
                    (np.abs(X - cx) <= hw)
                    & (np.abs(Y - cy) <= hw)
                    & (np.abs(Z - cz) <= hw)
                )
                grid[mask] = density

        # Compute FFT
        kspace = np.fft.fftn(grid)
        kspace = np.fft.fftshift(kspace)

        # Split into real and imaginary
        real_part = np.real(kspace).flatten()
        imag_part = np.imag(kspace).flatten()

        return np.concatenate([real_part, imag_part]).astype(np.float32)

    def _load_training_data(
        self, sims_dir: Path
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load training data from simulation directories."""
        from sbimaging.inverse_models.base import DataLoader

        loader = DataLoader(sims_dir)
        return loader.load()

    def _parse_hidden_layers(self, layers_str: str) -> list[int]:
        """Parse comma-separated hidden layer sizes string into list of ints."""
        try:
            return [int(x.strip()) for x in layers_str.split(",") if x.strip()]
        except ValueError:
            logger.warning(f"Invalid hidden layers string: {layers_str}, using default")
            return [4096, 2048, 1024]

    def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: list[str],
        model_type: str,
        test_fraction: float,
        progress_callback,
    ):
        """Train the inverse model."""
        if model_type.startswith("nn"):
            from sbimaging.inverse_models.nn.network import NeuralNetworkModel

            architecture = "cnn" if model_type == "nn_cnn" else "mlp"

            # Parse MLP configuration
            mlp_hidden_layers = self._parse_hidden_layers(
                self.state.inv_mlp_hidden_layers
            )
            mlp_dropout = float(self.state.inv_mlp_dropout)

            # Parse CNN configuration
            cnn_conv_channels = self._parse_hidden_layers(
                self.state.inv_cnn_conv_channels
            )
            cnn_pool_size = int(self.state.inv_cnn_pool_size)
            cnn_regressor_hidden = int(self.state.inv_cnn_regressor_hidden)
            cnn_dropout = float(self.state.inv_cnn_dropout)

            model = NeuralNetworkModel(
                name=model_type,
                architecture=architecture,
                mlp_hidden_layers=mlp_hidden_layers,
                mlp_dropout=mlp_dropout,
                cnn_conv_channels=cnn_conv_channels,
                cnn_pool_size=cnn_pool_size,
                cnn_regressor_hidden=cnn_regressor_hidden,
                cnn_dropout=cnn_dropout,
            )
            # NN model handles train/test split internally
            metrics = model.train(
                X,
                y,
                test_fraction=test_fraction,
                epochs=int(self.state.inv_epochs),
                batch_size=int(self.state.inv_batch_size),
                learning_rate=float(self.state.inv_learning_rate),
                sample_ids=sample_ids,
                progress_callback=progress_callback,
                early_stopping=bool(self.state.inv_early_stopping),
                early_stopping_patience=int(self.state.inv_early_stopping_patience),
            )
            test_ids = model.test_indices
        else:
            from sbimaging.inverse_models.base import train_test_split_by_index
            from sbimaging.inverse_models.gp.emulator import GaussianProcessModel

            # GP model needs manual split
            X_train, X_test, y_train, y_test, train_ids, test_ids = (
                train_test_split_by_index(X, y, sample_ids, test_fraction)
            )

            model = GaussianProcessModel(name=model_type)
            model.train(X_train, y_train)
            model.train_indices = train_ids
            model.test_indices = test_ids

            # Evaluate
            metrics = model.evaluate(X_test, y_test)

        return model, test_ids, metrics

    async def _train_kfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_ids: list[str],
        model_type: str,
        k: int,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[list, dict[str, np.ndarray], list[dict], list[str]]:
        """Train K models using K-fold cross validation.

        Args:
            X: Input features (n_samples, n_features).
            y: Target values (n_samples, n_outputs).
            sample_ids: List of sample identifiers (simulation hashes).
            model_type: Type of model to train ('nn_cnn', 'nn_mlp', 'gp').
            k: Number of folds.
            loop: Event loop for async execution.

        Returns:
            Tuple of:
                - models: List of K trained models
                - all_predictions: Dict mapping sample_id to prediction array
                - fold_metrics: List of metrics dicts for each fold
                - all_test_hashes: List of all sample IDs (each was tested once)
        """
        from sklearn.model_selection import KFold

        # Initialize K-fold data structures
        self._kfold_train_losses = [[] for _ in range(k)]
        self._kfold_test_losses = [[] for _ in range(k)]
        self._kfold_epochs = [[] for _ in range(k)]

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        sample_ids_arr = np.array(sample_ids)

        models = []
        all_predictions: dict[str, np.ndarray] = {}
        fold_metrics = []
        all_test_hashes: list[str] = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            self._current_fold = fold_idx
            self._log(f"Training fold {fold_idx + 1}/{k}...")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_ids = sample_ids_arr[test_idx].tolist()

            # Update progress
            fold_progress_base = 20 + int((fold_idx / k) * 70)
            self.state.inv_training_progress = fold_progress_base
            self.state.inv_training_message = f"Fold {fold_idx + 1}/{k}: Training..."
            await asyncio.sleep(0)

            # Create progress callback for this fold
            # Capture variables in closure by passing as default arguments
            def make_progress_callback(
                fold: int,
                fold_base: int,
                k_folds: int,
                page_ref: "InverseModelsPage",
                server_ref,
                loop_ref,
            ):
                def progress_callback(
                    epoch: int, total_epochs: int, train_loss: float, test_loss: float
                ):
                    fold_progress = fold_base + int(
                        (epoch / total_epochs) * (70 / k_folds)
                    )

                    def update():
                        page_ref._kfold_epochs[fold].append(epoch)
                        page_ref._kfold_train_losses[fold].append(train_loss)
                        page_ref._kfold_test_losses[fold].append(test_loss)

                        page_ref.state.inv_training_progress = fold_progress
                        page_ref.state.inv_training_message = (
                            f"Fold {fold + 1}/{k_folds} | Epoch {epoch}/{total_epochs} | "
                            f"Train: {train_loss:.6f} | Test: {test_loss:.6f}"
                        )

                        page_ref._update_loss_chart()
                        server_ref.state.flush()

                    loop_ref.call_soon_threadsafe(update)

                return progress_callback

            progress_cb = make_progress_callback(
                fold_idx, fold_progress_base, k, self, self.server, loop
            )

            # Train model for this fold
            # Bind loop variables via default arguments to avoid closure issues
            if model_type.startswith("nn"):
                model, metrics = await loop.run_in_executor(
                    None,
                    lambda xtr=X_train, ytr=y_train, xte=X_test, yte=y_test, mt=model_type, cb=progress_cb: (
                        self._train_single_fold(xtr, ytr, xte, yte, mt, cb)
                    ),
                )
            else:
                # GP model
                model, metrics = await loop.run_in_executor(
                    None,
                    lambda xtr=X_train, ytr=y_train, xte=X_test, yte=y_test, mt=model_type: (
                        self._train_single_fold_gp(xtr, ytr, xte, yte, mt)
                    ),
                )

            models.append(model)
            fold_metrics.append(metrics)

            # Generate predictions for test samples in this fold
            for i, sample_id in enumerate(test_ids):
                X_single = X_test[i : i + 1]
                pred = model.predict(X_single)
                all_predictions[sample_id] = pred.ravel()
                all_test_hashes.append(sample_id)

            self._log(
                f"Fold {fold_idx + 1}: Train={metrics['train_loss']:.6f}, "
                f"Test={metrics['test_loss']:.6f}"
            )

        return models, all_predictions, fold_metrics, all_test_hashes

    def _train_single_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str,
        progress_callback,
    ) -> tuple:
        """Train a single fold for neural network models."""
        from sbimaging.inverse_models.nn.network import NeuralNetworkModel

        architecture = "cnn" if model_type == "nn_cnn" else "mlp"

        # Parse architecture config
        mlp_hidden_layers = self._parse_hidden_layers(self.state.inv_mlp_hidden_layers)
        mlp_dropout = float(self.state.inv_mlp_dropout)
        cnn_conv_channels = self._parse_hidden_layers(self.state.inv_cnn_conv_channels)
        cnn_pool_size = int(self.state.inv_cnn_pool_size)
        cnn_regressor_hidden = int(self.state.inv_cnn_regressor_hidden)
        cnn_dropout = float(self.state.inv_cnn_dropout)

        model = NeuralNetworkModel(
            name=model_type,
            architecture=architecture,
            mlp_hidden_layers=mlp_hidden_layers,
            mlp_dropout=mlp_dropout,
            cnn_conv_channels=cnn_conv_channels,
            cnn_pool_size=cnn_pool_size,
            cnn_regressor_hidden=cnn_regressor_hidden,
            cnn_dropout=cnn_dropout,
        )

        # Train with pre-split data (test_fraction=0 since we already split)
        # Combine train/test for the model's internal handling
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.vstack([y_train, y_test])

        # Create sample IDs for combined data
        n_train = len(X_train)
        n_test = len(X_test)
        sample_ids = [f"train_{i}" for i in range(n_train)] + [
            f"test_{i}" for i in range(n_test)
        ]

        # Set test fraction to match our split
        test_fraction = n_test / (n_train + n_test)

        metrics = model.train(
            X_combined,
            y_combined,
            test_fraction=test_fraction,
            epochs=int(self.state.inv_epochs),
            batch_size=int(self.state.inv_batch_size),
            learning_rate=float(self.state.inv_learning_rate),
            sample_ids=sample_ids,
            progress_callback=progress_callback,
            early_stopping=bool(self.state.inv_early_stopping),
            early_stopping_patience=int(self.state.inv_early_stopping_patience),
        )

        return model, metrics

    def _train_single_fold_gp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str,
    ) -> tuple:
        """Train a single fold for Gaussian Process models."""
        from sbimaging.inverse_models.gp.emulator import GaussianProcessModel

        model = GaussianProcessModel(name=model_type)
        model.train(X_train, y_train)

        # Evaluate on test set
        metrics = model.evaluate(X_test, y_test)

        return model, metrics

    def _save_kfold_model(
        self,
        models: list,
        all_predictions: dict[str, np.ndarray],
        all_test_hashes: list[str],
        path: Path,
        metrics: dict,
    ):
        """Save K-fold trained models with predictions and metadata."""
        data = {
            "models": models,  # List of K models
            "model_type": self.state.inv_model_type,
            "model_name": self.state.inv_model_name.strip()
            if self.state.inv_model_name
            else None,
            "test_hashes": all_test_hashes,  # All samples (each tested once)
            "predictions": all_predictions,  # Dict: sample_id -> prediction
            "metrics": metrics,
            "kfold": True,
            "k": int(self.state.inv_kfold_k),
            "training_config": {
                "epochs": int(self.state.inv_epochs),
                "batch_size": int(self.state.inv_batch_size),
                "learning_rate": float(self.state.inv_learning_rate),
                "training_duration_seconds": self._training_duration_seconds,
                "early_stopping": bool(self.state.inv_early_stopping),
                "early_stopping_patience": int(self.state.inv_early_stopping_patience),
                "kfold": True,
                "k": int(self.state.inv_kfold_k),
            },
            "architecture_config": {
                "mlp_hidden_layers": self._parse_hidden_layers(
                    self.state.inv_mlp_hidden_layers
                ),
                "mlp_dropout": float(self.state.inv_mlp_dropout),
                "cnn_conv_channels": self._parse_hidden_layers(
                    self.state.inv_cnn_conv_channels
                ),
                "cnn_pool_size": int(self.state.inv_cnn_pool_size),
                "cnn_regressor_hidden": int(self.state.inv_cnn_regressor_hidden),
                "cnn_dropout": float(self.state.inv_cnn_dropout),
            },
            "kspace_config": {
                "grid_size": int(self.state.inv_kspace_grid_size),
            },
            "preprocessing_config": {
                "trim_timesteps": int(self.state.inv_trim_timesteps),
                "downsample_factor": int(self.state.inv_downsample_factor),
                "use_dynamic_compression": bool(self.state.inv_use_dynamic_compression),
                "compression_threshold": float(self.state.inv_compression_threshold),
                "compression_ratio": float(self.state.inv_compression_ratio),
                "use_normalization": bool(self.state.inv_use_normalization),
                "use_noise": bool(self.state.inv_use_noise),
                "noise_level": float(self.state.inv_noise_level),
            },
            "training_history": {
                "kfold_epochs": [list(e) for e in self._kfold_epochs],
                "kfold_train_losses": [
                    list(losses) for losses in self._kfold_train_losses
                ],
                "kfold_test_losses": [
                    list(losses) for losses in self._kfold_test_losses
                ],
            },
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        # Also save predictions to the predictions directory
        batch_name = self.state.inv_selected_batch
        predictions_dir = DEFAULT_DATA_DIR / batch_name / "predictions" / path.stem
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for sample_id, prediction in all_predictions.items():
            pred_file = predictions_dir / f"{sample_id}.pkl"
            with open(pred_file, "wb") as f:
                pickle.dump(prediction, f)

    def _save_model(self, model, path: Path, test_hashes: list[str], metrics: dict):
        """Save the trained model with metadata and training history."""
        data = {
            "model": model,
            "model_type": self.state.inv_model_type,
            "model_name": self.state.inv_model_name.strip()
            if self.state.inv_model_name
            else None,
            "test_hashes": test_hashes,
            "metrics": metrics,
            "training_config": {
                "epochs": int(self.state.inv_epochs),
                "epochs_completed": metrics.get(
                    "epochs_completed", int(self.state.inv_epochs)
                ),
                "batch_size": int(self.state.inv_batch_size),
                "learning_rate": float(self.state.inv_learning_rate),
                "test_fraction": float(self.state.inv_test_fraction),
                "training_duration_seconds": self._training_duration_seconds,
                "early_stopping": bool(self.state.inv_early_stopping),
                "early_stopping_patience": int(self.state.inv_early_stopping_patience),
                "stopped_early": metrics.get("stopped_early", False),
            },
            "architecture_config": {
                "mlp_hidden_layers": self._parse_hidden_layers(
                    self.state.inv_mlp_hidden_layers
                ),
                "mlp_dropout": float(self.state.inv_mlp_dropout),
                "cnn_conv_channels": self._parse_hidden_layers(
                    self.state.inv_cnn_conv_channels
                ),
                "cnn_pool_size": int(self.state.inv_cnn_pool_size),
                "cnn_regressor_hidden": int(self.state.inv_cnn_regressor_hidden),
                "cnn_dropout": float(self.state.inv_cnn_dropout),
            },
            "kspace_config": {
                "grid_size": int(self.state.inv_kspace_grid_size),
            },
            "preprocessing_config": {
                "trim_timesteps": int(self.state.inv_trim_timesteps),
                "downsample_factor": int(self.state.inv_downsample_factor),
                "use_dynamic_compression": bool(self.state.inv_use_dynamic_compression),
                "compression_threshold": float(self.state.inv_compression_threshold),
                "compression_ratio": float(self.state.inv_compression_ratio),
                "use_normalization": bool(self.state.inv_use_normalization),
                "use_noise": bool(self.state.inv_use_noise),
                "noise_level": float(self.state.inv_noise_level),
            },
            "training_history": {
                "epochs": self._loss_epochs.copy(),
                "train_loss": self._train_losses.copy(),
                "test_loss": self._test_losses.copy(),
            },
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _start_testing(self):
        """Start testing the selected model."""
        asyncio.create_task(self._test_model_async())

    async def _test_model_async(self):
        """Test the model and generate predictions."""
        batch_name = self.state.inv_test_selected_batch
        model_name = self.state.inv_test_selected_model

        if not batch_name or not model_name:
            return

        self.state.inv_is_testing = True
        self.state.inv_test_progress = 0
        self.state.inv_test_message = "Loading model..."

        batch_dir = DEFAULT_DATA_DIR / batch_name
        model_path = batch_dir / "inverse_models" / f"{model_name}.pkl"

        try:
            # Load model and its preprocessing config
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            # Handle K-fold models: extract the selected fold's model
            is_kfold = data.get("kfold", False)
            if is_kfold:
                models = data.get("models", [])
                selected_fold = int(self.state.inv_test_selected_fold)
                if selected_fold < len(models):
                    model = models[selected_fold]
                    logger.info(f"Using K-fold model from fold {selected_fold + 1}")
                else:
                    logger.error(f"Invalid fold index {selected_fold}")
                    self.state.inv_test_message = "Error: Invalid fold selection"
                    return
                # For K-fold, save predictions in a fold-specific subfolder
                predictions_dir = (
                    batch_dir / "predictions" / f"{model_name}_fold{selected_fold + 1}"
                )
            else:
                model = data.get("model")
                # Save predictions in a subfolder named after the model
                predictions_dir = batch_dir / "predictions" / model_name

            predictions_dir.mkdir(parents=True, exist_ok=True)

            if model is None:
                logger.error("No model found in saved data")
                self.state.inv_test_message = "Error: No model found"
                return

            test_hashes = data.get("test_hashes", [])
            preprocessing_config = data.get("preprocessing_config", {})

            # Extract preprocessing parameters (use defaults if not saved)
            trim_timesteps = preprocessing_config.get("trim_timesteps", 45)
            downsample_factor = preprocessing_config.get("downsample_factor", 2)
            use_compression = preprocessing_config.get("use_dynamic_compression", False)
            compression_threshold = preprocessing_config.get(
                "compression_threshold", 0.1
            )
            compression_ratio = preprocessing_config.get("compression_ratio", 4.0)
            use_normalization = preprocessing_config.get("use_normalization", False)
            # Note: We don't apply noise during testing - noise was used during
            # training to make the model robust, but we test on clean signals
            use_noise = preprocessing_config.get("use_noise", False)
            noise_level = preprocessing_config.get("noise_level", 0.0)

            logger.info(
                f"Testing with preprocessing: trim={trim_timesteps}, "
                f"downsample={downsample_factor}, compression={use_compression}"
                + (f", trained with noise={noise_level:.0%}" if use_noise else "")
            )

            self.state.inv_test_message = f"Testing on {len(test_hashes)} samples..."

            loop = asyncio.get_event_loop()
            server = self.server

            # Generate predictions for each test sample
            for i, sim_hash in enumerate(test_hashes):
                progress = int((i / len(test_hashes)) * 100)

                def update_progress(p=progress, h=sim_hash):
                    self.state.inv_test_progress = p
                    self.state.inv_test_message = f"Processing {h[:12]}..."
                    server.state.flush()

                loop.call_soon_threadsafe(update_progress)

                # Load and preprocess sensor data using model's saved config
                sensor_file = batch_dir / "simulations" / sim_hash / "sensor_data.pkl"
                if not sensor_file.exists():
                    logger.warning(f"Missing sensor_data.pkl for {sim_hash}")
                    continue

                X_test = self._process_sensor_data(
                    sensor_file,
                    trim_timesteps=trim_timesteps,
                    downsample_factor=downsample_factor,
                    use_compression=use_compression,
                    compression_threshold=compression_threshold,
                    compression_ratio=compression_ratio,
                    use_normalization=use_normalization,
                )

                X_test = np.asarray(X_test)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(1, -1)

                # Predict
                prediction = await loop.run_in_executor(
                    None, lambda x=X_test: model.predict(x)
                )

                # Save prediction
                pred_file = predictions_dir / f"{sim_hash}.pkl"
                with open(pred_file, "wb") as f:
                    pickle.dump(prediction.ravel(), f)

            self.state.inv_test_progress = 100
            self.state.inv_test_message = "Testing complete!"

            # Refresh test simulations list to show predictions
            self._on_test_model_selected(model_name)

        except Exception as e:
            logger.exception("Testing failed")
            self.state.inv_test_message = f"Error: {e}"

        finally:
            self.state.inv_is_testing = False

    def build_ui(self):
        """Build the inverse models page UI with tabs."""
        with v3.VContainer(fluid=True, classes="pa-0"):
            with v3.VCard(classes="fill-height"):
                # Tabs
                with v3.VTabs(
                    v_model=("inv_active_tab",), density="compact", align_tabs="center"
                ):
                    v3.VTab(value="train", text="Train")
                    v3.VTab(value="view", text="View")
                    v3.VTab(value="test", text="Test")

                with v3.VCardText(classes="pa-4"):
                    with v3.VWindow(v_model=("inv_active_tab",)):
                        # Train tab
                        with v3.VWindowItem(value="train"):
                            self._build_train_tab()

                        # View tab
                        with v3.VWindowItem(value="view"):
                            self._build_view_tab()

                        # Test tab
                        with v3.VWindowItem(value="test"):
                            self._build_test_tab()

    def _build_train_tab(self):
        """Build the Train tab content."""
        with v3.VRow():
            # Left side: Configuration
            with v3.VCol(cols=12, md=6):
                # === GENERAL SECTION ===
                html.Div("General", classes="text-subtitle-2 mb-2")
                with v3.VRow(align="center", dense=True):
                    with v3.VCol(cols=10):
                        v3.VSelect(
                            v_model=("inv_selected_batch",),
                            items=("inv_available_batches",),
                            label="Simulation Batch",
                            density="compact",
                            clearable=True,
                        )
                    with v3.VCol(cols=2):
                        v3.VBtn(
                            icon="mdi-refresh",
                            variant="text",
                            density="compact",
                            click=self._refresh_batch_list,
                        )

                v3.VSelect(
                    v_model=("inv_model_type",),
                    items=("inv_model_type_items",),
                    label="Model Type",
                    density="compact",
                )

                v3.VTextField(
                    v_model=("inv_model_name",),
                    label="Model Name",
                    density="compact",
                    placeholder="e.g., baseline, compressed_v1, high_lr",
                    hint="Optional custom name for this model",
                    persistent_hint=True,
                    clearable=True,
                )

                # === HYPERPARAMETERS SECTION ===
                with html.Div(v_show=("inv_model_type.startsWith('nn')",)):
                    v3.VDivider(classes="my-3")
                    html.Div("Hyperparameters", classes="text-subtitle-2 mb-2")
                    with v3.VRow(dense=True):
                        with v3.VCol(cols=6):
                            v3.VTextField(
                                v_model=("inv_epochs",),
                                label="Epochs",
                                type="number",
                                density="compact",
                            )
                        with v3.VCol(cols=6):
                            v3.VTextField(
                                v_model=("inv_batch_size",),
                                label="Batch Size",
                                type="number",
                                density="compact",
                            )
                    with v3.VRow(dense=True):
                        with v3.VCol(cols=6):
                            v3.VTextField(
                                v_model=("inv_learning_rate",),
                                label="Learning Rate",
                                type="number",
                                step="0.0001",
                                density="compact",
                            )
                        with v3.VCol(cols=6):
                            v3.VTextField(
                                v_model=("inv_test_fraction",),
                                label="Test Fraction",
                                type="number",
                                step="0.05",
                                density="compact",
                                hint="Fraction held for testing",
                                persistent_hint=True,
                                disabled=("inv_kfold_validation",),
                            )

                # === INPUT/OUTPUT SECTION ===
                v3.VDivider(classes="my-3")
                html.Div("Input / Output", classes="text-subtitle-2 mb-2")

                # Show input/output sizes when batch is selected
                with v3.VRow(
                    dense=True,
                    v_show=("inv_input_size > 0 || inv_output_size > 0",),
                    classes="mb-2",
                ):
                    with v3.VCol(cols=6):
                        with html.Div(classes="text-caption"):
                            html.Span("Input size: ")
                            html.Strong(
                                "{{ inv_input_size.toLocaleString() }}",
                                classes="text-primary",
                            )
                            html.Span(
                                " ({{ inv_num_sensors }} sensors × {{ Math.floor((inv_raw_timesteps - inv_trim_timesteps) / (inv_downsample_factor > 1 ? inv_downsample_factor : 1)) }} timesteps)",
                                classes="text-medium-emphasis",
                            )
                    with v3.VCol(cols=6):
                        with html.Div(classes="text-caption"):
                            html.Span("Output size: ")
                            html.Strong(
                                "{{ inv_output_size.toLocaleString() }}",
                                classes="text-primary",
                            )
                            html.Span(
                                " ({{ inv_kspace_grid_size }}³ × 2)",
                                classes="text-medium-emphasis",
                            )

                with v3.VRow(dense=True):
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("inv_trim_timesteps",),
                            label="Trim Timesteps",
                            type="number",
                            density="compact",
                            hint="Skip initial timesteps",
                            persistent_hint=True,
                        )
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("inv_downsample_factor",),
                            label="Downsample",
                            type="number",
                            density="compact",
                            hint=("inv_downsample_hint",),
                            persistent_hint=True,
                        )
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("inv_kspace_grid_size",),
                            label="K-Space Grid",
                            type="number",
                            density="compact",
                            hint="e.g., 64 = 64³ voxels",
                            persistent_hint=True,
                        )

                # === MLP ARCHITECTURE SECTION ===
                with html.Div(v_show=("inv_model_type === 'nn_mlp'",)):
                    v3.VDivider(classes="my-3")
                    html.Div("MLP Architecture", classes="text-subtitle-2 mb-2")
                    v3.VTextField(
                        v_model=("inv_mlp_hidden_layers",),
                        label="Hidden Layers",
                        density="compact",
                        hint="Comma-separated layer sizes (e.g., 8192, 16384, 32768)",
                        persistent_hint=True,
                    )
                    with v3.VRow(dense=True):
                        with v3.VCol(cols=6):
                            v3.VTextField(
                                v_model=("inv_mlp_dropout",),
                                label="Dropout",
                                type="number",
                                step="0.05",
                                density="compact",
                                hint="0-1, applied between hidden layers",
                                persistent_hint=True,
                            )

                # === CNN ARCHITECTURE SECTION ===
                with html.Div(v_show=("inv_model_type === 'nn_cnn'",)):
                    v3.VDivider(classes="my-3")
                    html.Div("CNN Architecture", classes="text-subtitle-2 mb-2")
                    v3.VTextField(
                        v_model=("inv_cnn_conv_channels",),
                        label="Conv Channels",
                        density="compact",
                        hint="Comma-separated channel sizes (e.g., 32, 64, 128)",
                        persistent_hint=True,
                    )
                    with v3.VRow(dense=True):
                        with v3.VCol(cols=4):
                            v3.VTextField(
                                v_model=("inv_cnn_pool_size",),
                                label="Pool Size",
                                type="number",
                                density="compact",
                                hint="Adaptive pool output size",
                                persistent_hint=True,
                            )
                        with v3.VCol(cols=4):
                            v3.VTextField(
                                v_model=("inv_cnn_regressor_hidden",),
                                label="Regressor Hidden",
                                type="number",
                                density="compact",
                                hint="MLP hidden size after conv",
                                persistent_hint=True,
                            )
                        with v3.VCol(cols=4):
                            v3.VTextField(
                                v_model=("inv_cnn_dropout",),
                                label="Dropout",
                                type="number",
                                step="0.05",
                                density="compact",
                                hint="0-1",
                                persistent_hint=True,
                            )

                # === SIGNAL PROCESSING SECTION ===
                v3.VDivider(classes="my-3")
                html.Div("Signal Processing", classes="text-subtitle-2 mb-2")
                with v3.VRow(dense=True):
                    # Left side: options
                    with v3.VCol(cols=5):
                        v3.VCheckbox(
                            v_model=("inv_use_dynamic_compression",),
                            label="Dynamic Compression",
                            density="compact",
                            hide_details=True,
                            classes="mt-0",
                        )

                        with v3.VRow(
                            dense=True,
                            v_show=("inv_use_dynamic_compression",),
                            classes="ml-4",
                        ):
                            with v3.VCol(cols=6):
                                v3.VTextField(
                                    v_model=("inv_compression_threshold",),
                                    label="Threshold",
                                    type="number",
                                    step="0.05",
                                    density="compact",
                                )
                            with v3.VCol(cols=6):
                                v3.VTextField(
                                    v_model=("inv_compression_ratio",),
                                    label="Ratio",
                                    type="number",
                                    step="0.5",
                                    density="compact",
                                )

                        v3.VCheckbox(
                            v_model=("inv_use_normalization",),
                            label="Normalization",
                            density="compact",
                            hide_details=True,
                            classes="mt-0",
                        )

                        v3.VCheckbox(
                            v_model=("inv_use_noise",),
                            label="Noise",
                            density="compact",
                            hide_details=True,
                            classes="mt-0",
                        )

                        with v3.VRow(
                            dense=True,
                            v_show=("inv_use_noise",),
                            classes="ml-4",
                        ):
                            with v3.VCol(cols=12):
                                v3.VTextField(
                                    v_model=("inv_noise_level",),
                                    label="Level (% of peak)",
                                    type="number",
                                    step="0.01",
                                    min="0",
                                    max="1",
                                    density="compact",
                                    hint="0-1 (e.g., 0.05 = 5%)",
                                    persistent_hint=True,
                                )

                    # Right side: signal preview chart
                    with v3.VCol(cols=7):
                        with html.Div(v_show=("inv_selected_batch",)):
                            with v3.VRow(dense=True, align="center", justify="center"):
                                with v3.VCol(cols="auto"):
                                    v3.VBtn(
                                        icon="mdi-chevron-left",
                                        variant="text",
                                        density="compact",
                                        click=self._prev_sensor_signal,
                                    )
                                with v3.VCol(cols="auto"):
                                    html.Span(
                                        "{{ inv_compression_preview_label }}",
                                        classes="text-caption",
                                    )
                                with v3.VCol(cols="auto"):
                                    v3.VBtn(
                                        icon="mdi-chevron-right",
                                        variant="text",
                                        density="compact",
                                        click=self._next_sensor_signal,
                                    )

                            with v3.VSheet(
                                rounded=True,
                                classes="d-flex align-center justify-center",
                                style="min-height: 120px;",
                            ):
                                self.compression_preview_widget = mpl_widgets.Figure(
                                    figure=None
                                )
                                self.compression_preview_widget.update(
                                    plt.figure(figsize=(5, 2))
                                )

                # === OPTIONS SECTION ===
                v3.VDivider(classes="my-3")
                html.Div("Options", classes="text-subtitle-2 mb-2")
                with v3.VRow(dense=True, align="center"):
                    with v3.VCol(cols="auto"):
                        v3.VCheckbox(
                            v_model=("inv_early_stopping",),
                            label="Early Stopping",
                            density="compact",
                            hide_details=True,
                            classes="mt-0",
                        )
                    with v3.VCol(cols=3, v_show=("inv_early_stopping",)):
                        v3.VTextField(
                            v_model=("inv_early_stopping_patience",),
                            label="Patience",
                            type="number",
                            density="compact",
                            hint="Epochs to wait for improvement",
                            persistent_hint=True,
                        )
                with v3.VRow(dense=True, align="center", classes="mt-1"):
                    with v3.VCol(cols="auto"):
                        v3.VCheckbox(
                            v_model=("inv_kfold_validation",),
                            label="K-Fold Cross Validation",
                            density="compact",
                            hide_details=True,
                            classes="mt-0",
                        )
                    with v3.VCol(cols=2, v_show=("inv_kfold_validation",)):
                        v3.VTextField(
                            v_model=("inv_kfold_k",),
                            label="K",
                            type="number",
                            density="compact",
                            hint="Number of folds",
                            persistent_hint=True,
                        )

                # Train button
                v3.VBtn(
                    "Train Model",
                    color="primary",
                    block=True,
                    disabled=("inv_is_training || !inv_selected_batch",),
                    click=self._start_training,
                    classes="mt-4",
                )

            # Right side: Progress and loss chart
            with v3.VCol(cols=12, md=6):
                # Progress
                v3.VProgressLinear(
                    v_if=("inv_is_training",),
                    model_value=("inv_training_progress",),
                    color="primary",
                    height=8,
                )

                v3.VAlert(
                    v_if=("inv_training_message",),
                    text=("inv_training_message",),
                    type="info",
                    density="compact",
                    classes="mt-2",
                )

                # Loss chart
                with v3.VSheet(
                    rounded=True,
                    classes="mt-2 d-flex align-center justify-center",
                    style="min-height: 200px;",
                ):
                    self.loss_chart_widget = mpl_widgets.Figure(figure=None)
                    self.loss_chart_widget.update(plt.figure(figsize=(6, 3)))

    def _build_view_tab(self):
        """Build the View tab content for inspecting trained models."""
        with v3.VRow():
            # Left side: Model selection (narrow column)
            with v3.VCol(cols=12, md=2):
                # Batch selection
                with v3.VRow(align="center", dense=True):
                    with v3.VCol(cols=10):
                        v3.VSelect(
                            v_model=("inv_view_selected_batch",),
                            items=("inv_view_available_batches",),
                            label="Simulation Batch",
                            density="compact",
                            clearable=True,
                        )
                    with v3.VCol(cols=2):
                        v3.VBtn(
                            icon="mdi-refresh",
                            variant="text",
                            density="compact",
                            click=self._refresh_batch_list,
                        )

                # Model selection
                v3.VSelect(
                    v_model=("inv_view_selected_model",),
                    items=("inv_view_available_models",),
                    label="Trained Model",
                    density="compact",
                    clearable=True,
                    disabled=("!inv_view_selected_batch",),
                )

            # Middle: Model info
            with v3.VCol(cols=12, md=5):
                with v3.VCard(
                    v_if=("Object.keys(inv_view_model_info).length > 0",),
                    variant="outlined",
                ):
                    v3.VCardTitle("Model Information", classes="text-subtitle-1")
                    with v3.VCardText():
                        with v3.VList(density="compact"):
                            # Model name (if custom)
                            with v3.VListItem(v_if=("inv_view_model_info.model_name",)):
                                with v3.VListItemTitle():
                                    html.Span("Name: ")
                                    html.Strong("{{ inv_view_model_info.model_name }}")

                            # Model type
                            with v3.VListItem():
                                with v3.VListItemTitle():
                                    html.Span("Model Type: ")
                                    html.Strong("{{ inv_view_model_info.model_type }}")

                            # K-fold indicator
                            with v3.VListItem(v_if=("inv_view_model_info.is_kfold",)):
                                with v3.VListItemTitle():
                                    html.Span("K-Fold: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.k }}-fold cross validation"
                                    )

                            # Architecture
                            with v3.VListItem(
                                v_if=("inv_view_model_info.architecture",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Architecture: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.architecture }}"
                                    )

                            # MLP hidden layers (only for MLP architecture)
                            with v3.VListItem(
                                v_if=("inv_view_model_info.mlp_hidden_layers_str",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Hidden Layers: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.mlp_hidden_layers_str }}"
                                    )

                            # MLP dropout
                            with v3.VListItem(
                                v_if=(
                                    "inv_view_model_info.mlp_dropout !== undefined && inv_view_model_info.architecture === 'mlp'",
                                )
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Dropout: ")
                                    html.Strong("{{ inv_view_model_info.mlp_dropout }}")

                            # CNN conv channels (only for CNN architecture)
                            with v3.VListItem(
                                v_if=(
                                    "inv_view_model_info.cnn_conv_channels_str && inv_view_model_info.architecture === 'cnn'",
                                )
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Conv Channels: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.cnn_conv_channels_str }}"
                                    )

                            # CNN pool size
                            with v3.VListItem(
                                v_if=(
                                    "inv_view_model_info.cnn_pool_size !== undefined && inv_view_model_info.architecture === 'cnn'",
                                )
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Pool Size: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.cnn_pool_size }}"
                                    )

                            # CNN regressor hidden
                            with v3.VListItem(
                                v_if=(
                                    "inv_view_model_info.cnn_regressor_hidden !== undefined && inv_view_model_info.architecture === 'cnn'",
                                )
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Regressor Hidden: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.cnn_regressor_hidden }}"
                                    )

                            # CNN dropout
                            with v3.VListItem(
                                v_if=(
                                    "inv_view_model_info.cnn_dropout !== undefined && inv_view_model_info.architecture === 'cnn'",
                                )
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Dropout: ")
                                    html.Strong("{{ inv_view_model_info.cnn_dropout }}")

                            # Input/Output dimensions
                            with v3.VListItem(v_if=("inv_view_model_info.input_dim",)):
                                with v3.VListItemTitle():
                                    html.Span("Input Size: ")
                                    html.Strong("{{ inv_view_model_info.input_dim }}")

                            with v3.VListItem(v_if=("inv_view_model_info.output_dim",)):
                                with v3.VListItemTitle():
                                    html.Span("Output Size: ")
                                    html.Strong("{{ inv_view_model_info.output_dim }}")

                            # Parameters
                            with v3.VListItem(
                                v_if=("inv_view_model_info.total_parameters",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Total Parameters: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.total_parameters }}"
                                    )

                        # Training configuration section
                        v3.VDivider(classes="my-2")
                        html.Div(
                            "Training Configuration",
                            classes="text-caption text-medium-emphasis mb-1",
                        )
                        with v3.VList(density="compact"):
                            with v3.VListItem(
                                v_if=("inv_view_model_info.training_epochs",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Epochs: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.training_epochs }}"
                                    )

                            with v3.VListItem(v_if=("inv_view_model_info.batch_size",)):
                                with v3.VListItemTitle():
                                    html.Span("Batch Size: ")
                                    html.Strong("{{ inv_view_model_info.batch_size }}")

                            with v3.VListItem(
                                v_if=("inv_view_model_info.learning_rate",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Learning Rate: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.learning_rate }}"
                                    )

                            with v3.VListItem(
                                v_if=("inv_view_model_info.test_fraction",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Test Fraction: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.test_fraction }}"
                                    )

                        # K-Space configuration section
                        with html.Div(v_if=("inv_view_model_info.kspace_grid_size",)):
                            v3.VDivider(classes="my-2")
                            html.Div(
                                "K-Space Configuration",
                                classes="text-caption text-medium-emphasis mb-1",
                            )
                            with v3.VList(density="compact"):
                                with v3.VListItem():
                                    with v3.VListItemTitle():
                                        html.Span("Grid Size: ")
                                        html.Strong(
                                            "{{ inv_view_model_info.kspace_grid_size }}³"
                                        )

                                with v3.VListItem(
                                    v_if=("inv_view_model_info.kspace_total_coeffs",)
                                ):
                                    with v3.VListItemTitle():
                                        html.Span("Total Coefficients: ")
                                        html.Strong(
                                            "{{ inv_view_model_info.kspace_total_coeffs }}"
                                        )

                        # Preprocessing configuration section
                        v3.VDivider(classes="my-2")
                        html.Div(
                            "Preprocessing",
                            classes="text-caption text-medium-emphasis mb-1",
                        )
                        with v3.VList(density="compact"):
                            with v3.VListItem(
                                v_if=(
                                    "inv_view_model_info.trim_timesteps !== undefined",
                                )
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Trim Timesteps: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.trim_timesteps }}"
                                    )

                            with v3.VListItem(
                                v_if=("inv_view_model_info.downsample_factor",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Downsample Factor: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.downsample_factor }}"
                                    )

                            with v3.VListItem():
                                with v3.VListItemTitle():
                                    html.Span("Dynamic Compression: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.use_dynamic_compression ? 'Enabled' : 'Disabled' }}"
                                    )
                            with v3.VListItem(
                                v_if=("inv_view_model_info.use_dynamic_compression",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Threshold: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.compression_threshold }}"
                                    )
                            with v3.VListItem(
                                v_if=("inv_view_model_info.use_dynamic_compression",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Ratio: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.compression_ratio }}:1"
                                    )
                            with v3.VListItem():
                                with v3.VListItemTitle():
                                    html.Span("Normalization: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.use_normalization ? 'Enabled' : 'Disabled' }}"
                                    )

                        # Results section
                        v3.VDivider(classes="my-2")
                        html.Div(
                            "Results",
                            classes="text-caption text-medium-emphasis mb-1",
                        )
                        with v3.VList(density="compact"):
                            with v3.VListItem(v_if=("inv_view_model_info.num_epochs",)):
                                with v3.VListItemTitle():
                                    html.Span("Epochs Completed: ")
                                    html.Strong("{{ inv_view_model_info.num_epochs }}")

                            with v3.VListItem(
                                v_if=("inv_view_model_info.num_test_samples",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Test Samples: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.num_test_samples }}"
                                    )

                            # Final losses (show "Avg" prefix for K-fold)
                            with v3.VListItem(
                                v_if=("inv_view_model_info.final_train_loss",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span(
                                        "{{ inv_view_model_info.is_kfold ? 'Avg Train Loss: ' : 'Final Train Loss: ' }}"
                                    )
                                    html.Strong(
                                        "{{ typeof inv_view_model_info.final_train_loss === 'number' ? inv_view_model_info.final_train_loss.toExponential(4) : inv_view_model_info.final_train_loss }}"
                                    )

                            with v3.VListItem(
                                v_if=("inv_view_model_info.final_test_loss",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span(
                                        "{{ inv_view_model_info.is_kfold ? 'Avg Test Loss: ' : 'Final Test Loss: ' }}"
                                    )
                                    html.Strong(
                                        "{{ typeof inv_view_model_info.final_test_loss === 'number' ? inv_view_model_info.final_test_loss.toExponential(4) : inv_view_model_info.final_test_loss }}"
                                    )

                            # Training duration
                            with v3.VListItem(
                                v_if=("inv_view_model_info.training_duration",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Training Time: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.training_duration }}"
                                    )

                            # Batch name
                            with v3.VListItem(v_if=("inv_view_model_info.batch_name",)):
                                with v3.VListItemTitle():
                                    html.Span("Trained on Batch: ")
                                    html.Strong("{{ inv_view_model_info.batch_name }}")

                        # K-Fold per-fold details section
                        with html.Div(
                            v_if=(
                                "inv_view_model_info.is_kfold && inv_view_model_info.fold_details",
                            )
                        ):
                            v3.VDivider(classes="my-2")
                            html.Div(
                                "Per-Fold Results",
                                classes="text-caption text-medium-emphasis mb-1",
                            )

                            # Summary statistics
                            with v3.VList(density="compact"):
                                with v3.VListItem():
                                    with v3.VListItemTitle():
                                        html.Span("Test Loss Mean: ")
                                        html.Strong(
                                            "{{ inv_view_model_info.fold_test_mean?.toExponential(4) }}"
                                        )
                                        html.Span(
                                            " ± {{ inv_view_model_info.fold_test_std?.toExponential(4) }}",
                                            classes="text-medium-emphasis",
                                        )
                                with v3.VListItem():
                                    with v3.VListItemTitle():
                                        html.Span("Test Loss Range: ")
                                        html.Strong(
                                            "{{ inv_view_model_info.fold_test_min?.toExponential(4) }} - {{ inv_view_model_info.fold_test_max?.toExponential(4) }}"
                                        )

                            # Per-fold table
                            with v3.VTable(density="compact", classes="mt-2"):
                                with html.Thead():
                                    with html.Tr():
                                        html.Th("Fold", classes="text-left")
                                        html.Th("Train Loss", classes="text-right")
                                        html.Th("Test Loss", classes="text-right")
                                with html.Tbody():
                                    with html.Tr(
                                        v_for="(fold, index) in inv_view_model_info.fold_details",
                                        key="index",
                                    ):
                                        html.Td("{{ fold.fold }}")
                                        html.Td(
                                            "{{ fold.train_loss.toExponential(4) }}",
                                            classes="text-right",
                                        )
                                        html.Td(
                                            "{{ fold.test_loss.toExponential(4) }}",
                                            classes="text-right",
                                        )

            # Right side: Training history chart
            with v3.VCol(cols=12, md=5):
                html.Div("Training History", classes="text-subtitle-1 mb-2")
                with v3.VSheet(
                    rounded=True,
                    classes="d-flex align-center justify-center",
                    style="min-height: 250px;",
                ):
                    self.view_loss_chart_widget = mpl_widgets.Figure(figure=None)
                    self.view_loss_chart_widget.update(plt.figure(figsize=(8, 4)))

    def _build_test_tab(self):
        """Build the Test tab content."""
        # Top section: Model selection and test controls
        with v3.VRow():
            # Batch selection
            with v3.VCol(cols=12, md=3):
                with v3.VRow(align="center", dense=True):
                    with v3.VCol(cols=10):
                        v3.VSelect(
                            v_model=("inv_test_selected_batch",),
                            items=("inv_test_available_batches",),
                            label="Simulation Batch",
                            density="compact",
                            clearable=True,
                        )
                    with v3.VCol(cols=2):
                        v3.VBtn(
                            icon="mdi-refresh",
                            variant="text",
                            density="compact",
                            click=self._refresh_batch_list,
                        )

            # Model selection
            with v3.VCol(cols=12, md=2):
                v3.VSelect(
                    v_model=("inv_test_selected_model",),
                    items=("inv_test_available_models",),
                    label="Trained Model",
                    density="compact",
                    clearable=True,
                    disabled=("!inv_test_selected_batch",),
                )

            # Fold selection (only visible for K-fold models)
            with v3.VCol(cols=12, md=1, v_if=("inv_test_is_kfold",)):
                v3.VSelect(
                    v_model=("inv_test_selected_fold",),
                    items=("inv_test_available_folds",),
                    label="Fold",
                    density="compact",
                )

            # Test button and progress
            with v3.VCol(cols=12, md=3):
                v3.VBtn(
                    "Run Tests",
                    color="primary",
                    block=True,
                    disabled=(
                        "inv_is_testing || !inv_test_selected_model || !inv_test_selected_batch",
                    ),
                    click=self._start_testing,
                )

                v3.VProgressLinear(
                    v_if=("inv_is_testing",),
                    model_value=("inv_test_progress",),
                    color="primary",
                    height=8,
                    classes="mt-2",
                )

                v3.VAlert(
                    v_if=("inv_test_message",),
                    text=("inv_test_message",),
                    type="info",
                    density="compact",
                    classes="mt-2",
                )

            # Test simulation selection
            with v3.VCol(cols=12, md=3):
                v3.VSelect(
                    v_model=("inv_selected_test_sim",),
                    items=("inv_test_simulations",),
                    label="View Test Result",
                    density="compact",
                    clearable=True,
                    disabled=("inv_test_simulations.length === 0",),
                )

        # Bottom section: Comparison visualization
        with v3.VRow(classes="mt-4"):
            with v3.VCol(cols=12):
                self._build_comparison_card()

    def _build_comparison_card(self):
        """Build the ground truth vs prediction comparison card."""
        with v3.VCard():
            v3.VCardTitle("Ground Truth vs Prediction")
            with v3.VCardText():
                # Visualization controls toolbar
                with v3.VToolbar(density="compact", color="surface", classes="mb-2"):
                    with v3.VRow(dense=True, align="center", classes="px-2"):
                        with v3.VCol(cols="auto"):
                            v3.VSelect(
                                v_model=("inv_selected_colormap",),
                                items=("inv_colormap_options",),
                                label="Colormap",
                                density="compact",
                                hide_details=True,
                                style="min-width: 160px;",
                            )
                        with v3.VCol(cols="auto"):
                            v3.VSelect(
                                v_model=("inv_selected_opacity",),
                                items=("inv_opacity_presets",),
                                label="Opacity",
                                density="compact",
                                hide_details=True,
                                style="min-width: 140px;",
                            )
                        with v3.VCol(cols="auto"):
                            v3.VTextField(
                                v_model=("inv_clim_min",),
                                label="Min",
                                type="number",
                                step="0.1",
                                density="compact",
                                hide_details=True,
                                disabled=("inv_auto_clim",),
                                style="max-width: 100px;",
                            )
                        with v3.VCol(cols="auto"):
                            v3.VTextField(
                                v_model=("inv_clim_max",),
                                label="Max",
                                type="number",
                                step="0.1",
                                density="compact",
                                hide_details=True,
                                disabled=("inv_auto_clim",),
                                style="max-width: 100px;",
                            )
                        with v3.VCol(cols="auto"):
                            v3.VCheckbox(
                                v_model=("inv_auto_clim",),
                                label="Auto (from ground truth)",
                                density="compact",
                                hide_details=True,
                            )
                        v3.VSpacer()
                        with v3.VCol(cols="auto"):
                            v3.VBtn(
                                "Sync Views",
                                color="primary",
                                variant="outlined",
                                density="compact",
                                click=self.sync_views,
                            )

                # Side-by-side comparison
                with v3.VRow(style="height: 725px;"):
                    # Ground truth
                    with v3.VCol(cols=6, style="height: 100%;"):
                        html.Div("Ground Truth", classes="text-center text-subtitle-1")
                        with v3.VSheet(style="height: calc(100% - 30px);"):
                            view_gt = plotter_ui(
                                self.plotter_ground_truth,
                                server=self.server,
                                add_menu=False,
                            )
                            self.ctrl.view_update_ground_truth = view_gt.update

                    # Prediction
                    with v3.VCol(cols=6, style="height: 100%;"):
                        html.Div("Prediction", classes="text-center text-subtitle-1")
                        with v3.VSheet(style="height: calc(100% - 30px);"):
                            view_pred = plotter_ui(
                                self.plotter_prediction,
                                server=self.server,
                                add_menu=False,
                            )
                            self.ctrl.view_update_prediction = view_pred.update
