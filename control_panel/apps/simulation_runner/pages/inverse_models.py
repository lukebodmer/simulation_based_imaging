"""Inverse Models page for training and testing inverse models."""

import asyncio
import pickle
import time
from pathlib import Path

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
        self.state.inv_test_fraction = 0.1
        self.state.inv_epochs = 500
        self.state.inv_batch_size = 5
        self.state.inv_learning_rate = 0.0001
        self.state.inv_kspace_grid_size = 64  # K-space resolution (grid_size^3 voxels)

        # Training state
        self.state.inv_is_training = False
        self.state.inv_training_progress = 0
        self.state.inv_training_message = ""
        self.state.inv_training_log = []

        # Loss history for plotting (used in train and view tabs)
        self._train_losses = []
        self._test_losses = []
        self._loss_epochs = []
        self.loss_chart_widget = None
        self.view_loss_chart_widget = None

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

        if not inv_view_selected_model or not self.state.inv_view_selected_batch:
            self._loss_epochs = []
            self._train_losses = []
            self._test_losses = []
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
                    self._loss_epochs = history.get("epochs", [])
                    self._train_losses = history.get("train_loss", [])
                    self._test_losses = history.get("test_loss", [])
                    self._update_view_loss_chart()
                    logger.info(
                        f"View tab: Loaded training history with {len(self._loss_epochs)} epochs"
                    )

            except Exception as e:
                logger.error(f"Failed to load model info: {e}")

    def _extract_model_info(self, data: dict) -> dict:
        """Extract model architecture and training info from saved data."""
        model = data.get("model")
        model_type = data.get("model_type", "unknown")
        metrics = data.get("metrics", {})
        test_hashes = data.get("test_hashes", [])
        history = data.get("training_history", {})
        training_config = data.get("training_config", {})

        info = {
            "model_type": model_type,
            "num_test_samples": len(test_hashes),
            "final_train_loss": metrics.get("train_loss", "N/A"),
            "final_test_loss": metrics.get("test_loss", "N/A"),
            "num_epochs": len(history.get("epochs", [])),
        }

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

        # Add k-space config if available
        kspace_config = data.get("kspace_config", {})
        if kspace_config:
            grid_size = kspace_config.get("grid_size")
            if grid_size:
                info["kspace_grid_size"] = grid_size
                info["kspace_total_coeffs"] = f"{grid_size**3 * 2:,}"  # real + imag

        # Extract architecture info from the model object
        if model is not None and hasattr(model, "_model") and model._model is not None:
            nn_model = model._model
            info["architecture"] = (
                model.architecture if hasattr(model, "architecture") else "unknown"
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
            if hasattr(model, "_input_dim"):
                info["input_dim"] = model._input_dim
            if hasattr(model, "_output_dim"):
                info["output_dim"] = model._output_dim

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

        # Load ground truth k-space from model_output.pkl
        sim_dir = batch_dir / "simulations" / sim_hash
        output_file = sim_dir / "model_output.pkl"

        # Load prediction
        pred_file = batch_dir / "predictions" / f"{sim_hash}.pkl"

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

        if len(self._loss_epochs) > 0:
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
        plt.tight_layout()

        self.loss_chart_widget.update(fig)
        plt.close(fig)

    def _update_view_loss_chart(self):
        """Update the loss chart in the view tab."""
        if self.view_loss_chart_widget is None:
            return

        fig, ax = plt.subplots(figsize=(8, 4))

        if len(self._loss_epochs) > 0:
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

            # Train model
            self.state.inv_training_message = "Training model..."
            self.state.inv_training_progress = 20
            await asyncio.sleep(0)

            model_type = self.state.inv_model_type
            test_fraction = float(self.state.inv_test_fraction)

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

            # Train the model with timing
            self._training_start_time = time.time()

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

            self._training_duration_seconds = time.time() - self._training_start_time

            # Log training time
            duration_str = self._format_duration(self._training_duration_seconds)
            self._log(f"Training completed in {duration_str}")

            # Save the model
            self.state.inv_training_message = "Saving model..."
            self.state.inv_training_progress = 95
            await asyncio.sleep(0)

            model_name = f"{model_type}_{len(sample_ids)}samples"
            model_path = models_dir / f"{model_name}.pkl"

            await loop.run_in_executor(
                None, lambda: self._save_model(model, model_path, test_hashes, metrics)
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

    def _check_training_data(self, sims_dir: Path) -> bool:
        """Check if training data exists for simulations."""
        if not sims_dir.exists():
            return False

        for sim_dir in sims_dir.iterdir():
            if not sim_dir.is_dir():
                continue
            if (sim_dir / "model_input.pkl").exists() and (
                sim_dir / "model_output.pkl"
            ).exists():
                return True

        return False

    def _generate_training_data(self, batch_dir: Path):
        """Generate model_input.pkl and model_output.pkl for all simulations."""
        sims_dir = batch_dir / "simulations"

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
                input_data = self._process_sensor_data(sensor_file)
                with open(sim_dir / "model_input.pkl", "wb") as f:
                    pickle.dump(input_data, f)

                # Generate model output (k-space from config)
                output_data = self._process_config_to_kspace(config_file)
                with open(sim_dir / "model_output.pkl", "wb") as f:
                    pickle.dump(output_data, f)

            except Exception as e:
                logger.error(f"Failed to process {sim_dir.name}: {e}")

    def _process_sensor_data(
        self, sensor_file: Path, downsample_factor: int = 2
    ) -> np.ndarray:
        """Process sensor data for model input."""
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
            sensor_data = sensor_data[:, 51:]  # Skip first 51 timesteps
            sensor_data = sensor_data[:, ::downsample_factor]

        return sensor_data.flatten().astype(np.float32)

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
            model = NeuralNetworkModel(
                name=model_type,
                architecture=architecture,
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

    def _save_model(self, model, path: Path, test_hashes: list[str], metrics: dict):
        """Save the trained model with metadata and training history."""
        data = {
            "model": model,
            "model_type": self.state.inv_model_type,
            "test_hashes": test_hashes,
            "metrics": metrics,
            "training_config": {
                "epochs": int(self.state.inv_epochs),
                "batch_size": int(self.state.inv_batch_size),
                "learning_rate": float(self.state.inv_learning_rate),
                "test_fraction": float(self.state.inv_test_fraction),
                "training_duration_seconds": self._training_duration_seconds,
            },
            "kspace_config": {
                "grid_size": int(self.state.inv_kspace_grid_size),
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
        predictions_dir = batch_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load model
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            model = data.get("model")
            test_hashes = data.get("test_hashes", [])

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

                # Load input
                input_file = batch_dir / "simulations" / sim_hash / "model_input.pkl"
                if not input_file.exists():
                    continue

                with open(input_file, "rb") as f:
                    X_test = pickle.load(f)

                if hasattr(X_test, "get"):
                    X_test = X_test.get()

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
                with v3.VTabs(v_model=("inv_active_tab",), density="compact"):
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
                # Batch selection
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

                # Model configuration
                v3.VSelect(
                    v_model=("inv_model_type",),
                    items=("inv_model_type_items",),
                    label="Model Type",
                    density="compact",
                )

                # NN-specific settings
                with v3.VRow(
                    dense=True,
                    v_show=("inv_model_type.startsWith('nn')",),
                ):
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("inv_epochs",),
                            label="Epochs",
                            type="number",
                            density="compact",
                        )
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("inv_batch_size",),
                            label="Batch Size",
                            type="number",
                            density="compact",
                        )
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("inv_learning_rate",),
                            label="Learning Rate",
                            type="number",
                            step="0.0001",
                            density="compact",
                        )

                with v3.VRow(dense=True):
                    with v3.VCol(cols=6):
                        v3.VTextField(
                            v_model=("inv_test_fraction",),
                            label="Test Fraction",
                            type="number",
                            step="0.05",
                            density="compact",
                            hint="Fraction held for testing",
                            persistent_hint=True,
                        )
                    with v3.VCol(cols=6):
                        v3.VTextField(
                            v_model=("inv_kspace_grid_size",),
                            label="K-Space Grid Size",
                            type="number",
                            density="compact",
                            hint="Resolution (e.g., 64 = 64³ voxels)",
                            persistent_hint=True,
                        )

                # Train button
                v3.VBtn(
                    "Train Model",
                    color="primary",
                    block=True,
                    disabled=("inv_is_training || !inv_selected_batch",),
                    click=self._start_training,
                    classes="mt-3",
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
            # Left side: Model selection
            with v3.VCol(cols=12, md=4):
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
            with v3.VCol(cols=12, md=4):
                with v3.VCard(
                    v_if=("Object.keys(inv_view_model_info).length > 0",),
                    variant="outlined",
                ):
                    v3.VCardTitle("Model Information", classes="text-subtitle-1")
                    with v3.VCardText():
                        with v3.VList(density="compact"):
                            # Model type
                            with v3.VListItem():
                                with v3.VListItemTitle():
                                    html.Span("Model Type: ")
                                    html.Strong("{{ inv_view_model_info.model_type }}")

                            # Architecture
                            with v3.VListItem(
                                v_if=("inv_view_model_info.architecture",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Architecture: ")
                                    html.Strong(
                                        "{{ inv_view_model_info.architecture }}"
                                    )

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

                            # Final losses
                            with v3.VListItem(
                                v_if=("inv_view_model_info.final_train_loss",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Final Train Loss: ")
                                    html.Strong(
                                        "{{ typeof inv_view_model_info.final_train_loss === 'number' ? inv_view_model_info.final_train_loss.toExponential(4) : inv_view_model_info.final_train_loss }}"
                                    )

                            with v3.VListItem(
                                v_if=("inv_view_model_info.final_test_loss",)
                            ):
                                with v3.VListItemTitle():
                                    html.Span("Final Test Loss: ")
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

            # Right side: Training history chart
            with v3.VCol(cols=12, md=4):
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
            with v3.VCol(cols=12, md=3):
                v3.VSelect(
                    v_model=("inv_test_selected_model",),
                    items=("inv_test_available_models",),
                    label="Trained Model",
                    density="compact",
                    clearable=True,
                    disabled=("!inv_test_selected_batch",),
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
