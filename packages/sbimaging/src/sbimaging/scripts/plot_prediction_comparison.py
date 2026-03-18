"""Generate a publication figure comparing ground truth vs predictions.

Creates a 4-row, 2-column figure showing ground truth (left) and
predictions (right) for multiple test samples from an inverse model.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv
import tomli

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbimaging.logging import get_logger

# Enable offscreen rendering
pv.OFF_SCREEN = True

# Default paths
DEFAULT_DATA_DIR = Path("/data/simulations")


def kspace_to_voxels(data: np.ndarray, grid_size: int = 32) -> np.ndarray:
    """Convert k-space coefficients to voxel density.

    Args:
        data: Flattened k-space data [real, imag] with shape (grid_size^3 * 2,).
        grid_size: Size of the cubic grid.

    Returns:
        3D voxel density array of shape (grid_size, grid_size, grid_size).
    """
    n_half = len(data) // 2
    real_part = data[:n_half].reshape((grid_size, grid_size, grid_size))
    imag_part = data[n_half:].reshape((grid_size, grid_size, grid_size))
    kspace = real_part + 1j * imag_part
    voxels = np.real(np.fft.ifftn(np.fft.ifftshift(kspace)))
    return voxels


def render_ground_truth(
    config_path: Path,
    window_size: list[int] | None = None,
    cube_color: str = "yellow",
    cube_opacity: float = 1.0,
) -> np.ndarray | None:
    """Render ground truth cubes from simulation config.

    Args:
        config_path: Path to the simulation config.toml file.
        window_size: Size of the render window [width, height].
        cube_color: Color for the inclusion cubes.
        cube_opacity: Opacity for the cubes.

    Returns:
        Rendered image as numpy array, or None on error.
    """
    if window_size is None:
        window_size = [400, 400]

    # Load config
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    mesh_config = config.get("mesh", {})
    cube_centers = mesh_config.get("cube_centers", [])
    cube_widths = mesh_config.get("cube_widths", [])
    box_size = mesh_config.get("box_size", 1.0)

    if not cube_centers or not cube_widths:
        return None

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    # Add each cube as a box mesh
    for center, width in zip(cube_centers, cube_widths):
        half_width = width / 2.0
        cube = pv.Box(
            bounds=[
                center[0] - half_width,
                center[0] + half_width,
                center[1] - half_width,
                center[1] + half_width,
                center[2] - half_width,
                center[2] + half_width,
            ]
        )
        plotter.add_mesh(cube, color=cube_color, opacity=cube_opacity)

    # Add domain bounding box for reference
    domain_box = pv.Box(bounds=[0, box_size, 0, box_size, 0, box_size])
    plotter.add_mesh(domain_box, style="wireframe", color="gray", line_width=1)

    # Show grid axes with smaller font
    plotter.show_grid(font_size=11)
    plotter.reset_camera()

    # Adjust camera: move left and down from the default diagonal view
    plotter.camera.azimuth = -10
    plotter.camera.elevation = -10
    plotter.camera.zoom(0.80)  # Zoom out to show full axes

    plotter.render()
    image = plotter.screenshot(return_img=True)
    plotter.close()

    return image


def render_volume(
    voxels: np.ndarray,
    clim: tuple[float, float],
    window_size: list[int] | None = None,
    opacity: str = "sigmoid",
    cmap: str = "viridis",
    camera_position: list | None = None,
    box_size: float = 1.0,
) -> np.ndarray | None:
    """Render a 3D volume and return the image as a numpy array.

    Matches the rendering style of the trame control panel.

    Args:
        voxels: 3D voxel density array.
        clim: Color limits (min, max).
        window_size: Size of the render window [width, height].
        opacity: Opacity preset (sigmoid, linear, geom, geom_r).
        cmap: Colormap name.
        camera_position: Optional camera position list.
        box_size: Size of the domain bounding box.

    Returns:
        Rendered image as numpy array, or None on error.
    """
    if window_size is None:
        window_size = [400, 400]

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    # Create PyVista ImageData grid (matching trame control panel)
    grid = pv.ImageData()
    grid.dimensions = voxels.shape
    grid.spacing = tuple(1.0 / s for s in voxels.shape)
    grid.origin = (0, 0, 0)
    grid["values"] = voxels.flatten(order="F")

    # Add volume with same settings as trame
    plotter.add_volume(
        grid,
        scalars="values",
        opacity=opacity,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=False,  # We'll handle this separately if needed
    )

    # Add domain bounding box for reference (matching ground truth)
    domain_box = pv.Box(bounds=[0, box_size, 0, box_size, 0, box_size])
    plotter.add_mesh(domain_box, style="wireframe", color="gray", line_width=1)

    # Show grid axes with smaller font (without scalar bar)
    plotter.show_grid(font_size=11)

    # Set camera position
    if camera_position is not None:
        plotter.camera_position = camera_position
    else:
        plotter.reset_camera()
        # Adjust camera: move left and down from the default diagonal view
        plotter.camera.azimuth = -10
        plotter.camera.elevation = -10
        plotter.camera.zoom(0.80)  # Zoom out to show full axes

    plotter.render()
    image = plotter.screenshot(return_img=True)
    plotter.close()

    return image


def load_model_metadata(model_path: Path) -> dict:
    """Load model metadata to get output configuration.

    Args:
        model_path: Path to the model pickle file.

    Returns:
        Dictionary with model metadata.
    """
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data


def create_comparison_figure(
    batch_name: str,
    model_name: str,
    output_path: Path,
    num_samples: int = 4,
    sample_indices: list[int] | None = None,
    grid_size: int = 32,
    figsize: tuple[float, float] = (8, 12),
    dpi: int = 300,
    cmap: str = "viridis",
    opacity: str = "sigmoid",
) -> None:
    """Create a figure comparing ground truth vs predictions.

    Args:
        batch_name: Name of the simulation batch.
        model_name: Name of the trained model.
        output_path: Path to save the output figure.
        num_samples: Number of samples to show (rows in the figure).
        sample_indices: Specific indices to use. If None, uses first num_samples.
        grid_size: Size of the voxel grid.
        figsize: Figure size in inches.
        dpi: Output resolution.
        cmap: Colormap for volume rendering.
        opacity: Opacity preset (sigmoid, linear, geom, geom_r).
    """
    logger = get_logger(__name__)

    # Set up paths
    batch_dir = DEFAULT_DATA_DIR / batch_name
    model_path = batch_dir / "inverse_models" / f"{model_name}.pkl"
    predictions_dir = batch_dir / "predictions" / model_name

    # Load model metadata
    logger.info(f"Loading model metadata from {model_path}")
    model_data = load_model_metadata(model_path)
    use_kspace = model_data.get("output_config", {}).get("use_kspace", False)
    logger.info(f"Model uses k-space: {use_kspace}")

    # Get list of prediction files
    pred_files = sorted(predictions_dir.glob("*.pkl"))
    logger.info(f"Found {len(pred_files)} predictions")

    if len(pred_files) == 0:
        logger.error("No predictions found")
        return

    # Select samples
    if sample_indices is not None:
        selected_files = [pred_files[i] for i in sample_indices if i < len(pred_files)]
    else:
        selected_files = pred_files[:num_samples]

    num_rows = len(selected_files)
    logger.info(f"Creating figure with {num_rows} samples")

    # Load prediction data (ground truth will be rendered from config)
    all_pred_voxels = []
    all_config_paths = []
    for pred_file in selected_files:
        sim_hash = pred_file.stem
        config_path = batch_dir / "simulations" / sim_hash / "config.toml"
        all_config_paths.append(config_path)
        with open(pred_file, "rb") as f:
            pred_data = pickle.load(f)
        # Prediction may be k-space
        if use_kspace:
            pred_voxels = kspace_to_voxels(pred_data, grid_size)
        else:
            pred_voxels = pred_data.reshape((grid_size, grid_size, grid_size))
        all_pred_voxels.append(pred_voxels)

    # Color limits for predictions: 0 to 2
    pred_clim = (0.0, 2.0)
    logger.info(f"Prediction color limits: {pred_clim}")

    # Create figure
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    # Render each sample
    for i in range(num_rows):
        sim_hash = selected_files[i].stem
        logger.info(f"Processing sample {i + 1}/{num_rows}: {sim_hash}")

        config_path = all_config_paths[i]
        pred_voxels = all_pred_voxels[i]

        # Render ground truth from config (shows actual cube locations)
        gt_image = render_ground_truth(config_path)

        # Render prediction with 0-2 range
        pred_image = render_volume(
            pred_voxels,
            clim=pred_clim,
            cmap=cmap,
            opacity=opacity,
        )

        # Add to figure
        if gt_image is not None:
            axes[i, 0].imshow(gt_image)
        axes[i, 0].axis("off")
        if pred_image is not None:
            axes[i, 1].imshow(pred_image)
        axes[i, 1].axis("off")

        # Add row label
        axes[i, 0].set_ylabel(
            f"Sample {i + 1}", fontsize=10, rotation=0, ha="right", va="center"
        )

    # Add column titles
    axes[0, 0].set_title("Ground Truth", fontsize=12, fontweight="light", fontfamily="serif")
    axes[0, 1].set_title("Prediction", fontsize=12, fontweight="light", fontfamily="serif")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Figure saved to {output_path}")


if __name__ == "__main__":
    from sbimaging import configure_logging

    parser = argparse.ArgumentParser(
        description="Generate ground truth vs prediction comparison figure"
    )
    parser.add_argument(
        "--batch",
        type=str,
        default="multi_cube_1000_p2",
        help="Simulation batch name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k_space_test_no_residual",
        help="Trained model name",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="prediction_comparison.png",
        help="Output file path",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to show",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        help="Specific sample indices to use (overrides --num-samples)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for volume rendering",
    )
    parser.add_argument(
        "--opacity",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "linear", "geom", "geom_r"],
        help="Opacity preset for volume rendering",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8, 9],
        help="Figure size in inches (width height)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution",
    )

    args = parser.parse_args()
    configure_logging()

    create_comparison_figure(
        batch_name=args.batch,
        model_name=args.model,
        output_path=Path(args.output),
        num_samples=args.num_samples,
        sample_indices=args.samples,
        cmap=args.cmap,
        opacity=args.opacity,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )
