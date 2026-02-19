#!/usr/bin/env python
"""Grid visualization of 3D simulation batch results.

Produces 5 images, each with 2 rows. Each row shows:
- Left: 3D wave speed visualization (inclusion geometry)
- Right: Sensor recordings matrix

Uses the first 10 simulations from the batch.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

# Nord color palette
NORD_COLORS = {
    "nord0": "#2e3440",
    "nord1": "#3b4252",
    "nord3": "#4c566a",
    "nord4": "#d8dee9",
    "nord6": "#eceff4",
    "nord8": "#88c0d0",
    "nord10": "#5e81ac",
    "nord11": "#bf616a",
    "nord12": "#d08770",
    "nord13": "#ebcb8b",
}


def create_nord_diverging_cmap() -> mcolors.LinearSegmentedColormap:
    """Create a diverging colormap using Nord colors (blue-dark-red)."""
    colors = [
        NORD_COLORS["nord10"],  # Dark blue (negative extreme)
        NORD_COLORS["nord8"],  # Cyan (negative mid)
        NORD_COLORS["nord1"],  # Dark gray (zero/center)
        NORD_COLORS["nord12"],  # Orange (positive mid)
        NORD_COLORS["nord11"],  # Red (positive extreme)
    ]
    return mcolors.LinearSegmentedColormap.from_list("nord_diverging", colors)


def apply_nord_style() -> None:
    """Apply Nord-themed matplotlib style settings."""
    plt.rcParams.update(
        {
            "figure.facecolor": NORD_COLORS["nord0"],
            "figure.edgecolor": NORD_COLORS["nord0"],
            "axes.facecolor": NORD_COLORS["nord1"],
            "axes.edgecolor": NORD_COLORS["nord3"],
            "axes.labelcolor": NORD_COLORS["nord4"],
            "axes.titlecolor": NORD_COLORS["nord6"],
            "axes.titleweight": "medium",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": NORD_COLORS["nord4"],
            "ytick.color": NORD_COLORS["nord4"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": NORD_COLORS["nord6"],
            "savefig.facecolor": NORD_COLORS["nord0"],
            "savefig.edgecolor": NORD_COLORS["nord0"],
        }
    )


def to_numpy(array):
    """Convert CuPy array to NumPy if needed."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def render_inclusion(
    mesh: dict,
    camera_azimuth: float = 30.0,
    camera_elevation: float = 30.0,
    camera_radius: float = 3.5,
    window_size: list[int] | None = None,
) -> np.ndarray:
    """Render the inclusion (wave speed) visualization.

    Args:
        mesh: Mesh data dictionary.
        camera_azimuth: Camera azimuth angle in degrees.
        camera_elevation: Camera elevation angle in degrees.
        camera_radius: Distance of camera from center.
        window_size: PyVista window size (width, height).

    Returns:
        Rendered image as numpy array.
    """
    if window_size is None:
        window_size = [600, 600]
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(NORD_COLORS["nord0"])

    cell_to_vertices = to_numpy(mesh["cell_to_vertices"])
    vertex_coordinates = to_numpy(mesh["vertex_coordinates"])
    speed = to_numpy(mesh["speed_per_cell"])

    num_cells = len(cell_to_vertices)

    # Build full tetrahedral grid
    cell_conn = np.hstack([np.full((num_cells, 1), 4), cell_to_vertices]).ravel()
    cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cell_conn, cell_types, vertex_coordinates)
    grid.cell_data["speed"] = speed

    # Find inclusion cells (cells with different wave speed)
    background_speed = np.median(speed)
    inclusion_mask = np.abs(speed - background_speed) > 0.01 * background_speed

    if np.any(inclusion_mask):
        # Render inclusion with solid color
        inclusion_cells = cell_to_vertices[inclusion_mask]
        num_inclusion_cells = len(inclusion_cells)

        inc_conn = np.hstack(
            [np.full((num_inclusion_cells, 1), 4), inclusion_cells]
        ).ravel()
        inc_types = np.full(num_inclusion_cells, pv.CellType.TETRA, dtype=np.uint8)
        inc_grid = pv.UnstructuredGrid(inc_conn, inc_types, vertex_coordinates)

        plotter.add_mesh(
            inc_grid,
            color=NORD_COLORS["nord13"],
            opacity=0.8,
            show_edges=False,
            show_scalar_bar=False,
        )

    # Compute camera position
    x = to_numpy(mesh["x"]).ravel()
    y = to_numpy(mesh["y"]).ravel()
    z = to_numpy(mesh["z"]).ravel()
    center = (
        (x.min() + x.max()) / 2,
        (y.min() + y.max()) / 2,
        (z.min() + z.max()) / 2,
    )

    az_rad = np.radians(camera_azimuth)
    el_rad = np.radians(camera_elevation)
    cam_x = center[0] + camera_radius * np.cos(el_rad) * np.cos(az_rad)
    cam_y = center[1] + camera_radius * np.cos(el_rad) * np.sin(az_rad)
    cam_z = center[2] + camera_radius * np.sin(el_rad)

    plotter.camera_position = [(cam_x, cam_y, cam_z), center, (0, 0, 1)]
    plotter.add_bounding_box(color="gray", line_width=1)

    plotter.render()
    image = plotter.screenshot(return_img=True)
    plotter.close()

    return image


def get_face_index(x: float, y: float, z: float, box_size: float, tol: float = 0.01) -> int:
    """Determine which face a sensor belongs to based on its coordinates.

    Args:
        x, y, z: Sensor coordinates.
        box_size: Size of the cubic domain.
        tol: Tolerance for boundary detection.

    Returns:
        Face index (0-5) or -1 if not on a face.
        Face ordering: 0=Z-bottom, 1=Z-top, 2=Y-front, 3=Y-back, 4=X-left, 5=X-right
    """
    if abs(z) < tol:
        return 0  # Z-bottom
    if abs(z - box_size) < tol:
        return 1  # Z-top
    if abs(y) < tol:
        return 2  # Y-front
    if abs(y - box_size) < tol:
        return 3  # Y-back
    if abs(x) < tol:
        return 4  # X-left
    if abs(x - box_size) < tol:
        return 5  # X-right
    return -1


def get_sensor_sort_key(x: float, y: float, z: float, face_idx: int) -> tuple:
    """Get sort key for a sensor within its face for consistent ordering.

    For each face, we want sensors ordered so that the "first" coordinate
    varies slowest and the "second" coordinate varies fastest, creating
    a consistent raster pattern across all faces.

    Face ordering uses (primary_coord, secondary_coord):
    - Face 0 (Z-bottom): (x, y)
    - Face 1 (Z-top): (x, y)
    - Face 2 (Y-front): (x, z)
    - Face 3 (Y-back): (x, z)
    - Face 4 (X-left): (y, z)
    - Face 5 (X-right): (y, z)
    """
    if face_idx in (0, 1):  # Z faces
        return (x, y)
    elif face_idx in (2, 3):  # Y faces
        return (x, z)
    else:  # X faces
        return (y, z)


def reorder_sensors_by_face(
    pressure: np.ndarray,
    locations: np.ndarray,
    box_size: float = 1.0,
) -> tuple[np.ndarray, list[int]]:
    """Reorder sensor data so sensors are grouped by face with consistent ordering.

    Args:
        pressure: Sensor pressure data, shape (num_sensors, num_timesteps).
        locations: Sensor coordinates, shape (num_sensors, 3).
        box_size: Size of the cubic domain.

    Returns:
        Tuple of (reordered_pressure, face_boundaries).
        face_boundaries is a list of row indices where each face starts.
    """
    num_sensors = pressure.shape[0]

    # Assign each sensor to a face and compute sort key
    sensor_info = []
    for i in range(num_sensors):
        x, y, z = locations[i]
        face_idx = get_face_index(x, y, z, box_size)
        sort_key = get_sensor_sort_key(x, y, z, face_idx)
        sensor_info.append((i, face_idx, sort_key))

    # Sort by face index first, then by sort key within face
    sensor_info.sort(key=lambda item: (item[1], item[2]))

    # Reorder pressure data
    sorted_indices = [item[0] for item in sensor_info]
    reordered_pressure = pressure[sorted_indices, :]

    # Find face boundaries
    face_boundaries = [0]
    current_face = sensor_info[0][1]
    for i, (_, face_idx, _) in enumerate(sensor_info):
        if face_idx != current_face:
            face_boundaries.append(i)
            current_face = face_idx

    return reordered_pressure, face_boundaries


def render_sensor_matrix(
    sensor_data: dict,
    ax=None,
    box_size: float = 1.0,
) -> np.ndarray | None:
    """Render sensor recordings as a matrix image.

    Args:
        sensor_data: Sensor data dictionary with 'pressure' and 'locations' keys.
        ax: Optional axes to render into. If None, creates standalone figure.
        box_size: Size of the cubic domain for face detection.

    Returns:
        Rendered image as numpy array (RGB), or None if ax was provided.
    """
    pressure = to_numpy(sensor_data["pressure"])
    locations = to_numpy(sensor_data.get("locations", None))

    # Reorder sensors by face if locations are available
    face_boundaries = []
    if locations is not None:
        pressure, face_boundaries = reorder_sensors_by_face(pressure, locations, box_size)

    # Fixed color limits for consistent visualization across all plots
    vmax = 0.3

    field_cmap = create_nord_diverging_cmap()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))
        apply_nord_style()

    extent = (0.0, float(pressure.shape[1]), 0.0, float(pressure.shape[0]))
    im = ax.imshow(
        pressure,
        aspect="auto",
        cmap=field_cmap,
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
    )

    # Draw face separator lines at actual face boundaries
    for boundary in face_boundaries[1:]:  # Skip first boundary (always 0)
        ax.axhline(y=boundary, color=NORD_COLORS["nord4"], linewidth=1.0, alpha=0.7)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_tick_params(color=NORD_COLORS["nord4"])
    cbar.outline.set_edgecolor(NORD_COLORS["nord3"])
    plt.setp(
        plt.getp(cbar.ax.axes, "yticklabels"),
        color=NORD_COLORS["nord4"],
    )
    cbar.set_label("Pressure", color=NORD_COLORS["nord4"])

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Sensor Index")
    ax.set_xlim(0, pressure.shape[1])

    if standalone:
        plt.tight_layout()
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return image

    return None


def load_simulation_data(sim_dir: Path) -> dict | None:
    """Load mesh and sensor data from a simulation directory.

    Args:
        sim_dir: Simulation output directory.

    Returns:
        Dictionary with 'mesh' and 'sensor_data', or None if data missing.
    """
    mesh_file = sim_dir / "mesh.pkl"
    sensor_file = sim_dir / "sensor_data.pkl"

    if not mesh_file.exists() or not sensor_file.exists():
        return None

    try:
        with open(mesh_file, "rb") as f:
            mesh = pickle.load(f)
        with open(sensor_file, "rb") as f:
            sensor_data = pickle.load(f)
        return {"mesh": mesh, "sensor_data": sensor_data}
    except Exception as e:
        print(f"Failed to load {sim_dir.name}: {e}")
        return None


def get_box_size_from_mesh(mesh: dict) -> float:
    """Extract box size from mesh data."""
    x = to_numpy(mesh["x"]).ravel()
    return float(x.max() - x.min())


def create_grid_image(
    simulations: list[dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Create a grid image with 2 rows of simulation visualizations.

    Args:
        simulations: List of 2 simulation data dictionaries.
        output_path: Output image path.
        dpi: Output image DPI.
    """
    apply_nord_style()

    # Create figure with 2 rows, each row has inclusion image + sensor plot
    fig = plt.figure(figsize=(14, 12))

    for row_idx, sim_data in enumerate(simulations):
        # Get box size from mesh
        box_size = get_box_size_from_mesh(sim_data["mesh"])

        # Left: inclusion visualization (rendered as image)
        ax_inc = fig.add_subplot(2, 2, row_idx * 2 + 1)
        inc_image = render_inclusion(sim_data["mesh"])
        ax_inc.imshow(inc_image)
        ax_inc.axis("off")

        # Right: sensor recordings (rendered directly into axes)
        ax_sensor = fig.add_subplot(2, 2, row_idx * 2 + 2)
        render_sensor_matrix(
            sim_data["sensor_data"],
            ax=ax_sensor,
            box_size=box_size,
        )

    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15, hspace=0.12
    )
    plt.savefig(output_path, dpi=dpi, facecolor=NORD_COLORS["nord0"])
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create grid visualizations of 3D simulation batch"
    )
    parser.add_argument(
        "batch_dir",
        type=str,
        help="Path to simulation batch directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for images (default: batch_dir)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="3d_batch_grid",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI",
    )
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir) if args.output_dir else batch_dir

    # Find simulation directories
    sims_dir = batch_dir / "simulations"
    if not sims_dir.exists():
        print(f"No simulations directory found in {batch_dir}")
        return

    sim_dirs = sorted(sims_dir.iterdir())
    sim_dirs = [d for d in sim_dirs if d.is_dir()]

    # Load first 10 simulations
    simulations = []
    for sim_dir in sim_dirs[:10]:
        data = load_simulation_data(sim_dir)
        if data is not None:
            simulations.append(data)
            print(f"Loaded {sim_dir.name}")

    if len(simulations) < 2:
        print(f"Need at least 2 simulations, found {len(simulations)}")
        return

    print(f"Loaded {len(simulations)} simulations")

    # Create 5 images with 2 simulations each
    num_images = min(5, len(simulations) // 2)
    for img_idx in range(num_images):
        start_idx = img_idx * 2
        sims_for_image = simulations[start_idx : start_idx + 2]

        output_path = output_dir / f"{args.prefix}_{img_idx + 1:02d}.png"
        create_grid_image(
            sims_for_image,
            output_path,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
