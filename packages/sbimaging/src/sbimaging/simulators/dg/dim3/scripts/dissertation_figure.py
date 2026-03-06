#!/usr/bin/env python
"""Generate dissertation-quality figures for 3D DG simulations.

Produces publication-ready PNG figures with a light color scheme
suitable for LaTeX dissertations and academic papers.
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

from sbimaging.logging import get_logger

pv.OFF_SCREEN = True


# Light color scheme for dissertation/publication figures
LIGHT_COLORS = {
    "background": "#ffffff",      # White background
    "text": "#2c3e50",            # Dark blue-gray text
    "axis": "#7f8c8d",            # Gray for axes
    "grid": "#bdc3c7",            # Light gray for grid lines
    "accent": "#e74c3c",          # Red accent
    "inclusion": "#f39c12",       # Orange/gold for inclusion
}


def create_diverging_cmap() -> mcolors.LinearSegmentedColormap:
    """Create a diverging colormap suitable for print (blue-gray-red).

    Uses darker, more saturated colors that pop off a white background.
    """
    colors = [
        "#08306b",  # Very dark blue (negative extreme)
        "#2171b5",  # Medium blue (negative mid)
        "#969696",  # Medium gray (zero/center)
        "#cb181d",  # Medium red (positive mid)
        "#67000d",  # Very dark red (positive extreme)
    ]
    return mcolors.LinearSegmentedColormap.from_list("diverging_light", colors)


def apply_light_style() -> None:
    """Apply light-themed matplotlib style for dissertation figures."""
    plt.rcParams.update({
        "figure.facecolor": LIGHT_COLORS["background"],
        "figure.edgecolor": LIGHT_COLORS["background"],
        "axes.facecolor": LIGHT_COLORS["background"],
        "axes.edgecolor": LIGHT_COLORS["axis"],
        "axes.labelcolor": LIGHT_COLORS["text"],
        "axes.titlecolor": LIGHT_COLORS["text"],
        "axes.titleweight": "normal",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.color": LIGHT_COLORS["text"],
        "ytick.color": LIGHT_COLORS["text"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": LIGHT_COLORS["text"],
        "savefig.facecolor": LIGHT_COLORS["background"],
        "savefig.edgecolor": LIGHT_COLORS["background"],
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })


def to_numpy(array):
    """Convert CuPy array to NumPy if needed."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def get_face_index(x: float, y: float, z: float, box_size: float, origin: float = 0.0, tol: float = 0.01) -> int:
    """Determine which face a sensor belongs to based on its coordinates."""
    if abs(z - origin) < tol:
        return 0  # Z-bottom
    if abs(z - (origin + box_size)) < tol:
        return 1  # Z-top
    if abs(y - origin) < tol:
        return 2  # Y-front
    if abs(y - (origin + box_size)) < tol:
        return 3  # Y-back
    if abs(x - origin) < tol:
        return 4  # X-left
    if abs(x - (origin + box_size)) < tol:
        return 5  # X-right
    return -1


def get_sensor_sort_key(x: float, y: float, z: float, face_idx: int) -> tuple:
    """Get sort key for a sensor within its face for consistent ordering."""
    if face_idx in (0, 1):  # Z faces
        return (x, y)
    elif face_idx in (2, 3):  # Y faces
        return (x, z)
    else:  # X faces
        return (y, z)


def reorder_sensors_by_face(
    pressure: np.ndarray,
    locations: np.ndarray,
    box_size: float,
    origin: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """Reorder sensor data so sensors are grouped by face with consistent ordering."""
    num_sensors = pressure.shape[0]

    sensor_info = []
    for i in range(num_sensors):
        x, y, z = locations[i]
        face_idx = get_face_index(x, y, z, box_size, origin)
        sort_key = get_sensor_sort_key(x, y, z, face_idx)
        sensor_info.append((i, face_idx, sort_key))

    sensor_info.sort(key=lambda item: (item[1], item[2]))

    sorted_indices = [item[0] for item in sensor_info]
    reordered_pressure = pressure[sorted_indices, :]

    face_boundaries = [0]
    current_face = sensor_info[0][1]
    for i, (_, face_idx, _) in enumerate(sensor_info):
        if face_idx != current_face:
            face_boundaries.append(i)
            current_face = face_idx

    return reordered_pressure, face_boundaries


def load_simulation_data(sim_dir: Path) -> dict:
    """Load all simulation data from a directory."""
    logger = get_logger(__name__)

    result = {
        "mesh": None,
        "timestep_files": [],
        "sensor_data": None,
        "config": None,
    }

    mesh_file = sim_dir / "mesh.pkl"
    if mesh_file.exists():
        with open(mesh_file, "rb") as f:
            result["mesh"] = pickle.load(f)
        logger.info(f"Loaded mesh from {mesh_file}")

    data_dir = sim_dir / "data"
    if data_dir.exists():
        result["timestep_files"] = sorted(data_dir.glob("*.pkl"))
        logger.info(f"Found {len(result['timestep_files'])} timestep files")

    sensor_file = sim_dir / "sensor_data.pkl"
    if sensor_file.exists():
        with open(sensor_file, "rb") as f:
            result["sensor_data"] = pickle.load(f)
        logger.info(f"Loaded sensor data from {sensor_file}")

    return result


def compute_camera_position(
    center: tuple[float, float, float],
    radius: float,
    azimuth: float,
    elevation: float,
) -> tuple[tuple, tuple, tuple]:
    """Compute camera position from azimuth and elevation angles."""
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)

    x = center[0] + radius * np.cos(el_rad) * np.cos(az_rad)
    y = center[1] + radius * np.cos(el_rad) * np.sin(az_rad)
    z = center[2] + radius * np.sin(el_rad)

    return (x, y, z), center, (0, 0, 1)


def add_inclusion_mesh(
    plotter: pv.Plotter,
    mesh: dict,
    opacity: float = 0.4,
) -> None:
    """Add inclusion (wave speed) visualization as overlay."""
    cell_to_vertices = to_numpy(mesh["cell_to_vertices"])
    vertex_coordinates = to_numpy(mesh["vertex_coordinates"])
    speed = to_numpy(mesh["speed_per_cell"])

    background_speed = np.median(speed)
    inclusion_mask = np.abs(speed - background_speed) > 0.01 * background_speed

    if not np.any(inclusion_mask):
        return

    inclusion_cells = cell_to_vertices[inclusion_mask]
    num_inclusion_cells = len(inclusion_cells)

    cell_conn = np.hstack(
        [np.full((num_inclusion_cells, 1), 4), inclusion_cells]
    ).ravel()
    cell_types = np.full(num_inclusion_cells, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cell_conn, cell_types, vertex_coordinates)

    plotter.add_mesh(
        grid,
        color=LIGHT_COLORS["inclusion"],
        opacity=opacity,
        show_edges=False,
        show_scalar_bar=False,
    )


def render_3d_points(
    plotter: pv.Plotter,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pressure: np.ndarray,
    clim: tuple[float, float],
    cmap,
    point_size: float = 5.0,
    mesh: dict | None = None,
    show_inclusion: bool = True,
    inclusion_opacity: float = 0.4,
    sensor_locations: np.ndarray | None = None,
) -> None:
    """Render pressure field as point cloud."""
    points = np.column_stack((x, y, z))

    # Symmetric opacity - hide near-zero values
    opacity = [0.9, 0.7, 0.5, 0.3, 0, 0.3, 0.5, 0.7, 0.9]

    threshold = 0.01 * max(abs(clim[0]), abs(clim[1]))
    mask = np.abs(pressure) > threshold

    if np.any(mask):
        plotter.add_points(
            points[mask],
            scalars=pressure[mask],
            cmap=cmap,
            clim=clim,
            point_size=point_size,
            render_points_as_spheres=True,
            opacity=opacity,
            show_scalar_bar=False,
        )

    if show_inclusion and mesh is not None:
        add_inclusion_mesh(plotter, mesh, opacity=inclusion_opacity)

    # Add sensor locations
    if sensor_locations is not None and len(sensor_locations) > 0:
        plotter.add_points(
            sensor_locations,
            color=LIGHT_COLORS["text"],
            point_size=6,
            render_points_as_spheres=True,
        )


def build_tetrahedral_grid(mesh: dict, pressure: np.ndarray) -> pv.UnstructuredGrid:
    """Build a PyVista UnstructuredGrid from tetrahedral mesh data."""
    cell_to_vertices = to_numpy(mesh["cell_to_vertices"])
    vertex_coordinates = to_numpy(mesh["vertex_coordinates"])
    num_cells = len(cell_to_vertices)
    num_vertices = len(vertex_coordinates)

    x_dg = to_numpy(mesh["x"])
    y_dg = to_numpy(mesh["y"])
    z_dg = to_numpy(mesh["z"])

    np_nodes = x_dg.shape[0]
    pressure_2d = pressure.reshape((np_nodes, num_cells), order="F")

    vertex_pressure = np.zeros(num_vertices)
    vertex_counts = np.zeros(num_vertices)

    for cell_idx in range(num_cells):
        cell_verts = cell_to_vertices[cell_idx]
        cell_x = x_dg[:, cell_idx]
        cell_y = y_dg[:, cell_idx]
        cell_z = z_dg[:, cell_idx]
        cell_p = pressure_2d[:, cell_idx]

        for vert_idx in cell_verts:
            vx, vy, vz = vertex_coordinates[vert_idx]
            distances = (cell_x - vx) ** 2 + (cell_y - vy) ** 2 + (cell_z - vz) ** 2
            nearest_node = np.argmin(distances)
            vertex_pressure[vert_idx] += cell_p[nearest_node]
            vertex_counts[vert_idx] += 1

    vertex_counts[vertex_counts == 0] = 1
    vertex_pressure /= vertex_counts

    cell_conn = np.hstack([np.full((num_cells, 1), 4), cell_to_vertices]).ravel()
    cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cell_conn, cell_types, vertex_coordinates)
    grid.point_data["pressure"] = vertex_pressure

    return grid


def render_3d_isosurface(
    plotter: pv.Plotter,
    mesh: dict,
    pressure: np.ndarray,
    clim: tuple[float, float],
    cmap,
    isosurface_opacity: float = 0.7,
    show_inclusion: bool = True,
    inclusion_opacity: float = 0.4,
    sensor_locations: np.ndarray | None = None,
) -> None:
    """Render pressure field as isosurfaces."""
    max_val = max(abs(clim[0]), abs(clim[1]))
    isosurface_levels = [
        -0.5 * max_val,
        -0.2 * max_val,
        -0.08 * max_val,
        0.08 * max_val,
        0.2 * max_val,
        0.5 * max_val,
    ]

    grid = build_tetrahedral_grid(mesh, pressure)

    try:
        contours = grid.contour(isosurfaces=isosurface_levels, scalars="pressure")
        if contours.n_points > 0:
            plotter.add_mesh(
                contours,
                scalars="pressure",
                cmap=cmap,
                clim=clim,
                opacity=isosurface_opacity,
                show_scalar_bar=False,
                smooth_shading=True,
                ambient=0.3,
                diffuse=0.8,
                specular=0.2,
            )
    except Exception:
        pass

    plotter.enable_3_lights()

    if show_inclusion and mesh is not None:
        add_inclusion_mesh(plotter, mesh, opacity=inclusion_opacity)

    if sensor_locations is not None and len(sensor_locations) > 0:
        plotter.add_points(
            sensor_locations,
            color=LIGHT_COLORS["text"],
            point_size=6,
            render_points_as_spheres=True,
        )


def generate_sensor_grid(
    box_size: float,
    sensors_per_face: int,
    origin: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Generate sensor grid on domain boundary faces."""
    grid_n = int(np.sqrt(sensors_per_face))
    if grid_n * grid_n != sensors_per_face:
        grid_n = max(1, round(np.sqrt(sensors_per_face)))

    sensors = []
    margin = box_size * 0.2
    coords = np.linspace(origin + margin, origin + box_size - margin, grid_n)

    for xi in coords:
        for yi in coords:
            sensors.append((xi, yi, origin))
    for xi in coords:
        for yi in coords:
            sensors.append((xi, yi, origin + box_size))
    for xi in coords:
        for zi in coords:
            sensors.append((xi, origin, zi))
    for xi in coords:
        for zi in coords:
            sensors.append((xi, origin + box_size, zi))
    for yi in coords:
        for zi in coords:
            sensors.append((origin, yi, zi))
    for yi in coords:
        for zi in coords:
            sensors.append((origin + box_size, yi, zi))

    return sensors


def render_dissertation_figure(
    sim_dir: Path | str,
    output_file: Path | str,
    vmin: float = -0.5,
    vmax: float = 0.5,
    render_mode: str = "points",
    show_inclusion: bool = True,
    inclusion_opacity: float = 0.4,
    isosurface_opacity: float = 0.7,
    camera_radius: float = 2.5,
    camera_elevation: float = 35.0,
    camera_azimuth: float = 45.0,
    timestep_index: int | None = None,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = 300,
    point_size: float = 5.0,
) -> bool:
    """Render a dissertation-quality figure showing 3D simulation and sensor data.

    Args:
        sim_dir: Simulation output directory.
        output_file: Output image file path (PNG or PDF).
        vmin: Minimum value for 3D pressure colormap.
        vmax: Maximum value for 3D pressure colormap.
        render_mode: "points" or "isosurface".
        show_inclusion: Whether to show the inclusion overlay.
        inclusion_opacity: Opacity for the inclusion mesh.
        isosurface_opacity: Opacity for isosurface meshes.
        camera_radius: Distance of camera from center.
        camera_elevation: Camera elevation angle in degrees.
        camera_azimuth: Camera azimuth angle in degrees.
        timestep_index: Specific timestep to render. If None, uses middle timestep.
        figsize: Figure size in inches (width, height).
        dpi: Output resolution.
        point_size: Size of points for point cloud rendering.

    Returns:
        True if figure was generated successfully.
    """
    logger = get_logger(__name__)
    sim_dir = Path(sim_dir)
    output_file = Path(output_file)

    # Load simulation data
    data = load_simulation_data(sim_dir)

    if data["mesh"] is None:
        logger.error("No mesh data found")
        return False

    if not data["timestep_files"]:
        logger.error("No timestep files found")
        return False

    mesh = data["mesh"]
    timestep_files = data["timestep_files"]

    # Select timestep
    if timestep_index is None:
        timestep_index = len(timestep_files) // 2
    timestep_index = max(0, min(timestep_index, len(timestep_files) - 1))

    logger.info(f"Rendering timestep {timestep_index} of {len(timestep_files)}")

    # Load timestep data
    with open(timestep_files[timestep_index], "rb") as f:
        ts_data = pickle.load(f)

    pressure = to_numpy(ts_data["fields"]["p"]).ravel(order="F")

    # Extract mesh coordinates
    if isinstance(mesh, dict):
        x = to_numpy(mesh["x"]).ravel(order="F")
        y = to_numpy(mesh["y"]).ravel(order="F")
        z = to_numpy(mesh["z"]).ravel(order="F")
    else:
        x = to_numpy(mesh.x).ravel(order="F")
        y = to_numpy(mesh.y).ravel(order="F")
        z = to_numpy(mesh.z).ravel(order="F")

    center = (
        (x.min() + x.max()) / 2,
        (y.min() + y.max()) / 2,
        (z.min() + z.max()) / 2,
    )

    box_size = x.max() - x.min()
    origin = x.min()

    # Process sensor data
    sensor_data = data.get("sensor_data")
    sensor_matrix = None
    face_boundaries: list[int] = []
    sensor_locations = None

    if sensor_data is not None and "pressure" in sensor_data:
        sensor_matrix = to_numpy(sensor_data["pressure"])

    if sensor_data is not None and "locations" in sensor_data:
        sensor_locations = to_numpy(sensor_data["locations"])
        logger.info(f"Found {len(sensor_locations)} sensor locations")

        if sensor_matrix is not None:
            sensor_matrix, face_boundaries = reorder_sensors_by_face(
                sensor_matrix, sensor_locations, box_size, origin
            )
    elif sensor_matrix is not None:
        num_sensors = sensor_matrix.shape[0]
        sensors_per_face = num_sensors // 6
        if sensors_per_face > 0:
            sensor_locations = np.array(
                generate_sensor_grid(box_size, sensors_per_face, origin)
            )
            sensor_matrix, face_boundaries = reorder_sensors_by_face(
                sensor_matrix, sensor_locations, box_size, origin
            )

    # Apply light styling
    apply_light_style()
    field_cmap = create_diverging_cmap()
    plt.colormaps.register(field_cmap, name="diverging_light", force=True)

    # Create PyVista plotter
    panel_width = int(figsize[0] / 2 * dpi)
    panel_height = int(figsize[1] * dpi * 1.1)
    plotter = pv.Plotter(off_screen=True, window_size=[panel_width, panel_height])
    plotter.set_background(LIGHT_COLORS["background"])

    # Compute camera position
    camera_pos, focal_point, view_up = compute_camera_position(
        center=center,
        radius=camera_radius,
        azimuth=camera_azimuth,
        elevation=camera_elevation,
    )

    # Render 3D view
    if render_mode == "isosurface":
        render_3d_isosurface(
            plotter=plotter,
            mesh=mesh,
            pressure=pressure,
            clim=(vmin, vmax),
            cmap=field_cmap,
            isosurface_opacity=isosurface_opacity,
            show_inclusion=show_inclusion,
            inclusion_opacity=inclusion_opacity,
            sensor_locations=sensor_locations,
        )
    else:
        render_3d_points(
            plotter=plotter,
            x=x, y=y, z=z,
            pressure=pressure,
            clim=(vmin, vmax),
            cmap=field_cmap,
            point_size=point_size,
            mesh=mesh if show_inclusion else None,
            show_inclusion=show_inclusion,
            inclusion_opacity=inclusion_opacity,
            sensor_locations=sensor_locations,
        )

    plotter.camera_position = [camera_pos, focal_point, view_up]
    plotter.add_bounding_box(color=LIGHT_COLORS["text"], line_width=2)

    plotter.render()
    image_3d = plotter.screenshot(return_img=True)
    plotter.close()

    if image_3d is None:
        logger.error("Failed to render 3D frame")
        return False

    # Get simulation time
    time = ts_data.get("time", timestep_index * ts_data.get("dt", 1.0))

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.12)

    # Left panel: 3D simulation
    ax_3d = fig.add_subplot(gs[0])
    ax_3d.imshow(image_3d)
    ax_3d.axis("off")
    ax_3d.set_title(f"3D Pressure Field ($t = {time:.4f}$ s)", pad=8)

    # Right panel: Sensor data
    ax_sensor = fig.add_subplot(gs[1])

    if sensor_matrix is not None:
        sensor_idx = min(timestep_index, sensor_matrix.shape[1] - 1)
        current_sensor_data = sensor_matrix[:, :sensor_idx + 1]

        sensor_extent = (
            0.0,
            float(sensor_matrix.shape[1]),
            0.0,
            float(sensor_matrix.shape[0]),
        )

        sensor_vmax = 0.3
        im_sensor = ax_sensor.imshow(
            current_sensor_data,
            aspect="auto",
            cmap=field_cmap,
            origin="lower",
            vmin=-sensor_vmax,
            vmax=sensor_vmax,
            extent=sensor_extent,
        )

        # Draw face separator lines
        if face_boundaries:
            for boundary in face_boundaries[1:]:
                ax_sensor.axhline(
                    y=boundary, color=LIGHT_COLORS["axis"], linewidth=0.8, alpha=0.7
                )

        cbar = plt.colorbar(
            im_sensor,
            ax=ax_sensor,
            label="Pressure",
            fraction=0.03,
            pad=0.02,
            shrink=0.8,
        )
        cbar.ax.tick_params(labelsize=8)

        ax_sensor.set_xlabel("Time Step")
        ax_sensor.set_ylabel("Sensor Index")
        ax_sensor.set_xlim(0, sensor_matrix.shape[1])
    else:
        ax_sensor.text(
            0.5, 0.5,
            "No sensor data",
            ha="center", va="center",
            transform=ax_sensor.transAxes,
        )

    ax_sensor.set_title("Sensor Recordings", pad=8)

    plt.tight_layout()

    # Save figure
    fig.savefig(
        output_file,
        dpi=dpi,
        facecolor=LIGHT_COLORS["background"],
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)

    logger.info(f"Dissertation figure saved to {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate dissertation-quality figures for 3D DG simulations"
    )
    parser.add_argument(
        "sim_dir",
        type=str,
        help="Simulation output directory",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dissertation_figure.png",
        help="Output file (PNG or PDF)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-0.5,
        help="Minimum value for 3D pressure colormap",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.5,
        help="Maximum value for 3D pressure colormap",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["points", "isosurface"],
        default="points",
        help="Rendering mode",
    )
    parser.add_argument(
        "--no-inclusion",
        action="store_true",
        help="Hide the inclusion overlay",
    )
    parser.add_argument(
        "--inclusion-opacity",
        type=float,
        default=0.4,
        help="Opacity for inclusion overlay (0-1)",
    )
    parser.add_argument(
        "--isosurface-opacity",
        type=float,
        default=0.7,
        help="Opacity for isosurface meshes (0-1)",
    )
    parser.add_argument(
        "--camera-radius",
        type=float,
        default=2.5,
        help="Camera distance from center",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=35.0,
        help="Camera elevation angle in degrees",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=45.0,
        help="Camera azimuth angle in degrees",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="Specific timestep index to render (default: middle)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 5],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Point size for point cloud rendering",
    )

    args = parser.parse_args()

    from sbimaging import configure_logging
    configure_logging()

    success = render_dissertation_figure(
        sim_dir=args.sim_dir,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        render_mode=args.render_mode,
        show_inclusion=not args.no_inclusion,
        inclusion_opacity=args.inclusion_opacity,
        isosurface_opacity=args.isosurface_opacity,
        camera_radius=args.camera_radius,
        camera_elevation=args.camera_elevation,
        camera_azimuth=args.camera_azimuth,
        timestep_index=args.timestep,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        point_size=args.point_size,
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
