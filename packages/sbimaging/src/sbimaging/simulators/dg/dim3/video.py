"""Video generation for 3D DG simulations.

Creates videos showing 3D simulation with rotating camera view
and accumulating sensor data side by side.
"""

import pickle
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import toml

matplotlib.use("Agg")  # Non-interactive backend for rendering
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from sbimaging.logging import get_logger

# Enable offscreen rendering
pv.OFF_SCREEN = True


# Nord color palette for consistent theming
NORD_COLORS = {
    "nord0": "#2e3440",
    "nord1": "#3b4252",
    "nord2": "#434c5e",
    "nord3": "#4c566a",
    "nord4": "#d8dee9",
    "nord5": "#e5e9f0",
    "nord6": "#eceff4",
    "nord7": "#8fbcbb",
    "nord8": "#88c0d0",
    "nord9": "#81a1c1",
    "nord10": "#5e81ac",
    "nord11": "#bf616a",
    "nord12": "#d08770",
    "nord13": "#ebcb8b",
    "nord14": "#a3be8c",
    "nord15": "#b48ead",
}

# Publication color palette (light background, suitable for dissertations/papers)
PUBLICATION_COLORS = {
    "background": "#ffffff",
    "panel_bg": "#ffffff",
    "text": "#333333",
    "text_light": "#666666",
    "border": "#cccccc",
    "accent": "#2c3e50",
    "inclusion": "#4c00b0",  # Gold/yellow for inclusions (contrasts with blue-orange isosurfaces)
    "sensors": "#34495e",  # Dark gray for sensors
}


def create_nord_diverging_cmap() -> mcolors.LinearSegmentedColormap:
    """Create a diverging colormap using Nord colors (blue-dark-red).

    Uses a dark center (nord1) instead of white for better visibility
    on dark backgrounds.
    """
    colors = [
        NORD_COLORS["nord10"],  # Dark blue (negative extreme)
        NORD_COLORS["nord8"],  # Cyan (negative mid)
        NORD_COLORS["nord1"],  # Dark gray (zero/center)
        NORD_COLORS["nord12"],  # Orange (positive mid)
        NORD_COLORS["nord11"],  # Red (positive extreme)
    ]
    return mcolors.LinearSegmentedColormap.from_list("nord_diverging", colors)


def create_publication_diverging_cmap() -> mcolors.LinearSegmentedColormap:
    """Create a diverging colormap suitable for publications (blue-white-red).

    Uses white center for print-friendly output and perceptually balanced
    blue/red extremes that work well in grayscale.
    """
    colors = [
        "#2166ac",  # Dark blue (negative extreme)
        "#67a9cf",  # Light blue (negative mid)
        "#f7f7f7",  # Near-white (zero/center)
        "#ef8a62",  # Light red/salmon (positive mid)
        "#b2182b",  # Dark red (positive extreme)
    ]
    return mcolors.LinearSegmentedColormap.from_list("publication_diverging", colors)


def create_publication_3d_cmap() -> mcolors.LinearSegmentedColormap:
    """Create a diverging colormap for 3D isosurfaces on white backgrounds.

    Uses saturated colors throughout with a neutral gray center for
    visibility against white backgrounds.
    """
    colors = [
        "#08519c",  # Dark blue (negative extreme)
        "#3182bd",  # Medium blue (negative mid)
        "#969696",  # Medium gray (zero/center) - visible on white
        "#e6550d",  # Orange (positive mid)
        "#a63603",  # Dark orange/brown (positive extreme)
    ]
    return mcolors.LinearSegmentedColormap.from_list("publication_3d", colors)


def create_publication_sensor_cmap() -> mcolors.LinearSegmentedColormap:
    """Create a colormap for sensor recordings that matches 3D isosurface colors.

    Uses the same saturated blue and orange as the 3D colormap but with
    a white center for clean appearance on white backgrounds.
    """
    colors = [
        "#08519c",  # Dark blue (negative extreme)
        "#2171b5",  # Medium-dark blue
        "#6baed6",  # Light blue
        "#ffffff",  # White (center)
        "#fd8d3c",  # Light orange
        "#d94701",  # Medium-dark orange
        "#a63603",  # Dark orange/brown (positive extreme)
    ]
    return mcolors.LinearSegmentedColormap.from_list("publication_sensor", colors)


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


def apply_publication_style() -> None:
    """Apply publication-friendly matplotlib style settings.

    Uses white/light background with dark text, suitable for
    PhD dissertations and academic papers.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": PUBLICATION_COLORS["background"],
            "figure.edgecolor": PUBLICATION_COLORS["background"],
            "axes.facecolor": PUBLICATION_COLORS["panel_bg"],
            "axes.edgecolor": PUBLICATION_COLORS["border"],
            "axes.labelcolor": PUBLICATION_COLORS["text"],
            "axes.titlecolor": PUBLICATION_COLORS["text"],
            "axes.titleweight": "medium",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": PUBLICATION_COLORS["text"],
            "ytick.color": PUBLICATION_COLORS["text"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": PUBLICATION_COLORS["text"],
            "savefig.facecolor": PUBLICATION_COLORS["background"],
            "savefig.edgecolor": PUBLICATION_COLORS["background"],
            # Use serif font for publication quality
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavuserif",
        }
    )


def to_numpy(array):
    """Convert CuPy array to NumPy if needed."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def get_face_index(x: float, y: float, z: float, box_size: float, origin: float = 0.0, tol: float = 0.01) -> int:
    """Determine which face a sensor belongs to based on its coordinates.

    Args:
        x, y, z: Sensor coordinates.
        box_size: Size of the cubic domain.
        origin: Minimum coordinate value.
        tol: Tolerance for boundary detection.

    Returns:
        Face index (0-5) or -1 if not on a face.
        Face ordering: 0=Z-bottom, 1=Z-top, 2=Y-front, 3=Y-back, 4=X-left, 5=X-right
    """
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
    box_size: float,
    origin: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """Reorder sensor data so sensors are grouped by face with consistent ordering.

    Args:
        pressure: Sensor pressure data, shape (num_sensors, num_timesteps).
        locations: Sensor coordinates, shape (num_sensors, 3).
        box_size: Size of the cubic domain.
        origin: Minimum coordinate value.

    Returns:
        Tuple of (reordered_pressure, face_boundaries).
        face_boundaries is a list of row indices where each face starts.
    """
    num_sensors = pressure.shape[0]

    # Assign each sensor to a face and compute sort key
    sensor_info = []
    for i in range(num_sensors):
        x, y, z = locations[i]
        face_idx = get_face_index(x, y, z, box_size, origin)
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


def generate_sensor_grid(
    box_size: float,
    sensors_per_face: int,
    origin: float = 0.0,
    exclude_regions: list[tuple[tuple[float, float, float], float]] | None = None,
) -> list[tuple[float, float, float]]:
    """Generate sensor grid on domain boundary faces.

    Args:
        box_size: Size of cubic domain.
        sensors_per_face: Total number of sensors per face (must be a perfect square).
        origin: Minimum coordinate value (domain goes from origin to origin+box_size).
        exclude_regions: List of (center, radius) tuples for excluded regions.

    Returns:
        List of (x, y, z) sensor coordinates.
    """
    grid_n = int(np.sqrt(sensors_per_face))
    if grid_n * grid_n != sensors_per_face:
        grid_n = max(1, round(np.sqrt(sensors_per_face)))

    sensors = []
    margin = box_size * 0.2
    coords = np.linspace(origin + margin, origin + box_size - margin, grid_n)

    # Group sensors by face (not interleaved) for correct visualization
    # Face 0: Z-bottom (z=origin)
    for xi in coords:
        for yi in coords:
            sensors.append((xi, yi, origin))

    # Face 1: Z-top (z=origin+box_size)
    for xi in coords:
        for yi in coords:
            sensors.append((xi, yi, origin + box_size))

    # Face 2: Y-front (y=origin)
    for xi in coords:
        for zi in coords:
            sensors.append((xi, origin, zi))

    # Face 3: Y-back (y=origin+box_size)
    for xi in coords:
        for zi in coords:
            sensors.append((xi, origin + box_size, zi))

    # Face 4: X-left (x=origin)
    for yi in coords:
        for zi in coords:
            sensors.append((origin, yi, zi))

    # Face 5: X-right (x=origin+box_size)
    for yi in coords:
        for zi in coords:
            sensors.append((origin + box_size, yi, zi))

    # Filter out sensors in excluded regions (near sources)
    if exclude_regions:
        filtered = []
        for sx, sy, sz in sensors:
            excluded = False
            for (cx, cy, cz), r in exclude_regions:
                dist2 = (sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2
                if dist2 < r**2:
                    excluded = True
                    break
            if not excluded:
                filtered.append((sx, sy, sz))
        sensors = filtered

    return sensors


def load_simulation_data(sim_dir: Path) -> dict:
    """Load all simulation data from a directory.

    Args:
        sim_dir: Simulation output directory.

    Returns:
        Dictionary with mesh, timestep files, sensor data, energy data, and config.
    """
    logger = get_logger(__name__)

    result = {
        "mesh": None,
        "timestep_files": [],
        "sensor_data": None,
        "energy_data": None,
        "config": None,
    }

    # Load mesh
    mesh_file = sim_dir / "mesh.pkl"
    if mesh_file.exists():
        with open(mesh_file, "rb") as f:
            result["mesh"] = pickle.load(f)
        logger.info(f"Loaded mesh from {mesh_file}")

    # Find timestep files
    data_dir = sim_dir / "data"
    if data_dir.exists():
        result["timestep_files"] = sorted(data_dir.glob("*.pkl"))
        logger.info(f"Found {len(result['timestep_files'])} timestep files")

    # Load sensor data
    sensor_file = sim_dir / "sensor_data.pkl"
    if sensor_file.exists():
        with open(sensor_file, "rb") as f:
            result["sensor_data"] = pickle.load(f)
        logger.info(f"Loaded sensor data from {sensor_file}")

    # Load energy data
    energy_file = sim_dir / "energy_data.pkl"
    if energy_file.exists():
        with open(energy_file, "rb") as f:
            result["energy_data"] = pickle.load(f)
        logger.info(f"Loaded energy data from {energy_file}")

    # Load config
    config_file = sim_dir / "config.toml"
    if config_file.exists():
        result["config"] = toml.load(config_file)
        logger.info(f"Loaded config from {config_file}")

    return result


def compute_camera_position_from_angles(
    center: tuple[float, float, float],
    radius: float,
    azimuth: float,
    elevation: float,
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    """Compute camera position from azimuth and elevation angles.

    Args:
        center: Center point to orbit around.
        radius: Distance from center.
        azimuth: Azimuth angle in degrees.
        elevation: Elevation angle in degrees.

    Returns:
        Tuple of (camera_position, focal_point, view_up).
    """
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)

    x = center[0] + radius * np.cos(el_rad) * np.cos(az_rad)
    y = center[1] + radius * np.cos(el_rad) * np.sin(az_rad)
    z = center[2] + radius * np.sin(el_rad)

    camera_position = (x, y, z)
    focal_point = center
    view_up = (0, 0, 1)

    return camera_position, focal_point, view_up


def compute_camera_position(
    frame_idx: int,
    total_frames: int,
    center: tuple[float, float, float],
    radius: float,
    elevation: float = 30.0,
    num_orbits: float = 1.0,
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    """Compute camera position for orbiting view.

    Args:
        frame_idx: Current frame index.
        total_frames: Total number of frames.
        center: Center point to orbit around.
        radius: Distance from center.
        elevation: Camera elevation angle in degrees.
        num_orbits: Number of complete orbits over the video.

    Returns:
        Tuple of (camera_position, focal_point, view_up).
    """
    azimuth = (frame_idx / total_frames) * 360.0 * num_orbits
    return compute_camera_position_from_angles(center, radius, azimuth, elevation)


def generate_frame_schedule(
    num_sim_frames: int,
    pause_at_fraction: float = 0.33,
    pan_frames: int = 60,
    pan_arc_degrees: float = 90.0,
    elevation_start: float = 25.0,
    elevation_end: float = 45.0,
    num_orbits: float = 1.0,
    final_orbit_seconds: float = 0.0,
    fps: int = 30,
) -> list[dict]:
    """Generate frame schedule with pause-and-pan effect.

    Creates a schedule where:
    1. Simulation plays while camera orbits to pause_at_fraction
    2. Simulation pauses, camera pans in an arc
    3. Simulation resumes, camera continues orbiting
    4. Simulation freezes at last frame, camera orbits for final_orbit_seconds

    Args:
        num_sim_frames: Number of simulation timestep files.
        pause_at_fraction: Fraction of simulation at which to pause (0-1).
        pan_frames: Number of frames for the pan animation.
        pan_arc_degrees: Degrees of arc to pan during pause.
        elevation_start: Starting elevation for pan.
        elevation_end: Ending elevation after pan.
        num_orbits: Total camera orbits over simulation.
        final_orbit_seconds: Seconds of additional orbiting at last frame.
        fps: Frames per second (used for final_orbit_seconds calculation).

    Returns:
        List of frame dictionaries with keys:
            - sim_idx: Index into timestep files
            - azimuth: Camera azimuth angle
            - elevation: Camera elevation angle
    """
    schedule = []

    # Phase 1: Simulation plays up to pause point
    pause_sim_idx = int(num_sim_frames * pause_at_fraction)
    phase1_orbit_fraction = pause_at_fraction

    for i in range(pause_sim_idx):
        progress = i / num_sim_frames
        azimuth = progress * 360.0 * num_orbits
        schedule.append(
            {
                "sim_idx": i,
                "azimuth": azimuth,
                "elevation": elevation_start,
            }
        )

    # Phase 2: Simulation paused, camera pans
    pause_azimuth_start = phase1_orbit_fraction * 360.0 * num_orbits
    pause_azimuth_end = pause_azimuth_start + pan_arc_degrees

    for i in range(pan_frames):
        t = i / (pan_frames - 1) if pan_frames > 1 else 0
        # Smooth easing (ease-in-out)
        t_smooth = 0.5 - 0.5 * np.cos(np.pi * t)

        azimuth = pause_azimuth_start + t_smooth * pan_arc_degrees
        elevation = elevation_start + t_smooth * (elevation_end - elevation_start)

        schedule.append(
            {
                "sim_idx": pause_sim_idx,  # Frozen simulation
                "azimuth": azimuth,
                "elevation": elevation,
            }
        )

    # Phase 3: Simulation resumes
    # Adjust orbit to account for the pan arc we just did
    resume_azimuth = pause_azimuth_end
    remaining_sim_frames = num_sim_frames - pause_sim_idx

    for i in range(remaining_sim_frames):
        sim_idx = pause_sim_idx + i
        # Progress from pause point to end
        progress = i / remaining_sim_frames if remaining_sim_frames > 0 else 0
        remaining_orbit = (
            1.0 - pause_at_fraction
        ) * 360.0 * num_orbits - pan_arc_degrees
        azimuth = resume_azimuth + progress * remaining_orbit

        schedule.append(
            {
                "sim_idx": sim_idx,
                "azimuth": azimuth,
                "elevation": elevation_end,
            }
        )

    # Phase 4: Final pause at last frame while camera continues orbiting
    if final_orbit_seconds > 0:
        final_orbit_frames = int(final_orbit_seconds * fps)
        last_sim_idx = num_sim_frames - 1
        # Continue from where phase 3 ended
        final_start_azimuth = resume_azimuth + remaining_orbit
        # Continue at the same orbital speed and direction as phase 3
        if remaining_sim_frames > 0:
            degrees_per_frame = remaining_orbit / remaining_sim_frames
        else:
            degrees_per_frame = 360.0 * num_orbits / num_sim_frames

        for i in range(final_orbit_frames):
            azimuth = final_start_azimuth + (i + 1) * degrees_per_frame

            schedule.append(
                {
                    "sim_idx": last_sim_idx,
                    "azimuth": azimuth,
                    "elevation": elevation_end,
                }
            )

    return schedule


# Opacity presets for 3D visualization
OPACITY_SYMMETRIC = [0.9, 0.7, 0.5, 0.3, 0, 0.3, 0.5, 0.7, 0.9]
OPACITY_LINEAR = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def add_inclusion_mesh(
    plotter: pv.Plotter,
    mesh: dict,
    opacity: float = 0.3,
    color: str | None = None,
) -> None:
    """Add inclusion (wave speed) visualization as overlay.

    Args:
        plotter: PyVista plotter to add mesh to.
        mesh: Mesh data dictionary.
        opacity: Opacity for the inclusion mesh.
        color: Solid color for inclusion. Defaults to Nord frost (light cyan).
    """
    if color is None:
        color = NORD_COLORS["nord13"]  # Yellow

    cell_to_vertices = to_numpy(mesh["cell_to_vertices"])
    vertex_coordinates = to_numpy(mesh["vertex_coordinates"])
    speed = to_numpy(mesh["speed_per_cell"])

    # Find inclusion cells (cells with different wave speed)
    background_speed = np.median(speed)
    inclusion_mask = np.abs(speed - background_speed) > 0.01 * background_speed

    if not np.any(inclusion_mask):
        return

    # Extract inclusion cells
    inclusion_cells = cell_to_vertices[inclusion_mask]
    num_inclusion_cells = len(inclusion_cells)

    # Build tetrahedral grid for inclusion only
    cell_conn = np.hstack(
        [np.full((num_inclusion_cells, 1), 4), inclusion_cells]
    ).ravel()
    cell_types = np.full(num_inclusion_cells, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cell_conn, cell_types, vertex_coordinates)

    # Add as surface mesh with solid color and transparency
    plotter.add_mesh(
        grid,
        color=color,
        opacity=opacity,
        show_edges=False,
        show_scalar_bar=False,
    )


def render_3d_frame(
    plotter: pv.Plotter,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pressure: np.ndarray,
    camera_position: tuple,
    focal_point: tuple,
    view_up: tuple,
    clim: tuple[float, float],
    point_size: float = 5.0,
    cmap: str = "RdBu_r",
    opacity: str | list = "symmetric",
    mesh: dict | None = None,
    show_inclusion: bool = True,
    inclusion_opacity: float = 0.3,
    sensor_locations: np.ndarray | None = None,
    inclusion_color: str | None = None,
    sensor_color: str | None = None,
) -> np.ndarray | None:
    """Render a single 3D frame.

    Args:
        plotter: PyVista plotter.
        x, y, z: Node coordinates (flattened).
        pressure: Pressure values at nodes.
        camera_position: Camera position.
        focal_point: Camera focal point.
        view_up: Camera up vector.
        clim: Color limits (min, max).
        point_size: Size of points.
        cmap: Colormap name.
        opacity: Opacity setting ("symmetric", "linear", or a list of values).
        mesh: Mesh data dictionary (for inclusion overlay).
        show_inclusion: Whether to show the inclusion overlay.
        inclusion_opacity: Opacity for the inclusion mesh.
        sensor_locations: Optional (N, 3) array of sensor coordinates to display.
        inclusion_color: Color for inclusion mesh. Defaults to Nord yellow.
        sensor_color: Color for sensor points. Defaults to Nord light gray.

    Returns:
        Rendered image as numpy array.
    """
    plotter.clear()

    # Create point cloud
    points = np.column_stack((x, y, z))

    # Resolve opacity preset
    if opacity == "symmetric":
        opacity_values = OPACITY_SYMMETRIC
    elif opacity == "linear":
        opacity_values = OPACITY_LINEAR
    else:
        opacity_values = opacity

    # Filter out points with near-zero pressure to avoid rendering transparent dots
    # Threshold is 1% of the color limit range
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
            opacity=opacity_values,
            show_scalar_bar=False,
        )

    # Add inclusion overlay if mesh data provided
    if show_inclusion and mesh is not None:
        add_inclusion_mesh(plotter, mesh, opacity=inclusion_opacity, color=inclusion_color)

    # Add sensor locations
    if sensor_locations is not None and len(sensor_locations) > 0:
        sensor_pt_color = sensor_color if sensor_color else NORD_COLORS["nord4"]
        plotter.add_points(
            sensor_locations,
            color=sensor_pt_color,
            point_size=8,
            render_points_as_spheres=True,
        )

    # Set camera
    plotter.camera_position = [camera_position, focal_point, view_up]

    # Add bounding box for reference
    plotter.add_bounding_box(color="gray", line_width=1)  # type: ignore[call-arg]

    # Render and capture
    plotter.render()
    image = plotter.screenshot(return_img=True)

    return image


def build_tetrahedral_grid(
    mesh: dict,
    pressure: np.ndarray,
) -> pv.UnstructuredGrid:
    """Build a PyVista UnstructuredGrid from tetrahedral mesh data.

    The DG method stores field values at nodal points within each cell, not at
    mesh vertices. This function interpolates from DG nodes to mesh vertices
    by averaging values from cells sharing each vertex.

    Args:
        mesh: Mesh data dictionary with cell_to_vertices, vertex_coordinates,
              and DG node coordinates (x, y, z).
        pressure: Pressure values at DG nodes (flattened, Fortran order).

    Returns:
        PyVista UnstructuredGrid with pressure interpolated to vertices.
    """
    cell_to_vertices = to_numpy(mesh["cell_to_vertices"])
    vertex_coordinates = to_numpy(mesh["vertex_coordinates"])
    num_cells = len(cell_to_vertices)
    num_vertices = len(vertex_coordinates)

    # Get DG node coordinates
    x_dg = to_numpy(mesh["x"])  # Shape: (Np, num_cells)
    y_dg = to_numpy(mesh["y"])
    z_dg = to_numpy(mesh["z"])

    # Reshape pressure to match DG node layout (Np, num_cells)
    np_nodes = x_dg.shape[0]  # Number of DG nodes per cell
    pressure_2d = pressure.reshape((np_nodes, num_cells), order="F")

    # Interpolate DG node values to mesh vertices
    # For each vertex, find cells that contain it and average the nearest DG node values
    vertex_pressure = np.zeros(num_vertices)
    vertex_counts = np.zeros(num_vertices)

    for cell_idx in range(num_cells):
        cell_verts = cell_to_vertices[cell_idx]  # 4 vertex indices
        cell_x = x_dg[:, cell_idx]
        cell_y = y_dg[:, cell_idx]
        cell_z = z_dg[:, cell_idx]
        cell_p = pressure_2d[:, cell_idx]

        for vert_idx in cell_verts:
            # Find the DG node closest to this vertex
            vx, vy, vz = vertex_coordinates[vert_idx]
            distances = (cell_x - vx) ** 2 + (cell_y - vy) ** 2 + (cell_z - vz) ** 2
            nearest_node = np.argmin(distances)

            vertex_pressure[vert_idx] += cell_p[nearest_node]
            vertex_counts[vert_idx] += 1

    # Average contributions from multiple cells
    vertex_counts[vertex_counts == 0] = 1  # Avoid division by zero
    vertex_pressure /= vertex_counts

    # Build cell connectivity array for PyVista
    cell_conn = np.hstack([np.full((num_cells, 1), 4), cell_to_vertices]).ravel()
    cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cell_conn, cell_types, vertex_coordinates)
    grid.point_data["pressure"] = vertex_pressure

    return grid


def render_3d_frame_isosurface(
    plotter: pv.Plotter,
    mesh: dict,
    pressure: np.ndarray,
    camera_position: tuple,
    focal_point: tuple,
    view_up: tuple,
    isosurface_levels: list[float] | None = None,
    clim: tuple[float, float] = (-0.5, 0.5),
    cmap: str = "RdBu_r",
    isosurface_opacity: float = 0.7,
    show_inclusion: bool = True,
    inclusion_opacity: float = 0.3,
    sensor_locations: np.ndarray | None = None,
    inclusion_color: str | None = None,
    sensor_color: str | None = None,
    publication_style: bool = False,
) -> np.ndarray | None:
    """Render a single 3D frame using isosurfaces.

    Args:
        plotter: PyVista plotter.
        mesh: Mesh data dictionary with cell_to_vertices and vertex_coordinates.
        pressure: Pressure values at vertices.
        camera_position: Camera position.
        focal_point: Camera focal point.
        view_up: Camera up vector.
        isosurface_levels: Pressure values for isosurfaces. Defaults to symmetric levels.
        clim: Color limits (min, max) for coloring isosurfaces.
        cmap: Colormap name.
        isosurface_opacity: Opacity for isosurface meshes.
        show_inclusion: Whether to show the inclusion overlay.
        inclusion_opacity: Opacity for the inclusion mesh.
        sensor_locations: Optional (N, 3) array of sensor coordinates to display.
        inclusion_color: Color for inclusion mesh. Defaults to Nord yellow.
        sensor_color: Color for sensor points. Defaults to Nord light gray.
        publication_style: Use publication-friendly isosurface levels (skip near-zero).

    Returns:
        Rendered image as numpy array.
    """
    plotter.clear()

    # Default isosurface levels: symmetric positive and negative
    if isosurface_levels is None:
        max_val = max(abs(clim[0]), abs(clim[1]))
        if publication_style:
            # Skip near-zero levels which would be near-white on white background
            isosurface_levels = [
                -0.5 * max_val,
                -0.25 * max_val,
                0.25 * max_val,
                0.5 * max_val,
            ]
        else:
            isosurface_levels = [
                -0.5 * max_val,
                -0.2 * max_val,
                -0.08 * max_val,
                0.08 * max_val,
                0.2 * max_val,
                0.5 * max_val,
            ]

    # Build tetrahedral grid with pressure data
    grid = build_tetrahedral_grid(mesh, pressure)

    # Extract isosurfaces
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
                ambient=1.0,
                diffuse=0.9,
                specular=0.1,
            )
    except Exception:
        # No isosurfaces found (pressure values don't cross the levels)
        pass

    # Add brighter lighting for isosurfaces
    plotter.enable_3_lights()

    # Add inclusion overlay
    if show_inclusion and mesh is not None:
        add_inclusion_mesh(plotter, mesh, opacity=inclusion_opacity, color=inclusion_color)

    # Add sensor locations
    if sensor_locations is not None and len(sensor_locations) > 0:
        sensor_pt_color = sensor_color if sensor_color else NORD_COLORS["nord4"]
        plotter.add_points(
            sensor_locations,
            color=sensor_pt_color,
            point_size=8,
            render_points_as_spheres=True,
        )

    # Set camera
    plotter.camera_position = [camera_position, focal_point, view_up]

    # Add bounding box for reference
    plotter.add_bounding_box(color="gray", line_width=1)  # type: ignore[call-arg]

    # Render and capture
    plotter.render()
    image = plotter.screenshot(return_img=True)

    return image


def create_video_with_sensors_3d(
    sim_dir: Path | str,
    output_file: Path | str,
    vmin: float = -1.0,
    vmax: float = 1.0,
    fps: int = 30,
    dpi: int = 240,
    figsize: tuple[float, float] = (16, 6),
    point_size: float = 5.0,
    camera_radius: float = 2.5,
    camera_elevation: float = 25.0,
    num_orbits: float = 1 / 3,
    sensors_per_face: int | None = None,
    use_nord_style: bool = True,
    window_size: list[int] | None = None,
    show_inclusion: bool = True,
    inclusion_opacity: float = 0.3,
    pause_and_pan: bool = True,
    pause_at_fraction: float = 0.33,
    pan_frames: int = 60,
    pan_arc_degrees: float = 90.0,
    pan_elevation_end: float = 45.0,
    final_orbit_seconds: float = 4.0,
    render_mode: str = "points",
    isosurface_levels: list[float] | None = None,
    isosurface_opacity: float = 0.7,
) -> bool:
    """Create video with 3D simulation and sensor data side by side.

    Args:
        sim_dir: Simulation output directory.
        output_file: Output video file path.
        vmin: Minimum value for pressure colormap. Defaults to -0.5.
        vmax: Maximum value for pressure colormap. Defaults to 0.5.
        fps: Frames per second.
        dpi: Image resolution.
        figsize: Figure size (width, height) in inches.
        point_size: Size of 3D points.
        camera_radius: Distance of camera from center.
        camera_elevation: Camera elevation angle in degrees.
        num_orbits: Number of camera orbits around the scene.
        sensors_per_face: Number of sensors per face for separator lines.
        use_nord_style: Apply Nord color theme.
        window_size: PyVista window size [width, height]. Defaults to [800, 600].
        show_inclusion: Whether to show the inclusion overlay.
        inclusion_opacity: Opacity for the inclusion mesh.
        pause_and_pan: Enable pause-and-pan effect at pause_at_fraction.
        pause_at_fraction: Fraction of simulation at which to pause (0-1).
        pan_frames: Number of frames for the pan animation during pause.
        pan_arc_degrees: Degrees of arc to pan during pause.
        pan_elevation_end: Camera elevation after pan completes.
        final_orbit_seconds: Seconds of orbiting at last simulation frame.
        render_mode: Rendering mode - "points" for point cloud, "isosurface" for isosurfaces.
        isosurface_levels: Pressure values for isosurfaces (only used if render_mode="isosurface").
        isosurface_opacity: Opacity for isosurface meshes.

    Returns:
        True if video was created successfully.
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
    sensor_data = data["sensor_data"]

    # Extract mesh coordinates (mesh can be a dict or MeshGeometry object)
    if isinstance(mesh, dict):
        x = to_numpy(mesh["x"]).ravel(order="F")
        y = to_numpy(mesh["y"]).ravel(order="F")
        z = to_numpy(mesh["z"]).ravel(order="F")
    else:
        x = to_numpy(mesh.x).ravel(order="F")
        y = to_numpy(mesh.y).ravel(order="F")
        z = to_numpy(mesh.z).ravel(order="F")

    # Compute domain center and size
    center = (
        (x.min() + x.max()) / 2,
        (y.min() + y.max()) / 2,
        (z.min() + z.max()) / 2,
    )

    logger.info(f"Color scale: [{vmin:.4g}, {vmax:.4g}]")

    # Compute domain bounds
    box_size = x.max() - x.min()  # Assume cubic domain
    origin = x.min()

    # Get sensor data matrix and locations
    sensor_matrix = None
    face_boundaries: list[int] = []

    if sensor_data is not None and "pressure" in sensor_data:
        sensor_matrix = to_numpy(sensor_data["pressure"])

    # Get sensor locations if available, or compute from config/sensors_per_face
    config = data.get("config")

    # Try to get sensors_per_face from config if not provided
    if sensors_per_face is None and config is not None:
        receivers = config.get("receivers", {})
        sensors_per_face = receivers.get("sensors_per_face")

    if sensor_data is not None and "locations" in sensor_data:
        sensor_locations = to_numpy(sensor_data["locations"])
        logger.info(f"Found {len(sensor_locations)} sensor locations in data")

        # Reorder sensor matrix by face if we have locations
        if sensor_matrix is not None:
            sensor_matrix, face_boundaries = reorder_sensors_by_face(
                sensor_matrix, sensor_locations, box_size, origin
            )
            logger.info(f"Reordered sensors by face, boundaries: {face_boundaries}")
    elif sensors_per_face is not None:
        # Compute sensor locations from mesh bounds and sensors_per_face
        # Get source exclusion regions from config
        exclude_regions = None
        if config is not None and "sources" in config:
            src = config["sources"]
            centers = src.get("centers", [])
            radii = src.get("radii", [])
            if centers and radii:
                exclude_regions = [
                    (tuple(c), r) for c, r in zip(centers, radii, strict=False)
                ]

        sensor_locations = np.array(
            generate_sensor_grid(box_size, sensors_per_face, origin, exclude_regions)
        )
        logger.info(f"Generated {len(sensor_locations)} sensor locations from grid")

        # Reorder sensor matrix by face if we have locations
        if sensor_matrix is not None:
            sensor_matrix, face_boundaries = reorder_sensors_by_face(
                sensor_matrix, sensor_locations, box_size, origin
            )
            logger.info(f"Reordered sensors by face, boundaries: {face_boundaries}")
    else:
        sensor_locations = None

    # Apply styling
    if use_nord_style:
        apply_nord_style()
        field_cmap = create_nord_diverging_cmap()
        # Register the Nord colormap with matplotlib so PyVista can use it
        plt.colormaps.register(field_cmap, name="nord_diverging", force=True)
        cmap_name = "nord_diverging"
    else:
        field_cmap = "RdBu_r"
        cmap_name = "RdBu_r"

    # Create PyVista plotter
    # Window size should match the aspect ratio of half the figure (since we have 2 panels)
    # Use a slightly taller height to match the sensor plot area (which has axes/labels)
    if window_size is None:
        panel_width = int(figsize[0] / 2 * dpi)
        panel_height = int(figsize[1] * dpi * 1.1)
        window_size = [panel_width, panel_height]
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(NORD_COLORS["nord0"] if use_nord_style else "white")  # type: ignore[arg-type]

    # Generate frame schedule
    if pause_and_pan:
        frame_schedule = generate_frame_schedule(
            num_sim_frames=len(timestep_files),
            pause_at_fraction=pause_at_fraction,
            pan_frames=pan_frames,
            pan_arc_degrees=pan_arc_degrees,
            elevation_start=camera_elevation,
            elevation_end=pan_elevation_end,
            num_orbits=num_orbits,
            final_orbit_seconds=final_orbit_seconds,
            fps=fps,
        )
        logger.info(
            f"Generated {len(frame_schedule)} frames "
            f"(pause at {pause_at_fraction:.0%}, {pan_frames} pan frames)"
        )
    else:
        # Simple schedule: one video frame per simulation frame
        frame_schedule = []
        for i in range(len(timestep_files)):
            progress = i / len(timestep_files)
            azimuth = progress * 360.0 * num_orbits
            frame_schedule.append(
                {
                    "sim_idx": i,
                    "azimuth": azimuth,
                    "elevation": camera_elevation,
                }
            )
        logger.info(f"Rendering {len(frame_schedule)} frames...")

    # Pre-load timestep data for efficiency
    timestep_cache: dict[int, dict] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for frame_idx, frame_info in enumerate(frame_schedule):
            sim_idx = frame_info["sim_idx"]
            azimuth = frame_info["azimuth"]
            elevation = frame_info["elevation"]

            # Load timestep data (with caching for pause frames)
            if sim_idx not in timestep_cache:
                with open(timestep_files[sim_idx], "rb") as f:
                    timestep_cache[sim_idx] = pickle.load(f)
                # Keep cache small - only keep current and previous
                if len(timestep_cache) > 2:
                    oldest = min(k for k in timestep_cache if k != sim_idx)
                    del timestep_cache[oldest]

            ts_data = timestep_cache[sim_idx]
            pressure = to_numpy(ts_data["fields"]["p"]).ravel(order="F")
            time = ts_data.get("time", sim_idx * ts_data.get("dt", 1.0))

            # Compute camera position from schedule
            camera_pos, focal_point, view_up = compute_camera_position_from_angles(
                center=center,
                radius=camera_radius,
                azimuth=azimuth,
                elevation=elevation,
            )

            # Render 3D view
            if render_mode == "isosurface":
                image_3d = render_3d_frame_isosurface(
                    plotter=plotter,
                    mesh=mesh,
                    pressure=pressure,
                    camera_position=camera_pos,
                    focal_point=focal_point,
                    view_up=view_up,
                    isosurface_levels=isosurface_levels,
                    clim=(vmin, vmax),
                    cmap=cmap_name,
                    isosurface_opacity=isosurface_opacity,
                    show_inclusion=show_inclusion,
                    inclusion_opacity=inclusion_opacity,
                    sensor_locations=sensor_locations,
                )
            else:
                image_3d = render_3d_frame(
                    plotter=plotter,
                    x=x,
                    y=y,
                    z=z,
                    pressure=pressure,
                    camera_position=camera_pos,
                    focal_point=focal_point,
                    view_up=view_up,
                    clim=(vmin, vmax),
                    point_size=point_size,
                    cmap=cmap_name,
                    mesh=mesh if show_inclusion else None,
                    show_inclusion=show_inclusion,
                    inclusion_opacity=inclusion_opacity,
                    sensor_locations=sensor_locations,
                )

            if image_3d is None:
                logger.error(f"Failed to render frame {frame_idx}")
                continue

            # Create combined figure
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.15)

            # Left panel: 3D simulation
            ax_3d = fig.add_subplot(gs[0])
            ax_3d.imshow(image_3d)
            ax_3d.axis("off")
            ax_3d.set_title(f"3D Pressure Field (t = {time:.4f} s)", pad=5)

            # Right panel: Sensor data
            ax_sensor = fig.add_subplot(gs[1])

            if sensor_matrix is not None:
                # Map simulation index to sensor data index
                sensor_idx = min(sim_idx, sensor_matrix.shape[1] - 1)
                current_sensor_data = sensor_matrix[:, : sensor_idx + 1]

                sensor_extent = (
                    0.0,
                    float(sensor_matrix.shape[1]),
                    0.0,
                    float(sensor_matrix.shape[0]),
                )

                # Fixed color scale for sensor data (matches plot_batch_grid.py)
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

                # Draw face separator lines at actual face boundaries
                if face_boundaries:
                    line_color = NORD_COLORS["nord4"] if use_nord_style else "white"
                    for boundary in face_boundaries[1:]:  # Skip first (always 0)
                        ax_sensor.axhline(
                            y=boundary, color=line_color, linewidth=1.0, alpha=0.7
                        )

                cbar = plt.colorbar(
                    im_sensor,
                    ax=ax_sensor,
                    label="Pressure",
                    fraction=0.03,
                    pad=0.02,
                    shrink=0.8,
                )
                cbar.ax.tick_params(labelsize=7)
                if use_nord_style:
                    cbar.ax.yaxis.set_tick_params(color=NORD_COLORS["nord4"])
                    cbar.outline.set_edgecolor(NORD_COLORS["nord3"])
                    plt.setp(
                        plt.getp(cbar.ax.axes, "yticklabels"),
                        color=NORD_COLORS["nord4"],
                    )
                    cbar.set_label("Pressure", color=NORD_COLORS["nord4"])

                ax_sensor.set_xlabel("Time Step")
                ax_sensor.set_ylabel("Sensor Index")
                ax_sensor.set_xlim(0, sensor_matrix.shape[1])
            else:
                ax_sensor.text(
                    0.5,
                    0.5,
                    "No sensor data",
                    ha="center",
                    va="center",
                    transform=ax_sensor.transAxes,
                )

            ax_sensor.set_title("Sensor Recordings")

            # Save frame
            frame_path = temp_path / f"frame_{frame_idx:05d}.png"
            fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            if (frame_idx + 1) % 20 == 0 or frame_idx == 0:
                logger.info(f"  Rendered {frame_idx + 1}/{len(frame_schedule)} frames")

        plotter.close()

        # Create video from images
        return create_video_from_images(temp_path, output_file, fps)


def render_dev_frame(
    sim_dir: Path | str,
    output_file: Path | str,
    vmin: float = -1.0,
    vmax: float = 1.0,
    render_mode: str = "points",
    show_inclusion: bool = True,
    inclusion_opacity: float = 0.3,
    isosurface_opacity: float = 0.7,
    camera_radius: float = 2.5,
    camera_elevation: float = 35.0,
    camera_azimuth: float = 45.0,
    publication_style: bool = False,
) -> bool:
    """Render a single frame from the middle of the simulation for quick testing.

    Args:
        sim_dir: Simulation output directory.
        output_file: Output image file path (PNG).
        vmin: Minimum value for pressure colormap.
        vmax: Maximum value for pressure colormap.
        render_mode: "points" or "isosurface".
        show_inclusion: Whether to show the inclusion overlay.
        inclusion_opacity: Opacity for the inclusion mesh.
        isosurface_opacity: Opacity for isosurface meshes.
        camera_radius: Distance of camera from center.
        camera_elevation: Camera elevation angle in degrees.
        camera_azimuth: Camera azimuth angle in degrees.
        publication_style: Use publication-friendly colors (white background, serif fonts).

    Returns:
        True if frame was rendered successfully.
    """
    logger = get_logger(__name__)
    sim_dir = Path(sim_dir)
    output_file = Path(output_file)

    # Ensure output is a PNG
    if output_file.suffix.lower() != ".png":
        output_file = output_file.with_suffix(".png")

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

    # Pick middle timestep
    mid_idx = len(timestep_files) // 2
    logger.info(f"Rendering frame {mid_idx} of {len(timestep_files)} (dev mode)")

    # Load timestep data
    with open(timestep_files[mid_idx], "rb") as f:
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

    # Compute domain bounds
    box_size = x.max() - x.min()
    origin = x.min()

    # Get sensor locations and matrix
    sensor_data = data.get("sensor_data")
    config = data.get("config")
    sensors_per_face = None
    sensor_matrix = None
    face_boundaries: list[int] = []

    if config is not None:
        receivers = config.get("receivers", {})
        sensors_per_face = receivers.get("sensors_per_face")

    # Get sensor matrix first
    if sensor_data is not None and "pressure" in sensor_data:
        sensor_matrix = to_numpy(sensor_data["pressure"])

    # Get sensor locations
    if sensor_data is not None and "locations" in sensor_data:
        sensor_locations = to_numpy(sensor_data["locations"])
        logger.info(f"Found {len(sensor_locations)} sensor locations in data")

        # Reorder sensor matrix by face if we have locations
        if sensor_matrix is not None:
            sensor_matrix, face_boundaries = reorder_sensors_by_face(
                sensor_matrix, sensor_locations, box_size, origin
            )
            logger.info(f"Reordered sensors by face, boundaries: {face_boundaries}")
    elif sensor_data is not None and "pressure" in sensor_data:
        # Try to infer sensors_per_face from total sensors (6 faces) if not in config
        if sensors_per_face is None:
            num_sensors = sensor_data["pressure"].shape[0]
            sensors_per_face = num_sensors // 6
        if sensors_per_face is not None and sensors_per_face > 0:
            # Get source exclusion regions from config
            exclude_regions = None
            if config is not None and "sources" in config:
                src = config["sources"]
                centers = src.get("centers", [])
                radii = src.get("radii", [])
                if centers and radii:
                    exclude_regions = [
                        (tuple(c), r) for c, r in zip(centers, radii, strict=False)
                    ]

            sensor_locations = np.array(
                generate_sensor_grid(box_size, sensors_per_face, origin, exclude_regions)
            )
            logger.info(f"Generated {len(sensor_locations)} sensor locations from grid")

            # Reorder sensor matrix by face
            if sensor_matrix is not None:
                sensor_matrix, face_boundaries = reorder_sensors_by_face(
                    sensor_matrix, sensor_locations, box_size, origin
                )
                logger.info(f"Reordered sensors by face, boundaries: {face_boundaries}")
        else:
            sensor_locations = None
    else:
        sensor_locations = None

    # Figure settings (same as main video)
    figsize = (16, 6)
    dpi = 240

    # Apply styling based on mode
    if publication_style:
        apply_publication_style()
        cmap_3d = create_publication_3d_cmap()
        cmap_sensor = create_publication_sensor_cmap()
        plt.colormaps.register(cmap_3d, name="publication_3d", force=True)
        plt.colormaps.register(cmap_sensor, name="publication_sensor", force=True)
        cmap_name = "publication_3d"
        sensor_cmap_name = "publication_sensor"
        bg_color = PUBLICATION_COLORS["background"]
        inclusion_color = PUBLICATION_COLORS["inclusion"]
        sensor_color = PUBLICATION_COLORS["sensors"]
        text_color = PUBLICATION_COLORS["text"]
    else:
        apply_nord_style()
        field_cmap = create_nord_diverging_cmap()
        plt.colormaps.register(field_cmap, name="nord_diverging", force=True)
        cmap_name = "nord_diverging"
        sensor_cmap_name = "nord_diverging"
        bg_color = NORD_COLORS["nord0"]
        inclusion_color = None  # Use default (nord13)
        sensor_color = None  # Use default (nord4)
        text_color = NORD_COLORS["nord4"]

    # Create plotter with proper size for the panel
    # Use a slightly taller height to match the sensor plot area (which has axes/labels)
    panel_width = int(figsize[0] / 2 * dpi)
    panel_height = int(figsize[1] * dpi * 1.1)
    plotter = pv.Plotter(off_screen=True, window_size=[panel_width, panel_height])
    plotter.set_background(bg_color)

    # Compute camera position
    camera_pos, focal_point, view_up = compute_camera_position_from_angles(
        center=center,
        radius=camera_radius,
        azimuth=camera_azimuth,
        elevation=camera_elevation,
    )

    # Render based on mode
    if render_mode == "isosurface":
        image_3d = render_3d_frame_isosurface(
            plotter=plotter,
            mesh=mesh,
            pressure=pressure,
            camera_position=camera_pos,
            focal_point=focal_point,
            view_up=view_up,
            clim=(vmin, vmax),
            cmap=cmap_name,
            isosurface_opacity=isosurface_opacity,
            show_inclusion=show_inclusion,
            inclusion_opacity=inclusion_opacity,
            sensor_locations=sensor_locations,
            inclusion_color=inclusion_color,
            sensor_color=sensor_color,
            publication_style=publication_style,
        )
    else:
        image_3d = render_3d_frame(
            plotter=plotter,
            x=x,
            y=y,
            z=z,
            pressure=pressure,
            camera_position=camera_pos,
            focal_point=focal_point,
            view_up=view_up,
            clim=(vmin, vmax),
            cmap=cmap_name,
            mesh=mesh if show_inclusion else None,
            show_inclusion=show_inclusion,
            inclusion_opacity=inclusion_opacity,
            sensor_locations=sensor_locations,
            inclusion_color=inclusion_color,
            sensor_color=sensor_color,
        )

    plotter.close()

    if image_3d is None:
        logger.error("Failed to render 3D frame")
        return False

    # Get simulation time
    time = ts_data.get("time", mid_idx * ts_data.get("dt", 1.0))

    # Create combined figure (same as main video)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.15)

    # Left panel: 3D simulation
    ax_3d = fig.add_subplot(gs[0])
    ax_3d.imshow(image_3d)
    ax_3d.axis("off")
    ax_3d.set_title(f"3D Pressure Field (t = {time:.4f} s)", pad=5)

    # Right panel: Sensor data
    ax_sensor = fig.add_subplot(gs[1])

    if sensor_matrix is not None:
        # Show full sensor matrix up to mid_idx
        sensor_idx = min(mid_idx, sensor_matrix.shape[1] - 1)
        current_sensor_data = sensor_matrix[:, : sensor_idx + 1]

        sensor_extent = (
            0.0,
            float(sensor_matrix.shape[1]),
            0.0,
            float(sensor_matrix.shape[0]),
        )

        # Fixed color scale for sensor data (matches plot_batch_grid.py)
        sensor_vmax = 0.3
        im_sensor = ax_sensor.imshow(
            current_sensor_data,
            aspect="auto",
            cmap=sensor_cmap_name,
            origin="lower",
            vmin=-sensor_vmax,
            vmax=sensor_vmax,
            extent=sensor_extent,
        )

        # Draw face separator lines at actual face boundaries
        if face_boundaries:
            line_color = text_color
            for boundary in face_boundaries[1:]:  # Skip first (always 0)
                ax_sensor.axhline(
                    y=boundary, color=line_color, linewidth=1.0, alpha=0.7
                )

        cbar = plt.colorbar(
            im_sensor,
            ax=ax_sensor,
            label="Pressure",
            fraction=0.03,
            pad=0.02,
            shrink=0.8,
        )
        cbar.ax.tick_params(labelsize=7)
        border_color = (
            PUBLICATION_COLORS["border"] if publication_style else NORD_COLORS["nord3"]
        )
        cbar.ax.yaxis.set_tick_params(color=text_color)
        cbar.outline.set_edgecolor(border_color)
        plt.setp(
            plt.getp(cbar.ax.axes, "yticklabels"),
            color=text_color,
        )
        cbar.set_label("Pressure", color=text_color)

        ax_sensor.set_xlabel("Time Step")
        ax_sensor.set_ylabel("Sensor Index")
        ax_sensor.set_xlim(0, sensor_matrix.shape[1])
    else:
        ax_sensor.text(
            0.5,
            0.5,
            "No sensor data",
            ha="center",
            va="center",
            transform=ax_sensor.transAxes,
        )

    ax_sensor.set_title("Sensor Recordings")

    plt.tight_layout()

    # Save the complete frame
    fig.savefig(output_file, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    logger.info(f"Dev frame saved to {output_file}")
    return True


def create_video_from_images(
    image_dir: Path,
    output_file: Path,
    fps: int = 30,
    codec: str = "libx264",
    crf: int = 23,
) -> bool:
    """Create video from PNG images using ffmpeg.

    Args:
        image_dir: Directory containing frame_*.png images.
        output_file: Output video file path.
        fps: Frames per second.
        codec: Video codec.
        crf: Constant rate factor (quality).

    Returns:
        True if video was created successfully.
    """
    logger = get_logger(__name__)

    image_pattern = image_dir / "frame_%05d.png"

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(image_pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_file),
    ]

    logger.info(f"Creating video: {output_file}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Video created successfully: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        return False


if __name__ == "__main__":
    import argparse

    from sbimaging import configure_logging

    parser = argparse.ArgumentParser(description="Create 3D simulation video")
    parser.add_argument(
        "sim_dir",
        type=str,
        help="Simulation output directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="simulation_3d.mp4",
        help="Output video file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-1.0,
        help="Minimum value for pressure colormap",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1.0,
        help="Maximum value for pressure colormap",
    )
    parser.add_argument(
        "--orbits",
        type=float,
        default=1 / 3,
        help="Number of camera orbits",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Point size for 3D rendering",
    )
    parser.add_argument(
        "--no-inclusion",
        action="store_true",
        help="Hide the inclusion overlay",
    )
    parser.add_argument(
        "--inclusion-opacity",
        type=float,
        default=0.3,
        help="Opacity for inclusion overlay (0-1)",
    )
    parser.add_argument(
        "--no-pause-pan",
        action="store_true",
        help="Disable pause-and-pan effect",
    )
    parser.add_argument(
        "--pause-at",
        type=float,
        default=0.33,
        help="Fraction of simulation at which to pause (0-1)",
    )
    parser.add_argument(
        "--pan-frames",
        type=int,
        default=60,
        help="Number of frames for pan animation during pause",
    )
    parser.add_argument(
        "--pan-arc",
        type=float,
        default=90.0,
        help="Degrees of arc to pan during pause",
    )
    parser.add_argument(
        "--pan-elevation-end",
        type=float,
        default=45.0,
        help="Camera elevation after pan completes",
    )
    parser.add_argument(
        "--final-orbit",
        type=float,
        default=4.0,
        help="Seconds of orbiting at the last simulation frame",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["points", "isosurface"],
        default="points",
        help="Rendering mode: 'points' for point cloud, 'isosurface' for isosurfaces",
    )
    parser.add_argument(
        "--isosurface-opacity",
        type=float,
        default=0.7,
        help="Opacity for isosurface meshes (0-1)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode: render single frame from middle for fast testing",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Use publication-friendly colors (white background, serif fonts)",
    )

    args = parser.parse_args()
    configure_logging()

    # Development mode: render single frame from middle
    if args.dev:
        success = render_dev_frame(
            sim_dir=args.sim_dir,
            output_file=args.output,
            vmin=args.vmin,
            vmax=args.vmax,
            render_mode=args.render_mode,
            show_inclusion=not args.no_inclusion,
            inclusion_opacity=args.inclusion_opacity,
            isosurface_opacity=args.isosurface_opacity,
            publication_style=args.publication,
        )
        if not success:
            exit(1)
        exit(0)

    success = create_video_with_sensors_3d(
        sim_dir=args.sim_dir,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        fps=args.fps,
        num_orbits=args.orbits,
        point_size=args.point_size,
        show_inclusion=not args.no_inclusion,
        inclusion_opacity=args.inclusion_opacity,
        pause_and_pan=not args.no_pause_pan,
        pause_at_fraction=args.pause_at,
        pan_frames=args.pan_frames,
        pan_arc_degrees=args.pan_arc,
        pan_elevation_end=args.pan_elevation_end,
        final_orbit_seconds=args.final_orbit,
        render_mode=args.render_mode,
        isosurface_opacity=args.isosurface_opacity,
    )

    if not success:
        exit(1)
