"""Video generation for 3D DG simulations.

Creates videos showing 3D simulation with rotating camera view
and accumulating sensor data side by side.
"""

import pickle
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def create_nord_diverging_cmap() -> mcolors.LinearSegmentedColormap:
    """Create a diverging colormap using Nord colors (blue-white-red)."""
    colors = [
        NORD_COLORS["nord10"],
        NORD_COLORS["nord8"],
        NORD_COLORS["nord6"],
        NORD_COLORS["nord12"],
        NORD_COLORS["nord11"],
    ]
    return mcolors.LinearSegmentedColormap.from_list("nord_diverging", colors)


def apply_nord_style() -> None:
    """Apply Nord-themed matplotlib style settings."""
    plt.rcParams.update({
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
    })


def to_numpy(array):
    """Convert CuPy array to NumPy if needed."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def load_simulation_data(sim_dir: Path) -> dict:
    """Load all simulation data from a directory.

    Args:
        sim_dir: Simulation output directory.

    Returns:
        Dictionary with mesh, timestep files, sensor data, and energy data.
    """
    logger = get_logger(__name__)

    result = {
        "mesh": None,
        "timestep_files": [],
        "sensor_data": None,
        "energy_data": None,
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

    return result


def compute_camera_position_from_angles(
    center: tuple[float, float, float],
    radius: float,
    azimuth: float,
    elevation: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
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
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
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
        schedule.append({
            "sim_idx": i,
            "azimuth": azimuth,
            "elevation": elevation_start,
        })

    # Phase 2: Simulation paused, camera pans
    pause_azimuth_start = phase1_orbit_fraction * 360.0 * num_orbits
    pause_azimuth_end = pause_azimuth_start + pan_arc_degrees

    for i in range(pan_frames):
        t = i / (pan_frames - 1) if pan_frames > 1 else 0
        # Smooth easing (ease-in-out)
        t_smooth = 0.5 - 0.5 * np.cos(np.pi * t)

        azimuth = pause_azimuth_start + t_smooth * pan_arc_degrees
        elevation = elevation_start + t_smooth * (elevation_end - elevation_start)

        schedule.append({
            "sim_idx": pause_sim_idx,  # Frozen simulation
            "azimuth": azimuth,
            "elevation": elevation,
        })

    # Phase 3: Simulation resumes
    # Adjust orbit to account for the pan arc we just did
    resume_azimuth = pause_azimuth_end
    remaining_sim_frames = num_sim_frames - pause_sim_idx

    for i in range(remaining_sim_frames):
        sim_idx = pause_sim_idx + i
        # Progress from pause point to end
        progress = i / remaining_sim_frames if remaining_sim_frames > 0 else 0
        remaining_orbit = (1.0 - pause_at_fraction) * 360.0 * num_orbits - pan_arc_degrees
        azimuth = resume_azimuth + progress * remaining_orbit

        schedule.append({
            "sim_idx": sim_idx,
            "azimuth": azimuth,
            "elevation": elevation_end,
        })

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

            schedule.append({
                "sim_idx": last_sim_idx,
                "azimuth": azimuth,
                "elevation": elevation_end,
            })

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
        color = NORD_COLORS["nord8"]  # Light cyan/frost color

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
        )

    # Add inclusion overlay if mesh data provided
    if show_inclusion and mesh is not None:
        add_inclusion_mesh(plotter, mesh, opacity=inclusion_opacity)

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
    vmin: float = -0.5,
    vmax: float = 0.5,
    fps: int = 30,
    dpi: int = 150,
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
    final_orbit_seconds: float = 2.0,
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

    # Get sensor data matrix
    if sensor_data is not None and "pressure" in sensor_data:
        sensor_matrix = to_numpy(sensor_data["pressure"])
        sensor_vmax = np.abs(sensor_matrix).max() if sensor_matrix.size > 0 else 1.0
    else:
        sensor_matrix = None
        sensor_vmax = 1.0

    # Apply styling
    if use_nord_style:
        apply_nord_style()
        field_cmap = create_nord_diverging_cmap()
        cmap_name = "seismic"  # PyVista uses string names
    else:
        field_cmap = "RdBu_r"
        cmap_name = "RdBu_r"

    # Create PyVista plotter
    if window_size is None:
        window_size = [800, 600]
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
            frame_schedule.append({
                "sim_idx": i,
                "azimuth": azimuth,
                "elevation": camera_elevation,
            })
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
            ax_3d.set_title(f"3D Pressure Field (t = {time:.4f} s)")

            # Right panel: Sensor data
            ax_sensor = fig.add_subplot(gs[1])

            if sensor_matrix is not None:
                # Map simulation index to sensor data index
                sensor_idx = min(sim_idx, sensor_matrix.shape[1] - 1)
                current_sensor_data = sensor_matrix[:, :sensor_idx + 1]

                sensor_extent = (
                    0.0,
                    float(sensor_matrix.shape[1]),
                    0.0,
                    float(sensor_matrix.shape[0]),
                )

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
                if sensors_per_face is not None:
                    line_color = NORD_COLORS["nord6"] if use_nord_style else "white"
                    num_faces = sensor_matrix.shape[0] // sensors_per_face
                    for face_idx in range(1, num_faces):
                        y_line = face_idx * sensors_per_face
                        ax_sensor.axhline(
                            y=y_line, color=line_color, linewidth=1.5, linestyle="-"
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
        default=2.0,
        help="Seconds of orbiting at the last simulation frame",
    )

    args = parser.parse_args()
    configure_logging()

    success = create_video_with_sensors_3d(
        sim_dir=args.sim_dir,
        output_file=args.output,
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
    )

    if not success:
        exit(1)
