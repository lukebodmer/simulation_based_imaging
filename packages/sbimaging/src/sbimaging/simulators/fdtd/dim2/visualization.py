"""Visualization utilities for 2D FDTD simulations.

Creates plots and videos from simulation output.
"""

from pathlib import Path
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sbimaging.logging import get_logger


# Nord color palette for consistent theming
NORD_COLORS = {
    # Polar Night - Dark backgrounds
    "nord0": "#2e3440",
    "nord1": "#3b4252",
    "nord2": "#434c5e",
    "nord3": "#4c566a",
    # Snow Storm - Light text
    "nord4": "#d8dee9",
    "nord5": "#e5e9f0",
    "nord6": "#eceff4",
    # Frost - Blues
    "nord7": "#8fbcbb",
    "nord8": "#88c0d0",
    "nord9": "#81a1c1",
    "nord10": "#5e81ac",
    # Aurora - Accents
    "nord11": "#bf616a",  # Red
    "nord12": "#d08770",  # Orange
    "nord13": "#ebcb8b",  # Yellow
    "nord14": "#a3be8c",  # Green
    "nord15": "#b48ead",  # Purple
}


def create_nord_diverging_cmap() -> mcolors.LinearSegmentedColormap:
    """Create a diverging colormap using Nord colors (blue-white-red)."""
    colors = [
        NORD_COLORS["nord10"],  # Dark blue (negative)
        NORD_COLORS["nord8"],   # Light blue
        NORD_COLORS["nord6"],   # White (zero)
        NORD_COLORS["nord12"],  # Orange
        NORD_COLORS["nord11"],  # Red (positive)
    ]
    return mcolors.LinearSegmentedColormap.from_list("nord_diverging", colors)


def apply_nord_style() -> None:
    """Apply Nord-themed matplotlib style settings."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor": NORD_COLORS["nord0"],
        "figure.edgecolor": NORD_COLORS["nord0"],

        # Axes
        "axes.facecolor": NORD_COLORS["nord1"],
        "axes.edgecolor": NORD_COLORS["nord3"],
        "axes.labelcolor": NORD_COLORS["nord4"],
        "axes.titlecolor": NORD_COLORS["nord6"],
        "axes.titleweight": "medium",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.labelweight": "normal",

        # Grid
        "axes.grid": False,
        "grid.color": NORD_COLORS["nord3"],
        "grid.alpha": 0.3,

        # Ticks
        "xtick.color": NORD_COLORS["nord4"],
        "ytick.color": NORD_COLORS["nord4"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # Text
        "text.color": NORD_COLORS["nord6"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Nimbus Sans", "FreeSans", "Cantarell", "sans-serif"],

        # Legend
        "legend.facecolor": NORD_COLORS["nord2"],
        "legend.edgecolor": NORD_COLORS["nord3"],
        "legend.labelcolor": NORD_COLORS["nord4"],

        # Savefig
        "savefig.facecolor": NORD_COLORS["nord0"],
        "savefig.edgecolor": NORD_COLORS["nord0"],
    })


def plot_pressure_field(
    pressure: np.ndarray,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
) -> Figure | None:
    """Plot a 2D pressure field.

    Args:
        pressure: 2D pressure array.
        x_coords: x-coordinates for axis labels.
        y_coords: y-coordinates for axis labels.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        cmap: Matplotlib colormap name.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        ax: Axes to plot on. If None, creates new figure.

    Returns:
        Figure if ax was None, else None.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True

    if vmin is None and vmax is None:
        max_abs = np.abs(pressure).max()
        if max_abs > 0:
            vmin, vmax = -max_abs, max_abs
        else:
            vmin, vmax = -1, 1

    extent: tuple[float, float, float, float] | None = None
    if x_coords is not None and y_coords is not None:
        extent = (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1]))

    im = ax.imshow(
        pressure.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
    )

    plt.colorbar(im, ax=ax, label="Pressure [Pa]")

    if x_coords is not None:
        ax.set_xlabel("x [m]")
    if y_coords is not None:
        ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title)

    if created_fig:
        return fig
    return None


def render_frames_to_images(
    frame_dir: Path | str,
    output_dir: Path | str,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    dpi: int = 100,
    figsize: tuple[float, float] = (8, 6),
) -> int:
    """Render saved pressure frames to PNG images.

    Args:
        frame_dir: Directory containing pressure .npy files.
        output_dir: Directory for output PNG images.
        x_coords: x-coordinates for axis labels.
        y_coords: y-coordinates for axis labels.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        cmap: Matplotlib colormap name.
        dpi: Image resolution.
        figsize: Figure size (width, height) in inches.

    Returns:
        Number of frames rendered.
    """
    logger = get_logger(__name__)
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frame_dir.glob("pressure_*.npy"))
    if not frame_files:
        logger.warning(f"No pressure files found in {frame_dir}")
        return 0

    if vmin is None or vmax is None:
        logger.info("Scanning frames to determine color scale...")
        max_abs = 0.0
        for frame_file in frame_files:
            pressure = np.load(frame_file)
            max_abs = max(max_abs, np.abs(pressure).max())
        if max_abs > 0:
            vmin, vmax = -max_abs, max_abs
        else:
            vmin, vmax = -1, 1
        logger.info(f"Color scale: [{vmin:.4g}, {vmax:.4g}]")

    logger.info(f"Rendering {len(frame_files)} frames...")

    for i, frame_file in enumerate(frame_files):
        pressure = np.load(frame_file)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        extent: tuple[float, float, float, float] | None = None
        if x_coords is not None and y_coords is not None:
            extent = (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1]))

        im = ax.imshow(
            pressure.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
        )

        plt.colorbar(im, ax=ax, label="Pressure [Pa]")

        if x_coords is not None:
            ax.set_xlabel("x [m]")
        if y_coords is not None:
            ax.set_ylabel("y [m]")

        frame_num = int(frame_file.stem.split("_")[-1])
        ax.set_title(f"Pressure Field (frame {frame_num})")

        output_file = output_dir / f"frame_{i:05d}.png"
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if (i + 1) % 50 == 0:
            logger.info(f"  Rendered {i + 1}/{len(frame_files)} frames")

    logger.info(f"Rendered {len(frame_files)} frames to {output_dir}")
    return len(frame_files)


def create_video_from_images(
    image_dir: Path | str,
    output_file: Path | str,
    fps: int = 30,
    codec: str = "libx264",
    crf: int = 23,
) -> bool:
    """Create video from PNG images using ffmpeg.

    Args:
        image_dir: Directory containing frame_*.png images.
        output_file: Output video file path.
        fps: Frames per second.
        codec: Video codec (libx264 for h264).
        crf: Constant rate factor (quality, lower = better, 18-28 typical).

    Returns:
        True if video was created successfully.
    """
    logger = get_logger(__name__)
    image_dir = Path(image_dir)
    output_file = Path(output_file)

    image_pattern = image_dir / "frame_%05d.png"

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(image_pattern),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", codec,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_file),
    ]

    logger.info(f"Creating video: {output_file}")
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Video created successfully: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        return False


def create_video_from_frames(
    frame_dir: Path | str,
    output_file: Path | str,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    fps: int = 30,
    dpi: int = 100,
    figsize: tuple[float, float] = (8, 6),
) -> bool:
    """Create video from saved pressure frames.

    This is a convenience function that renders frames to images
    and then creates a video.

    Args:
        frame_dir: Directory containing pressure .npy files.
        output_file: Output video file path.
        x_coords: x-coordinates for axis labels.
        y_coords: y-coordinates for axis labels.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        cmap: Matplotlib colormap name.
        fps: Frames per second.
        dpi: Image resolution.
        figsize: Figure size (width, height) in inches.

    Returns:
        True if video was created successfully.
    """
    import tempfile

    frame_dir = Path(frame_dir)
    output_file = Path(output_file)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        num_frames = render_frames_to_images(
            frame_dir=frame_dir,
            output_dir=temp_path,
            x_coords=x_coords,
            y_coords=y_coords,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            dpi=dpi,
            figsize=figsize,
        )

        if num_frames == 0:
            return False

        return create_video_from_images(
            image_dir=temp_path,
            output_file=output_file,
            fps=fps,
        )


class CircleOverlay:
    """Circle overlay for marking inclusions on plots."""

    def __init__(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        color: str | None = None,
        linewidth: float = 2.0,
        linestyle: str = "-",
    ):
        """Initialize circle overlay.

        Args:
            center_x: x-coordinate of circle center [m].
            center_y: y-coordinate of circle center [m].
            radius: Circle radius [m].
            color: Line color. If None, uses Nord dark background.
            linewidth: Line width.
            linestyle: Line style ('-', '--', ':', etc.).
        """
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.color = color if color is not None else NORD_COLORS["nord0"]
        self.linewidth = linewidth
        self.linestyle = linestyle

    def add_to_axes(self, ax: Axes) -> None:
        """Add circle to matplotlib axes.

        Args:
            ax: Matplotlib axes to draw on.
        """
        from matplotlib.patches import Circle

        circle = Circle(
            (self.center_x, self.center_y),
            self.radius,
            fill=False,
            edgecolor=self.color,
            linewidth=self.linewidth,
            linestyle=self.linestyle,
        )
        ax.add_patch(circle)


class TriangleOverlay:
    """Triangle overlay for marking inclusions on plots."""

    def __init__(
        self,
        vertices: list[tuple[float, float]],
        color: str | None = None,
        linewidth: float = 2.0,
        linestyle: str = "-",
    ):
        """Initialize triangle overlay.

        Args:
            vertices: List of 3 (x, y) tuples defining triangle vertices [m].
            color: Line color. If None, uses Nord dark background.
            linewidth: Line width.
            linestyle: Line style ('-', '--', ':', etc.).
        """
        if len(vertices) != 3:
            raise ValueError("Triangle requires exactly 3 vertices")
        self.vertices = vertices
        self.color = color if color is not None else NORD_COLORS["nord0"]
        self.linewidth = linewidth
        self.linestyle = linestyle

    def add_to_axes(self, ax: Axes) -> None:
        """Add triangle to matplotlib axes.

        Args:
            ax: Matplotlib axes to draw on.
        """
        from matplotlib.patches import Polygon

        triangle = Polygon(
            self.vertices,
            fill=False,
            edgecolor=self.color,
            linewidth=self.linewidth,
            linestyle=self.linestyle,
            closed=True,
        )
        ax.add_patch(triangle)


class RectangleOverlay:
    """Rectangle overlay for marking inclusions on plots."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        width: float,
        height: float,
        color: str | None = None,
        linewidth: float = 2.0,
        linestyle: str = "-",
    ):
        """Initialize rectangle overlay.

        Args:
            x_min: Left edge x-coordinate [m].
            y_min: Bottom edge y-coordinate [m].
            width: Rectangle width [m].
            height: Rectangle height [m].
            color: Line color. If None, uses Nord dark background.
            linewidth: Line width.
            linestyle: Line style ('-', '--', ':', etc.).
        """
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height
        self.color = color if color is not None else NORD_COLORS["nord0"]
        self.linewidth = linewidth
        self.linestyle = linestyle

    def add_to_axes(self, ax: Axes) -> None:
        """Add rectangle to matplotlib axes.

        Args:
            ax: Matplotlib axes to draw on.
        """
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (self.x_min, self.y_min),
            self.width,
            self.height,
            fill=False,
            edgecolor=self.color,
            linewidth=self.linewidth,
            linestyle=self.linestyle,
        )
        ax.add_patch(rect)


def create_video_from_memory(
    frames: list[np.ndarray],
    times: list[float],
    output_file: Path | str,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    fps: int = 30,
    dpi: int = 100,
    figsize: tuple[float, float] = (8, 6),
    overlays: list[CircleOverlay] | None = None,
) -> bool:
    """Create video from frames stored in memory.

    Args:
        frames: List of pressure field arrays.
        times: List of simulation times for each frame.
        output_file: Output video file path.
        x_coords: x-coordinates for axis labels.
        y_coords: y-coordinates for axis labels.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        cmap: Matplotlib colormap name.
        fps: Frames per second.
        dpi: Image resolution.
        figsize: Figure size (width, height) in inches.
        overlays: List of overlay objects (e.g., CircleOverlay) to draw on each frame.

    Returns:
        True if video was created successfully.
    """
    import tempfile

    logger = get_logger(__name__)

    if not frames:
        logger.warning("No frames provided")
        return False

    output_file = Path(output_file)

    if vmin is None or vmax is None:
        max_abs = max(np.abs(frame).max() for frame in frames)
        if max_abs > 0:
            vmin, vmax = -max_abs, max_abs
        else:
            vmin, vmax = -1, 1

    logger.info(f"Rendering {len(frames)} frames...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, (pressure, time) in enumerate(zip(frames, times)):
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            extent: tuple[float, float, float, float] | None = None
            if x_coords is not None and y_coords is not None:
                extent = (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1]))

            im = ax.imshow(
                pressure.T,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                aspect="equal",
            )

            plt.colorbar(im, ax=ax, label="Pressure [Pa]")

            if overlays:
                for overlay in overlays:
                    overlay.add_to_axes(ax)

            if x_coords is not None:
                ax.set_xlabel("x [m]")
            if y_coords is not None:
                ax.set_ylabel("y [m]")

            ax.set_title(f"Pressure Field (t = {time*1000:.3f} ms)")

            output_path = temp_path / f"frame_{i:05d}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        return create_video_from_images(
            image_dir=temp_path,
            output_file=output_file,
            fps=fps,
        )


class SensorOverlay:
    """Overlay for marking sensor positions on plots."""

    def __init__(
        self,
        locations: np.ndarray,
        color: str = "green",
        marker: str = "o",
        size: float = 30,
    ):
        """Initialize sensor overlay.

        Args:
            locations: Array of (x, y) sensor coordinates.
            color: Marker color.
            marker: Marker style.
            size: Marker size.
        """
        self.locations = locations
        self.color = color
        self.marker = marker
        self.size = size

    def add_to_axes(self, ax: Axes) -> None:
        """Add sensor markers to matplotlib axes.

        Args:
            ax: Matplotlib axes to draw on.
        """
        ax.scatter(
            self.locations[:, 0],
            self.locations[:, 1],
            c=self.color,
            marker=self.marker,
            s=self.size,
            zorder=10,
        )


def create_video_with_sensors(
    frames: list[np.ndarray],
    times: list[float],
    sensor_data: np.ndarray,
    output_file: Path | str,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    sensor_locations: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    fps: int = 30,
    dpi: int = 100,
    figsize: tuple[float, float] = (14, 5),
    overlays: list[CircleOverlay] | None = None,
    sensors_per_side: int | None = None,
    use_nord_style: bool = True,
) -> bool:
    """Create video with pressure field and sensor data side by side.

    The left panel shows the pressure field with optional overlays.
    The right panel shows the sensor data matrix accumulating over time.

    Args:
        frames: List of pressure field arrays.
        times: List of simulation times for each frame.
        sensor_data: Sensor recordings (num_sensors, num_timesteps).
        output_file: Output video file path.
        x_coords: x-coordinates for axis labels.
        y_coords: y-coordinates for axis labels.
        sensor_locations: Array of (x, y) sensor coordinates for plotting.
        vmin: Minimum value for pressure colormap.
        vmax: Maximum value for pressure colormap.
        cmap: Matplotlib colormap name. If None and use_nord_style, uses Nord colormap.
        fps: Frames per second.
        dpi: Image resolution.
        figsize: Figure size (width, height) in inches.
        overlays: List of overlay objects to draw on pressure field.
        sensors_per_side: Number of sensors per side, used to draw face separators.
        use_nord_style: If True, apply Nord color theme to plots.

    Returns:
        True if video was created successfully.
    """
    import tempfile

    logger = get_logger(__name__)

    if not frames:
        logger.warning("No frames provided")
        return False

    output_file = Path(output_file)

    # Apply Nord style if requested
    if use_nord_style:
        apply_nord_style()
        if cmap is None:
            field_cmap = create_nord_diverging_cmap()
        else:
            field_cmap = cmap
    else:
        field_cmap = cmap if cmap is not None else "RdBu_r"

    if vmin is None or vmax is None:
        max_abs = max(np.abs(frame).max() for frame in frames)
        if max_abs > 0:
            vmin, vmax = -max_abs, max_abs
        else:
            vmin, vmax = -1, 1

    sensor_vmax = np.abs(sensor_data).max() if sensor_data.size > 0 else 1.0

    logger.info(f"Rendering {len(frames)} frames with sensor data...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, (pressure, time) in enumerate(zip(frames, times)):
            # Use GridSpec for better control over subplot sizes
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.3)

            ax_field = fig.add_subplot(gs[0])
            ax_sensor = fig.add_subplot(gs[1])

            # Left panel: Pressure field
            extent: tuple[float, float, float, float] | None = None
            if x_coords is not None and y_coords is not None:
                extent = (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1]))

            im_field = ax_field.imshow(
                pressure.T,
                origin="lower",
                cmap=field_cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                aspect="equal",
            )

            cbar_field = plt.colorbar(im_field, ax=ax_field, label="Pressure", fraction=0.03, pad=0.02, shrink=0.8)
            cbar_field.ax.tick_params(labelsize=7)

            if overlays:
                for overlay in overlays:
                    overlay.add_to_axes(ax_field)

            if sensor_locations is not None:
                # Use Nord green for sensor markers
                sensor_color = NORD_COLORS["nord14"] if use_nord_style else "lime"
                edge_color = NORD_COLORS["nord0"] if use_nord_style else "black"
                ax_field.scatter(
                    sensor_locations[:, 0],
                    sensor_locations[:, 1],
                    c=sensor_color,
                    marker="o",
                    s=15,
                    edgecolors=edge_color,
                    linewidths=0.5,
                    zorder=10,
                )

            if x_coords is not None:
                ax_field.set_xlabel("x [m]")
            if y_coords is not None:
                ax_field.set_ylabel("y [m]")

            ax_field.set_title(f"Pressure Field (t = {time:.3f} s)")

            # Right panel: Sensor data matrix (accumulated up to current frame)
            current_sensor_data = sensor_data[:, :i+1] if i < sensor_data.shape[1] else sensor_data

            # Use auto aspect to allow the sensor plot to fill its space properly
            sensor_extent: tuple[float, float, float, float] = (
                0.0, float(len(times)), 0.0, float(sensor_data.shape[0])
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

            # Draw horizontal lines to separate sensor faces
            if sensors_per_side is not None:
                line_color = NORD_COLORS["nord6"] if use_nord_style else "white"
                for face_idx in range(1, 4):
                    y_line = face_idx * sensors_per_side
                    ax_sensor.axhline(y=y_line, color=line_color, linewidth=1.5, linestyle="-")

            cbar_sensor = plt.colorbar(im_sensor, ax=ax_sensor, label="Pressure", fraction=0.03, pad=0.12, shrink=0.8)
            cbar_sensor.ax.tick_params(labelsize=7)

            ax_sensor.set_xlabel("Time Step")
            ax_sensor.set_ylabel("Sensor Index")
            ax_sensor.set_title("Sensor Recordings")
            ax_sensor.set_xlim(0, len(times))

            # Add face labels on the right side (between plot and colorbar)
            if sensors_per_side is not None:
                face_labels = ["Bottom", "Top", "Left", "Right"]
                for face_idx, label in enumerate(face_labels):
                    y_pos = (face_idx + 0.5) * sensors_per_side
                    # Position labels in axes coordinates, just past the right edge
                    ax_sensor.text(
                        1.02, y_pos / sensor_data.shape[0], label,
                        va="center", ha="left", fontsize=7,
                        transform=ax_sensor.transAxes,
                    )

            output_path = temp_path / f"frame_{i:05d}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        return create_video_from_images(
            image_dir=temp_path,
            output_file=output_file,
            fps=fps,
        )
