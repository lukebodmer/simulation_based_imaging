"""Visualization utilities for 1D FDTD simulations."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle


def create_video(
    frames: list[np.ndarray],
    times: list[float],
    output_file: str | Path,
    x_coords: np.ndarray,
    inclusion_region: tuple[float, float] | None = None,
    sensor_locations: list[float] | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    fps: int = 30,
    dpi: int = 150,
    figsize: tuple[float, float] = (12, 4),
) -> bool:
    """Create video of 1D pressure field evolution.

    Args:
        frames: List of pressure field arrays.
        times: List of simulation times for each frame.
        output_file: Output video file path.
        x_coords: x-coordinates for plotting.
        inclusion_region: Tuple (x_min, x_max) for inclusion shading.
        sensor_locations: List of sensor x-coordinates.
        vmin: Minimum pressure for y-axis.
        vmax: Maximum pressure for y-axis.
        fps: Frames per second.
        dpi: Resolution.
        figsize: Figure size in inches.

    Returns:
        True if successful.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Nord color scheme
    nord_bg = "#2e3440"
    nord_fg = "#eceff4"
    nord_blue = "#5e81ac"
    nord_cyan = "#88c0d0"
    nord_red = "#bf616a"

    fig, ax = plt.subplots(figsize=figsize, facecolor=nord_bg)
    ax.set_facecolor(nord_bg)

    # Style axes
    ax.spines["bottom"].set_color(nord_fg)
    ax.spines["top"].set_color(nord_fg)
    ax.spines["left"].set_color(nord_fg)
    ax.spines["right"].set_color(nord_fg)
    ax.tick_params(colors=nord_fg)
    ax.xaxis.label.set_color(nord_fg)
    ax.yaxis.label.set_color(nord_fg)

    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Pressure")

    # Add inclusion region shading
    if inclusion_region is not None:
        x_min, x_max = inclusion_region
        rect = Rectangle(
            (x_min, vmin),
            x_max - x_min,
            vmax - vmin,
            facecolor=nord_blue,
            alpha=0.3,
            edgecolor=nord_cyan,
            linewidth=2,
        )
        ax.add_patch(rect)

    # Add sensor markers
    if sensor_locations:
        for x in sensor_locations:
            ax.axvline(x, color=nord_red, linestyle="--", alpha=0.5, linewidth=1)

    # Initialize line
    (line,) = ax.plot([], [], color=nord_cyan, linewidth=2)
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, color=nord_fg, fontsize=12
    )

    def init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def animate(i):
        line.set_data(x_coords, frames[i])
        time_text.set_text(f"t = {times[i]:.4f} s")
        return line, time_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(frames), interval=1000 / fps, blit=True
    )

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        anim.save(str(output_file), writer=writer, dpi=dpi)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        plt.close(fig)
        return False


def create_video_with_sensors(
    frames: list[np.ndarray],
    times: list[float],
    sensor_data: np.ndarray,
    output_file: str | Path,
    x_coords: np.ndarray,
    sensor_locations: list[float],
    inclusion_region: tuple[float, float] | None = None,
    inclusion_alpha: float = 0.3,
    vmin: float = -1.0,
    vmax: float = 1.0,
    fps: int = 30,
    dpi: int = 150,
    figsize: tuple[float, float] = (14, 5),
) -> bool:
    """Create video with pressure field and sensor data side by side.

    Args:
        frames: List of pressure field arrays.
        times: List of simulation times for each frame.
        sensor_data: Sensor recordings (num_sensors, num_frames).
        output_file: Output video file path.
        x_coords: x-coordinates for plotting.
        sensor_locations: List of sensor x-coordinates.
        inclusion_region: Tuple (x_min, x_max) for inclusion shading.
        inclusion_alpha: Alpha (opacity) for inclusion region shading.
        vmin: Minimum pressure for y-axis.
        vmax: Maximum pressure for y-axis.
        fps: Frames per second.
        dpi: Resolution.
        figsize: Figure size in inches.

    Returns:
        True if successful.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Nord color scheme
    nord_bg = "#2e3440"
    nord_fg = "#eceff4"
    nord_blue = "#5e81ac"
    nord_cyan = "#88c0d0"
    nord_red = "#bf616a"

    fig = plt.figure(figsize=figsize, facecolor=nord_bg, layout="constrained")
    # Left column: pressure field on top, tunnel visualization below
    # Right column: top = left sensor, bottom = right sensor
    # Use GridSpec for more control over subplot sizes
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(
        8, 2,
        figure=fig,
        width_ratios=[1, 1],
        height_ratios=[1, 1, 1, 1, 1, 1, 1, 1],
    )

    ax1 = fig.add_subplot(gs[0:7, 0])  # Pressure field (top 7/8 of left column)
    ax_tunnel = fig.add_subplot(gs[7, 0])  # Tunnel visualization (bottom 1/8 of left column)
    ax2 = fig.add_subplot(gs[0:4, 1])  # Left sensor (top half of right column)
    ax3 = fig.add_subplot(gs[4:8, 1])  # Right sensor (bottom half of right column)

    for ax in [ax1, ax2, ax3, ax_tunnel]:
        ax.set_facecolor(nord_bg)
        ax.spines["bottom"].set_color(nord_fg)
        ax.spines["top"].set_color(nord_fg)
        ax.spines["left"].set_color(nord_fg)
        ax.spines["right"].set_color(nord_fg)
        ax.tick_params(colors=nord_fg)
        ax.xaxis.label.set_color(nord_fg)
        ax.yaxis.label.set_color(nord_fg)
        ax.title.set_color(nord_fg)

    # Pressure field plot
    ax1.set_xlim(x_coords[0], x_coords[-1])
    ax1.set_ylim(vmin, vmax)
    ax1.set_xlabel("Position (m)")
    ax1.set_ylabel("Pressure")
    ax1.set_title("Pressure Field")

    # Add inclusion region
    if inclusion_region is not None:
        x_min, x_max = inclusion_region
        rect = Rectangle(
            (x_min, vmin),
            x_max - x_min,
            vmax - vmin,
            facecolor=nord_blue,
            alpha=inclusion_alpha,
            edgecolor=nord_cyan,
            linewidth=2,
        )
        ax1.add_patch(rect)

    # Sensor markers on pressure plot (dots that will move with pressure)
    (left_sensor_dot,) = ax1.plot([], [], "o", color=nord_red, markersize=10)
    (right_sensor_dot,) = ax1.plot([], [], "o", color=nord_cyan, markersize=10)

    (line,) = ax1.plot([], [], color=nord_cyan, linewidth=2)
    time_text = ax1.text(
        0.02, 0.95, "", transform=ax1.transAxes, color=nord_fg, fontsize=12
    )

    # Sensor data plots - separate subplots for each sensor
    sensor_vmax = max(abs(sensor_data.min()), abs(sensor_data.max()))

    # Left sensor (top right subplot)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Pressure")
    ax2.set_title("Left Sensor")
    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(-sensor_vmax * 1.1, sensor_vmax * 1.1)
    (left_sensor_line,) = ax2.plot([], [], color=nord_red, linewidth=2)

    # Right sensor (bottom right subplot)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Pressure")
    ax3.set_title("Right Sensor")
    ax3.set_xlim(times[0], times[-1])
    ax3.set_ylim(-sensor_vmax * 1.1, sensor_vmax * 1.1)
    (right_sensor_line,) = ax3.plot([], [], color=nord_cyan, linewidth=2)

    # Tunnel visualization (horizontal colorbar showing pressure)
    ax_tunnel.set_xlim(x_coords[0], x_coords[-1])
    ax_tunnel.set_ylim(0, 1)
    ax_tunnel.set_xlabel("Position (m)")
    ax_tunnel.set_yticks([])

    # Add inclusion region to tunnel view
    if inclusion_region is not None:
        x_min, x_max = inclusion_region
        tunnel_rect = Rectangle(
            (x_min, 0),
            x_max - x_min,
            1,
            facecolor=nord_blue,
            alpha=min(inclusion_alpha + 0.2, 1.0),
            edgecolor=nord_cyan,
            linewidth=2,
        )
        ax_tunnel.add_patch(tunnel_rect)

    # Initialize tunnel pressure image
    tunnel_data = np.zeros((1, len(x_coords)))
    tunnel_im = ax_tunnel.imshow(
        tunnel_data,
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        extent=[x_coords[0], x_coords[-1], 0, 1],
    )

    # Add sensor markers to tunnel
    ax_tunnel.plot(sensor_locations[0], 0.5, "o", color=nord_red, markersize=8, zorder=10)
    ax_tunnel.plot(sensor_locations[1], 0.5, "o", color=nord_cyan, markersize=8, zorder=10)

    def init():
        line.set_data([], [])
        time_text.set_text("")
        left_sensor_dot.set_data([], [])
        right_sensor_dot.set_data([], [])
        left_sensor_line.set_data([], [])
        right_sensor_line.set_data([], [])
        tunnel_im.set_array(np.zeros((1, len(x_coords))))
        return (line, time_text, left_sensor_dot, right_sensor_dot, left_sensor_line, right_sensor_line, tunnel_im)

    def animate(i):
        line.set_data(x_coords, frames[i])
        time_text.set_text(f"t = {times[i]:.4f} s")

        # Update sensor dots on pressure field
        left_sensor_dot.set_data([sensor_locations[0]], [sensor_data[0, i]])
        right_sensor_dot.set_data([sensor_locations[1]], [sensor_data[1, i]])

        # Update sensor lines (show data up to current time)
        time_slice = times[: i + 1]
        left_sensor_line.set_data(time_slice, sensor_data[0, : i + 1])
        right_sensor_line.set_data(time_slice, sensor_data[1, : i + 1])

        # Update tunnel visualization
        tunnel_im.set_array(frames[i].reshape(1, -1))

        return (line, time_text, left_sensor_dot, right_sensor_dot, left_sensor_line, right_sensor_line, tunnel_im)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(frames), interval=1000 / fps, blit=True
    )

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        anim.save(str(output_file), writer=writer, dpi=dpi)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        plt.close(fig)
        return False
