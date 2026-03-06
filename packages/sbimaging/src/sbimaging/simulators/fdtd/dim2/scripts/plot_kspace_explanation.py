#!/usr/bin/env python
"""Generate a figure explaining k-space representation of inclusions.

Shows a 2D inclusion domain on the left and its k-space representation
on the right, with Nord color scheme.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from numpy.fft import fft2, fftshift


# Nord color palette
NORD_COLORS = {
    "nord0": "#2e3440",   # Dark background
    "nord1": "#3b4252",   # Lighter background
    "nord2": "#434c5e",
    "nord3": "#4c566a",   # Light gray
    "nord4": "#d8dee9",   # Light text
    "nord5": "#e5e9f0",
    "nord6": "#eceff4",   # White text
    "nord7": "#8fbcbb",   # Frost cyan-green
    "nord8": "#88c0d0",   # Frost cyan
    "nord9": "#81a1c1",   # Frost blue
    "nord10": "#5e81ac",  # Frost dark blue
    "nord11": "#bf616a",  # Aurora red
    "nord12": "#d08770",  # Aurora orange
    "nord13": "#ebcb8b",  # Aurora yellow
    "nord14": "#a3be8c",  # Aurora green
    "nord15": "#b48ead",  # Aurora purple
}


def create_nord_cmap() -> mcolors.LinearSegmentedColormap:
    """Create a colormap using Nord colors for k-space magnitude."""
    colors = [
        NORD_COLORS["nord0"],   # Dark (low values)
        NORD_COLORS["nord10"],  # Dark blue
        NORD_COLORS["nord9"],   # Blue
        NORD_COLORS["nord8"],   # Cyan
        NORD_COLORS["nord13"],  # Yellow (high values)
    ]
    return mcolors.LinearSegmentedColormap.from_list("nord_mag", colors)


def apply_nord_style() -> None:
    """Apply Nord-themed matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": NORD_COLORS["nord0"],
        "figure.edgecolor": NORD_COLORS["nord0"],
        "axes.facecolor": NORD_COLORS["nord1"],
        "axes.edgecolor": NORD_COLORS["nord3"],
        "axes.labelcolor": NORD_COLORS["nord4"],
        "axes.titlecolor": NORD_COLORS["nord6"],
        "axes.titleweight": "medium",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.color": NORD_COLORS["nord4"],
        "ytick.color": NORD_COLORS["nord4"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": NORD_COLORS["nord6"],
        "savefig.facecolor": NORD_COLORS["nord0"],
        "savefig.edgecolor": NORD_COLORS["nord0"],
    })


def create_inclusion_image(
    inclusion_type: str,
    center_x: float,
    center_y: float,
    inclusion_size: float,
    domain_size: float,
    grid_size: int = 128,
) -> np.ndarray:
    """Create a 2D image with the inclusion mask.

    Args:
        inclusion_type: Type of inclusion ("circle", "square", "triangle").
        center_x: x-coordinate of inclusion center.
        center_y: y-coordinate of inclusion center.
        inclusion_size: Characteristic size (diameter for circle, side for others).
        domain_size: Size of the square domain.
        grid_size: Resolution of the output image.

    Returns:
        2D array of shape (grid_size, grid_size).
    """
    x = np.linspace(0, domain_size, grid_size)
    y = np.linspace(0, domain_size, grid_size)
    X, Y = np.meshgrid(x, y)

    image = np.zeros((grid_size, grid_size), dtype=np.float32)

    if inclusion_type == "circle":
        radius = inclusion_size / 2
        mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius**2

    elif inclusion_type == "square":
        half_side = inclusion_size / 2
        mask = (np.abs(X - center_x) <= half_side) & (np.abs(Y - center_y) <= half_side)

    elif inclusion_type == "triangle":
        side = inclusion_size
        height = (np.sqrt(3) / 2) * side
        v0 = np.array([center_x - side / 2, center_y - height / 3])
        v1 = np.array([center_x + side / 2, center_y - height / 3])
        v2 = np.array([center_x, center_y + 2 * height / 3])
        mask = _point_in_triangle(X, Y, v0, v1, v2)

    else:
        raise ValueError(f"Unknown inclusion type: {inclusion_type}")

    image[mask] = 1.0
    return image


def _point_in_triangle(
    X: np.ndarray,
    Y: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Check if points are inside a triangle using barycentric coordinates."""
    def sign(px, py, x1, y1, x2, y2):
        return (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)

    d1 = sign(X, Y, v0[0], v0[1], v1[0], v1[1])
    d2 = sign(X, Y, v1[0], v1[1], v2[0], v2[1])
    d3 = sign(X, Y, v2[0], v2[1], v0[0], v0[1])

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return ~(has_neg & has_pos)


def add_inclusion_patch(
    ax,
    inclusion_type: str,
    center_x: float,
    center_y: float,
    inclusion_size: float,
) -> None:
    """Add an inclusion outline patch to the axes."""
    if inclusion_type == "circle":
        patch = Circle(
            (center_x, center_y),
            inclusion_size / 2,
            fill=True,
            facecolor=NORD_COLORS["nord13"],
            edgecolor=NORD_COLORS["nord6"],
            linewidth=2,
            alpha=0.9,
        )
    elif inclusion_type == "square":
        half = inclusion_size / 2
        patch = Rectangle(
            (center_x - half, center_y - half),
            inclusion_size,
            inclusion_size,
            fill=True,
            facecolor=NORD_COLORS["nord13"],
            edgecolor=NORD_COLORS["nord6"],
            linewidth=2,
            alpha=0.9,
        )
    elif inclusion_type == "triangle":
        side = inclusion_size
        height = (np.sqrt(3) / 2) * side
        vertices = [
            (center_x - side / 2, center_y - height / 3),
            (center_x + side / 2, center_y - height / 3),
            (center_x, center_y + 2 * height / 3),
        ]
        patch = Polygon(
            vertices,
            closed=True,
            fill=True,
            facecolor=NORD_COLORS["nord13"],
            edgecolor=NORD_COLORS["nord6"],
            linewidth=2,
            alpha=0.9,
        )
    else:
        return

    ax.add_patch(patch)


def plot_kspace_explanation(
    output_file: Path | str,
    inclusion_type: str = "circle",
    center_x: float = 0.6,
    center_y: float = 0.5,
    inclusion_size: float = 0.25,
    domain_size: float = 1.0,
    grid_size: int = 128,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = 200,
    vmax: float | None = None,
) -> None:
    """Generate a figure explaining k-space representation.

    Args:
        output_file: Output file path (PNG).
        inclusion_type: Type of inclusion ("circle", "square", "triangle").
        center_x: x-coordinate of inclusion center.
        center_y: y-coordinate of inclusion center.
        inclusion_size: Characteristic size of inclusion.
        domain_size: Size of the square domain.
        grid_size: Resolution of the k-space grid.
        figsize: Figure size in inches.
        dpi: Output resolution.
        vmax: Maximum value for k-space colorbar. If None, uses data maximum.
    """
    apply_nord_style()

    # Create inclusion image and compute k-space
    image = create_inclusion_image(
        inclusion_type=inclusion_type,
        center_x=center_x,
        center_y=center_y,
        inclusion_size=inclusion_size,
        domain_size=domain_size,
        grid_size=grid_size,
    )

    kspace = fftshift(fft2(image))
    kspace_magnitude = np.abs(kspace)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: Spatial domain
    ax_spatial = axes[0]
    ax_spatial.set_facecolor(NORD_COLORS["nord1"])
    ax_spatial.set_xlim(0, domain_size)
    ax_spatial.set_ylim(0, domain_size)
    ax_spatial.set_aspect("equal")

    # Add inclusion shape
    add_inclusion_patch(ax_spatial, inclusion_type, center_x, center_y, inclusion_size)

    # Add domain boundary
    rect = Rectangle(
        (0, 0), domain_size, domain_size,
        fill=False,
        edgecolor=NORD_COLORS["nord4"],
        linewidth=2,
    )
    ax_spatial.add_patch(rect)

    ax_spatial.set_xlabel("$x$")
    ax_spatial.set_ylabel("$y$")
    ax_spatial.set_title("Spatial Domain $\\rho(x, y)$")

    # Add grid lines
    ax_spatial.grid(True, color=NORD_COLORS["nord3"], linewidth=0.5, alpha=0.5)

    # Right panel: K-space
    ax_kspace = axes[1]

    # Create colormap
    kspace_cmap = create_nord_cmap()

    # Compute frequency axes
    freq = np.fft.fftshift(np.fft.fftfreq(grid_size, d=domain_size / grid_size))
    extent = [freq[0], freq[-1], freq[0], freq[-1]]

    im = ax_kspace.imshow(
        kspace_magnitude,
        extent=extent,
        origin="lower",
        cmap=kspace_cmap,
        aspect="equal",
        vmin=0,
        vmax=vmax,
    )

    ax_kspace.set_xlabel("$k_x$")
    ax_kspace.set_ylabel("$k_y$")
    ax_kspace.set_title("K-Space $\\hat{\\rho}(k_x, k_y)$")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_kspace, fraction=0.046, pad=0.04)
    cbar.set_label("$|\\hat{\\rho}|$", color=NORD_COLORS["nord4"])
    cbar.ax.tick_params(colors=NORD_COLORS["nord4"])
    cbar.outline.set_edgecolor(NORD_COLORS["nord3"])

    plt.tight_layout()

    # Save figure
    output_file = Path(output_file)
    fig.savefig(output_file, dpi=dpi, facecolor=NORD_COLORS["nord0"], edgecolor="none")
    plt.close(fig)

    print(f"Saved k-space explanation figure to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate k-space explanation figure for dissertation"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="kspace_explanation.png",
        help="Output file path",
    )
    parser.add_argument(
        "--inclusion-type",
        type=str,
        choices=["circle", "square", "triangle"],
        default="circle",
        help="Type of inclusion shape",
    )
    parser.add_argument(
        "--center-x",
        type=float,
        default=0.6,
        help="x-coordinate of inclusion center",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        default=0.5,
        help="y-coordinate of inclusion center",
    )
    parser.add_argument(
        "--inclusion-size",
        type=float,
        default=0.25,
        help="Size of inclusion",
    )
    parser.add_argument(
        "--domain-size",
        type=float,
        default=1.0,
        help="Size of the square domain",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="Resolution of k-space grid",
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
        default=200,
        help="Output resolution",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1000.0,
        help="Maximum value for k-space colorbar (default: 1000)",
    )

    args = parser.parse_args()

    plot_kspace_explanation(
        output_file=args.output,
        inclusion_type=args.inclusion_type,
        center_x=args.center_x,
        center_y=args.center_y,
        inclusion_size=args.inclusion_size,
        domain_size=args.domain_size,
        grid_size=args.grid_size,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
