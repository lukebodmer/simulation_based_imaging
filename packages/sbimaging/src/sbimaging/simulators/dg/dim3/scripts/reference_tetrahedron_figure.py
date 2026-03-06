#!/usr/bin/env python
"""Generate a figure explaining the reference tetrahedron mapping.

Shows the reference tetrahedron with nodal points on the left and a
mapped physical tetrahedron on the right, illustrating the isoparametric
mapping used in the DG method.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sbimaging.simulators.dg.dim3.nodes import compute_warp_and_blend_nodes


# Light color scheme for dissertation
LIGHT_COLORS = {
    "background": "#ffffff",
    "text": "#2c3e50",
    "axis": "#7f8c8d",
    "grid": "#bdc3c7",
    "reference_face": "#3498db",  # Blue for reference
    "physical_face": "#e74c3c",  # Red for physical
    "reference_node": "#2980b9",  # Darker blue for nodes
    "physical_node": "#c0392b",  # Darker red for nodes
    "vertex": "#2c3e50",  # Dark for vertices
    "edge": "#34495e",  # Dark gray for edges
    "arrow": "#27ae60",  # Green for mapping arrow
}


def apply_light_style() -> None:
    """Apply light color scheme for dissertation figures."""
    plt.rcParams.update({
        "figure.facecolor": LIGHT_COLORS["background"],
        "figure.edgecolor": LIGHT_COLORS["background"],
        "axes.facecolor": LIGHT_COLORS["background"],
        "axes.edgecolor": LIGHT_COLORS["axis"],
        "axes.labelcolor": LIGHT_COLORS["text"],
        "axes.titlecolor": LIGHT_COLORS["text"],
        "axes.titleweight": "medium",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.color": LIGHT_COLORS["text"],
        "ytick.color": LIGHT_COLORS["text"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": LIGHT_COLORS["text"],
        "font.family": "serif",
        "savefig.facecolor": LIGHT_COLORS["background"],
        "savefig.edgecolor": LIGHT_COLORS["background"],
    })


def get_reference_tetrahedron_vertices() -> np.ndarray:
    """Get vertices of the reference tetrahedron.

    Returns:
        (4, 3) array of vertex coordinates.
    """
    return np.array([
        [-1, -1, -1],  # Vertex 0
        [1, -1, -1],   # Vertex 1
        [-1, 1, -1],   # Vertex 2
        [-1, -1, 1],   # Vertex 3
    ], dtype=np.float64)


def map_reference_to_physical(
    r: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
    physical_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map reference coordinates to physical coordinates.

    Uses the same isoparametric mapping as the DG code.

    Args:
        r, s, t: Reference coordinates.
        physical_vertices: (4, 3) array of physical vertex coordinates.

    Returns:
        Tuple of (x, y, z) physical coordinates.
    """
    va = physical_vertices[0]
    vb = physical_vertices[1]
    vc = physical_vertices[2]
    vd = physical_vertices[3]

    x = 0.5 * (-(1 + r + s + t) * va[0] + (1 + r) * vb[0] + (1 + s) * vc[0] + (1 + t) * vd[0])
    y = 0.5 * (-(1 + r + s + t) * va[1] + (1 + r) * vb[1] + (1 + s) * vc[1] + (1 + t) * vd[1])
    z = 0.5 * (-(1 + r + s + t) * va[2] + (1 + r) * vb[2] + (1 + s) * vc[2] + (1 + t) * vd[2])

    return x, y, z


def create_tetrahedron_faces(vertices: np.ndarray) -> list[np.ndarray]:
    """Create face polygons for a tetrahedron.

    Args:
        vertices: (4, 3) array of vertex coordinates.

    Returns:
        List of 4 face arrays, each (3, 3) for the 3 vertices of each triangular face.
    """
    # Face definitions (vertex indices for each face)
    face_indices = [
        [0, 1, 2],  # Face 0: t = -1 (bottom)
        [0, 1, 3],  # Face 1: s = -1
        [1, 2, 3],  # Face 2: r + s + t = -1 (slanted)
        [0, 2, 3],  # Face 3: r = -1
    ]
    return [vertices[idx] for idx in face_indices]


def plot_tetrahedron(
    ax: Axes3D,
    vertices: np.ndarray,
    nodes: np.ndarray | None = None,
    face_color: str = LIGHT_COLORS["reference_face"],
    node_color: str = LIGHT_COLORS["reference_node"],
    edge_color: str = LIGHT_COLORS["edge"],
    face_alpha: float = 0.15,
    node_size: float = 30,
    show_vertex_labels: bool = True,
    vertex_labels: list[str] | None = None,
    label_offset: float | None = None,
) -> None:
    """Plot a tetrahedron with optional nodal points.

    Args:
        ax: 3D axes to plot on.
        vertices: (4, 3) array of vertex coordinates.
        nodes: (N, 3) array of nodal point coordinates, or None.
        face_color: Color for tetrahedron faces.
        node_color: Color for nodal points.
        edge_color: Color for edges.
        face_alpha: Transparency of faces.
        node_size: Size of nodal point markers.
        show_vertex_labels: Whether to label vertices.
        vertex_labels: Custom vertex labels, or None for default.
    """
    # Draw faces
    faces = create_tetrahedron_faces(vertices)
    face_collection = Poly3DCollection(
        faces,
        alpha=face_alpha,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=1.5,
    )
    ax.add_collection3d(face_collection)

    # Draw edges more prominently
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3),
    ]
    for i, j in edges:
        ax.plot3D(
            [vertices[i, 0], vertices[j, 0]],
            [vertices[i, 1], vertices[j, 1]],
            [vertices[i, 2], vertices[j, 2]],
            color=edge_color,
            linewidth=2,
            zorder=5,
        )

    # Draw vertices
    ax.scatter3D(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        color=LIGHT_COLORS["vertex"],
        s=80,
        zorder=10,
        depthshade=False,
    )

    # Label vertices
    if show_vertex_labels:
        if vertex_labels is None:
            vertex_labels = [
                r"$(-1,-1,-1)$",
                r"$(1,-1,-1)$",
                r"$(-1,1,-1)$",
                r"$(-1,-1,1)$",
            ]
        # Compute offset based on tetrahedron size if not provided
        if label_offset is None:
            tet_size = np.max(vertices.max(axis=0) - vertices.min(axis=0))
            label_offset = tet_size * 0.08
        for i, (v, label) in enumerate(zip(vertices, vertex_labels)):
            ax.text(
                v[0] + label_offset,
                v[1] + label_offset,
                v[2] + label_offset,
                label,
                fontsize=9,
                color=LIGHT_COLORS["text"],
                ha="left",
            )

    # Draw nodal points
    if nodes is not None:
        ax.scatter3D(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            color=node_color,
            s=node_size,
            alpha=0.8,
            zorder=8,
            depthshade=False,
        )


def create_reference_tetrahedron_figure(
    output_file: Path | str,
    polynomial_order: int = 3,
    figsize: tuple[float, float] = (14, 6),
    dpi: int = 200,
    elevation: float = 20,
    azimuth: float = -60,
) -> None:
    """Generate the reference tetrahedron mapping figure.

    Args:
        output_file: Output file path (PNG or PDF).
        polynomial_order: Polynomial order for nodal points.
        figsize: Figure size in inches.
        dpi: Output resolution.
        elevation: Camera elevation angle.
        azimuth: Camera azimuth angle.
    """
    apply_light_style()

    # Compute nodal points on reference tetrahedron
    r, s, t = compute_warp_and_blend_nodes(polynomial_order)
    ref_nodes = np.column_stack([r, s, t])
    ref_vertices = get_reference_tetrahedron_vertices()

    # Define a physical tetrahedron - irregular shape typical of unstructured mesh
    # Small, skewed tetrahedron like you'd find near curved geometry
    phys_vertices = np.array([
        [0.72, 0.48, 0.45],   # Base vertex
        [0.88, 0.51, 0.43],   # Short edge along x
        [0.78, 0.64, 0.42],   # Offset in y, slightly flat
        [0.81, 0.55, 0.59],   # Peak - not centered, creating asymmetry
    ], dtype=np.float64)

    # Map nodes to physical tetrahedron
    x, y, z = map_reference_to_physical(r, s, t, phys_vertices)
    phys_nodes = np.column_stack([x, y, z])

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Left panel: Reference tetrahedron
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_facecolor(LIGHT_COLORS["background"])

    plot_tetrahedron(
        ax1,
        ref_vertices,
        nodes=ref_nodes,
        face_color=LIGHT_COLORS["reference_face"],
        node_color=LIGHT_COLORS["reference_node"],
    )

    ax1.set_xlabel(r"$r$", fontsize=12, labelpad=8)
    ax1.set_ylabel(r"$s$", fontsize=12, labelpad=8)
    ax1.set_zlabel(r"$t$", fontsize=12, labelpad=8)
    ax1.set_title(f"Reference Tetrahedron\n({len(r)} nodes, order {polynomial_order})", fontsize=14, pad=10)

    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_zlim(-1.3, 1.3)
    ax1.view_init(elev=elevation, azim=azimuth + 90)

    # Style the 3D axes
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor(LIGHT_COLORS["grid"])
    ax1.yaxis.pane.set_edgecolor(LIGHT_COLORS["grid"])
    ax1.zaxis.pane.set_edgecolor(LIGHT_COLORS["grid"])
    ax1.grid(True, alpha=0.3, color=LIGHT_COLORS["grid"])

    # Right panel: Physical tetrahedron
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_facecolor(LIGHT_COLORS["background"])

    # Physical vertex labels
    phys_labels = [
        r"$\mathbf{v}_a$",
        r"$\mathbf{v}_b$",
        r"$\mathbf{v}_c$",
        r"$\mathbf{v}_d$",
    ]

    plot_tetrahedron(
        ax2,
        phys_vertices,
        nodes=phys_nodes,
        face_color=LIGHT_COLORS["physical_face"],
        node_color=LIGHT_COLORS["physical_node"],
        vertex_labels=phys_labels,
    )

    ax2.set_xlabel(r"$x$", fontsize=12, labelpad=8)
    ax2.set_ylabel(r"$y$", fontsize=12, labelpad=8)
    ax2.set_zlabel(r"$z$", fontsize=12, labelpad=8)
    ax2.set_title("Physical Tetrahedron\n(mapped nodes)", fontsize=14, pad=10)

    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_zlim(0.0, 1.0)
    ax2.view_init(elev=elevation, azim=azimuth)

    # Style the 3D axes
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_edgecolor(LIGHT_COLORS["grid"])
    ax2.yaxis.pane.set_edgecolor(LIGHT_COLORS["grid"])
    ax2.zaxis.pane.set_edgecolor(LIGHT_COLORS["grid"])
    ax2.grid(True, alpha=0.3, color=LIGHT_COLORS["grid"])

    plt.tight_layout()

    # Add mapping arrow annotation between plots (after tight_layout)
    fig.text(
        0.5, 0.62,
        r"$\Phi: (r,s,t) \mapsto (x,y,z)$",
        fontsize=14,
        ha="center",
        va="center",
        color=LIGHT_COLORS["text"],
        fontweight="bold",
    )
    fig.text(
        0.5, 0.56,
        r"$\longrightarrow$",
        fontsize=24,
        ha="center",
        va="center",
        color=LIGHT_COLORS["text"],
    )

    # Adjust subplot positions to make room for labels and center annotation
    plt.subplots_adjust(left=0.05, right=0.92, wspace=0.5)

    # Save figure
    output_file = Path(output_file)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", pad_inches=0.35,
                facecolor=LIGHT_COLORS["background"], edgecolor="none")
    plt.close(fig)

    print(f"Saved reference tetrahedron figure to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference tetrahedron mapping figure for dissertation"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="reference_tetrahedron.png",
        help="Output file path",
    )
    parser.add_argument(
        "-p", "--polynomial-order",
        type=int,
        default=3,
        help="Polynomial order for nodal points (default: 3)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[14, 6],
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
        "--elevation",
        type=float,
        default=20,
        help="Camera elevation angle (default: 20)",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=-60,
        help="Camera azimuth angle (default: -60)",
    )

    args = parser.parse_args()

    create_reference_tetrahedron_figure(
        output_file=args.output,
        polynomial_order=args.polynomial_order,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )


if __name__ == "__main__":
    main()
