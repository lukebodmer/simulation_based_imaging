"""Warp & Blend node generation for 3D tetrahedra.

Implements the Hesthaven-Warburton algorithm for generating
well-conditioned interpolation nodes on the reference tetrahedron.

Reference: Hesthaven & Warburton, "Nodal Discontinuous Galerkin Methods", 2008
"""

import numpy as np

from sbimaging.simulators.dg.dim3.basis import (
    compute_gauss_lobatto_points,
    compute_nodes_per_tetrahedron,
)


# Optimal blending parameters from Hesthaven-Warburton
_ALPHA_OPTIMAL = [
    0, 0, 0, 0.1002, 1.1332, 1.5608, 1.3413, 1.2577, 1.1603,
    1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655,
]


def compute_warp_and_blend_nodes(
    polynomial_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Warp & Blend nodes on the reference tetrahedron.

    Args:
        polynomial_order: Polynomial order of the approximation.

    Returns:
        Tuple of (r, s, t) coordinate arrays on the reference tetrahedron
        with vertices at (-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1).
    """
    n_nodes = compute_nodes_per_tetrahedron(polynomial_order)
    p = polynomial_order

    # Create equidistributed nodes
    equi_r, equi_s, equi_t = _compute_equidistributed_nodes(p, n_nodes)

    # Get barycentric coordinates
    l1 = (1 + equi_t) / 2
    l2 = (1 + equi_s) / 2
    l3 = -(1 + equi_r + equi_s + equi_t) / 2
    l4 = (1 + equi_r) / 2

    # Reshape for matrix operations
    l1 = l1.reshape(-1, 1)
    l2 = l2.reshape(-1, 1)
    l3 = l3.reshape(-1, 1)
    l4 = l4.reshape(-1, 1)

    # Get equilateral tetrahedron vertices
    v1 = np.array([[-1, -1 / np.sqrt(3), -1 / np.sqrt(6)]])
    v2 = np.array([[1, -1 / np.sqrt(3), -1 / np.sqrt(6)]])
    v3 = np.array([[0, 2 / np.sqrt(3), -1 / np.sqrt(6)]])
    v4 = np.array([[0, 0, 3 / np.sqrt(6)]])

    # Find tangent vectors at each face
    t1, t2 = _compute_face_tangents(v1, v2, v3, v4)

    # Form undeformed coordinates on equilateral tetrahedron
    xyz = l3 @ v1 + l4 @ v2 + l2 @ v3 + l1 @ v4

    # Apply warp and blend
    xyz = _warp_and_blend(p, xyz, t1, t2, l1, l2, l3, l4)

    # Map from equilateral to reference tetrahedron
    r, s, t = _map_equilateral_to_reference(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    return r, s, t


def find_face_node_indices(
    r: np.ndarray, s: np.ndarray, t: np.ndarray, tolerance: float = 1e-6
) -> np.ndarray:
    """Find indices of nodes on each face of the reference tetrahedron.

    Uses a larger tolerance to account for warp & blend effects.

    Args:
        r: First coordinate array.
        s: Second coordinate array.
        t: Third coordinate array.
        tolerance: Tolerance for face detection.

    Returns:
        Array of indices, ordered by face (face 0, face 1, face 2, face 3).
    """
    face_0 = np.where(np.abs(1 + t) < tolerance)[0]  # t = -1
    face_1 = np.where(np.abs(1 + s) < tolerance)[0]  # s = -1
    face_2 = np.where(np.abs(1 + r + s + t) < tolerance)[0]  # r+s+t = -1
    face_3 = np.where(np.abs(1 + r) < tolerance)[0]  # r = -1

    return np.concatenate((face_0, face_1, face_2, face_3))


def _compute_equidistributed_nodes(
    polynomial_order: int, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute equidistributed nodes on the reference tetrahedron."""
    x = np.zeros(n_nodes)
    y = np.zeros(n_nodes)
    z = np.zeros(n_nodes)
    p = polynomial_order

    node_index = 0
    for n in range(1, p + 2):
        for m in range(1, p + 3 - n):
            for q in range(1, p + 4 - n - m):
                x[node_index] = -1 + (q - 1) * 2 / p
                y[node_index] = -1 + (m - 1) * 2 / p
                z[node_index] = -1 + (n - 1) * 2 / p
                node_index += 1

    return x, y, z


def _compute_face_tangents(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute orthogonal tangent vectors for each face."""
    t1 = np.zeros((4, 3))
    t2 = np.zeros((4, 3))

    t1[0, :] = v2 - v1
    t1[1, :] = v2 - v1
    t1[2, :] = v3 - v2
    t1[3, :] = v3 - v1

    t2[0, :] = v3 - 0.5 * (v1 + v2)
    t2[1, :] = v4 - 0.5 * (v1 + v2)
    t2[2, :] = v4 - 0.5 * (v2 + v3)
    t2[3, :] = v4 - 0.5 * (v1 + v3)

    for n in range(4):
        t1[n, :] = t1[n, :] / np.linalg.norm(t1[n, :])
        t2[n, :] = t2[n, :] / np.linalg.norm(t2[n, :])

    return t1, t2


def _get_alpha(n: int) -> float:
    """Get optimal blending parameter for polynomial order n."""
    if n <= 15 and n > 0:
        return _ALPHA_OPTIMAL[n - 1]
    elif n > 15:
        return 1.0
    return 0.0


def _warp_and_blend(
    p: int,
    xyz: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
    l4: np.ndarray,
) -> np.ndarray:
    """Apply warp and blend transformation to equidistributed nodes."""
    alpha = _get_alpha(p)
    shift = np.zeros_like(xyz)
    tol = 1e-10

    for face in range(4):
        la, lb, lc, ld = _select_barycentric_for_face(face, l1, l2, l3, l4)

        # Compute warp tangential to the face
        warp1, warp2 = _warp_shift_face_3d(p, alpha, lb, lc, ld)

        # Compute volume blending
        blend = lb * lc * ld

        # Modify linear blend
        denominator = (lb + 0.5 * la) * (lc + 0.5 * la) * (ld + 0.5 * la)
        valid = np.where(denominator > tol)[0]
        blend[valid] = (1 + (alpha * la[valid]) ** 2) * blend[valid] / denominator[valid]

        # Compute warp & blend
        shift += (blend * warp1) @ t1[face, :].reshape(1, 3) + (
            blend * warp2
        ) @ t2[face, :].reshape(1, 3)

        # Fix face warp for edge/vertex nodes
        on_face = np.where(
            (la < tol)
            & (
                (lb > tol).astype(int)
                + (lc > tol).astype(int)
                + (ld > tol).astype(int)
                < 3
            )
        )[0]
        shift[on_face, :] = warp1[on_face, :] * t1[face, :] + warp2[on_face, :] * t2[face, :]

    return xyz + shift


def _select_barycentric_for_face(
    face: int,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
    l4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select correct barycentric coordinates for each face."""
    if face == 0:
        return l1, l2, l3, l4
    elif face == 1:
        return l2, l1, l3, l4
    elif face == 2:
        return l3, l1, l4, l2
    else:  # face == 3
        return l4, l1, l3, l2


def _warp_shift_face_3d(
    p: int, pval: float, l2: np.ndarray, l3: np.ndarray, l4: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute warp factor used in creating 3D Warp & Blend nodes."""
    dtan1, dtan2 = _eval_shift(p, pval, l2, l3, l4)
    return dtan1, dtan2


def _eval_shift(
    p: int, pval: float, l1: np.ndarray, l2: np.ndarray, l3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two-dimensional Warp & Blend transform."""
    # Compute Gauss-Lobatto-Legendre node distribution
    gauss_x = -compute_gauss_lobatto_points(0, 0, p)

    # Compute blending function at each node for each edge
    blend1 = l2 * l3
    blend2 = l1 * l3
    blend3 = l1 * l2

    # Amount of warp for each node, for each edge
    warpfactor1 = 4 * _compute_edge_warp(p, gauss_x, l3 - l2)
    warpfactor2 = 4 * _compute_edge_warp(p, gauss_x, l1 - l3)
    warpfactor3 = 4 * _compute_edge_warp(p, gauss_x, l2 - l1)

    # Combine blend & warp
    warp1 = blend1 * warpfactor1 * (1 + (pval * l1) ** 2)
    warp2 = blend2 * warpfactor2 * (1 + (pval * l2) ** 2)
    warp3 = blend3 * warpfactor3 * (1 + (pval * l3) ** 2)

    # Evaluate shift in equilateral triangle
    dx = warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
    dy = np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3

    return dx, dy


def _compute_edge_warp(
    p: int, xnodes: np.ndarray, xout: np.ndarray
) -> np.ndarray:
    """Compute one-dimensional edge warping function."""
    warp = np.zeros_like(xout)

    xeq = np.zeros(p + 1)
    for i in range(p + 1):
        xeq[i] = -1 + 2 * (p - i) / p

    for i in range(1, p + 2):
        d = xnodes[i - 1] - xeq[i - 1]

        for j in range(2, p + 1):
            if i != j:
                d = d * (xout - xeq[j - 1]) / (xeq[i - 1] - xeq[j - 1])

        if i != 1:
            d = -d / (xeq[i - 1] - xeq[0])

        if i != (p + 1):
            d = d / (xeq[i - 1] - xeq[p])

        warp += d

    return warp


def _map_equilateral_to_reference(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map from equilateral tetrahedron to reference tetrahedron."""
    v1 = np.array([-1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
    v2 = np.array([1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
    v3 = np.array([0, 2 / np.sqrt(3), -1 / np.sqrt(6)])
    v4 = np.array([0, 0, 3 / np.sqrt(6)])

    rhs = (np.array([x, y, z]).T - np.array([0.5 * (v2 + v3 + v4 - v1)])).T
    a_matrix = np.column_stack([0.5 * (v2 - v1), 0.5 * (v3 - v1), 0.5 * (v4 - v1)])
    rst = np.linalg.solve(a_matrix, rhs)

    return rst[0, :].T, rst[1, :].T, rst[2, :].T
