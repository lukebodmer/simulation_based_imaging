"""Reference element for 3D tetrahedral DG methods.

Provides the Lagrange finite element on the reference tetrahedron with
vertices at (-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1).
"""

import numpy as np

from sbimaging.simulators.dg.dim3.basis import (
    compute_nodes_per_face,
    compute_nodes_per_tetrahedron,
    evaluate_simplex_basis_2d,
    evaluate_simplex_basis_3d,
    evaluate_simplex_basis_3d_gradient,
)
from sbimaging.simulators.dg.dim3.nodes import (
    compute_warp_and_blend_nodes,
    find_face_node_indices,
)


class ReferenceTetrahedron:
    """Reference tetrahedron with high-order Lagrange basis.

    Attributes:
        polynomial_order: Order of polynomial approximation.
        nodes_per_cell: Number of interpolation nodes per tetrahedron.
        nodes_per_face: Number of nodes per triangular face.
        num_faces: Number of faces (always 4 for tetrahedron).
        r, s, t: Node coordinates on reference element.
        face_node_indices: Indices of nodes on each face.
        vertices: Vertex coordinates of reference tetrahedron.
    """

    def __init__(self, polynomial_order: int):
        """Initialize reference tetrahedron.

        Args:
            polynomial_order: Order of polynomial approximation (p >= 1).
        """
        if polynomial_order < 1:
            raise ValueError("Polynomial order must be at least 1")

        self.polynomial_order = polynomial_order
        self.nodes_per_cell = compute_nodes_per_tetrahedron(polynomial_order)
        self.nodes_per_face = compute_nodes_per_face(polynomial_order)
        self.num_faces = 4

        self.vertices = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ])

        self.r, self.s, self.t = compute_warp_and_blend_nodes(polynomial_order)
        self.face_node_indices = find_face_node_indices(self.r, self.s, self.t)


class ReferenceOperators:
    """Precomputed operators on the reference tetrahedron.

    Includes Vandermonde matrices, differentiation matrices, mass matrix,
    and the lift matrix for DG surface integrals.

    Attributes:
        vandermonde: 3D Vandermonde matrix V.
        vandermonde_r: Derivative of V with respect to r.
        vandermonde_s: Derivative of V with respect to s.
        vandermonde_t: Derivative of V with respect to t.
        mass_matrix: Mass matrix M = (V V^T)^{-1}.
        diff_r: Differentiation matrix D_r.
        diff_s: Differentiation matrix D_s.
        diff_t: Differentiation matrix D_t.
        lift: Surface-to-volume lift operator.
    """

    def __init__(self, reference_element: ReferenceTetrahedron):
        """Initialize reference operators.

        Args:
            reference_element: The reference tetrahedron to compute operators for.
        """
        self._element = reference_element
        self._build_vandermonde_3d()
        self._build_vandermonde_gradient_3d()
        self._build_mass_matrix()
        self._build_differentiation_matrices()
        self._build_lift_matrix()

    def _build_vandermonde_3d(self) -> None:
        """Build the 3D Vandermonde matrix."""
        n = self._element.polynomial_order
        n_nodes = self._element.nodes_per_cell
        r, s, t = self._element.r, self._element.s, self._element.t

        v = np.zeros((n_nodes, n_nodes))
        col = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    v[:, col] = evaluate_simplex_basis_3d(r, s, t, i, j, k)
                    col += 1

        self.vandermonde = v
        self._inv_vandermonde = np.linalg.inv(v)

    def _build_vandermonde_gradient_3d(self) -> None:
        """Build gradient of the 3D Vandermonde matrix."""
        n = self._element.polynomial_order
        n_nodes = self._element.nodes_per_cell
        r, s, t = self._element.r, self._element.s, self._element.t

        vr = np.zeros((n_nodes, n_nodes))
        vs = np.zeros((n_nodes, n_nodes))
        vt = np.zeros((n_nodes, n_nodes))

        col = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    vr[:, col], vs[:, col], vt[:, col] = evaluate_simplex_basis_3d_gradient(
                        r, s, t, i, j, k
                    )
                    col += 1

        self.vandermonde_r = vr
        self.vandermonde_s = vs
        self.vandermonde_t = vt

    def _build_mass_matrix(self) -> None:
        """Build the mass matrix M = (V^{-T})(V^{-1})."""
        inv_v = self._inv_vandermonde
        self.mass_matrix = inv_v.T @ inv_v

    def _build_differentiation_matrices(self) -> None:
        """Build differentiation matrices D_r, D_s, D_t."""
        inv_v = self._inv_vandermonde
        self.diff_r = self.vandermonde_r @ inv_v
        self.diff_s = self.vandermonde_s @ inv_v
        self.diff_t = self.vandermonde_t @ inv_v

    def _build_lift_matrix(self) -> None:
        """Build the surface-to-volume lift operator."""
        n = self._element.polynomial_order
        n_nodes = self._element.nodes_per_cell
        n_face = self._element.nodes_per_face
        num_faces = self._element.num_faces
        face_indices = self._element.face_node_indices.reshape(4, -1).T
        r, s, t = self._element.r, self._element.s, self._element.t
        v = self.vandermonde

        epsilon = np.zeros((n_nodes, num_faces * n_face))

        for face in range(num_faces):
            if face == 0:
                face_r = r[face_indices[:, 0]]
                face_s = s[face_indices[:, 0]]
            elif face == 1:
                face_r = r[face_indices[:, 1]]
                face_s = t[face_indices[:, 1]]
            elif face == 2:
                face_r = s[face_indices[:, 2]]
                face_s = t[face_indices[:, 2]]
            else:  # face == 3
                face_r = s[face_indices[:, 3]]
                face_s = t[face_indices[:, 3]]

            v2d = self._build_vandermonde_2d(face_r, face_s, n)
            face_mass = np.linalg.inv(v2d @ v2d.T)

            row_idx = face_indices[:, face]
            col_idx = np.arange(face * n_face, (face + 1) * n_face)
            epsilon[row_idx[:, np.newaxis], col_idx] += face_mass

        self.lift = v @ (v.T @ epsilon)

    def _build_vandermonde_2d(
        self, r: np.ndarray, s: np.ndarray, n: int
    ) -> np.ndarray:
        """Build 2D Vandermonde matrix for face integration."""
        n_face = self._element.nodes_per_face
        v2d = np.zeros((n_face, n_face))

        col = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                v2d[:, col] = evaluate_simplex_basis_2d(r, s, i, j)
                col += 1

        return v2d
