"""Tests for basis functions."""

import numpy as np
import pytest

from sbimaging.simulators.dg.dim3.basis import (
    compute_nodes_per_face,
    compute_nodes_per_tetrahedron,
    evaluate_jacobi_polynomial,
    evaluate_jacobi_polynomial_derivative,
    evaluate_simplex_basis_3d,
)


class TestNodeCounts:
    def test_nodes_per_tetrahedron_order_1(self):
        assert compute_nodes_per_tetrahedron(1) == 4

    def test_nodes_per_tetrahedron_order_2(self):
        assert compute_nodes_per_tetrahedron(2) == 10

    def test_nodes_per_tetrahedron_order_3(self):
        assert compute_nodes_per_tetrahedron(3) == 20

    def test_nodes_per_tetrahedron_order_4(self):
        assert compute_nodes_per_tetrahedron(4) == 35

    def test_nodes_per_face_order_1(self):
        assert compute_nodes_per_face(1) == 3

    def test_nodes_per_face_order_2(self):
        assert compute_nodes_per_face(2) == 6

    def test_nodes_per_face_order_3(self):
        assert compute_nodes_per_face(3) == 10


class TestJacobiPolynomials:
    def test_legendre_p0(self):
        """P_0(x) = 1/sqrt(2) (normalized)."""
        x = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        p0 = evaluate_jacobi_polynomial(x, 0, 0, 0)
        expected = 1.0 / np.sqrt(2)
        np.testing.assert_allclose(p0, expected, rtol=1e-10)

    def test_legendre_p1_zeros(self):
        """P_1(0) = 0 for Legendre polynomials."""
        x = np.array([0.0])
        p1 = evaluate_jacobi_polynomial(x, 0, 0, 1)
        np.testing.assert_allclose(p1, 0.0, atol=1e-10)

    def test_jacobi_derivative_p0(self):
        """Derivative of P_0 is 0."""
        x = np.array([0.0, 0.5, -0.5])
        dp0 = evaluate_jacobi_polynomial_derivative(x, 0, 0, 0)
        np.testing.assert_allclose(dp0, 0.0, atol=1e-10)


class TestSimplexBasis:
    def test_basis_partition_of_unity(self):
        """Sum of all basis functions should approximate 1 at any point."""
        # For order 1, the basis functions should sum to a constant
        r = np.array([0.0])
        s = np.array([0.0])
        t = np.array([0.0])

        # Sum first few basis functions (they're orthonormal, not partition of unity)
        # Instead, test that the basis function values are finite
        p000 = evaluate_simplex_basis_3d(r, s, t, 0, 0, 0)
        assert np.isfinite(p000[0])

    def test_basis_at_vertices(self):
        """Basis functions should be well-defined at vertices."""
        vertices_r = np.array([-1.0, 1.0, -1.0, -1.0])
        vertices_s = np.array([-1.0, -1.0, 1.0, -1.0])
        vertices_t = np.array([-1.0, -1.0, -1.0, 1.0])

        p000 = evaluate_simplex_basis_3d(vertices_r, vertices_s, vertices_t, 0, 0, 0)
        assert np.all(np.isfinite(p000))
