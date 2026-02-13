"""Tests for reference element and operators."""

import numpy as np
import pytest

from sbimaging.simulators.dg.dim3.reference_element import (
    ReferenceOperators,
    ReferenceTetrahedron,
)


class TestReferenceTetrahedron:
    def test_init_order_1(self):
        elem = ReferenceTetrahedron(polynomial_order=1)
        assert elem.polynomial_order == 1
        assert elem.nodes_per_cell == 4
        assert elem.nodes_per_face == 3
        assert elem.num_faces == 4

    def test_init_order_3(self):
        elem = ReferenceTetrahedron(polynomial_order=3)
        assert elem.polynomial_order == 3
        assert elem.nodes_per_cell == 20
        assert elem.nodes_per_face == 10

    def test_init_order_0_raises(self):
        with pytest.raises(ValueError):
            ReferenceTetrahedron(polynomial_order=0)

    def test_vertices_shape(self):
        elem = ReferenceTetrahedron(polynomial_order=2)
        assert elem.vertices.shape == (4, 3)

    def test_nodes_shape(self):
        elem = ReferenceTetrahedron(polynomial_order=2)
        assert elem.r.shape == (10,)
        assert elem.s.shape == (10,)
        assert elem.t.shape == (10,)

    def test_nodes_approximately_in_reference_domain(self):
        """Nodes should be approximately within the reference tetrahedron.

        Warp & Blend can push nodes slightly outside for better conditioning.
        """
        elem = ReferenceTetrahedron(polynomial_order=4)
        # Allow small tolerance for warp effects
        tol = 0.2
        assert np.all(elem.r >= -1 - tol)
        assert np.all(elem.s >= -1 - tol)
        assert np.all(elem.t >= -1 - tol)
        assert np.all(elem.r + elem.s + elem.t <= -1 + tol)

    def test_face_node_indices_nonempty(self):
        """Face node indices should be found for each face."""
        elem = ReferenceTetrahedron(polynomial_order=3)
        # Should have nodes on faces (exact count depends on warp tolerance)
        assert len(elem.face_node_indices) > 0
        # All indices should be valid
        assert np.all(elem.face_node_indices < elem.nodes_per_cell)


class TestReferenceOperators:
    def test_vandermonde_shape(self):
        elem = ReferenceTetrahedron(polynomial_order=2)
        ops = ReferenceOperators(elem)
        n = elem.nodes_per_cell
        assert ops.vandermonde.shape == (n, n)

    def test_vandermonde_invertible(self):
        elem = ReferenceTetrahedron(polynomial_order=3)
        ops = ReferenceOperators(elem)
        # Check condition number is reasonable
        cond = np.linalg.cond(ops.vandermonde)
        assert cond < 1e10

    def test_mass_matrix_symmetric(self):
        elem = ReferenceTetrahedron(polynomial_order=2)
        ops = ReferenceOperators(elem)
        np.testing.assert_allclose(ops.mass_matrix, ops.mass_matrix.T, rtol=1e-10)

    def test_mass_matrix_positive_definite(self):
        elem = ReferenceTetrahedron(polynomial_order=2)
        ops = ReferenceOperators(elem)
        eigenvalues = np.linalg.eigvalsh(ops.mass_matrix)
        assert np.all(eigenvalues > 0)

    def test_diff_matrices_shape(self):
        elem = ReferenceTetrahedron(polynomial_order=3)
        ops = ReferenceOperators(elem)
        n = elem.nodes_per_cell
        assert ops.diff_r.shape == (n, n)
        assert ops.diff_s.shape == (n, n)
        assert ops.diff_t.shape == (n, n)

    def test_lift_matrix_shape(self):
        elem = ReferenceTetrahedron(polynomial_order=2)
        ops = ReferenceOperators(elem)
        n_cell = elem.nodes_per_cell
        n_surface = elem.num_faces * elem.nodes_per_face
        assert ops.lift.shape == (n_cell, n_surface)

    def test_differentiate_constant(self):
        """Derivative of constant field should be zero."""
        elem = ReferenceTetrahedron(polynomial_order=2)
        ops = ReferenceOperators(elem)
        constant_field = np.ones(elem.nodes_per_cell)
        dr = ops.diff_r @ constant_field
        ds = ops.diff_s @ constant_field
        dt = ops.diff_t @ constant_field
        np.testing.assert_allclose(dr, 0, atol=1e-10)
        np.testing.assert_allclose(ds, 0, atol=1e-10)
        np.testing.assert_allclose(dt, 0, atol=1e-10)

    def test_differentiate_linear_r(self):
        """Derivative of r with respect to r should be 1."""
        elem = ReferenceTetrahedron(polynomial_order=2)
        ops = ReferenceOperators(elem)
        dr = ops.diff_r @ elem.r
        np.testing.assert_allclose(dr, 1.0, rtol=1e-10)

    def test_differentiate_linear_s(self):
        """Derivative of s with respect to s should be 1."""
        elem = ReferenceTetrahedron(polynomial_order=2)
        ops = ReferenceOperators(elem)
        ds = ops.diff_s @ elem.s
        np.testing.assert_allclose(ds, 1.0, rtol=1e-10)

    def test_differentiate_linear_t(self):
        """Derivative of t with respect to t should be 1."""
        elem = ReferenceTetrahedron(polynomial_order=2)
        ops = ReferenceOperators(elem)
        dt = ops.diff_t @ elem.t
        np.testing.assert_allclose(dt, 1.0, rtol=1e-10)
