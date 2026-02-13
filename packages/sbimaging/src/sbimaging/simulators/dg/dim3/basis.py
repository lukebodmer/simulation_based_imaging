"""Orthonormal polynomial basis functions for 3D tetrahedra.

Implements Jacobi polynomials and their derivatives for constructing
high-order nodal basis functions on the reference tetrahedron.
"""

import numpy as np
from scipy.linalg import eig
from scipy.special import gamma


def compute_nodes_per_tetrahedron(polynomial_order: int) -> int:
    """Compute number of nodes for a given polynomial order."""
    n = polynomial_order
    return (n + 1) * (n + 2) * (n + 3) // 6


def compute_nodes_per_face(polynomial_order: int) -> int:
    """Compute number of nodes per face for a given polynomial order."""
    n = polynomial_order
    return (n + 1) * (n + 2) // 2


def evaluate_jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """Evaluate the n-th order Jacobi polynomial P_n^{alpha,beta}(x).

    Uses the three-term recurrence relation for numerical stability.

    Args:
        x: Evaluation points.
        alpha: First Jacobi parameter (> -1).
        beta: Second Jacobi parameter (> -1).
        n: Polynomial order.

    Returns:
        Values of P_n^{alpha,beta}(x) at each point.
    """
    pl = np.zeros((n + 1, len(x)))

    gamma0 = (
        2 ** (alpha + beta + 1)
        / (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 1)
    )
    pl[0, :] = 1.0 / np.sqrt(gamma0)

    if n == 0:
        return pl[0, :]

    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    pl[1, :] = ((alpha + beta + 2) * x / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)

    if n == 1:
        return pl[n, :]

    a_old = 2 / (2 + alpha + beta) * np.sqrt(
        (alpha + 1) * (beta + 1) / (alpha + beta + 3)
    )

    for i in range(1, n):
        h1 = 2 * i + alpha + beta
        a_new = (
            2
            / (h1 + 2)
            * np.sqrt(
                (i + 1)
                * (i + 1 + alpha + beta)
                * (i + 1 + alpha)
                * (i + 1 + beta)
                / (h1 + 1)
                / (h1 + 3)
            )
        )
        b_new = -(alpha**2 - beta**2) / h1 / (h1 + 2)
        pl[i + 1, :] = 1 / a_new * (-a_old * pl[i - 1, :] + (x - b_new) * pl[i, :])
        a_old = a_new

    return pl[n, :]


def evaluate_jacobi_polynomial_derivative(
    x: np.ndarray, alpha: float, beta: float, n: int
) -> np.ndarray:
    """Evaluate the derivative of the n-th order Jacobi polynomial.

    Args:
        x: Evaluation points.
        alpha: First Jacobi parameter.
        beta: Second Jacobi parameter.
        n: Polynomial order.

    Returns:
        Values of dP_n^{alpha,beta}/dx at each point.
    """
    if n == 0:
        return np.zeros(len(x))
    return np.sqrt(n * (n + alpha + beta + 1)) * evaluate_jacobi_polynomial(
        x, alpha + 1, beta + 1, n - 1
    )


def compute_jacobi_gauss_quadrature(
    alpha: float, beta: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gauss quadrature points and weights for Jacobi polynomials.

    Args:
        alpha: First Jacobi parameter (> -1).
        beta: Second Jacobi parameter (> -1).
        n: Number of quadrature points minus 1.

    Returns:
        Tuple of (points, weights) arrays.
    """
    x = np.zeros(n + 1)
    w = np.zeros(n + 1)

    if n == 0:
        x[0] = -(alpha - beta) / (alpha + beta + 2)
        w[0] = 2
        return x, w

    h1 = 2 * np.arange(n + 1) + alpha + beta
    j_matrix = np.diag(-0.5 * (alpha**2 - beta**2) / (h1 + 2) / h1) + np.diag(
        2
        / (h1[0:n] + 2)
        * np.sqrt(
            np.arange(1, n + 1)
            * (np.arange(1, n + 1) + alpha + beta)
            * (np.arange(1, n + 1) + alpha)
            * (np.arange(1, n + 1) + beta)
            / (h1[0:n] + 1)
            / (h1[0:n] + 3)
        ),
        1,
    )

    if alpha + beta < 10 * np.finfo(float).eps:
        j_matrix[0, 0] = 0.0

    j_matrix = j_matrix + j_matrix.T

    eigenvalues, eigenvectors = eig(j_matrix)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    x = np.real(eigenvalues)

    w = np.real(
        (eigenvectors[0, :] ** 2)
        * 2 ** (alpha + beta + 1)
        / (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 1)
    )

    return x, w


def compute_gauss_lobatto_points(alpha: float, beta: float, n: int) -> np.ndarray:
    """Compute Gauss-Lobatto quadrature points for Jacobi polynomials.

    Args:
        alpha: First Jacobi parameter.
        beta: Second Jacobi parameter.
        n: Polynomial order.

    Returns:
        Array of quadrature points including endpoints.
    """
    x = np.zeros(n + 1)

    if n == 1:
        x[0] = -1.0
        x[1] = 1.0
        return x

    x_interior, _ = compute_jacobi_gauss_quadrature(alpha + 1, beta + 1, n - 2)
    x[0] = -1
    x[1:n] = x_interior
    x[n] = 1

    return x


def evaluate_simplex_basis_2d(
    r: np.ndarray, s: np.ndarray, i: int, j: int
) -> np.ndarray:
    """Evaluate 2D orthonormal polynomial on simplex at (r,s) of order (i,j).

    Args:
        r: First coordinate array.
        s: Second coordinate array.
        i: Order in first direction.
        j: Order in second direction.

    Returns:
        Values of the basis function at each point.
    """
    a, b = _rs_to_ab(r, s)
    h1 = evaluate_jacobi_polynomial(a, 0, 0, i)
    h2 = evaluate_jacobi_polynomial(b, 2 * i + 1, 0, j)
    return np.sqrt(2.0) * h1 * h2 * (1 - b) ** i


def evaluate_simplex_basis_3d(
    r: np.ndarray, s: np.ndarray, t: np.ndarray, i: int, j: int, k: int
) -> np.ndarray:
    """Evaluate 3D orthonormal polynomial on simplex at (r,s,t) of order (i,j,k).

    Args:
        r: First coordinate array.
        s: Second coordinate array.
        t: Third coordinate array.
        i: Order in first direction.
        j: Order in second direction.
        k: Order in third direction.

    Returns:
        Values of the basis function at each point.
    """
    a, b, c = _rst_to_abc(r, s, t)
    h1 = evaluate_jacobi_polynomial(a, 0, 0, i)
    h2 = evaluate_jacobi_polynomial(b, 2 * i + 1, 0, j)
    h3 = evaluate_jacobi_polynomial(c, 2 * (i + j) + 2, 0, k)
    return 2 * np.sqrt(2) * h1 * h2 * ((1 - b) ** i) * h3 * ((1 - c) ** (i + j))


def evaluate_simplex_basis_3d_gradient(
    r: np.ndarray, s: np.ndarray, t: np.ndarray, i: int, j: int, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate gradient of 3D orthonormal polynomial on simplex.

    Args:
        r: First coordinate array.
        s: Second coordinate array.
        t: Third coordinate array.
        i: Order in first direction.
        j: Order in second direction.
        k: Order in third direction.

    Returns:
        Tuple of (dPdr, dPds, dPdt) gradient components.
    """
    a, b, c = _rst_to_abc(r, s, t)

    fa = evaluate_jacobi_polynomial(a, 0, 0, i)
    dfa = evaluate_jacobi_polynomial_derivative(a, 0, 0, i)
    gb = evaluate_jacobi_polynomial(b, 2 * i + 1, 0, j)
    dgb = evaluate_jacobi_polynomial_derivative(b, 2 * i + 1, 0, j)
    hc = evaluate_jacobi_polynomial(c, 2 * (i + j) + 2, 0, k)
    dhc = evaluate_jacobi_polynomial_derivative(c, 2 * (i + j) + 2, 0, k)

    # r derivative
    vr = dfa * (gb * hc)
    if i > 0:
        vr *= (0.5 * (1 - b)) ** (i - 1)
    if i + j > 0:
        vr *= (0.5 * (1 - c)) ** (i + j - 1)

    # s derivative
    vs = 0.5 * (1 + a) * vr
    tmp = dgb * ((0.5 * (1 - b)) ** i)
    if i > 0:
        tmp += (-0.5 * i) * (gb * ((0.5 * (1 - b)) ** (i - 1)))
    if i + j > 0:
        tmp *= (0.5 * (1 - c)) ** (i + j - 1)
    tmp = fa * (tmp * hc)
    vs += tmp

    # t derivative
    vt = 0.5 * (1 + a) * vr + 0.5 * (1 + b) * tmp
    tmp = dhc * ((0.5 * (1 - c)) ** (i + j))
    if i + j > 0:
        tmp -= 0.5 * (i + j) * (hc * ((0.5 * (1 - c)) ** (i + j - 1)))
    tmp = fa * (gb * tmp)
    tmp *= (0.5 * (1 - b)) ** i
    vt += tmp

    # normalize
    scale = 2 ** (2 * i + j + 1.5)
    vr *= scale
    vs *= scale
    vt *= scale

    return vr, vs, vt


def _rs_to_ab(r: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map from (r,s) to (a,b) coordinates for Jacobi polynomial evaluation."""
    a = np.zeros(len(r))
    for n in range(len(r)):
        if s[n] != 1:
            a[n] = 2 * (1 + r[n]) / (1 - s[n]) - 1
        else:
            a[n] = -1
    return a, s


def _rst_to_abc(
    r: np.ndarray, s: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map from (r,s,t) to (a,b,c) coordinates for Jacobi polynomial evaluation."""
    n_points = len(r)
    a = np.zeros(n_points)
    b = np.zeros(n_points)
    c = np.zeros(n_points)

    for n in range(n_points):
        if s[n] + t[n] != 0:
            a[n] = 2 * (1 + r[n]) / (-s[n] - t[n]) - 1
        else:
            a[n] = -1

        if t[n] != 1:
            b[n] = 2 * (1 + s[n]) / (1 - t[n]) - 1
        else:
            b[n] = -1

        c[n] = t[n]

    return a, b, c
