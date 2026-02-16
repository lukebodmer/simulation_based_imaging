"""2D k-space representation for inverse problems.

Converts inclusion parameters (shape, position, size) to frequency domain
representation suitable for neural network training.
"""

from dataclasses import dataclass

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift


@dataclass
class KSpace2D:
    """2D k-space representation of a domain.

    Attributes:
        real: Real part of k-space coefficients (grid_size, grid_size).
        imag: Imaginary part of k-space coefficients (grid_size, grid_size).
        grid_size: Size of the k-space grid.
    """

    real: np.ndarray
    imag: np.ndarray
    grid_size: int

    def to_flat(self) -> np.ndarray:
        """Flatten to 1D array [real, imag] for neural network output.

        Returns:
            1D array of shape (grid_size^2 * 2,).
        """
        return np.concatenate([self.real.ravel(), self.imag.ravel()])

    @classmethod
    def from_flat(cls, data: np.ndarray, grid_size: int) -> "KSpace2D":
        """Reconstruct from flattened representation.

        Args:
            data: 1D array of shape (grid_size^2 * 2,).
            grid_size: Size of the k-space grid.

        Returns:
            KSpace2D instance.
        """
        n_coeffs = grid_size * grid_size
        real = data[:n_coeffs].reshape(grid_size, grid_size)
        imag = data[n_coeffs:].reshape(grid_size, grid_size)
        return cls(real=real, imag=imag, grid_size=grid_size)

    def to_complex(self) -> np.ndarray:
        """Convert to complex k-space array.

        Returns:
            Complex array of shape (grid_size, grid_size).
        """
        return self.real + 1j * self.imag


def inclusion_to_kspace(
    inclusion_type: str,
    center_x: float,
    center_y: float,
    inclusion_size: float,
    domain_size: float,
    grid_size: int = 64,
    background_value: float = 0.0,
    inclusion_value: float = 1.0,
) -> KSpace2D:
    """Convert inclusion parameters to k-space representation.

    Creates a 2D grid with the inclusion mask, then computes the 2D FFT.

    Args:
        inclusion_type: Type of inclusion ("circle", "square", "triangle").
        center_x: x-coordinate of inclusion center.
        center_y: y-coordinate of inclusion center.
        inclusion_size: Characteristic size (diameter for circle, side for others).
        domain_size: Size of the square domain.
        grid_size: Resolution of the k-space grid.
        background_value: Value outside the inclusion.
        inclusion_value: Value inside the inclusion.

    Returns:
        KSpace2D representation.
    """
    image = create_inclusion_image(
        inclusion_type=inclusion_type,
        center_x=center_x,
        center_y=center_y,
        inclusion_size=inclusion_size,
        domain_size=domain_size,
        grid_size=grid_size,
        background_value=background_value,
        inclusion_value=inclusion_value,
    )

    kspace = fftshift(fft2(image))

    return KSpace2D(
        real=np.real(kspace).astype(np.float32),
        imag=np.imag(kspace).astype(np.float32),
        grid_size=grid_size,
    )


def kspace_to_image(kspace: KSpace2D) -> np.ndarray:
    """Convert k-space representation back to real-space image.

    Args:
        kspace: K-space representation.

    Returns:
        Real-space image of shape (grid_size, grid_size).
    """
    complex_kspace = kspace.to_complex()
    image = ifft2(ifftshift(complex_kspace))
    return np.real(image)


def create_inclusion_image(
    inclusion_type: str,
    center_x: float,
    center_y: float,
    inclusion_size: float,
    domain_size: float,
    grid_size: int = 64,
    background_value: float = 0.0,
    inclusion_value: float = 1.0,
) -> np.ndarray:
    """Create a 2D image with the inclusion mask.

    Args:
        inclusion_type: Type of inclusion ("circle", "square", "triangle").
        center_x: x-coordinate of inclusion center.
        center_y: y-coordinate of inclusion center.
        inclusion_size: Characteristic size (diameter for circle, side for others).
        domain_size: Size of the square domain.
        grid_size: Resolution of the output image.
        background_value: Value outside the inclusion.
        inclusion_value: Value inside the inclusion.

    Returns:
        2D array of shape (grid_size, grid_size).
    """
    x = np.linspace(0, domain_size, grid_size)
    y = np.linspace(0, domain_size, grid_size)
    X, Y = np.meshgrid(x, y)

    image = np.full((grid_size, grid_size), background_value, dtype=np.float32)

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

    image[mask] = inclusion_value
    return image


def _point_in_triangle(
    X: np.ndarray,
    Y: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Check if points are inside a triangle using barycentric coordinates.

    Args:
        X: x-coordinates grid.
        Y: y-coordinates grid.
        v0, v1, v2: Triangle vertices as (x, y) arrays.

    Returns:
        Boolean mask of points inside the triangle.
    """

    def sign(px, py, x1, y1, x2, y2):
        return (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)

    d1 = sign(X, Y, v0[0], v0[1], v1[0], v1[1])
    d2 = sign(X, Y, v1[0], v1[1], v2[0], v2[1])
    d3 = sign(X, Y, v2[0], v2[1], v0[0], v0[1])

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return ~(has_neg & has_pos)
