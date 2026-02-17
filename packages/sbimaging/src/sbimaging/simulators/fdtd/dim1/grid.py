"""Grid class for 1D FDTD simulation.

Implements a staggered Yee grid for acoustic wave propagation with
pressure at cell centers and velocity at cell faces.
"""

import numpy as np

from sbimaging.array.backend import xp


class Grid:
    """Staggered Yee grid for 1D acoustic FDTD.

    The grid uses staggered storage for pressure and velocity fields:
    - Pressure p is stored at cell centers: p[i] at position i*dx
    - Velocity vx is stored at cell faces: vx[i] at position (i+0.5)*dx

    Attributes:
        nx: Number of cells.
        dx: Cell size [m].
        size_x: Total domain size [m].
        p: Pressure field array (nx+1,).
        vx: Velocity field array (nx+1,).
    """

    def __init__(self, nx: int, dx: float):
        """Initialize the computational grid.

        Args:
            nx: Number of cells.
            dx: Cell size [m].
        """
        self.nx = nx
        self.dx = dx
        self.size_x = nx * dx

        self._initialize_fields()

    @classmethod
    def from_domain_size(cls, size_x: float, nx: int) -> "Grid":
        """Create grid from domain size and number of cells.

        Args:
            size_x: Domain size [m].
            nx: Number of cells.

        Returns:
            Grid instance.
        """
        dx = size_x / nx
        return cls(nx, dx)

    def _initialize_fields(self) -> None:
        """Initialize field arrays to zero."""
        self.p = xp.zeros(self.nx + 1, dtype=np.float64)
        self.vx = xp.zeros(self.nx + 1, dtype=np.float64)

    def reset(self) -> None:
        """Reset all fields to zero."""
        self.p.fill(0.0)
        self.vx.fill(0.0)

    def x_coordinates(self) -> np.ndarray:
        """Get x coordinates of pressure grid points.

        Returns:
            1D array of x coordinates [m].
        """
        return np.arange(self.nx + 1) * self.dx

    def index_at_position(self, x: float) -> int:
        """Get grid index for a physical position.

        Args:
            x: x-coordinate [m].

        Returns:
            Grid index i.
        """
        i = int(round(x / self.dx))
        return max(0, min(i, self.nx))

    def __repr__(self) -> str:
        return f"Grid(nx={self.nx}, dx={self.dx:.6g}, size={self.size_x:.4g} m)"
