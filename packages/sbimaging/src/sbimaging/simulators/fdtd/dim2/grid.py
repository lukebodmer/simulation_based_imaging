"""Grid class for 2D FDTD simulation.

Implements a staggered Yee grid for acoustic wave propagation with
pressure at cell centers and velocity at cell faces.
"""

import numpy as np

from sbimaging.array.backend import xp


class Grid:
    """Staggered Yee grid for 2D acoustic FDTD.

    The grid uses staggered storage for pressure and velocity fields:
    - Pressure p is stored at cell centers: p[i,j] at position (i*dx, j*dy)
    - x-velocity vx is stored at x-faces: vx[i,j] at position ((i+0.5)*dx, j*dy)
    - y-velocity vy is stored at y-faces: vy[i,j] at position (i*dx, (j+0.5)*dy)

    Attributes:
        nx: Number of cells in x direction.
        ny: Number of cells in y direction.
        dx: Cell size in x direction [m].
        dy: Cell size in y direction [m].
        size_x: Total domain size in x [m].
        size_y: Total domain size in y [m].
        p: Pressure field array (nx+1, ny+1).
        vx: x-velocity field array (nx+1, ny).
        vy: y-velocity field array (nx, ny+1).
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
    ):
        """Initialize the computational grid.

        Args:
            nx: Number of cells in x direction.
            ny: Number of cells in y direction.
            dx: Cell size in x direction [m].
            dy: Cell size in y direction [m].
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        self.size_x = nx * dx
        self.size_y = ny * dy

        self._initialize_fields()

    @classmethod
    def from_domain_size(
        cls,
        size_x: float,
        size_y: float,
        nx: int,
        ny: int,
    ) -> "Grid":
        """Create grid from domain size and number of cells.

        Args:
            size_x: Domain size in x direction [m].
            size_y: Domain size in y direction [m].
            nx: Number of cells in x direction.
            ny: Number of cells in y direction.

        Returns:
            Grid instance.
        """
        dx = size_x / nx
        dy = size_y / ny
        return cls(nx, ny, dx, dy)

    def _initialize_fields(self) -> None:
        """Initialize field arrays to zero.

        Array dimensions account for staggered grid layout:
        - p: (nx+1, ny+1) - values at cell corners/centers
        - vx: (nx+1, ny) - values at x-faces
        - vy: (nx, ny+1) - values at y-faces
        """
        self.p = xp.zeros((self.nx + 1, self.ny + 1), dtype=np.float64)
        self.vx = xp.zeros((self.nx + 1, self.ny), dtype=np.float64)
        self.vy = xp.zeros((self.nx, self.ny + 1), dtype=np.float64)

    def reset(self) -> None:
        """Reset all fields to zero."""
        self.p.fill(0.0)
        self.vx.fill(0.0)
        self.vy.fill(0.0)

    def x_coordinates(self) -> np.ndarray:
        """Get x coordinates of pressure grid points.

        Returns:
            1D array of x coordinates [m].
        """
        return np.arange(self.nx + 1) * self.dx

    def y_coordinates(self) -> np.ndarray:
        """Get y coordinates of pressure grid points.

        Returns:
            1D array of y coordinates [m].
        """
        return np.arange(self.ny + 1) * self.dy

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Get 2D meshgrid of pressure grid coordinates.

        Returns:
            Tuple of (X, Y) meshgrid arrays.
        """
        x = self.x_coordinates()
        y = self.y_coordinates()
        return np.meshgrid(x, y, indexing="ij")

    def index_at_position(self, x: float, y: float) -> tuple[int, int]:
        """Get grid indices for a physical position.

        Args:
            x: x-coordinate [m].
            y: y-coordinate [m].

        Returns:
            Tuple of (i, j) grid indices.
        """
        i = int(round(x / self.dx))
        j = int(round(y / self.dy))
        i = max(0, min(i, self.nx))
        j = max(0, min(j, self.ny))
        return i, j

    def __repr__(self) -> str:
        return (
            f"Grid(nx={self.nx}, ny={self.ny}, "
            f"dx={self.dx:.6g}, dy={self.dy:.6g}, "
            f"size=({self.size_x:.4g} x {self.size_y:.4g}) m)"
        )
