"""Material properties for 2D FDTD simulation.

Stores spatially-varying density and wave speed, and computes
derived quantities needed for the FDTD update equations.
"""

import numpy as np

from sbimaging.array.backend import xp, to_numpy
from sbimaging.simulators.fdtd.dim2.grid import Grid


class Material:
    """Acoustic material properties for FDTD simulation.

    Stores density and wave speed at each grid point. Computes derived
    quantities like bulk modulus and averaged densities at staggered
    velocity locations.

    Attributes:
        density: Material density at pressure points [kg/m^3].
        wave_speed: Wave speed at pressure points [m/s].
        bulk_modulus: Bulk modulus kappa = rho * c^2 [Pa].
        density_x: Density averaged to vx locations [kg/m^3].
        density_y: Density averaged to vy locations [kg/m^3].
    """

    def __init__(
        self,
        grid: Grid,
        density: np.ndarray,
        wave_speed: np.ndarray,
    ):
        """Initialize material properties.

        Args:
            grid: Computational grid.
            density: Density array at pressure points (nx+1, ny+1) [kg/m^3].
            wave_speed: Wave speed array at pressure points (nx+1, ny+1) [m/s].
        """
        self._grid = grid
        self.density = xp.asarray(density, dtype=np.float64)
        self.wave_speed = xp.asarray(wave_speed, dtype=np.float64)

        self._compute_derived_quantities()

    @classmethod
    def uniform(
        cls,
        grid: Grid,
        density: float,
        wave_speed: float,
    ) -> "Material":
        """Create uniform material.

        Args:
            grid: Computational grid.
            density: Uniform density [kg/m^3].
            wave_speed: Uniform wave speed [m/s].

        Returns:
            Material instance with uniform properties.
        """
        rho = np.full((grid.nx + 1, grid.ny + 1), density, dtype=np.float64)
        c = np.full((grid.nx + 1, grid.ny + 1), wave_speed, dtype=np.float64)
        return cls(grid, rho, c)

    def set_rectangular_inclusion(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        density: float,
        wave_speed: float,
    ) -> None:
        """Set material properties in a rectangular region.

        Args:
            x_min: Left boundary of rectangle [m].
            x_max: Right boundary of rectangle [m].
            y_min: Bottom boundary of rectangle [m].
            y_max: Top boundary of rectangle [m].
            density: Density in the region [kg/m^3].
            wave_speed: Wave speed in the region [m/s].
        """
        grid = self._grid
        i_min, j_min = grid.index_at_position(x_min, y_min)
        i_max, j_max = grid.index_at_position(x_max, y_max)

        i_min = max(0, i_min)
        i_max = min(grid.nx, i_max)
        j_min = max(0, j_min)
        j_max = min(grid.ny, j_max)

        self.density[i_min : i_max + 1, j_min : j_max + 1] = density
        self.wave_speed[i_min : i_max + 1, j_min : j_max + 1] = wave_speed

        self._compute_derived_quantities()

    def set_circular_inclusion(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        density: float,
        wave_speed: float,
    ) -> None:
        """Set material properties in a circular region.

        Args:
            center_x: x-coordinate of circle center [m].
            center_y: y-coordinate of circle center [m].
            radius: Circle radius [m].
            density: Density in the region [kg/m^3].
            wave_speed: Wave speed in the region [m/s].
        """
        grid = self._grid
        X, Y = grid.meshgrid()

        distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        mask = distance <= radius

        density_np = to_numpy(self.density)
        wave_speed_np = to_numpy(self.wave_speed)

        density_np[mask] = density
        wave_speed_np[mask] = wave_speed

        self.density = xp.asarray(density_np)
        self.wave_speed = xp.asarray(wave_speed_np)

        self._compute_derived_quantities()

    def set_triangular_inclusion(
        self,
        vertices: list[tuple[float, float]],
        density: float,
        wave_speed: float,
    ) -> None:
        """Set material properties in a triangular region.

        Args:
            vertices: List of 3 (x, y) tuples defining triangle vertices [m].
            density: Density in the region [kg/m^3].
            wave_speed: Wave speed in the region [m/s].
        """
        if len(vertices) != 3:
            raise ValueError("Triangle requires exactly 3 vertices")

        grid = self._grid
        X, Y = grid.meshgrid()

        # Use barycentric coordinates to determine if points are inside triangle
        v0 = np.array(vertices[0])
        v1 = np.array(vertices[1])
        v2 = np.array(vertices[2])

        # Compute vectors
        v0v1 = v1 - v0
        v0v2 = v2 - v0

        # Compute dot products for barycentric coordinates
        dot00 = np.dot(v0v1, v0v1)
        dot01 = np.dot(v0v1, v0v2)
        dot11 = np.dot(v0v2, v0v2)

        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)

        # For each grid point, compute barycentric coordinates
        v0p_x = X - v0[0]
        v0p_y = Y - v0[1]

        dot02 = v0v1[0] * v0p_x + v0v1[1] * v0p_y
        dot12 = v0v2[0] * v0p_x + v0v2[1] * v0p_y

        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Point is inside triangle if u >= 0, v >= 0, and u + v <= 1
        mask = (u >= 0) & (v >= 0) & (u + v <= 1)

        density_np = to_numpy(self.density)
        wave_speed_np = to_numpy(self.wave_speed)

        density_np[mask] = density
        wave_speed_np[mask] = wave_speed

        self.density = xp.asarray(density_np)
        self.wave_speed = xp.asarray(wave_speed_np)

        self._compute_derived_quantities()

    def _compute_derived_quantities(self) -> None:
        """Compute bulk modulus and averaged densities."""
        self.bulk_modulus = self.density * (self.wave_speed ** 2)

        density_np = to_numpy(self.density)
        grid = self._grid

        density_x = np.zeros((grid.nx + 1, grid.ny), dtype=np.float64)
        for j in range(grid.ny):
            density_x[:, j] = 0.5 * (density_np[:, j] + density_np[:, j + 1])

        density_y = np.zeros((grid.nx, grid.ny + 1), dtype=np.float64)
        for i in range(grid.nx):
            density_y[i, :] = 0.5 * (density_np[i, :] + density_np[i + 1, :])

        self.density_x = xp.asarray(density_x)
        self.density_y = xp.asarray(density_y)

    @property
    def max_wave_speed(self) -> float:
        """Maximum wave speed in the domain."""
        return float(to_numpy(self.wave_speed).max())

    def __repr__(self) -> str:
        rho_min = float(to_numpy(self.density).min())
        rho_max = float(to_numpy(self.density).max())
        c_min = float(to_numpy(self.wave_speed).min())
        c_max = float(to_numpy(self.wave_speed).max())
        return (
            f"Material(density=[{rho_min:.2f}, {rho_max:.2f}] kg/m^3, "
            f"wave_speed=[{c_min:.2f}, {c_max:.2f}] m/s)"
        )
