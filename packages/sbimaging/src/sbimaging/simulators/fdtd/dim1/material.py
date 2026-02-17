"""Material properties for 1D FDTD simulation.

Stores spatially-varying density and wave speed, and computes
derived quantities needed for the FDTD update equations.
"""

import numpy as np

from sbimaging.array.backend import xp, to_numpy
from sbimaging.simulators.fdtd.dim1.grid import Grid


class Material:
    """Acoustic material properties for 1D FDTD simulation.

    Stores density and wave speed at each grid point. Computes derived
    quantities like bulk modulus and averaged densities at staggered
    velocity locations.

    Attributes:
        density: Material density at pressure points [kg/m^3].
        wave_speed: Wave speed at pressure points [m/s].
        bulk_modulus: Bulk modulus kappa = rho * c^2 [Pa].
        density_x: Density averaged to vx locations [kg/m^3].
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
            density: Density array at pressure points (nx+1,) [kg/m^3].
            wave_speed: Wave speed array at pressure points (nx+1,) [m/s].
        """
        self._grid = grid
        self.density = xp.asarray(density, dtype=np.float64)
        self.wave_speed = xp.asarray(wave_speed, dtype=np.float64)

        self._compute_derived_quantities()

    @classmethod
    def uniform(cls, grid: Grid, density: float, wave_speed: float) -> "Material":
        """Create uniform material.

        Args:
            grid: Computational grid.
            density: Uniform density [kg/m^3].
            wave_speed: Uniform wave speed [m/s].

        Returns:
            Material instance with uniform properties.
        """
        rho = np.full(grid.nx + 1, density, dtype=np.float64)
        c = np.full(grid.nx + 1, wave_speed, dtype=np.float64)
        return cls(grid, rho, c)

    def set_inclusion(
        self,
        x_min: float,
        x_max: float,
        density: float,
        wave_speed: float,
    ) -> None:
        """Set material properties in a region (1D inclusion).

        Args:
            x_min: Left boundary of inclusion [m].
            x_max: Right boundary of inclusion [m].
            density: Density in the region [kg/m^3].
            wave_speed: Wave speed in the region [m/s].
        """
        grid = self._grid
        i_min = grid.index_at_position(x_min)
        i_max = grid.index_at_position(x_max)

        i_min = max(0, i_min)
        i_max = min(grid.nx, i_max)

        self.density[i_min : i_max + 1] = density
        self.wave_speed[i_min : i_max + 1] = wave_speed

        self._compute_derived_quantities()

    def _compute_derived_quantities(self) -> None:
        """Compute bulk modulus and averaged densities."""
        self.bulk_modulus = self.density * (self.wave_speed**2)

        density_np = to_numpy(self.density)
        grid = self._grid

        # Average density to velocity locations (staggered grid)
        density_x = np.zeros(grid.nx + 1, dtype=np.float64)
        density_x[:-1] = 0.5 * (density_np[:-1] + density_np[1:])
        density_x[-1] = density_np[-1]  # Boundary

        self.density_x = xp.asarray(density_x)

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
