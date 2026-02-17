"""Update coefficients for 1D FDTD simulation.

Precomputes coefficients for the velocity and pressure update
equations based on material properties and grid spacing.
"""

import numpy as np

from sbimaging.array.backend import xp
from sbimaging.simulators.fdtd.dim1.grid import Grid
from sbimaging.simulators.fdtd.dim1.material import Material


class UpdateCoefficients:
    """Precomputed update coefficients for 1D FDTD time stepping.

    The update equations use these coefficients:

    Velocity:
        vx^{n+1/2} = Cvxvx * vx^{n-1/2} + Cvxp * (p[i+1] - p[i])

    Pressure:
        p^{n+1} = Cpp * p^n + Cpvx * (vx[i] - vx[i-1])

    For a lossless medium without damping:
        Cvxvx = Cpp = 1.0
        Cvxp = -dt / (rho_x * dx)
        Cpvx = -bulk * dt / dx

    Attributes:
        Cvxvx: vx self-coefficient (nx+1,).
        Cvxp: vx coefficient for pressure gradient (nx+1,).
        Cpp: p self-coefficient (nx+1,).
        Cpvx: p coefficient for vx divergence (nx+1,).
    """

    def __init__(self, grid: Grid, material: Material, dt: float):
        """Compute update coefficients.

        Args:
            grid: Computational grid.
            material: Material properties.
            dt: Time step size [s].
        """
        self.dt = dt
        self._grid = grid
        self._material = material

        self._compute_coefficients()

    def _compute_coefficients(self) -> None:
        """Compute all update coefficients."""
        grid = self._grid
        material = self._material
        dt = self.dt
        dx = grid.dx

        self.Cvxvx = xp.ones(grid.nx + 1, dtype=np.float64)
        self.Cvxp = -dt / (material.density_x * dx)

        self.Cpp = xp.ones(grid.nx + 1, dtype=np.float64)
        self.Cpvx = -material.bulk_modulus * dt / dx

    def recompute(self, dt: float | None = None) -> None:
        """Recompute coefficients, optionally with new time step.

        Args:
            dt: New time step size [s], or None to keep current.
        """
        if dt is not None:
            self.dt = dt
        self._compute_coefficients()


def compute_cfl_timestep(
    grid: Grid,
    material: Material,
    courant_factor: float = 0.9,
) -> float:
    """Compute stable time step from CFL condition.

    The CFL condition for 1D FDTD requires:
        dt <= dx / c_max

    Args:
        grid: Computational grid.
        material: Material properties.
        courant_factor: Safety factor (< 1.0).

    Returns:
        Stable time step size [s].
    """
    c_max = material.max_wave_speed
    dx = grid.dx

    dt_max = dx / c_max
    return courant_factor * dt_max
