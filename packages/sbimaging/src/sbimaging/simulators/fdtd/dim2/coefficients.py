"""Update coefficients for 2D FDTD simulation.

Precomputes coefficients for the velocity and pressure update
equations based on material properties and grid spacing.
"""

import numpy as np

from sbimaging.array.backend import xp
from sbimaging.simulators.fdtd.dim2.grid import Grid
from sbimaging.simulators.fdtd.dim2.material import Material


class UpdateCoefficients:
    """Precomputed update coefficients for FDTD time stepping.

    The update equations use these coefficients:

    Velocity:
        vx^{n+1/2} = Cvxvx * vx^{n-1/2} + Cvxp * (p[i+1,j] - p[i,j])
        vy^{n+1/2} = Cvyvy * vy^{n-1/2} + Cvyp * (p[i,j+1] - p[i,j])

    Pressure:
        p^{n+1} = Cpp * p^n + Cpvx * (vx[i,j] - vx[i-1,j])
                            + Cpvy * (vy[i,j] - vy[i,j-1])

    For a lossless medium without damping:
        Cvxvx = Cvyvy = Cpp = 1.0
        Cvxp = -dt / (rho_x * dx)
        Cvyp = -dt / (rho_y * dy)
        Cpvx = -bulk * dt / dx
        Cpvy = -bulk * dt / dy

    Attributes:
        Cvxvx: vx self-coefficient (nx+1, ny).
        Cvxp: vx coefficient for pressure gradient (nx+1, ny).
        Cvyvy: vy self-coefficient (nx, ny+1).
        Cvyp: vy coefficient for pressure gradient (nx, ny+1).
        Cpp: p self-coefficient (nx+1, ny+1).
        Cpvx: p coefficient for vx divergence (nx+1, ny+1).
        Cpvy: p coefficient for vy divergence (nx+1, ny+1).
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
        dy = grid.dy

        self.Cvxvx = xp.ones((grid.nx + 1, grid.ny), dtype=np.float64)
        self.Cvxp = -dt / (material.density_x * dx)

        self.Cvyvy = xp.ones((grid.nx, grid.ny + 1), dtype=np.float64)
        self.Cvyp = -dt / (material.density_y * dy)

        self.Cpp = xp.ones((grid.nx + 1, grid.ny + 1), dtype=np.float64)
        self.Cpvx = -material.bulk_modulus * dt / dx
        self.Cpvy = -material.bulk_modulus * dt / dy

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

    The CFL condition for 2D FDTD requires:
        dt <= 1 / (c_max * sqrt(1/dx^2 + 1/dy^2))

    Args:
        grid: Computational grid.
        material: Material properties.
        courant_factor: Safety factor (< 1.0).

    Returns:
        Stable time step size [s].
    """
    c_max = material.max_wave_speed
    dx = grid.dx
    dy = grid.dy

    dt_max = 1.0 / (c_max * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
    return courant_factor * dt_max
