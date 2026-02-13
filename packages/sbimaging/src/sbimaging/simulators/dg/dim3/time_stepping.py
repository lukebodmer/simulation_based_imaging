"""Time stepping methods for DG simulations.

Implements low-storage Runge-Kutta time integration for
explicit DG methods.
"""

from collections.abc import Callable

import numpy as np

from sbimaging.array.backend import xp
from sbimaging.logging import get_logger
from sbimaging.simulators.dg.dim3.acoustics import AcousticsOperator


class LowStorageRungeKutta:
    """Low-storage 5-stage 4th-order Runge-Kutta integrator.

    Uses Carpenter-Kennedy coefficients for memory-efficient
    explicit time stepping.

    Attributes:
        physics: AcousticsOperator providing right-hand side.
        dt: Time step size.
        t: Current simulation time.
        t_final: Final simulation time.
        current_step: Current time step number.
        num_steps: Total number of time steps.
    """

    RK4A = np.array([
        0.0,
        -567301805773.0 / 1357537059087.0,
        -2404267990393.0 / 2016746695238.0,
        -3550918686646.0 / 2091501179385.0,
        -1275806237668.0 / 842570457699.0,
    ])

    RK4B = np.array([
        1432997174477.0 / 9575080441755.0,
        5161836677717.0 / 13612068292357.0,
        1720146321549.0 / 2090206949498.0,
        3134564353537.0 / 4481467310338.0,
        2277821191437.0 / 14882151754819.0,
    ])

    def __init__(
        self,
        physics: AcousticsOperator,
        dt: float,
        t_initial: float = 0.0,
        t_final: float | None = None,
        num_steps: int | None = None,
    ):
        """Initialize time stepper.

        Args:
            physics: AcousticsOperator for computing right-hand side.
            dt: Time step size.
            t_initial: Initial time.
            t_final: Final time (specify either this or num_steps).
            num_steps: Number of time steps (specify either this or t_final).

        Raises:
            ValueError: If neither or both t_final and num_steps specified.
        """
        if (t_final is None) == (num_steps is None):
            raise ValueError("Specify exactly one of 't_final' or 'num_steps'.")

        self.physics = physics
        self.dt = dt
        self.t = t_initial
        self.current_step = 0

        if t_final is None:
            self.num_steps = num_steps
            self.t_final = num_steps * dt
        else:
            self.t_final = t_final
            self.num_steps = int(np.ceil(t_final / dt))

        npc = physics.mesh._element.nodes_per_cell
        k = physics.mesh.num_cells

        self._res_u = xp.zeros((npc, k))
        self._res_v = xp.zeros((npc, k))
        self._res_w = xp.zeros((npc, k))
        self._res_p = xp.zeros((npc, k))

        self._rk4a = xp.asarray(self.RK4A)
        self._rk4b = xp.asarray(self.RK4B)

        self._log_info()

    def step(self) -> None:
        """Advance solution by one time step."""
        dt = self.dt
        physics = self.physics

        for i in range(5):
            rhs_u, rhs_v, rhs_w, rhs_p = physics.compute_rhs(time=self.t)

            self._res_u = self._rk4a[i] * self._res_u + dt * rhs_u
            physics.u = physics.u + self._rk4b[i] * self._res_u

            self._res_v = self._rk4a[i] * self._res_v + dt * rhs_v
            physics.v = physics.v + self._rk4b[i] * self._res_v

            self._res_w = self._rk4a[i] * self._res_w + dt * rhs_w
            physics.w = physics.w + self._rk4b[i] * self._res_w

            self._res_p = self._rk4a[i] * self._res_p + dt * rhs_p
            physics.p = physics.p + self._rk4b[i] * self._res_p

        self.t += dt
        self.current_step += 1

    def run(self, callback: Callable | None = None, callback_interval: int = 1) -> None:
        """Run simulation to completion.

        Args:
            callback: Optional function called every callback_interval steps.
                Signature: callback(stepper, step_number).
            callback_interval: Steps between callback invocations.
        """
        logger = get_logger(__name__)

        while self.current_step < self.num_steps:
            self.step()

            if callback and self.current_step % callback_interval == 0:
                callback(self, self.current_step)

        logger.info(f"Simulation completed: {self.num_steps} steps, t={self.t:.6g}s")

    def _log_info(self) -> None:
        """Log time stepping parameters."""
        logger = get_logger(__name__)
        logger.info(f"Time step size: {self.dt:.6g}s")
        logger.info(f"Total time steps: {self.num_steps}")
        logger.info(f"Final time: {self.t_final:.6g}s")


def compute_cfl_timestep(
    smallest_diameter: float,
    max_speed: float,
    polynomial_order: int,
    cfl_factor: float = 0.9,
) -> float:
    """Compute stable time step from CFL condition.

    Uses Hesthaven-Warburton formula for DG stability.

    Args:
        smallest_diameter: Smallest cell inscribed diameter.
        max_speed: Maximum wave speed in domain.
        polynomial_order: Polynomial order of approximation.
        cfl_factor: Safety factor (< 1).

    Returns:
        Stable time step size.
    """
    p = polynomial_order
    return cfl_factor * smallest_diameter / ((2 * p + 1) * max_speed)
