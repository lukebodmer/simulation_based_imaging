"""Simulation class for 1D FDTD acoustic wave propagation.

Orchestrates the FDTD simulation including field updates,
boundary conditions, source injection, and output.
"""

from collections.abc import Callable

import numpy as np

from sbimaging.array.backend import to_numpy
from sbimaging.logging import get_logger
from sbimaging.simulators.fdtd.dim1.coefficients import UpdateCoefficients, compute_cfl_timestep
from sbimaging.simulators.fdtd.dim1.grid import Grid
from sbimaging.simulators.fdtd.dim1.material import Material
from sbimaging.simulators.fdtd.dim1.sensors import SensorArray
from sbimaging.simulators.fdtd.dim1.source import BoundarySource, Source


class Simulation:
    """1D FDTD acoustic wave simulation.

    Manages the complete simulation including:
    - Field initialization and updates
    - Source injection
    - Boundary condition enforcement
    - Time stepping and output

    Attributes:
        grid: Computational grid with field arrays.
        material: Material properties.
        coefficients: Precomputed update coefficients.
        dt: Time step size [s].
        time: Current simulation time [s].
        step: Current time step number.
        sources: List of pressure sources.
    """

    def __init__(
        self,
        grid: Grid,
        material: Material,
        dt: float | None = None,
        courant_factor: float = 0.9,
    ):
        """Initialize FDTD simulation.

        Args:
            grid: Computational grid.
            material: Material properties.
            dt: Time step size [s]. If None, computed from CFL condition.
            courant_factor: Safety factor for CFL time step.
        """
        self.grid = grid
        self.material = material

        if dt is None:
            self.dt = compute_cfl_timestep(grid, material, courant_factor)
        else:
            self.dt = dt

        self.coefficients = UpdateCoefficients(grid, material, self.dt)

        self.time = 0.0
        self.step = 0
        self.sources: list[Source] = []
        self.boundary_sources: list[BoundarySource] = []
        self.sensors: SensorArray | None = None

        self._logger = get_logger(__name__)
        self._log_info()

    def add_source(self, source: Source) -> None:
        """Add a pressure source to the simulation.

        Args:
            source: Source object to add.
        """
        source.locate_on_grid(self.grid)
        self.sources.append(source)
        self._logger.info(f"Added source at x={source.x:.4g}")

    def add_boundary_source(self, source: BoundarySource) -> None:
        """Add a boundary pressure source to the simulation.

        Args:
            source: BoundarySource object to add.
        """
        self.boundary_sources.append(source)
        self._logger.info(f"Added boundary source on {source.boundary} wall")

    def set_sensors(self, sensors: SensorArray) -> None:
        """Set the sensor array for recording pressure.

        Args:
            sensors: SensorArray object for recording.
        """
        self.sensors = sensors
        self._logger.info(f"Added {sensors.num_sensors} sensors")

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.grid.reset()
        self.time = 0.0
        self.step = 0
        if self.sensors:
            self.sensors.reset()

    def update_velocity(self) -> None:
        """Update velocity field from pressure gradient.

        vx^{n+1/2} = Cvxvx * vx^{n-1/2} - (dt/rho_x/dx) * (p[i+1] - p[i])
        """
        grid = self.grid
        coeff = self.coefficients

        # Update interior velocity points
        grid.vx[:-1] = (
            coeff.Cvxvx[:-1] * grid.vx[:-1]
            + coeff.Cvxp[:-1] * (grid.p[1:] - grid.p[:-1])
        )

    def apply_velocity_bc(self) -> None:
        """Apply velocity boundary conditions.

        For perfectly reflecting (rigid wall) boundaries,
        velocity at domain boundaries is zero.
        """
        grid = self.grid
        grid.vx[0] = 0.0
        grid.vx[-1] = 0.0

    def _set_boundary_source_pressure(self) -> None:
        """Set boundary pressure to source value."""
        for source in self.boundary_sources:
            p_source = source.get_pressure(self.time)
            if source.boundary == "left":
                self.grid.p[0] = p_source
            else:  # right
                self.grid.p[-1] = p_source

    def _cache_boundary_source_velocity(self) -> None:
        """Cache velocity at source boundaries after update."""
        self._cached_source_velocity: list[tuple[str, float]] = []

        for source in self.boundary_sources:
            if source.boundary == "left":
                self._cached_source_velocity.append(("left", float(self.grid.vx[0])))
            else:  # right
                self._cached_source_velocity.append(("right", float(self.grid.vx[-1])))

    def apply_pressure_bc(self) -> None:
        """Apply pressure boundary conditions.

        For rigid wall (reflecting) boundaries, the pressure
        gradient is zero (Neumann BC). Implemented by mirroring.
        """
        grid = self.grid
        grid.p[0] = grid.p[1]
        grid.p[-1] = grid.p[-2]

    def update_pressure(self) -> None:
        """Update pressure field from velocity divergence.

        p^{n+1} = Cpp * p^n - kappa*dt/dx * (vx[i] - vx[i-1])
        """
        grid = self.grid
        coeff = self.coefficients

        # Update interior pressure points
        grid.p[1:-1] = (
            coeff.Cpp[1:-1] * grid.p[1:-1]
            + coeff.Cpvx[1:-1] * (grid.vx[1:-1] - grid.vx[:-2])
        )

    def apply_sources(self) -> None:
        """Apply all pressure sources at current time."""
        for source in self.sources:
            source.apply(self.grid.p, self.time)

    def _restore_boundary_source_velocity(self) -> None:
        """Restore boundary source velocity after BC zeroed it."""
        if not hasattr(self, "_cached_source_velocity"):
            return
        for boundary, value in self._cached_source_velocity:
            if boundary == "left":
                self.grid.vx[0] = value
            else:  # right
                self.grid.vx[-1] = value

    def record_sensors(self) -> None:
        """Record pressure at all sensor locations."""
        if self.sensors is not None:
            self.sensors.record(self.grid.p, self.time)

    def step_forward(self) -> None:
        """Advance simulation by one time step.

        The update sequence is:
        1. Set boundary source pressure
        2. Update velocity from pressure gradient
        3. Cache velocity at source boundaries
        4. Apply velocity boundary conditions
        5. Restore velocity at source boundaries
        6. Update pressure from velocity divergence
        7. Apply pressure boundary conditions
        8. Set boundary source pressure
        9. Apply interior sources
        10. Record sensor data
        11. Advance time
        """
        self._set_boundary_source_pressure()
        self.update_velocity()
        self._cache_boundary_source_velocity()
        self.apply_velocity_bc()
        self._restore_boundary_source_velocity()
        self.update_pressure()
        self.apply_pressure_bc()
        self._set_boundary_source_pressure()
        self.apply_sources()
        self.record_sensors()

        self.time += self.dt
        self.step += 1

    def run(
        self,
        num_steps: int,
        callback: Callable[["Simulation"], None] | None = None,
        callback_interval: int = 1,
    ) -> None:
        """Run simulation for specified number of steps.

        Args:
            num_steps: Number of time steps to run.
            callback: Optional function called every callback_interval steps.
            callback_interval: Steps between callback invocations.
        """
        self._logger.info(f"Running {num_steps} time steps...")

        for _ in range(num_steps):
            self.step_forward()

            if callback and self.step % callback_interval == 0:
                callback(self)

        self._logger.info(
            f"Simulation completed: {self.step} steps, t={self.time:.6g}s"
        )

    def run_until(
        self,
        t_final: float,
        callback: Callable[["Simulation"], None] | None = None,
        callback_interval: int = 1,
    ) -> None:
        """Run simulation until specified time.

        Args:
            t_final: Final simulation time [s].
            callback: Optional function called every callback_interval steps.
            callback_interval: Steps between callback invocations.
        """
        num_steps = int(np.ceil((t_final - self.time) / self.dt))
        self.run(num_steps, callback, callback_interval)

    def get_pressure(self) -> np.ndarray:
        """Get current pressure field as numpy array.

        Returns:
            Pressure field array (nx+1,).
        """
        return to_numpy(self.grid.p)

    def _log_info(self) -> None:
        """Log simulation parameters."""
        grid = self.grid
        self._logger.info(f"Grid: {grid.nx} cells, dx={grid.dx:.4g}m")
        self._logger.info(f"Domain: {grid.size_x:.4g}m")
        self._logger.info(f"Time step: {self.dt:.6g}s")
        self._logger.info(f"Material: {self.material}")


class FrameRecorder:
    """Callback for recording simulation frames.

    Use as a callback to Simulation.run() to save pressure
    fields at regular intervals.

    Attributes:
        frames: List of saved pressure arrays.
        times: List of simulation times for each frame.
    """

    def __init__(self, keep_in_memory: bool = True):
        """Initialize frame recorder.

        Args:
            keep_in_memory: If True, stores frames in memory.
        """
        self.keep_in_memory = keep_in_memory
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []

    def __call__(self, sim: Simulation) -> None:
        """Record current frame.

        Args:
            sim: Simulation instance.
        """
        pressure = sim.get_pressure()
        self.times.append(sim.time)

        if self.keep_in_memory:
            self.frames.append(pressure.copy())
