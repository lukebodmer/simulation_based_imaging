"""Simulation class for 2D FDTD acoustic wave propagation.

Orchestrates the FDTD simulation including field updates,
boundary conditions, source injection, and output.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from sbimaging.array.backend import to_numpy
from sbimaging.logging import get_logger
from sbimaging.simulators.fdtd.dim2.coefficients import UpdateCoefficients, compute_cfl_timestep
from sbimaging.simulators.fdtd.dim2.grid import Grid
from sbimaging.simulators.fdtd.dim2.material import Material
from sbimaging.simulators.fdtd.dim2.sensors import SensorArray
from sbimaging.simulators.fdtd.dim2.source import BoundarySource, Source


class Simulation:
    """2D FDTD acoustic wave simulation.

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
        self._logger.info(f"Added source at ({source.x:.4g}, {source.y:.4g})")

    def add_boundary_source(self, source: BoundarySource) -> None:
        """Add a boundary pressure source to the simulation.

        Args:
            source: BoundarySource object to add.
        """
        source.locate_on_grid(self.grid)
        self.boundary_sources.append(source)
        self._logger.info(
            f"Added boundary source on {source.boundary} wall "
            f"at position {source.position:.4g}, width {source.width:.4g}"
        )

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
        """Update velocity fields from pressure gradient.

        vx^{n+1/2} = Cvxvx * vx^{n-1/2} - (dt/rho_x/dx) * (p[i+1,j] - p[i,j])
        vy^{n+1/2} = Cvyvy * vy^{n-1/2} - (dt/rho_y/dy) * (p[i,j+1] - p[i,j])
        """
        grid = self.grid
        coeff = self.coefficients

        grid.vx[:-1, :] = (
            coeff.Cvxvx[:-1, :] * grid.vx[:-1, :]
            + coeff.Cvxp[:-1, :] * (grid.p[1:, :-1] - grid.p[:-1, :-1])
        )

        grid.vy[:, :-1] = (
            coeff.Cvyvy[:, :-1] * grid.vy[:, :-1]
            + coeff.Cvyp[:, :-1] * (grid.p[:-1, 1:] - grid.p[:-1, :-1])
        )

    def apply_velocity_bc(self) -> None:
        """Apply velocity boundary conditions.

        For perfectly reflecting (rigid wall) boundaries,
        normal velocity at domain boundaries is zero.
        """
        grid = self.grid

        grid.vx[0, :] = 0.0
        grid.vx[-1, :] = 0.0
        grid.vy[:, 0] = 0.0
        grid.vy[:, -1] = 0.0

    def _set_boundary_source_pressure(self) -> None:
        """Set boundary pressure to source value.

        This is called before velocity update so the pressure gradient
        at the boundary drives flow into/out of the domain.
        """
        for source in self.boundary_sources:
            p_source = source.get_pressure(self.time)

            if source.boundary == "left" and source._j_indices is not None:
                self.grid.p[0, source._j_indices] = p_source
            elif source.boundary == "right" and source._j_indices is not None:
                self.grid.p[-1, source._j_indices] = p_source
            elif source.boundary == "bottom" and source._i_indices is not None:
                self.grid.p[source._i_indices, 0] = p_source
            elif source.boundary == "top" and source._i_indices is not None:
                self.grid.p[source._i_indices, -1] = p_source

    def _cache_boundary_source_velocity(self) -> None:
        """Cache velocity at source boundaries after update_velocity computed them.

        This is called after update_velocity() but before apply_velocity_bc()
        so we can restore the computed values after BC zeros them.
        """
        self._cached_source_velocity: list[tuple[str, np.ndarray, np.ndarray]] = []

        for source in self.boundary_sources:
            if source.boundary == "left" and source._j_indices is not None:
                j_idx_v = source._j_indices[:-1]
                vx_bnd = self.grid.vx[0, j_idx_v].copy()
                self._cached_source_velocity.append(("left", j_idx_v.copy(), vx_bnd))
            elif source.boundary == "right" and source._j_indices is not None:
                j_idx_v = source._j_indices[:-1]
                vx_bnd = self.grid.vx[-1, j_idx_v].copy()
                self._cached_source_velocity.append(("right", j_idx_v.copy(), vx_bnd))
            elif source.boundary == "bottom" and source._i_indices is not None:
                i_idx_v = source._i_indices[:-1]
                vy_bnd = self.grid.vy[i_idx_v, 0].copy()
                self._cached_source_velocity.append(("bottom", i_idx_v.copy(), vy_bnd))
            elif source.boundary == "top" and source._i_indices is not None:
                i_idx_v = source._i_indices[:-1]
                vy_bnd = self.grid.vy[i_idx_v, -1].copy()
                self._cached_source_velocity.append(("top", i_idx_v.copy(), vy_bnd))

    def apply_boundary_source_conditions(self) -> None:
        """Legacy method - now split into _set_boundary_source_pressure and _cache_boundary_source_velocity."""
        self._set_boundary_source_pressure()
        self._cache_boundary_source_velocity()

    def apply_pressure_bc(self) -> None:
        """Apply pressure boundary conditions.

        For rigid wall (reflecting) boundaries, the normal pressure
        gradient is zero (Neumann BC). This is implemented by copying
        the interior pressure to the boundary (mirror condition).
        """
        grid = self.grid

        # Left and right boundaries (x = 0 and x = size_x)
        grid.p[0, :] = grid.p[1, :]
        grid.p[-1, :] = grid.p[-2, :]

        # Bottom and top boundaries (y = 0 and y = size_y)
        grid.p[:, 0] = grid.p[:, 1]
        grid.p[:, -1] = grid.p[:, -2]

    def update_pressure(self) -> None:
        """Update pressure field from velocity divergence.

        p^{n+1} = Cpp * p^n - kappa*dt/dx * (vx[i,j] - vx[i-1,j])
                            - kappa*dt/dy * (vy[i,j] - vy[i,j-1])
        """
        grid = self.grid
        coeff = self.coefficients

        grid.p[1:-1, 1:-1] = (
            coeff.Cpp[1:-1, 1:-1] * grid.p[1:-1, 1:-1]
            + coeff.Cpvx[1:-1, 1:-1] * (grid.vx[1:-1, 1:] - grid.vx[:-2, 1:])
            + coeff.Cpvy[1:-1, 1:-1] * (grid.vy[1:, 1:-1] - grid.vy[1:, :-2])
        )

    def apply_sources(self) -> None:
        """Apply all pressure sources at current time."""
        for source in self.sources:
            source.apply(self.grid.p, self.time)

    def apply_boundary_sources(self) -> None:
        """Apply all boundary sources at current time.

        Boundary sources override the pressure BC on their region.
        """
        for source in self.boundary_sources:
            source.apply_pressure_bc(self.grid.p, self.time)

    def _restore_boundary_source_velocity(self) -> None:
        """Restore boundary source velocity after apply_velocity_bc zeroed it.

        This uses cached values computed in apply_boundary_source_conditions.
        """
        if not hasattr(self, "_cached_source_velocity"):
            return
        for boundary, indices, values in self._cached_source_velocity:
            if boundary == "left":
                self.grid.vx[0, indices] = values
            elif boundary == "right":
                self.grid.vx[-1, indices] = values
            elif boundary == "bottom":
                self.grid.vy[indices, 0] = values
            elif boundary == "top":
                self.grid.vy[indices, -1] = values

    def _restore_boundary_source_pressure(self) -> None:
        """Restore boundary source pressure after apply_pressure_bc overwrote it.

        This uses cached values computed in apply_boundary_source_conditions.
        """
        if not hasattr(self, "_cached_source_pressure"):
            return
        for boundary, indices, values in self._cached_source_pressure:
            if boundary == "left":
                self.grid.p[0, indices] = values
            elif boundary == "right":
                self.grid.p[-1, indices] = values
            elif boundary == "bottom":
                self.grid.p[indices, 0] = values
            elif boundary == "top":
                self.grid.p[indices, -1] = values

    def record_sensors(self) -> None:
        """Record pressure at all sensor locations."""
        if self.sensors is not None:
            self.sensors.record(self.grid.p, self.time)

    def step_forward(self) -> None:
        """Advance simulation by one time step.

        The update sequence is:
        1. Set boundary source pressure (so velocity update sees source pressure)
        2. Update velocity from pressure gradient
        3. Cache velocity at source boundaries (before BC zeros it)
        4. Apply velocity boundary conditions (zero velocity at rigid walls)
        5. Restore velocity at source boundaries (override the zero)
        6. Update pressure from velocity divergence
        7. Apply pressure boundary conditions (mirror for rigid walls)
        8. Set boundary source pressure (override the mirror)
        9. Apply interior sources
        10. Record sensor data
        11. Advance time
        """
        # Set boundary source pressure so velocity update sees it
        self._set_boundary_source_pressure()
        self.update_velocity()
        # Cache velocity at source boundaries AFTER update but BEFORE BC zeros it
        self._cache_boundary_source_velocity()
        self.apply_velocity_bc()
        # Restore velocity at source boundaries (was zeroed by apply_velocity_bc)
        self._restore_boundary_source_velocity()
        self.update_pressure()
        self.apply_pressure_bc()
        # Set source pressure again (was overwritten by apply_pressure_bc)
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
                Signature: callback(simulation).
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
            Pressure field array (nx+1, ny+1).
        """
        return to_numpy(self.grid.p)

    def save_frame(self, output_dir: Path | str, frame_number: int | None = None) -> None:
        """Save current pressure field to file.

        Args:
            output_dir: Directory for output files.
            frame_number: Frame number for filename. Defaults to current step.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if frame_number is None:
            frame_number = self.step

        filename = output_dir / f"pressure_{frame_number:05d}.npy"
        np.save(filename, self.get_pressure())

    def _log_info(self) -> None:
        """Log simulation parameters."""
        grid = self.grid
        self._logger.info(
            f"Grid: {grid.nx}x{grid.ny} cells, "
            f"dx={grid.dx:.4g}m, dy={grid.dy:.4g}m"
        )
        self._logger.info(
            f"Domain: {grid.size_x:.4g}m x {grid.size_y:.4g}m"
        )
        self._logger.info(f"Time step: {self.dt:.6g}s")
        self._logger.info(f"Material: {self.material}")


class FrameRecorder:
    """Callback for recording simulation frames.

    Use as a callback to Simulation.run() to save pressure
    fields at regular intervals.

    Attributes:
        output_dir: Directory for saved frames.
        frames: List of saved pressure arrays (if keep_in_memory=True).
        times: List of simulation times for each frame.
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        keep_in_memory: bool = False,
    ):
        """Initialize frame recorder.

        Args:
            output_dir: Directory for saved frames. If None, only keeps in memory.
            keep_in_memory: If True, also stores frames in memory.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.keep_in_memory = keep_in_memory
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, sim: Simulation) -> None:
        """Record current frame.

        Args:
            sim: Simulation instance.
        """
        pressure = sim.get_pressure()
        self.times.append(sim.time)

        if self.keep_in_memory:
            self.frames.append(pressure.copy())

        if self.output_dir:
            filename = self.output_dir / f"pressure_{len(self.times):05d}.npy"
            np.save(filename, pressure)
