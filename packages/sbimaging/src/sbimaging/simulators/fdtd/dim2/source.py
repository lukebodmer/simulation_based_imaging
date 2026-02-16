"""Pressure sources for 2D FDTD simulation.

Implements various source types and waveforms for injecting
acoustic energy into the simulation domain.
"""

from abc import ABC, abstractmethod

import numpy as np

from sbimaging.simulators.fdtd.dim2.grid import Grid


class Waveform(ABC):
    """Abstract base class for source waveforms."""

    @abstractmethod
    def __call__(self, time: float) -> float:
        """Evaluate waveform at given time.

        Args:
            time: Current simulation time [s].

        Returns:
            Waveform amplitude at the given time.
        """
        pass


class GaussianPulse(Waveform):
    """Gaussian pulse waveform.

    Produces a smooth pulse: A * exp(-(t - t0)^2 / (2 * sigma^2))

    Attributes:
        amplitude: Peak amplitude.
        frequency: Characteristic frequency [Hz] (determines pulse width).
        delay: Pulse center time t0 [s].
        sigma: Pulse width parameter [s].
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1000.0,
        delay: float | None = None,
    ):
        """Initialize Gaussian pulse.

        Args:
            amplitude: Peak amplitude.
            frequency: Characteristic frequency [Hz].
            delay: Pulse center time [s]. Defaults to 1/(2*frequency).
        """
        self.amplitude = amplitude
        self.frequency = frequency

        if delay is None:
            self.delay = 1.0 / (2.0 * frequency)
        else:
            self.delay = delay

        self.sigma = self.delay / 4.0

    def __call__(self, time: float) -> float:
        """Evaluate Gaussian pulse at given time."""
        t0 = self.delay
        sigma = self.sigma
        return self.amplitude * np.exp(-((time - t0) ** 2) / (2.0 * sigma**2))


class SineWave(Waveform):
    """Continuous sinusoidal waveform.

    Produces: A * sin(2 * pi * f * t)

    Attributes:
        amplitude: Wave amplitude.
        frequency: Wave frequency [Hz].
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1000.0,
    ):
        """Initialize sine wave.

        Args:
            amplitude: Wave amplitude.
            frequency: Wave frequency [Hz].
        """
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, time: float) -> float:
        """Evaluate sine wave at given time."""
        return self.amplitude * np.sin(2.0 * np.pi * self.frequency * time)


class RickerWavelet(Waveform):
    """Ricker wavelet (Mexican hat) waveform.

    Common in seismic simulations. Produces:
    A * (1 - 2*(pi*f*(t-t0))^2) * exp(-(pi*f*(t-t0))^2)

    Attributes:
        amplitude: Peak amplitude.
        frequency: Central frequency [Hz].
        delay: Wavelet center time [s].
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1000.0,
        delay: float | None = None,
    ):
        """Initialize Ricker wavelet.

        Args:
            amplitude: Peak amplitude.
            frequency: Central frequency [Hz].
            delay: Wavelet center time [s]. Defaults to 1.5/frequency.
        """
        self.amplitude = amplitude
        self.frequency = frequency

        if delay is None:
            self.delay = 1.5 / frequency
        else:
            self.delay = delay

    def __call__(self, time: float) -> float:
        """Evaluate Ricker wavelet at given time."""
        f = self.frequency
        t0 = self.delay
        tau = np.pi * f * (time - t0)
        return self.amplitude * (1.0 - 2.0 * tau**2) * np.exp(-(tau**2))


class Source:
    """Pressure source at a specific location.

    Injects a time-varying pressure signal at a point in the domain.
    Can be configured as either a soft source (adds to existing field)
    or hard source (overwrites field value).

    Attributes:
        x: Source x-coordinate [m].
        y: Source y-coordinate [m].
        waveform: Waveform object defining temporal behavior.
        is_hard: If True, overwrites pressure; if False, adds to it.
        grid_i: Grid index in x direction.
        grid_j: Grid index in y direction.
    """

    def __init__(
        self,
        x: float,
        y: float,
        waveform: Waveform | None = None,
        frequency: float = 1000.0,
        amplitude: float = 1.0,
        is_hard: bool = False,
    ):
        """Initialize pressure source.

        Args:
            x: Source x-coordinate [m].
            y: Source y-coordinate [m].
            waveform: Waveform object. If None, uses GaussianPulse.
            frequency: Frequency for default waveform [Hz].
            amplitude: Amplitude for default waveform.
            is_hard: If True, overwrites pressure; if False, adds to it.
        """
        self.x = x
        self.y = y
        self.is_hard = is_hard

        if waveform is None:
            self.waveform = GaussianPulse(amplitude=amplitude, frequency=frequency)
        else:
            self.waveform = waveform

        self.grid_i: int | None = None
        self.grid_j: int | None = None

    def locate_on_grid(self, grid: Grid) -> None:
        """Find grid indices corresponding to source location.

        Args:
            grid: Computational grid.
        """
        self.grid_i, self.grid_j = grid.index_at_position(self.x, self.y)

    def get_pressure(self, time: float) -> float:
        """Get source pressure at given time.

        Args:
            time: Current simulation time [s].

        Returns:
            Source pressure value.
        """
        return self.waveform(time)

    def apply(self, p: np.ndarray, time: float) -> None:
        """Apply source to pressure field.

        Args:
            p: Pressure field array (modified in place).
            time: Current simulation time [s].
        """
        if self.grid_i is None or self.grid_j is None:
            raise RuntimeError("Source not located on grid. Call locate_on_grid first.")

        pressure = self.get_pressure(time)

        if self.is_hard:
            p[self.grid_i, self.grid_j] = pressure
        else:
            p[self.grid_i, self.grid_j] += pressure

    def __repr__(self) -> str:
        return (
            f"Source(x={self.x:.4g}, y={self.y:.4g}, "
            f"waveform={self.waveform.__class__.__name__}, "
            f"hard={self.is_hard})"
        )


class BoundarySource:
    """Pressure source on domain boundary.

    Applies a pressure boundary condition on a segment of the domain
    boundary, similar to how the 3D DG code handles sources. This avoids
    the artifacts that can occur with interior hard sources.

    The source sets pressure on the boundary and adjusts the normal
    velocity using an impedance-matched condition.

    Attributes:
        boundary: Which boundary ('left', 'right', 'bottom', 'top').
        y_min: Minimum y-coordinate of source region (for left/right).
        y_max: Maximum y-coordinate of source region (for left/right).
        x_min: Minimum x-coordinate of source region (for bottom/top).
        x_max: Maximum x-coordinate of source region (for bottom/top).
        waveform: Waveform object defining temporal behavior.
    """

    def __init__(
        self,
        boundary: str,
        position: float,
        width: float,
        waveform: Waveform | None = None,
        frequency: float = 1000.0,
        amplitude: float = 1.0,
    ):
        """Initialize boundary source.

        Args:
            boundary: Which boundary ('left', 'right', 'bottom', 'top').
            position: Center position along the boundary [m].
            width: Width of source region [m].
            waveform: Waveform object. If None, uses GaussianPulse.
            frequency: Frequency for default waveform [Hz].
            amplitude: Amplitude for default waveform.
        """
        self.boundary = boundary.lower()
        if self.boundary not in ("left", "right", "bottom", "top"):
            raise ValueError(f"Invalid boundary: {boundary}")

        self.position = position
        self.width = width

        if waveform is None:
            self.waveform = GaussianPulse(amplitude=amplitude, frequency=frequency)
        else:
            self.waveform = waveform

        # Grid indices for source region (set by locate_on_grid)
        self._j_indices: np.ndarray | None = None  # For left/right boundaries
        self._i_indices: np.ndarray | None = None  # For bottom/top boundaries

    def locate_on_grid(self, grid: Grid) -> None:
        """Find grid indices for source region.

        Args:
            grid: Computational grid.
        """
        half_width = self.width / 2.0

        if self.boundary in ("left", "right"):
            # Source spans y from position - half_width to position + half_width
            y_min = self.position - half_width
            y_max = self.position + half_width
            j_min = max(0, int(np.floor(y_min / grid.dy)))
            j_max = min(grid.ny, int(np.ceil(y_max / grid.dy)))
            self._j_indices = np.arange(j_min, j_max + 1)
        else:
            # Source spans x from position - half_width to position + half_width
            x_min = self.position - half_width
            x_max = self.position + half_width
            i_min = max(0, int(np.floor(x_min / grid.dx)))
            i_max = min(grid.nx, int(np.ceil(x_max / grid.dx)))
            self._i_indices = np.arange(i_min, i_max + 1)

    def get_pressure(self, time: float) -> float:
        """Get source pressure at given time.

        Args:
            time: Current simulation time [s].

        Returns:
            Source pressure value.
        """
        return self.waveform(time)

    def apply_pressure_bc(self, p: np.ndarray, time: float) -> None:
        """Apply source pressure boundary condition.

        Args:
            p: Pressure field array (modified in place).
            time: Current simulation time [s].
        """
        if self._j_indices is None and self._i_indices is None:
            raise RuntimeError("Source not located on grid. Call locate_on_grid first.")

        pressure = self.get_pressure(time)

        if self.boundary == "left" and self._j_indices is not None:
            p[0, self._j_indices] = pressure
        elif self.boundary == "right" and self._j_indices is not None:
            p[-1, self._j_indices] = pressure
        elif self.boundary == "bottom" and self._i_indices is not None:
            p[self._i_indices, 0] = pressure
        elif self.boundary == "top" and self._i_indices is not None:
            p[self._i_indices, -1] = pressure

    def apply_velocity_bc(self, vx: np.ndarray, vy: np.ndarray) -> None:
        """Apply source velocity boundary condition.

        For a pressure source on a rigid boundary, the normal velocity
        is zero (the source drives pressure, not velocity).

        Args:
            vx: x-velocity field array (modified in place).
            vy: y-velocity field array (modified in place).
        """
        if self._j_indices is None and self._i_indices is None:
            raise RuntimeError("Source not located on grid. Call locate_on_grid first.")

        # Normal velocity remains zero at source boundary
        # (source drives pressure, wall remains rigid for velocity)
        if self.boundary == "left" and self._j_indices is not None:
            vx[0, self._j_indices[:-1]] = 0.0
        elif self.boundary == "right" and self._j_indices is not None:
            vx[-1, self._j_indices[:-1]] = 0.0
        elif self.boundary == "bottom" and self._i_indices is not None:
            vy[self._i_indices[:-1], 0] = 0.0
        elif self.boundary == "top" and self._i_indices is not None:
            vy[self._i_indices[:-1], -1] = 0.0

    def __repr__(self) -> str:
        return (
            f"BoundarySource(boundary='{self.boundary}', "
            f"position={self.position:.4g}, width={self.width:.4g}, "
            f"waveform={self.waveform.__class__.__name__})"
        )
