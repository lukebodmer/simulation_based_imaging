"""Pressure sources for 1D FDTD simulation.

Implements various source types and waveforms for injecting
acoustic energy into the simulation domain.
"""

from abc import ABC, abstractmethod

import numpy as np

from sbimaging.simulators.fdtd.dim1.grid import Grid


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
        waveform: Waveform object defining temporal behavior.
        is_hard: If True, overwrites pressure; if False, adds to it.
        grid_i: Grid index.
    """

    def __init__(
        self,
        x: float,
        waveform: Waveform | None = None,
        frequency: float = 1000.0,
        amplitude: float = 1.0,
        is_hard: bool = False,
    ):
        """Initialize pressure source.

        Args:
            x: Source x-coordinate [m].
            waveform: Waveform object. If None, uses GaussianPulse.
            frequency: Frequency for default waveform [Hz].
            amplitude: Amplitude for default waveform.
            is_hard: If True, overwrites pressure; if False, adds to it.
        """
        self.x = x
        self.is_hard = is_hard

        if waveform is None:
            self.waveform = GaussianPulse(amplitude=amplitude, frequency=frequency)
        else:
            self.waveform = waveform

        self.grid_i: int | None = None

    def locate_on_grid(self, grid: Grid) -> None:
        """Find grid index corresponding to source location.

        Args:
            grid: Computational grid.
        """
        self.grid_i = grid.index_at_position(self.x)

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
        if self.grid_i is None:
            raise RuntimeError("Source not located on grid. Call locate_on_grid first.")

        pressure = self.get_pressure(time)

        if self.is_hard:
            p[self.grid_i] = pressure
        else:
            p[self.grid_i] += pressure

    def __repr__(self) -> str:
        return (
            f"Source(x={self.x:.4g}, "
            f"waveform={self.waveform.__class__.__name__}, "
            f"hard={self.is_hard})"
        )


class BoundarySource:
    """Pressure source on domain boundary.

    Applies a pressure boundary condition at one end of the 1D domain.

    Attributes:
        boundary: Which boundary ('left' or 'right').
        waveform: Waveform object defining temporal behavior.
    """

    def __init__(
        self,
        boundary: str,
        waveform: Waveform | None = None,
        frequency: float = 1000.0,
        amplitude: float = 1.0,
    ):
        """Initialize boundary source.

        Args:
            boundary: Which boundary ('left' or 'right').
            waveform: Waveform object. If None, uses GaussianPulse.
            frequency: Frequency for default waveform [Hz].
            amplitude: Amplitude for default waveform.
        """
        self.boundary = boundary.lower()
        if self.boundary not in ("left", "right"):
            raise ValueError(f"Invalid boundary: {boundary}. Must be 'left' or 'right'.")

        if waveform is None:
            self.waveform = GaussianPulse(amplitude=amplitude, frequency=frequency)
        else:
            self.waveform = waveform

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
        pressure = self.get_pressure(time)

        if self.boundary == "left":
            p[0] = pressure
        else:  # right
            p[-1] = pressure

    def apply_velocity_bc(self, vx: np.ndarray) -> None:
        """Apply source velocity boundary condition.

        For a pressure source on a rigid boundary, the normal velocity
        is zero (the source drives pressure, not velocity).

        Args:
            vx: Velocity field array (modified in place).
        """
        if self.boundary == "left":
            vx[0] = 0.0
        else:  # right
            vx[-1] = 0.0

    def __repr__(self) -> str:
        return (
            f"BoundarySource(boundary='{self.boundary}', "
            f"waveform={self.waveform.__class__.__name__})"
        )
