"""1D FDTD acoustic wave simulation.

This module provides a complete implementation for 1D linear acoustics
using the finite-difference time-domain (FDTD) method with a staggered
Yee grid.

Example:
    >>> from sbimaging.simulators.fdtd.dim1 import (
    ...     Grid, Material, Simulation, BoundarySource, GaussianPulse
    ... )
    >>> grid = Grid.from_domain_size(size_x=1.0, nx=500)
    >>> material = Material.uniform(grid, density=1.0, wave_speed=1.0)
    >>> sim = Simulation(grid, material)
    >>> source = BoundarySource("left", waveform=GaussianPulse(frequency=5.0))
    >>> sim.add_boundary_source(source)
    >>> sim.run(1000)
"""

from sbimaging.simulators.fdtd.dim1.grid import Grid
from sbimaging.simulators.fdtd.dim1.material import Material
from sbimaging.simulators.fdtd.dim1.coefficients import UpdateCoefficients, compute_cfl_timestep
from sbimaging.simulators.fdtd.dim1.source import (
    Waveform,
    GaussianPulse,
    SineWave,
    RickerWavelet,
    Source,
    BoundarySource,
)
from sbimaging.simulators.fdtd.dim1.sensors import SensorArray, generate_boundary_sensors
from sbimaging.simulators.fdtd.dim1.simulation import Simulation, FrameRecorder

__all__ = [
    "Grid",
    "Material",
    "UpdateCoefficients",
    "compute_cfl_timestep",
    "Waveform",
    "GaussianPulse",
    "SineWave",
    "RickerWavelet",
    "Source",
    "BoundarySource",
    "SensorArray",
    "generate_boundary_sensors",
    "Simulation",
    "FrameRecorder",
]
