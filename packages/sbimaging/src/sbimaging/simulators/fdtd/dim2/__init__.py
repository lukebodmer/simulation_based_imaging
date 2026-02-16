"""2D FDTD simulator for linear acoustics.

This module implements the Finite-Difference Time-Domain method for
simulating 2D acoustic wave propagation using a staggered Yee grid.

Example:
    >>> from sbimaging.simulators.fdtd.dim2 import Grid, Material, Simulation
    >>> grid = Grid(nx=200, ny=200, dx=0.001, dy=0.001)
    >>> material = Material.uniform(grid, density=1.0, wave_speed=343.0)
    >>> sim = Simulation(grid, material)
    >>> sim.add_source(Source(x=0.1, y=0.1, frequency=1000, amplitude=1.0))
    >>> sim.run(num_steps=500)
"""

from sbimaging.simulators.fdtd.dim2.grid import Grid
from sbimaging.simulators.fdtd.dim2.material import Material
from sbimaging.simulators.fdtd.dim2.coefficients import UpdateCoefficients, compute_cfl_timestep
from sbimaging.simulators.fdtd.dim2.source import Source, BoundarySource, GaussianPulse, SineWave, RickerWavelet
from sbimaging.simulators.fdtd.dim2.sensors import SensorArray, generate_boundary_sensors
from sbimaging.simulators.fdtd.dim2.simulation import Simulation, FrameRecorder

__all__ = [
    "Grid",
    "Material",
    "UpdateCoefficients",
    "compute_cfl_timestep",
    "Source",
    "BoundarySource",
    "GaussianPulse",
    "SineWave",
    "RickerWavelet",
    "SensorArray",
    "generate_boundary_sensors",
    "Simulation",
    "FrameRecorder",
]
