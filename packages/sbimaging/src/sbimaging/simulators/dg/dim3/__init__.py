"""3D Discontinuous Galerkin simulator for acoustic wave propagation.

This module provides a complete DG solver for linear acoustics in 3D,
including:

- Reference element with Warp & Blend nodes
- Mesh loading from Gmsh files
- Acoustics operator with upwind flux
- Low-storage Runge-Kutta time stepping
- Configuration from TOML files
- Sensor field evaluation
- Output handling for data and energy

Example usage:
    from pathlib import Path
    from sbimaging.simulators.dg.dim3 import run_simulation

    run_simulation(
        config_path=Path("config.toml"),
        output_dir=Path("output"),
        mesh_file=Path("mesh.msh"),
    )
"""

from sbimaging.simulators.dg.dim3.acoustics import AcousticsOperator, Source
from sbimaging.simulators.dg.dim3.config import (
    MaterialConfig,
    MeshConfig,
    OutputConfig,
    ReceiverConfig,
    SimulationConfig,
    SolverConfig,
    SourceConfig,
)
from sbimaging.simulators.dg.dim3.mesh import MeshGeometry, MeshLoader
from sbimaging.simulators.dg.dim3.output import EnergyCalculator, SimulationOutput
from sbimaging.simulators.dg.dim3.reference_element import (
    ReferenceOperators,
    ReferenceTetrahedron,
)
from sbimaging.simulators.dg.dim3.runner import SimulationRunner, run_simulation
from sbimaging.simulators.dg.dim3.sensors import SensorArray, generate_grid_sensors
from sbimaging.simulators.dg.dim3.time_stepping import (
    LowStorageRungeKutta,
    compute_cfl_timestep,
)

__all__ = [
    # Core components
    "AcousticsOperator",
    "LowStorageRungeKutta",
    "MeshGeometry",
    "MeshLoader",
    "ReferenceOperators",
    "ReferenceTetrahedron",
    "Source",
    "compute_cfl_timestep",
    # Configuration
    "MaterialConfig",
    "MeshConfig",
    "OutputConfig",
    "ReceiverConfig",
    "SimulationConfig",
    "SolverConfig",
    "SourceConfig",
    # Sensors and output
    "EnergyCalculator",
    "SensorArray",
    "SimulationOutput",
    "generate_grid_sensors",
    # Runner
    "SimulationRunner",
    "run_simulation",
]
