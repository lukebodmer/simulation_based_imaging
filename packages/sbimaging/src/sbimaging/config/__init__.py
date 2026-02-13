"""Configuration presets for simulation batches.

This module provides bundled configuration presets for common
simulation scenarios, including base configs and parameter spaces.
"""

from sbimaging.config.preset import (
    ConfigPreset,
    CubeConfig,
    InclusionConfig,
    get_base_config_path,
    list_presets,
    load_preset,
)
from sbimaging.config.simulation import CubeConfig as CubeRangeConfig
from sbimaging.config.simulation import (
    InclusionGeometryConfig,
    InclusionMaterialConfig,
    InclusionTypeConfig,
    MeshConfig,
    OuterMaterialConfig,
    OutputConfig,
    ParameterRange,
    ReceiverConfig,
    SimulationConfig,
    SolverConfig,
    SourceConfig,
)

__all__ = [
    # Preset system
    "ConfigPreset",
    "CubeConfig",
    "InclusionConfig",
    "get_base_config_path",
    "list_presets",
    "load_preset",
    # Simulation config
    "CubeRangeConfig",
    "InclusionGeometryConfig",
    "InclusionMaterialConfig",
    "InclusionTypeConfig",
    "MeshConfig",
    "OuterMaterialConfig",
    "OutputConfig",
    "ParameterRange",
    "ReceiverConfig",
    "SimulationConfig",
    "SolverConfig",
    "SourceConfig",
]
