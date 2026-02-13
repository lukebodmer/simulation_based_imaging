"""Unified simulation configuration model.

Provides a comprehensive configuration model that combines sweep parameters
(varied per sample) with fixed parameters (same across batch). This serves
as the single source of truth for all simulation parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import toml
import tomli


@dataclass
class SourceConfig:
    """Source configuration for simulation.

    Controls acoustic sources placed on the domain boundary faces.

    Attributes:
        number: Number of sources (typically 6, one per face).
        centers: List of [x, y, z] coordinates for each source center.
        radii: Radius of each source.
        amplitudes: Amplitude of each source.
        frequencies: Frequency of each source in Hz.
    """

    number: int = 6
    centers: list[list[float]] = field(
        default_factory=lambda: [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.0, 0.5],
            [0.5, 1.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
        ]
    )
    radii: list[float] = field(default_factory=lambda: [0.05] * 6)
    amplitudes: list[float] = field(default_factory=lambda: [1.0] * 6)
    frequencies: list[float] = field(default_factory=lambda: [3.0] * 6)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for TOML serialization."""
        return {
            "number": self.number,
            "centers": self.centers,
            "radii": self.radii,
            "amplitudes": self.amplitudes,
            "frequencies": self.frequencies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceConfig":
        """Create from dictionary."""
        return cls(
            number=data.get("number", 6),
            centers=data.get(
                "centers",
                [
                    [0.5, 0.5, 0.0],
                    [0.5, 0.5, 1.0],
                    [0.5, 0.0, 0.5],
                    [0.5, 1.0, 0.5],
                    [0.0, 0.5, 0.5],
                    [1.0, 0.5, 0.5],
                ],
            ),
            radii=data.get("radii", [0.05] * 6),
            amplitudes=data.get("amplitudes", [1.0] * 6),
            frequencies=data.get("frequencies", [3.0] * 6),
        )


@dataclass
class OuterMaterialConfig:
    """Outer (background) material properties.

    These are fixed values, not swept.

    Attributes:
        wave_speed: Wave speed in the outer domain.
        density: Density of the outer domain.
    """

    wave_speed: float = 2.0
    density: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for TOML serialization."""
        return {
            "outer_wave_speed": self.wave_speed,
            "outer_density": self.density,
        }


@dataclass
class MeshConfig:
    """Mesh generation settings.

    Attributes:
        grid_size: Target element size for mesh generation.
        box_size: Size of the simulation domain (cubic).
        inclusion_center: Center point of the inclusion [x, y, z].
    """

    grid_size: float = 0.04
    box_size: float = 1.0
    inclusion_center: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for TOML serialization."""
        return {
            "grid_size": self.grid_size,
            "box_size": self.box_size,
            "inclusion_center": self.inclusion_center,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshConfig":
        """Create from dictionary."""
        return cls(
            grid_size=data.get("grid_size", 0.04),
            box_size=data.get("box_size", 1.0),
            inclusion_center=data.get("inclusion_center", [0.5, 0.5, 0.5]),
        )


@dataclass
class SolverConfig:
    """DG solver settings.

    Attributes:
        polynomial_order: Polynomial order for DG method.
        number_of_timesteps: Total number of timesteps to simulate.
    """

    polynomial_order: int = 1
    number_of_timesteps: int = 10000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for TOML serialization."""
        return {
            "polynomial_order": self.polynomial_order,
            "number_of_timesteps": self.number_of_timesteps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolverConfig":
        """Create from dictionary."""
        return cls(
            polynomial_order=data.get("polynomial_order", 1),
            number_of_timesteps=data.get("number_of_timesteps", 10000),
        )


@dataclass
class ReceiverConfig:
    """Sensor/receiver configuration.

    Attributes:
        sensors_per_face: Number of sensors on each face of the domain.
    """

    sensors_per_face: int = 25

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for TOML serialization."""
        return {
            "sensors_per_face": self.sensors_per_face,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReceiverConfig":
        """Create from dictionary."""
        return cls(
            sensors_per_face=data.get("sensors_per_face", 25),
        )


@dataclass
class OutputConfig:
    """Output interval settings.

    Controls how often various outputs are written during simulation.

    Attributes:
        image: Timesteps between image outputs.
        data: Timesteps between data file outputs.
        points: Timesteps between point data outputs.
        energy: Timesteps between energy computation outputs.
    """

    image: int = 1000
    data: int = 1000
    points: int = 10
    energy: int = 500

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for TOML serialization."""
        return {
            "image": self.image,
            "data": self.data,
            "points": self.points,
            "energy": self.energy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputConfig":
        """Create from dictionary."""
        return cls(
            image=data.get("image", 1000),
            data=data.get("data", 1000),
            points=data.get("points", 10),
            energy=data.get("energy", 500),
        )


@dataclass
class ParameterRange:
    """Range specification for a swept parameter.

    Attributes:
        min_val: Minimum value.
        max_val: Maximum value.
    """

    min_val: float
    max_val: float

    def to_list(self) -> list[float]:
        """Convert to [min, max] list."""
        return [self.min_val, self.max_val]

    @classmethod
    def from_list(cls, data: list[float]) -> "ParameterRange":
        """Create from [min, max] list."""
        return cls(min_val=data[0], max_val=data[1])


@dataclass
class InclusionMaterialConfig:
    """Inclusion material properties (swept).

    These ranges are sampled per simulation.

    Attributes:
        density_range: Range for inclusion density.
        wave_speed_range: Range for inclusion wave speed.
    """

    density_range: ParameterRange = field(
        default_factory=lambda: ParameterRange(1.5, 4.0)
    )
    wave_speed_range: ParameterRange = field(
        default_factory=lambda: ParameterRange(1.5, 4.0)
    )


@dataclass
class InclusionGeometryConfig:
    """Inclusion geometry configuration (swept).

    Attributes:
        scaling_x_range: Range for X-axis scaling.
        scaling_y_range: Range for Y-axis scaling.
        scaling_z_range: Range for Z-axis scaling.
        allow_rotation: Whether inclusion can rotate.
        allow_movement: Whether inclusion can move from center.
        boundary_buffer: Minimum distance from domain boundary.
    """

    scaling_x_range: ParameterRange = field(
        default_factory=lambda: ParameterRange(0.1, 0.3)
    )
    scaling_y_range: ParameterRange = field(
        default_factory=lambda: ParameterRange(0.1, 0.3)
    )
    scaling_z_range: ParameterRange = field(
        default_factory=lambda: ParameterRange(0.1, 0.3)
    )
    allow_rotation: bool = False
    allow_movement: bool = False
    boundary_buffer: float = 0.05


@dataclass
class InclusionTypeConfig:
    """Inclusion type flags.

    Attributes:
        is_sphere: Use spherical inclusion.
        is_ellipsoid_of_revolution: Use ellipsoid of revolution.
        is_multi_cubes: Use multiple cube inclusions.
        is_cube_in_ellipsoid: Use cube embedded in ellipsoid.
    """

    is_sphere: bool = False
    is_ellipsoid_of_revolution: bool = False
    is_multi_cubes: bool = False
    is_cube_in_ellipsoid: bool = False

    @property
    def inclusion_type(self) -> str:
        """Get the inclusion type as a string."""
        if self.is_sphere:
            return "sphere"
        if self.is_multi_cubes:
            return "multi_cubes"
        if self.is_cube_in_ellipsoid:
            return "cube_in_ellipsoid"
        return "ellipsoid"

    @classmethod
    def from_type_string(cls, type_str: str) -> "InclusionTypeConfig":
        """Create from type string."""
        return cls(
            is_sphere=type_str == "sphere",
            is_ellipsoid_of_revolution=type_str == "ellipsoid_of_revolution",
            is_multi_cubes=type_str == "multi_cubes",
            is_cube_in_ellipsoid=type_str == "cube_in_ellipsoid",
        )


@dataclass
class CubeConfig:
    """Cube inclusion configuration (swept).

    Attributes:
        quantity_range: Range for number of cubes [min, max].
        width_range: Range for cube widths.
    """

    quantity_range: tuple[int, int] = (1, 3)
    width_range: ParameterRange = field(
        default_factory=lambda: ParameterRange(0.05, 0.2)
    )


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    Combines fixed parameters (same for all samples in batch) with
    sweep parameters (varied per sample). This is the single source
    of truth for the control panel.

    Fixed parameters:
        - sources: Source configuration
        - outer_material: Background material properties
        - mesh: Mesh generation settings
        - solver: DG solver settings
        - receivers: Sensor configuration
        - output: Output interval settings

    Sweep parameters:
        - inclusion_material: Material property ranges
        - inclusion_geometry: Geometry ranges
        - inclusion_type: Type flags
        - cubes: Cube-specific configuration
    """

    # Fixed parameters
    sources: SourceConfig = field(default_factory=SourceConfig)
    outer_material: OuterMaterialConfig = field(default_factory=OuterMaterialConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    receivers: ReceiverConfig = field(default_factory=ReceiverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Sweep parameters
    inclusion_material: InclusionMaterialConfig = field(
        default_factory=InclusionMaterialConfig
    )
    inclusion_geometry: InclusionGeometryConfig = field(
        default_factory=InclusionGeometryConfig
    )
    inclusion_type: InclusionTypeConfig = field(default_factory=InclusionTypeConfig)
    cubes: CubeConfig = field(default_factory=CubeConfig)

    # Batch metadata
    batch_name: str = "my_batch"
    batch_description: str = ""
    num_samples: int = 100

    def to_base_toml_dict(self) -> dict[str, Any]:
        """Convert fixed parameters to TOML-compatible dictionary.

        This produces a dict suitable for writing as a base config,
        with sweep parameters at their default/center values.
        """
        return {
            "sources": self.sources.to_dict(),
            "material": {
                **self.outer_material.to_dict(),
                "inclusion_density": (
                    self.inclusion_material.density_range.min_val
                    + self.inclusion_material.density_range.max_val
                )
                / 2,
                "inclusion_wave_speed": (
                    self.inclusion_material.wave_speed_range.min_val
                    + self.inclusion_material.wave_speed_range.max_val
                )
                / 2,
            },
            "mesh": {
                **self.mesh.to_dict(),
                "inclusion_scaling": [
                    (
                        self.inclusion_geometry.scaling_x_range.min_val
                        + self.inclusion_geometry.scaling_x_range.max_val
                    )
                    / 2,
                    (
                        self.inclusion_geometry.scaling_y_range.min_val
                        + self.inclusion_geometry.scaling_y_range.max_val
                    )
                    / 2,
                    (
                        self.inclusion_geometry.scaling_z_range.min_val
                        + self.inclusion_geometry.scaling_z_range.max_val
                    )
                    / 2,
                ],
                "inclusion_semi_major_axis_direction": [0.0, 0.0, 1.0],
            },
            "solver": self.solver.to_dict(),
            "receivers": self.receivers.to_dict(),
            "output_intervals": self.output.to_dict(),
        }

    def get_fixed_overrides(self) -> dict[str, Any]:
        """Get fixed parameter overrides for the generator.

        Returns a dict of sections to override in the base config
        when generating parameter files.
        """
        return {
            "sources": self.sources.to_dict(),
            "material": self.outer_material.to_dict(),
            "mesh": {
                "grid_size": self.mesh.grid_size,
                "box_size": self.mesh.box_size,
                "inclusion_center": self.mesh.inclusion_center,
            },
            "solver": self.solver.to_dict(),
            "receivers": self.receivers.to_dict(),
            "output_intervals": self.output.to_dict(),
        }

    @classmethod
    def from_toml(cls, path: Path) -> "SimulationConfig":
        """Load configuration from a TOML file.

        This loads a complete TOML config and extracts the relevant
        sections into a SimulationConfig.
        """
        with open(path, "rb") as f:
            data = tomli.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        """Create from dictionary (e.g., loaded from TOML)."""
        sources_data = data.get("sources", {})
        material_data = data.get("material", {})
        mesh_data = data.get("mesh", {})
        solver_data = data.get("solver", {})
        receivers_data = data.get("receivers", {})
        output_data = data.get("output_intervals", data.get("output", {}))

        return cls(
            sources=SourceConfig.from_dict(sources_data),
            outer_material=OuterMaterialConfig(
                wave_speed=material_data.get("outer_wave_speed", 2.0),
                density=material_data.get("outer_density", 2.0),
            ),
            mesh=MeshConfig.from_dict(mesh_data),
            solver=SolverConfig.from_dict(solver_data),
            receivers=ReceiverConfig.from_dict(receivers_data),
            output=OutputConfig.from_dict(output_data),
        )

    def write_base_config(self, path: Path) -> None:
        """Write this configuration as a base TOML file."""
        with open(path, "w") as f:
            toml.dump(self.to_base_toml_dict(), f)
