"""Configuration preset loading and management.

Provides functions to discover and load bundled configuration presets
for simulation batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli

from sbimaging.batch.generator import ParameterRange, ParameterSpace

if TYPE_CHECKING:
    from sbimaging.config.simulation import SimulationConfig


@dataclass
class InclusionConfig:
    """Configuration for inclusion geometry and behavior.

    Attributes:
        wave_speed_range: Range for wave speed [min, max].
        density_range: Range for density [min, max].
        scaling_range: Per-axis scaling ranges [[min, max], ...].
        allow_rotation: Whether inclusion can rotate.
        allow_movement: Whether inclusion can move from center.
        is_sphere: Spherical inclusion type.
        is_ellipsoid_of_revolution: Ellipsoid of revolution type.
        is_multi_cubes: Multiple cubes type.
        is_cube_in_ellipsoid: Cube embedded in ellipsoid type.
    """

    wave_speed_range: tuple[float, float] = (1.0, 4.0)
    density_range: tuple[float, float] = (1.0, 4.0)
    scaling_range: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.1, 0.3), (0.1, 0.3), (0.1, 0.3)]
    )
    allow_rotation: bool = False
    allow_movement: bool = False
    is_sphere: bool = False
    is_ellipsoid_of_revolution: bool = False
    is_multi_cubes: bool = False
    is_cube_in_ellipsoid: bool = False


@dataclass
class CubeConfig:
    """Configuration for cube inclusions.

    Attributes:
        quantity_range: Range for number of cubes [min, max].
        width_range: Range for cube widths [min, max].
    """

    quantity_range: tuple[int, int] = (1, 3)
    width_range: tuple[float, float] = (0.05, 0.2)


@dataclass
class SourcePreset:
    """Source configuration preset.

    Attributes:
        number: Number of sources.
        frequency: Default frequency for all sources.
        amplitude: Default amplitude for all sources.
        radius: Default radius for all sources.
    """

    number: int = 6
    frequency: float = 3.0
    amplitude: float = 1.0
    radius: float = 0.05


@dataclass
class OuterMaterialPreset:
    """Outer material preset.

    Attributes:
        wave_speed: Wave speed in outer domain.
        density: Density of outer domain.
    """

    wave_speed: float = 2.0
    density: float = 2.0


@dataclass
class MeshPreset:
    """Mesh configuration preset.

    Attributes:
        grid_size: Target element size.
        box_size: Domain size.
    """

    grid_size: float = 0.04
    box_size: float = 1.0


@dataclass
class SolverPreset:
    """Solver configuration preset.

    Attributes:
        polynomial_order: DG polynomial order.
        number_of_timesteps: Total timesteps (mutually exclusive with total_time).
        total_time: Total simulation time in seconds (mutually exclusive with number_of_timesteps).
    """

    polynomial_order: int = 1
    number_of_timesteps: int | None = 10000
    total_time: float | None = None


@dataclass
class ReceiverPreset:
    """Receiver configuration preset.

    Attributes:
        sensors_per_face: Number of sensors per domain face.
    """

    sensors_per_face: int = 25


@dataclass
class OutputPreset:
    """Output intervals preset.

    Attributes:
        image: Timesteps between image outputs.
        data: Timesteps between data outputs.
        points: Timesteps between point outputs.
        energy: Timesteps between energy outputs.
        save_last_timestep_only: If True, only save image/data on final timestep.
    """

    image: int = 1000
    data: int = 1000
    points: int = 10
    energy: int = 500
    save_last_timestep_only: bool = False


@dataclass
class ConfigPreset:
    """A complete simulation batch preset.

    Attributes:
        name: Unique preset identifier.
        description: Human-readable description.
        base_config: Name of base config file.
        default_num_samples: Suggested number of samples.
        inclusion: Inclusion geometry configuration.
        cubes: Cube-specific configuration.
        boundary_buffer: Minimum distance from domain boundary.
        sources: Optional source configuration preset.
        outer_material: Optional outer material preset.
        mesh: Optional mesh configuration preset.
        solver: Optional solver configuration preset.
        receivers: Optional receiver configuration preset.
        output: Optional output intervals preset.
    """

    name: str
    description: str
    base_config: str
    default_num_samples: int
    inclusion: InclusionConfig
    cubes: CubeConfig
    boundary_buffer: float = 0.05
    # Optional fixed parameter presets
    sources: SourcePreset | None = None
    outer_material: OuterMaterialPreset | None = None
    mesh: MeshPreset | None = None
    solver: SolverPreset | None = None
    receivers: ReceiverPreset | None = None
    output: OutputPreset | None = None

    def to_parameter_space(self) -> ParameterSpace:
        """Convert preset to a ParameterSpace for the generator.

        Returns:
            ParameterSpace configured from this preset.
        """
        return ParameterSpace(
            inclusion_density=ParameterRange(
                self.inclusion.density_range[0],
                self.inclusion.density_range[1],
            ),
            inclusion_speed=ParameterRange(
                self.inclusion.wave_speed_range[0],
                self.inclusion.wave_speed_range[1],
            ),
            inclusion_scaling=ParameterRange(
                self.inclusion.scaling_range[0][0],
                self.inclusion.scaling_range[0][1],
            ),
            cube_width=ParameterRange(
                self.cubes.width_range[0],
                self.cubes.width_range[1],
            ),
            cube_count=self.cubes.quantity_range,
            boundary_buffer=self.boundary_buffer,
        )

    def to_simulation_config(self) -> "SimulationConfig":
        """Convert preset to a SimulationConfig for the generator.

        Returns:
            SimulationConfig with fixed parameters from this preset.
        """
        from sbimaging.config.simulation import (
            MeshConfig,
            OuterMaterialConfig,
            OutputConfig,
            ReceiverConfig,
            SimulationConfig,
            SolverConfig,
            SourceConfig,
        )

        sources = SourceConfig(
            number=self.sources.number if self.sources else 6,
            radii=[self.sources.radius if self.sources else 0.05] * (self.sources.number if self.sources else 6),
            amplitudes=[self.sources.amplitude if self.sources else 1.0] * (self.sources.number if self.sources else 6),
            frequencies=[self.sources.frequency if self.sources else 3.0] * (self.sources.number if self.sources else 6),
        )

        outer_material = OuterMaterialConfig(
            wave_speed=self.outer_material.wave_speed if self.outer_material else 2.0,
            density=self.outer_material.density if self.outer_material else 2.0,
        )

        mesh = MeshConfig(
            grid_size=self.mesh.grid_size if self.mesh else 0.04,
            box_size=self.mesh.box_size if self.mesh else 1.0,
        )

        solver = SolverConfig(
            polynomial_order=self.solver.polynomial_order if self.solver else 1,
            number_of_timesteps=self.solver.number_of_timesteps if self.solver else None,
            total_time=self.solver.total_time if self.solver else None,
        )

        receivers = ReceiverConfig(
            sensors_per_face=self.receivers.sensors_per_face if self.receivers else 25,
        )

        output = OutputConfig(
            image=self.output.image if self.output else 1000,
            data=self.output.data if self.output else 1000,
            points=self.output.points if self.output else 10,
            energy=self.output.energy if self.output else 500,
            save_last_timestep_only=self.output.save_last_timestep_only if self.output else False,
        )

        return SimulationConfig(
            sources=sources,
            outer_material=outer_material,
            mesh=mesh,
            solver=solver,
            receivers=receivers,
            output=output,
            batch_name=self.name,
            batch_description=self.description,
            num_samples=self.default_num_samples,
        )


def _get_config_package():
    """Get the config package for resource loading."""
    return resources.files("sbimaging.config")


def _get_presets_package():
    """Get the presets subpackage for resource loading."""
    return resources.files("sbimaging.config.presets")


def get_base_config_path(filename: str = "base_parameters.toml") -> Path:
    """Get the path to a bundled base configuration file.

    Args:
        filename: Name of the base config file.

    Returns:
        Path to the configuration file.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    config_pkg = _get_config_package()
    resource = config_pkg.joinpath(filename)

    with resources.as_file(resource) as path:
        if not path.exists():
            raise FileNotFoundError(f"Base config not found: {filename}")
        return Path(path)


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        List of preset names (without .toml extension).
    """
    presets_pkg = _get_presets_package()
    preset_names = []

    for item in presets_pkg.iterdir():
        if item.name.endswith(".toml"):
            preset_names.append(item.name[:-5])

    return sorted(preset_names)


def load_preset(name: str) -> ConfigPreset:
    """Load a configuration preset by name.

    Args:
        name: Name of the preset (without .toml extension).

    Returns:
        Loaded ConfigPreset instance.

    Raises:
        FileNotFoundError: If the preset doesn't exist.
        ValueError: If the preset file is invalid.
    """
    presets_pkg = _get_presets_package()
    resource = presets_pkg.joinpath(f"{name}.toml")

    with resources.as_file(resource) as path:
        if not path.exists():
            raise FileNotFoundError(f"Preset not found: {name}")

        with open(path, "rb") as f:
            data = tomli.load(f)

    return _parse_preset(data)


def _parse_preset(data: dict[str, Any]) -> ConfigPreset:
    """Parse a preset dictionary into a ConfigPreset object."""
    preset_section = data.get("preset", {})
    sweep = data.get("sweep", {})
    fixed = data.get("fixed", {})
    inclusion_data = sweep.get("inclusion", {})
    cubes_data = sweep.get("cubes", {})
    geometry_data = sweep.get("geometry", {})

    inclusion = InclusionConfig(
        wave_speed_range=tuple(inclusion_data.get("wave_speed_range", [1.0, 4.0])),
        density_range=tuple(inclusion_data.get("density_range", [1.0, 4.0])),
        scaling_range=[
            tuple(r)
            for r in inclusion_data.get(
                "scaling_range", [[0.1, 0.3], [0.1, 0.3], [0.1, 0.3]]
            )
        ],
        allow_rotation=inclusion_data.get("allow_rotation", False),
        allow_movement=inclusion_data.get("allow_movement", False),
        is_sphere=inclusion_data.get("is_sphere", False),
        is_ellipsoid_of_revolution=inclusion_data.get(
            "is_ellipsoid_of_revolution", False
        ),
        is_multi_cubes=inclusion_data.get("is_multi_cubes", False),
        is_cube_in_ellipsoid=inclusion_data.get("is_cube_in_ellipsoid", False),
    )

    cubes = CubeConfig(
        quantity_range=tuple(cubes_data.get("quantity_range", [1, 3])),
        width_range=tuple(cubes_data.get("width_range", [0.05, 0.2])),
    )

    # Parse optional fixed parameter sections
    sources = None
    if "sources" in fixed:
        src = fixed["sources"]
        sources = SourcePreset(
            number=src.get("number", 6),
            frequency=src.get("frequency", 3.0),
            amplitude=src.get("amplitude", 1.0),
            radius=src.get("radius", 0.05),
        )

    outer_material = None
    if "outer_material" in fixed:
        om = fixed["outer_material"]
        outer_material = OuterMaterialPreset(
            wave_speed=om.get("wave_speed", 2.0),
            density=om.get("density", 2.0),
        )

    mesh = None
    if "mesh" in fixed:
        m = fixed["mesh"]
        mesh = MeshPreset(
            grid_size=m.get("grid_size", 0.04),
            box_size=m.get("box_size", 1.0),
        )

    solver = None
    if "solver" in fixed:
        s = fixed["solver"]
        solver = SolverPreset(
            polynomial_order=s.get("polynomial_order", 1),
            number_of_timesteps=s.get("number_of_timesteps"),
            total_time=s.get("total_time"),
        )

    receivers = None
    if "receivers" in fixed:
        r = fixed["receivers"]
        receivers = ReceiverPreset(
            sensors_per_face=r.get("sensors_per_face", 25),
        )

    output = None
    if "output" in fixed:
        o = fixed["output"]
        output = OutputPreset(
            image=o.get("image", 1000),
            data=o.get("data", 1000),
            points=o.get("points", 10),
            energy=o.get("energy", 500),
            save_last_timestep_only=o.get("save_last_timestep_only", False),
        )

    return ConfigPreset(
        name=preset_section.get("name", "unnamed"),
        description=preset_section.get("description", ""),
        base_config=preset_section.get("base_config", "base_parameters.toml"),
        default_num_samples=preset_section.get("default_num_samples", 100),
        inclusion=inclusion,
        cubes=cubes,
        boundary_buffer=geometry_data.get("boundary_buffer", 0.05),
        sources=sources,
        outer_material=outer_material,
        mesh=mesh,
        solver=solver,
        receivers=receivers,
        output=output,
    )
