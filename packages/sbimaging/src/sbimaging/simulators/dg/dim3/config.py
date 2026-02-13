"""Configuration dataclasses for 3D DG simulations.

Provides structured configuration for sources, materials, mesh,
solver settings, receivers, and output intervals.
"""

from dataclasses import dataclass, field
from pathlib import Path

import tomli


@dataclass
class SourceConfig:
    """Configuration for pressure sources.

    Attributes:
        centers: List of (x, y, z) source center coordinates.
        radii: List of source radii.
        amplitudes: List of source pressure amplitudes.
        frequencies: List of source frequencies in Hz.
    """

    centers: list[list[float]] = field(default_factory=lambda: [[0.5, 0.5, 0.0]])
    radii: list[float] = field(default_factory=lambda: [0.05])
    amplitudes: list[float] = field(default_factory=lambda: [0.1])
    frequencies: list[float] = field(default_factory=lambda: [30.0])

    def __post_init__(self) -> None:
        n = len(self.centers)
        if len(self.radii) != n:
            raise ValueError(f"radii length ({len(self.radii)}) must match centers ({n})")
        if len(self.amplitudes) != n:
            raise ValueError(f"amplitudes length ({len(self.amplitudes)}) must match centers ({n})")
        if len(self.frequencies) != n:
            raise ValueError(f"frequencies length ({len(self.frequencies)}) must match centers ({n})")
        for i, center in enumerate(self.centers):
            if len(center) != 3:
                raise ValueError(f"Center {i} must have 3 elements, got {len(center)}")


@dataclass
class MaterialConfig:
    """Configuration for material properties.

    Attributes:
        outer_density: Background material density.
        outer_wave_speed: Background wave speed.
        inclusion_density: Inclusion material density.
        inclusion_wave_speed: Inclusion wave speed.
    """

    outer_density: float = 1.0
    outer_wave_speed: float = 1.5
    inclusion_density: float = 8.0
    inclusion_wave_speed: float = 3.0


@dataclass
class MeshConfig:
    """Configuration for mesh generation/loading.

    Attributes:
        msh_file: Path to existing Gmsh file (optional).
        grid_size: Target element size for mesh generation.
        box_size: Domain size (cube side length).
        inclusion_center: Center of inclusion region.
        inclusion_scaling: Scaling factors for inclusion.
        inclusion_semi_major_axis_direction: Orientation of ellipsoid.
        number_of_cubes: Number of cube inclusions.
        cube_centers: Centers of cube inclusions.
        cube_widths: Widths of cube inclusions.
    """

    msh_file: str | None = None
    grid_size: float = 0.008
    box_size: float = 0.25
    inclusion_center: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    inclusion_scaling: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    inclusion_semi_major_axis_direction: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0]
    )
    number_of_cubes: int = 0
    cube_centers: list[list[float]] = field(default_factory=list)
    cube_widths: list[float] = field(default_factory=list)


@dataclass
class SolverConfig:
    """Configuration for solver settings.

    Attributes:
        polynomial_order: DG polynomial order.
        total_time: Total simulation time (specify this or num_steps).
        num_steps: Number of time steps (specify this or total_time).
        cfl_factor: CFL safety factor for time step calculation.
    """

    polynomial_order: int = 3
    total_time: float | None = None
    num_steps: int | None = None
    cfl_factor: float = 0.9

    def __post_init__(self) -> None:
        if (self.total_time is None) == (self.num_steps is None):
            raise ValueError("Specify exactly one of 'total_time' or 'num_steps'")


@dataclass
class ReceiverConfig:
    """Configuration for sensor/receiver locations.

    Attributes:
        pressure: List of (x, y, z) pressure sensor locations.
        sensors_per_face: Number of sensors per boundary face.
        additional_sensors: Extra sensor locations.
    """

    pressure: list[list[float]] = field(default_factory=list)
    sensors_per_face: int | None = None
    additional_sensors: list[list[float]] = field(default_factory=list)


@dataclass
class OutputConfig:
    """Configuration for output intervals.

    Attributes:
        image_interval: Steps between image saves (0 to disable).
        data_interval: Steps between full data saves (0 to disable).
        sensor_interval: Steps between sensor readings.
        energy_interval: Steps between energy calculations (0 to disable).
    """

    image_interval: int = 0
    data_interval: int = 0
    sensor_interval: int = 10
    energy_interval: int = 0


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    Combines all configuration sections into a single object.
    Can be loaded from TOML files.
    """

    sources: SourceConfig = field(default_factory=SourceConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    receivers: ReceiverConfig = field(default_factory=ReceiverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_toml(cls, path: Path) -> "SimulationConfig":
        """Load configuration from TOML file.

        Args:
            path: Path to TOML configuration file.

        Returns:
            SimulationConfig instance.
        """
        with open(path, "rb") as f:
            data = tomli.load(f)

        return cls._from_data(data)

    @classmethod
    def _from_data(cls, data: dict) -> "SimulationConfig":
        """Create config from dictionary, handling field name mappings."""
        sources_data = _filter_keys(data.get("sources", {}), SourceConfig)
        material_data = _filter_keys(data.get("material", {}), MaterialConfig)
        mesh_data = _filter_keys(data.get("mesh", {}), MeshConfig)
        solver_data = _map_solver_fields(data.get("solver", {}))
        receivers_data = _filter_keys(data.get("receivers", {}), ReceiverConfig)
        output_data = _map_output_fields(
            data.get("output", data.get("output_intervals", {}))
        )

        return cls(
            sources=SourceConfig(**sources_data),
            material=MaterialConfig(**material_data),
            mesh=MeshConfig(**mesh_data),
            solver=SolverConfig(**solver_data),
            receivers=ReceiverConfig(**receivers_data),
            output=OutputConfig(**output_data),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """Load configuration from dictionary.

        Args:
            data: Dictionary with configuration sections.

        Returns:
            SimulationConfig instance.
        """
        return cls._from_data(data)


def _filter_keys(data: dict, dataclass_type: type) -> dict:
    """Filter dictionary to only include keys that exist in the dataclass."""
    import dataclasses

    valid_keys = {f.name for f in dataclasses.fields(dataclass_type)}
    return {k: v for k, v in data.items() if k in valid_keys}


def _map_solver_fields(data: dict) -> dict:
    """Map solver field names from TOML to dataclass."""
    result = {}

    if "polynomial_order" in data:
        result["polynomial_order"] = data["polynomial_order"]

    if "number_of_timesteps" in data:
        result["num_steps"] = data["number_of_timesteps"]
    elif "num_steps" in data:
        result["num_steps"] = data["num_steps"]

    if "total_time" in data:
        result["total_time"] = data["total_time"]

    if "cfl_factor" in data:
        result["cfl_factor"] = data["cfl_factor"]

    return result


def _map_output_fields(data: dict) -> dict:
    """Map output field names from TOML to dataclass."""
    result = {}

    if "image" in data:
        result["image_interval"] = data["image"]
    elif "image_interval" in data:
        result["image_interval"] = data["image_interval"]

    if "data" in data:
        result["data_interval"] = data["data"]
    elif "data_interval" in data:
        result["data_interval"] = data["data_interval"]

    if "points" in data:
        result["sensor_interval"] = data["points"]
    elif "sensor_interval" in data:
        result["sensor_interval"] = data["sensor_interval"]

    if "energy" in data:
        result["energy_interval"] = data["energy"]
    elif "energy_interval" in data:
        result["energy_interval"] = data["energy_interval"]

    return result
