"""Batch execution for 2D FDTD simulations.

Generates random inclusion configurations and runs simulations,
saving parameters and sensor data for training inverse models.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import time

import numpy as np

from sbimaging.logging import get_logger
from sbimaging.simulators.fdtd.dim2 import (
    Grid,
    Material,
    Simulation,
    BoundarySource,
    GaussianPulse,
    SensorArray,
    generate_boundary_sensors,
)


class InclusionType(Enum):
    """Types of inclusions for 2D simulations."""

    CIRCLE = "circle"
    TRIANGLE = "triangle"
    SQUARE = "square"


@dataclass
class SimulationParameters:
    """Parameters for a single 2D simulation.

    Attributes:
        sim_id: Unique simulation identifier.
        inclusion_type: Type of inclusion (circle, triangle, square).
        center_x: x-coordinate of inclusion center.
        center_y: y-coordinate of inclusion center.
        inclusion_size: Size parameter (radius for circle, side length for square/triangle).
        domain_size: Size of square domain [m].
    """

    sim_id: str
    inclusion_type: InclusionType
    center_x: float
    center_y: float
    inclusion_size: float
    domain_size: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sim_id": self.sim_id,
            "inclusion_type": self.inclusion_type.value,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "inclusion_size": self.inclusion_size,
            "domain_size": self.domain_size,
        }

    @classmethod
    def from_dict(cls, data: dict, config: "BatchConfig | None" = None) -> "SimulationParameters":
        """Create from dictionary.

        Args:
            data: Dictionary with parameter values.
            config: Optional batch config for backwards compatibility with old files
                    that don't have inclusion_size/domain_size.
        """
        inclusion_size = data.get("inclusion_size")
        domain_size = data.get("domain_size")

        if inclusion_size is None or domain_size is None:
            if config is None:
                raise ValueError(
                    "Parameter file missing inclusion_size/domain_size and no config provided"
                )
            inclusion_size = config.inclusion_size
            domain_size = config.domain_size

        return cls(
            sim_id=data["sim_id"],
            inclusion_type=InclusionType(data["inclusion_type"]),
            center_x=data["center_x"],
            center_y=data["center_y"],
            inclusion_size=inclusion_size,
            domain_size=domain_size,
        )


@dataclass
class BatchConfig:
    """Configuration for a batch of 2D simulations.

    Attributes:
        num_simulations: Number of simulations to run.
        domain_size: Size of square domain [m].
        grid_size: Number of grid cells per dimension.
        final_time: Simulation end time [s].
        min_boundary_distance: Minimum distance from inclusion center to boundary [m].
        inclusion_size: Characteristic width of inclusions (diameter for circle, side for square/triangle).
        background_density: Density of background medium [kg/m^3].
        background_speed: Wave speed in background [m/s].
        inclusion_density: Density inside inclusion [kg/m^3].
        inclusion_speed: Wave speed inside inclusion [m/s].
        source_frequency: Source pulse frequency [Hz].
        sensors_per_side: Number of sensors per boundary side.
    """

    num_simulations: int = 500
    domain_size: float = 1.0
    grid_size: int = 500
    final_time: float = 2.0
    min_boundary_distance: float = 0.25
    inclusion_size: float = 0.18
    background_density: float = 1.0
    background_speed: float = 1.0
    inclusion_density: float = 2.0
    inclusion_speed: float = 2.0
    source_frequency: float = 3.0
    sensors_per_side: int = 10

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_simulations": self.num_simulations,
            "domain_size": self.domain_size,
            "grid_size": self.grid_size,
            "final_time": self.final_time,
            "min_boundary_distance": self.min_boundary_distance,
            "inclusion_size": self.inclusion_size,
            "background_density": self.background_density,
            "background_speed": self.background_speed,
            "inclusion_density": self.inclusion_density,
            "inclusion_speed": self.inclusion_speed,
            "source_frequency": self.source_frequency,
            "sensors_per_side": self.sensors_per_side,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchConfig":
        """Create from dictionary."""
        return cls(**data)


class Batch2DPlanner:
    """Plans and generates parameters for 2D FDTD batch simulations."""

    def __init__(self, batch_dir: Path, config: BatchConfig):
        """Initialize batch planner.

        Args:
            batch_dir: Root directory for batch data.
            config: Batch configuration.
        """
        self.batch_dir = Path(batch_dir)
        self.config = config
        self.parameters_dir = self.batch_dir / "parameters"
        self.simulations_dir = self.batch_dir / "simulations"
        self._logger = get_logger(__name__)

    def setup(self) -> None:
        """Create batch directories and save configuration."""
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.parameters_dir.mkdir(exist_ok=True)
        self.simulations_dir.mkdir(exist_ok=True)

        config_path = self.batch_dir / "batch_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self._logger.info(f"Batch directory created at {self.batch_dir}")

    def generate_parameters(self, seed: int | None = None) -> list[SimulationParameters]:
        """Generate random parameters for all simulations.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            List of simulation parameters.
        """
        if seed is not None:
            np.random.seed(seed)

        parameters = []
        inclusion_types = list(InclusionType)

        # Valid center range: [min_dist, domain_size - min_dist]
        min_center = self.config.min_boundary_distance
        max_center = self.config.domain_size - self.config.min_boundary_distance

        for i in range(self.config.num_simulations):
            sim_id = f"sim_{i:05d}"
            inclusion_type = inclusion_types[np.random.randint(len(inclusion_types))]
            center_x = np.random.uniform(min_center, max_center)
            center_y = np.random.uniform(min_center, max_center)

            params = SimulationParameters(
                sim_id=sim_id,
                inclusion_type=inclusion_type,
                center_x=center_x,
                center_y=center_y,
                inclusion_size=self.config.inclusion_size,
                domain_size=self.config.domain_size,
            )
            parameters.append(params)

            # Save individual parameter file
            param_path = self.parameters_dir / f"{sim_id}.json"
            with open(param_path, "w") as f:
                json.dump(params.to_dict(), f, indent=2)

        self._logger.info(f"Generated {len(parameters)} parameter files")
        return parameters

    def find_pending_simulations(self) -> list[str]:
        """Find simulations that haven't completed yet.

        Returns:
            List of simulation IDs that need to run.
        """
        # Get all parameter files
        param_files = list(self.parameters_dir.glob("sim_*.json"))
        all_sim_ids = {f.stem for f in param_files}

        # Find completed simulations (those with sensor_data.npy)
        completed_ids = set()
        if self.simulations_dir.exists():
            for d in self.simulations_dir.iterdir():
                if d.is_dir() and (d / "sensor_data.npy").exists():
                    completed_ids.add(d.name)

        pending = sorted(all_sim_ids - completed_ids)
        self._logger.info(f"Found {len(pending)} pending simulations")
        return pending

    def load_parameters(self, sim_id: str) -> SimulationParameters:
        """Load parameters for a simulation.

        Args:
            sim_id: Simulation identifier.

        Returns:
            Simulation parameters.
        """
        param_path = self.parameters_dir / f"{sim_id}.json"
        with open(param_path) as f:
            data = json.load(f)
        return SimulationParameters.from_dict(data, config=self.config)


class Batch2DExecutor:
    """Executes 2D FDTD simulations."""

    def __init__(self, batch_dir: Path, config: BatchConfig):
        """Initialize batch executor.

        Args:
            batch_dir: Root directory for batch data.
            config: Batch configuration.
        """
        self.batch_dir = Path(batch_dir)
        self.config = config
        self.parameters_dir = self.batch_dir / "parameters"
        self.simulations_dir = self.batch_dir / "simulations"
        self._logger = get_logger(__name__)

    def run_simulation(self, params: SimulationParameters) -> bool:
        """Run a single simulation.

        Args:
            params: Simulation parameters.

        Returns:
            True if simulation completed successfully.
        """
        try:
            self._run_simulation_impl(params)
            return True
        except Exception as e:
            self._logger.error(f"Simulation {params.sim_id} failed: {e}")
            return False

    def _run_simulation_impl(self, params: SimulationParameters) -> None:
        """Implementation of single simulation run."""
        config = self.config

        # Create grid
        grid = Grid.from_domain_size(
            size_x=config.domain_size,
            size_y=config.domain_size,
            nx=config.grid_size,
            ny=config.grid_size,
        )

        # Create material with background properties
        material = Material.uniform(
            grid,
            density=config.background_density,
            wave_speed=config.background_speed,
        )

        # Add inclusion based on type
        # inclusion_size is the characteristic width (diameter for circle, side for square/triangle)
        if params.inclusion_type == InclusionType.CIRCLE:
            material.set_circular_inclusion(
                center_x=params.center_x,
                center_y=params.center_y,
                radius=config.inclusion_size / 2,
                density=config.inclusion_density,
                wave_speed=config.inclusion_speed,
            )
        elif params.inclusion_type == InclusionType.SQUARE:
            half_side = config.inclusion_size / 2
            material.set_rectangular_inclusion(
                x_min=params.center_x - half_side,
                x_max=params.center_x + half_side,
                y_min=params.center_y - half_side,
                y_max=params.center_y + half_side,
                density=config.inclusion_density,
                wave_speed=config.inclusion_speed,
            )
        elif params.inclusion_type == InclusionType.TRIANGLE:
            # Equilateral triangle
            side = config.inclusion_size
            height = (np.sqrt(3) / 2) * side
            vertices = [
                (params.center_x - side / 2, params.center_y - height / 3),
                (params.center_x + side / 2, params.center_y - height / 3),
                (params.center_x, params.center_y + 2 * height / 3),
            ]
            material.set_triangular_inclusion(
                vertices=vertices,
                density=config.inclusion_density,
                wave_speed=config.inclusion_speed,
            )

        # Create simulation
        sim = Simulation(grid, material, courant_factor=0.9)

        # Add boundary source
        waveform = GaussianPulse(
            amplitude=1.0,
            frequency=config.source_frequency,
        )
        source = BoundarySource(
            boundary="left",
            position=0.5,
            width=0.005,
            waveform=waveform,
        )
        sim.add_boundary_source(source)

        # Add sensors
        sensor_locations = generate_boundary_sensors(
            grid,
            sensors_per_side=config.sensors_per_side,
            margin_fraction=0.05,
            offset_cells=2,
        )
        sensors = SensorArray(grid, sensor_locations)
        sim.set_sensors(sensors)

        # Run simulation
        num_steps = int(config.final_time / sim.dt)
        sim.run(num_steps)

        # Save results
        output_dir = self.simulations_dir / params.sim_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save sensor data
        sensor_data = sensors.get_data_matrix()
        np.save(output_dir / "sensor_data.npy", sensor_data)

        # Copy parameter file to output dir for convenience
        param_path = self.parameters_dir / f"{params.sim_id}.json"
        with open(param_path) as f:
            param_data = json.load(f)
        with open(output_dir / "parameters.json", "w") as f:
            json.dump(param_data, f, indent=2)

    def run_batch(
        self,
        pending: list[str],
        progress_callback=None,
    ) -> tuple[int, int]:
        """Run all pending simulations.

        Args:
            pending: List of simulation IDs to run.
            progress_callback: Optional callback(completed, failed, total).

        Returns:
            Tuple of (completed_count, failed_count).
        """
        total = len(pending)
        completed = 0
        failed = 0

        start_time = time.time()

        for i, sim_id in enumerate(pending):
            self._logger.info(f"[{i + 1}/{total}] Running simulation {sim_id}")

            param_path = self.parameters_dir / f"{sim_id}.json"
            with open(param_path) as f:
                data = json.load(f)
            params = SimulationParameters.from_dict(data, config=self.config)

            if self.run_simulation(params):
                completed += 1
            else:
                failed += 1

            if progress_callback:
                progress_callback(completed, failed, total)

        elapsed = time.time() - start_time
        self._logger.info(
            f"Batch complete: {completed} succeeded, {failed} failed in {elapsed:.1f}s"
        )

        return completed, failed


def run_2d_batch(
    batch_dir: Path | str,
    num_simulations: int = 500,
    seed: int | None = None,
    resume: bool = True,
) -> tuple[int, int]:
    """Run a batch of 2D FDTD simulations.

    Args:
        batch_dir: Root directory for batch data.
        num_simulations: Number of simulations to generate/run.
        seed: Random seed for parameter generation.
        resume: If True, skip already completed simulations.

    Returns:
        Tuple of (completed_count, failed_count).
    """
    from sbimaging import configure_logging

    configure_logging()
    logger = get_logger(__name__)

    batch_dir = Path(batch_dir)
    config = BatchConfig(num_simulations=num_simulations)

    # Check if batch already exists
    config_path = batch_dir / "batch_config.json"
    if config_path.exists():
        logger.info(f"Loading existing batch from {batch_dir}")
        with open(config_path) as f:
            config = BatchConfig.from_dict(json.load(f))
    else:
        logger.info(f"Creating new batch at {batch_dir}")
        planner = Batch2DPlanner(batch_dir, config)
        planner.setup()
        planner.generate_parameters(seed=seed)

    # Find pending simulations
    planner = Batch2DPlanner(batch_dir, config)
    pending = planner.find_pending_simulations()

    if not pending:
        logger.info("All simulations completed")
        return 0, 0

    if not resume:
        # If not resuming, run all
        pending = sorted([f.stem for f in planner.parameters_dir.glob("sim_*.json")])

    logger.info(f"Running {len(pending)} simulations")

    # Run simulations
    executor = Batch2DExecutor(batch_dir, config)
    return executor.run_batch(pending)
