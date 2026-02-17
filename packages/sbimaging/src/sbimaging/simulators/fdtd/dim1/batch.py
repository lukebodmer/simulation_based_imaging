"""Batch execution for 1D FDTD simulations.

Generates random inclusion configurations and runs simulations,
saving parameters and sensor data for training inverse models.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np

from sbimaging.logging import get_logger
from sbimaging.simulators.fdtd.dim1 import (
    Grid,
    Material,
    Simulation,
    BoundarySource,
    GaussianPulse,
    SensorArray,
    generate_boundary_sensors,
)


@dataclass
class SimulationParameters:
    """Parameters for a single 1D simulation.

    Attributes:
        sim_id: Unique simulation identifier.
        inclusion_center: x-coordinate of inclusion center.
        inclusion_size: Width of the inclusion.
        inclusion_density: Density inside inclusion.
        inclusion_speed: Wave speed inside inclusion.
        domain_size: Size of domain [m].
    """

    sim_id: str
    inclusion_center: float
    inclusion_size: float
    inclusion_density: float
    inclusion_speed: float
    domain_size: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sim_id": self.sim_id,
            "inclusion_center": self.inclusion_center,
            "inclusion_size": self.inclusion_size,
            "inclusion_density": self.inclusion_density,
            "inclusion_speed": self.inclusion_speed,
            "domain_size": self.domain_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationParameters":
        """Create from dictionary."""
        return cls(
            sim_id=data["sim_id"],
            inclusion_center=data["inclusion_center"],
            inclusion_size=data["inclusion_size"],
            inclusion_density=data["inclusion_density"],
            inclusion_speed=data["inclusion_speed"],
            domain_size=data["domain_size"],
        )


@dataclass
class BatchConfig:
    """Configuration for a batch of 1D simulations.

    Attributes:
        num_simulations: Number of simulations to run.
        domain_size: Size of domain [m].
        grid_size: Number of grid cells.
        final_time: Simulation end time [s].
        boundary_margin: Minimum distance from inclusion edge to boundary [m].
        min_inclusion_size: Minimum inclusion width.
        max_inclusion_size: Maximum inclusion width.
        min_inclusion_density: Minimum density inside inclusion.
        max_inclusion_density: Maximum density inside inclusion.
        min_inclusion_speed: Minimum wave speed inside inclusion.
        max_inclusion_speed: Maximum wave speed inside inclusion.
        background_density: Density of background medium [kg/m^3].
        background_speed: Wave speed in background [m/s].
        source_frequency: Source pulse frequency [Hz].
    """

    num_simulations: int = 500
    domain_size: float = 1.0
    grid_size: int = 500
    final_time: float = 4.0
    boundary_margin: float = 0.1
    min_inclusion_size: float = 0.1
    max_inclusion_size: float = 0.6
    min_inclusion_density: float = 2.0
    max_inclusion_density: float = 4.0
    min_inclusion_speed: float = 2.0
    max_inclusion_speed: float = 4.0
    background_density: float = 1.0
    background_speed: float = 1.0
    source_frequency: float = 5.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_simulations": self.num_simulations,
            "domain_size": self.domain_size,
            "grid_size": self.grid_size,
            "final_time": self.final_time,
            "boundary_margin": self.boundary_margin,
            "min_inclusion_size": self.min_inclusion_size,
            "max_inclusion_size": self.max_inclusion_size,
            "min_inclusion_density": self.min_inclusion_density,
            "max_inclusion_density": self.max_inclusion_density,
            "min_inclusion_speed": self.min_inclusion_speed,
            "max_inclusion_speed": self.max_inclusion_speed,
            "background_density": self.background_density,
            "background_speed": self.background_speed,
            "source_frequency": self.source_frequency,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchConfig":
        """Create from dictionary."""
        return cls(**data)

    def compute_fixed_timestep(self, courant_factor: float = 0.9) -> float:
        """Compute a fixed timestep valid for all simulations in the batch.

        Uses the maximum possible wave speed to ensure CFL stability
        for all simulations regardless of their specific inclusion properties.

        Args:
            courant_factor: Safety factor for CFL condition.

        Returns:
            Fixed timestep [s].
        """
        dx = self.domain_size / self.grid_size
        # Use maximum wave speed across background and all possible inclusions
        max_speed = max(self.background_speed, self.max_inclusion_speed)
        dt = courant_factor * dx / max_speed
        return dt


class Batch1DPlanner:
    """Plans and generates parameters for 1D FDTD batch simulations."""

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
        config = self.config

        for i in range(config.num_simulations):
            sim_id = f"sim_{i:05d}"

            # Random inclusion size
            inclusion_size = np.random.uniform(
                config.min_inclusion_size, config.max_inclusion_size
            )

            # Valid center range: inclusion must stay boundary_margin away from edges
            # center - size/2 >= boundary_margin  =>  center >= boundary_margin + size/2
            # center + size/2 <= domain_size - boundary_margin  =>  center <= domain_size - boundary_margin - size/2
            min_center = config.boundary_margin + inclusion_size / 2
            max_center = config.domain_size - config.boundary_margin - inclusion_size / 2

            inclusion_center = np.random.uniform(min_center, max_center)

            # Random density and wave speed
            inclusion_density = np.random.uniform(
                config.min_inclusion_density, config.max_inclusion_density
            )
            inclusion_speed = np.random.uniform(
                config.min_inclusion_speed, config.max_inclusion_speed
            )

            params = SimulationParameters(
                sim_id=sim_id,
                inclusion_center=inclusion_center,
                inclusion_size=inclusion_size,
                inclusion_density=inclusion_density,
                inclusion_speed=inclusion_speed,
                domain_size=config.domain_size,
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
        return SimulationParameters.from_dict(data)


class Batch1DExecutor:
    """Executes 1D FDTD simulations."""

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
            nx=config.grid_size,
        )

        # Create material with background properties
        material = Material.uniform(
            grid,
            density=config.background_density,
            wave_speed=config.background_speed,
        )

        # Add inclusion
        inclusion_x_min = params.inclusion_center - params.inclusion_size / 2
        inclusion_x_max = params.inclusion_center + params.inclusion_size / 2
        material.set_inclusion(
            x_min=inclusion_x_min,
            x_max=inclusion_x_max,
            density=params.inclusion_density,
            wave_speed=params.inclusion_speed,
        )

        # Create simulation with fixed timestep for consistent sensor data across batch
        fixed_dt = config.compute_fixed_timestep(courant_factor=0.9)
        sim = Simulation(grid, material, dt=fixed_dt)

        # Add boundary source
        waveform = GaussianPulse(
            amplitude=1.0,
            frequency=config.source_frequency,
        )
        source = BoundarySource(
            boundary="left",
            waveform=waveform,
        )
        sim.add_boundary_source(source)

        # Add sensors at both ends
        sensor_locations = generate_boundary_sensors(grid, offset_cells=2)
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
            params = SimulationParameters.from_dict(data)

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


def run_1d_batch(
    batch_dir: Path | str,
    num_simulations: int = 500,
    seed: int | None = None,
    resume: bool = True,
) -> tuple[int, int]:
    """Run a batch of 1D FDTD simulations.

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
        planner = Batch1DPlanner(batch_dir, config)
        planner.setup()
        planner.generate_parameters(seed=seed)

    # Find pending simulations
    planner = Batch1DPlanner(batch_dir, config)
    pending = planner.find_pending_simulations()

    if not pending:
        logger.info("All simulations completed")
        return 0, 0

    if not resume:
        # If not resuming, run all
        pending = sorted([f.stem for f in planner.parameters_dir.glob("sim_*.json")])

    logger.info(f"Running {len(pending)} simulations")

    # Run simulations
    executor = Batch1DExecutor(batch_dir, config)
    return executor.run_batch(pending)
