"""Batch execution for running multiple simulations.

Handles sequential execution of simulation batches with progress
tracking and error handling.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from sbimaging.logging import get_logger
from sbimaging.simulators.dg.dim3.config import SimulationConfig
from sbimaging.simulators.dg.dim3.runner import SimulationRunner

if TYPE_CHECKING:
    from sbimaging.batch.generator import ParameterSpace
    from sbimaging.config.simulation import SimulationConfig as UISimulationConfig


class BatchExecutor:
    """Executes batches of simulations.

    Runs pending simulations with shared time step and mesh reuse.

    Attributes:
        batch_dir: Root directory for batch data.
        mesh_dir: Directory for mesh files.
        simulations_dir: Directory for simulation outputs.
    """

    def __init__(self, batch_dir: Path):
        """Initialize batch executor.

        Args:
            batch_dir: Root directory for batch data.
        """
        self.batch_dir = Path(batch_dir)
        self.parameter_dir = self.batch_dir / "parameter_files"
        self.mesh_dir = self.batch_dir / "meshes"
        self.simulations_dir = self.batch_dir / "simulations"

        self._logger = get_logger(__name__)
        self._completed = 0
        self._failed = 0

    def run_all(
        self,
        pending: list[str],
        global_dt: float,
        mesh_file_resolver: Callable | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> tuple[int, int]:
        """Run all pending simulations.

        Args:
            pending: List of simulation hashes to run.
            global_dt: Global time step to use for all simulations.
            mesh_file_resolver: Optional function to resolve mesh file paths.
                Signature: (sim_hash: str) -> Path
            progress_callback: Optional callback for progress updates.
                Signature: (pending: int, completed: int, failed: int) -> None

        Returns:
            Tuple of (completed_count, failed_count).
        """
        self._logger.info(f"Running {len(pending)} simulations")
        self._completed = 0
        self._failed = 0
        total = len(pending)

        start_time = time.time()

        if progress_callback:
            progress_callback(total, 0, 0)

        for i, sim_hash in enumerate(pending):
            self._logger.info(f"[{i + 1}/{total}] Running simulation {sim_hash}")

            try:
                self._run_single(sim_hash, global_dt, mesh_file_resolver)
                self._completed += 1
            except Exception as e:
                self._logger.error(f"Simulation {sim_hash} failed: {e}")
                self._failed += 1

            if progress_callback:
                remaining = total - (self._completed + self._failed)
                progress_callback(remaining, self._completed, self._failed)

        elapsed = time.time() - start_time
        self._logger.info(
            f"Batch complete: {self._completed} succeeded, "
            f"{self._failed} failed in {elapsed:.1f}s"
        )

        return self._completed, self._failed

    def run_single(
        self,
        sim_hash: str,
        global_dt: float,
        mesh_file: Path | None = None,
    ) -> bool:
        """Run a single simulation.

        Args:
            sim_hash: Simulation hash (parameter file stem).
            global_dt: Time step to use.
            mesh_file: Path to mesh file (optional).

        Returns:
            True if simulation completed successfully.
        """
        try:
            self._run_single(sim_hash, global_dt, lambda _: mesh_file)
            return True
        except Exception as e:
            self._logger.error(f"Simulation {sim_hash} failed: {e}")
            return False

    def _run_single(
        self,
        sim_hash: str,
        global_dt: float,
        mesh_resolver: Callable | None,
    ) -> None:
        """Execute a single simulation."""
        param_file = self.parameter_dir / f"{sim_hash}.toml"
        if not param_file.exists():
            raise FileNotFoundError(f"Parameter file not found: {param_file}")

        output_dir = self.simulations_dir / sim_hash
        output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(param_file, output_dir / "config.toml")

        config = SimulationConfig.from_toml(param_file)

        # Convert total_time to num_steps using global_dt
        if config.solver.num_steps is None and config.solver.total_time is not None:
            config.solver.num_steps = int(config.solver.total_time / global_dt)
            config.solver.total_time = None
        elif config.solver.num_steps is None:
            raise ValueError(f"Cannot determine num_steps for {sim_hash}")

        mesh_file = None
        if mesh_resolver:
            mesh_file = mesh_resolver(sim_hash)

        if mesh_file is None and config.mesh.msh_file:
            mesh_file = Path(config.mesh.msh_file)

        runner = SimulationRunner(
            config=config,
            output_dir=output_dir,
            mesh_file=mesh_file,
            global_dt=global_dt,
        )
        runner.setup()

        # Save mesh as pickle for visualization
        mesh_pkl_path = output_dir / "mesh.pkl"
        if not mesh_pkl_path.exists():
            self._save_mesh_with_materials(runner, mesh_pkl_path)

        runner.run()

    @property
    def completed_count(self) -> int:
        """Number of completed simulations."""
        return self._completed

    @property
    def failed_count(self) -> int:
        """Number of failed simulations."""
        return self._failed

    def _save_mesh_with_materials(
        self, runner: SimulationRunner, output_path: Path
    ) -> None:
        """Save mesh geometry and material properties to pickle.

        Args:
            runner: SimulationRunner with loaded mesh and materials.
            output_path: Path to save pickle file.
        """
        import pickle

        from sbimaging.array.backend import to_numpy

        mesh = runner._mesh
        if mesh is None:
            self._logger.warning("Cannot save mesh pickle: mesh not loaded")
            return

        data = {
            "num_vertices": mesh.num_vertices,
            "num_cells": mesh.num_cells,
            "smallest_diameter": mesh.smallest_diameter,
            "vertex_coordinates": to_numpy(mesh.vertex_coordinates),
            "cell_to_vertices": to_numpy(mesh.cell_to_vertices),
            "x": to_numpy(mesh.x),
            "y": to_numpy(mesh.y),
            "z": to_numpy(mesh.z),
            "cell_to_cells": to_numpy(mesh.cell_to_cells),
            "cell_to_faces": to_numpy(mesh.cell_to_faces),
            "jacobians": to_numpy(mesh.jacobians),
            "speed": to_numpy(runner._speed),
            "density": to_numpy(runner._density),
            "speed_per_cell": to_numpy(runner._speed[0, :]),
        }

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        self._logger.info(f"Saved mesh pickle to {output_path}")


def resume_batch(
    batch_dir: Path,
    progress_callback: Callable[[int, int, int], None] | None = None,
    total_simulations_callback: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    """Resume a batch of simulations from where it left off.

    This function skips parameter and mesh generation, assuming the batch
    was previously set up. It finds pending simulations and runs them.

    Args:
        batch_dir: Root directory for batch data.
        progress_callback: Optional callback for progress updates.
            Signature: (pending: int, completed: int, failed: int) -> None
        total_simulations_callback: Optional callback to report total simulations.
            Signature: (total: int) -> None

    Returns:
        Tuple of (completed_count, failed_count).

    Raises:
        FileNotFoundError: If batch_metadata.toml doesn't exist.
        ValueError: If no pending simulations found or metadata invalid.
    """
    from sbimaging.batch.planner import BatchPlanner

    logger = get_logger(__name__)
    batch_dir = Path(batch_dir)

    planner = BatchPlanner(batch_dir)

    # Load existing metadata (required for resume)
    if not planner.load_metadata():
        raise FileNotFoundError(
            f"No batch_metadata.toml found in {batch_dir}. "
            "Cannot resume - the batch may not have been set up properly."
        )

    planner.compute_mesh_hashes()
    pending = planner.find_pending_simulations()

    if not pending:
        logger.info("No pending simulations - batch is complete")
        return 0, 0

    global_dt = planner.global_dt
    if global_dt is None:
        raise ValueError("No global time step in metadata - batch may be corrupted")

    def mesh_resolver(sim_hash: str) -> Path | None:
        mesh_hash = planner.get_mesh_hash_for_simulation(sim_hash)
        if mesh_hash:
            return planner.get_mesh_file(mesh_hash)
        return None

    # Report total simulations count
    if total_simulations_callback:
        total_simulations_callback(len(pending))

    logger.info(f"Resuming batch with {len(pending)} pending simulations")

    executor = BatchExecutor(batch_dir)
    return executor.run_all(
        pending=pending,
        global_dt=global_dt,
        mesh_file_resolver=mesh_resolver,
        progress_callback=progress_callback,
    )


def run_batch(
    batch_dir: Path,
    base_config_path: Path | None = None,
    num_samples: int | None = None,
    mesh_file: Path | None = None,
    parameter_space: ParameterSpace | None = None,
    geometry_type: str = "ellipsoid",
    simulation_config: UISimulationConfig | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    mesh_progress_callback: Callable[[int, int], None] | None = None,
    total_simulations_callback: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    """Run a batch of simulations.

    This is a convenience function that combines planning and execution.

    Args:
        batch_dir: Root directory for batch data.
        base_config_path: Path to base config for generating new samples.
        num_samples: Number of samples to generate (if base_config provided).
        mesh_file: Path to mesh file for all simulations (overrides generation).
        parameter_space: Optional ParameterSpace defining sweep ranges.
        geometry_type: Type of geometry to generate ("ellipsoid", "sphere",
            "multi_cubes", "cube_in_ellipsoid").
        simulation_config: Optional SimulationConfig with fixed parameter overrides.
        progress_callback: Optional callback for progress updates.
            Signature: (pending: int, completed: int, failed: int) -> None
        mesh_progress_callback: Optional callback for mesh generation progress.
            Signature: (generated: int, total: int) -> None
        total_simulations_callback: Optional callback to report total simulations.
            Signature: (total: int) -> None

    Returns:
        Tuple of (completed_count, failed_count).
    """
    from sbimaging.batch.generator import ParameterGenerator
    from sbimaging.batch.planner import BatchPlanner
    from sbimaging.meshing import GeometryType

    logger = get_logger(__name__)
    batch_dir = Path(batch_dir)

    geometry_type_enum = GeometryType(geometry_type)

    if base_config_path and num_samples:
        logger.info(f"Generating {num_samples} parameter files")
        generator = ParameterGenerator(
            base_config_path=base_config_path,
            output_dir=batch_dir / "parameter_files",
            space=parameter_space,
            simulation_config=simulation_config,
        )
        if geometry_type in ("multi_cubes", "cube_in_ellipsoid"):
            generator.generate_cube_samples(num_samples)
        else:
            generator.generate(num_samples)

    planner = BatchPlanner(batch_dir)
    planner.compute_mesh_hashes()
    pending = planner.find_pending_simulations()

    if not pending:
        logger.info("No pending simulations")
        return 0, 0

    if not planner.load_metadata():
        if mesh_file:
            from sbimaging.simulators.dg.dim3.mesh import MeshLoader
            from sbimaging.simulators.dg.dim3.reference_element import (
                ReferenceOperators,
                ReferenceTetrahedron,
            )

            for mesh_hash, info in planner.mesh_info.items():
                elem = ReferenceTetrahedron(info.polynomial_order)
                ops = ReferenceOperators(elem)
                loader = MeshLoader(mesh_file)
                mesh = loader.load(elem, ops)
                planner.update_mesh_diameter(mesh_hash, mesh.smallest_diameter)
                loader.close()
        else:
            logger.info("Generating meshes for unique configurations")
            planner.generate_missing_meshes(
                geometry_type=geometry_type_enum,
                progress_callback=mesh_progress_callback,
            )

        global_dt = planner.compute_global_timestep()
        planner.save_metadata()
    else:
        global_dt = planner.global_dt

    if global_dt is None:
        raise ValueError("Could not determine global time step")

    def mesh_resolver(sim_hash: str) -> Path | None:
        if mesh_file:
            return mesh_file
        mesh_hash = planner.get_mesh_hash_for_simulation(sim_hash)
        if mesh_hash:
            return planner.get_mesh_file(mesh_hash)
        return None

    # Report total simulations count
    if total_simulations_callback:
        total_simulations_callback(len(pending))

    executor = BatchExecutor(batch_dir)
    return executor.run_all(
        pending=pending,
        global_dt=global_dt,
        mesh_file_resolver=mesh_resolver,
        progress_callback=progress_callback,
    )
