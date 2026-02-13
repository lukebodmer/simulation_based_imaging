"""Batch planning for simulation runs.

Handles parameter file discovery, mesh management, and time step
calculation across batches of simulations.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import toml
import tomli

from sbimaging.logging import get_logger
from sbimaging.meshing import GeometryType, generate_mesh_from_config
from sbimaging.simulators.dg.dim3.time_stepping import compute_cfl_timestep


@dataclass
class MeshInfo:
    """Information about a mesh configuration.

    Attributes:
        mesh_hash: Hash identifying the mesh geometry.
        param_file: Path to a parameter file using this mesh.
        polynomial_order: Polynomial order for this mesh.
        max_wave_speed: Maximum wave speed in the domain.
        smallest_diameter: Smallest element diameter (set after generation).
    """

    mesh_hash: str
    param_file: Path
    polynomial_order: int
    max_wave_speed: float
    smallest_diameter: float | None = None


class BatchPlanner:
    """Plans batch simulation runs.

    Discovers parameter files, computes mesh hashes, determines
    which simulations need to run, and calculates global time step.

    Attributes:
        batch_dir: Root directory for batch data.
        parameter_dir: Directory containing parameter files.
        mesh_dir: Directory for mesh files.
        simulations_dir: Directory for simulation outputs.
    """

    def __init__(self, batch_dir: Path):
        """Initialize batch planner.

        Args:
            batch_dir: Root directory for batch data.
        """
        self.batch_dir = Path(batch_dir)
        self.parameter_dir = self.batch_dir / "parameter_files"
        self.mesh_dir = self.batch_dir / "meshes"
        self.simulations_dir = self.batch_dir / "simulations"

        self._logger = get_logger(__name__)
        self._mesh_info: dict[str, MeshInfo] = {}
        self._pending_simulations: list[str] = []
        self._global_dt: float | None = None

    def discover_parameter_files(self) -> list[Path]:
        """Find all parameter files in the batch directory.

        Returns:
            List of paths to parameter files.
        """
        if not self.parameter_dir.exists():
            return []

        files = sorted(self.parameter_dir.glob("*.toml"))
        self._logger.info(f"Found {len(files)} parameter files")
        return files

    def compute_mesh_hashes(self) -> dict[str, MeshInfo]:
        """Compute mesh hashes for all parameter files.

        Returns:
            Dictionary mapping mesh hashes to MeshInfo objects.
        """
        self._mesh_info = {}

        for param_file in self.discover_parameter_files():
            config = self._load_config(param_file)
            mesh_hash = self._compute_mesh_hash(config)

            if mesh_hash not in self._mesh_info:
                material = config.get("material", {})
                solver = config.get("solver", {})

                max_speed = max(
                    material.get("inclusion_wave_speed", 1.0),
                    material.get("outer_wave_speed", 1.0),
                )

                self._mesh_info[mesh_hash] = MeshInfo(
                    mesh_hash=mesh_hash,
                    param_file=param_file,
                    polynomial_order=solver.get("polynomial_order", 3),
                    max_wave_speed=max_speed,
                )

        self._logger.info(f"Found {len(self._mesh_info)} unique mesh configurations")
        return self._mesh_info

    def find_pending_simulations(self) -> list[str]:
        """Find parameter files that haven't been simulated.

        Returns:
            List of parameter file hashes (stems) that need simulation.
        """
        param_hashes = {f.stem for f in self.discover_parameter_files()}

        completed_hashes = set()
        if self.simulations_dir.exists():
            completed_hashes = {
                d.name for d in self.simulations_dir.iterdir() if d.is_dir()
            }

        self._pending_simulations = sorted(param_hashes - completed_hashes)

        if self._pending_simulations:
            self._logger.info(
                f"Found {len(self._pending_simulations)} pending simulations"
            )
        else:
            self._logger.info("All simulations completed")

        return self._pending_simulations

    def compute_global_timestep(self) -> float:
        """Compute global minimum time step across all meshes.

        Requires mesh diameters to be set via update_mesh_diameter().

        Returns:
            Global minimum time step.

        Raises:
            ValueError: If no valid meshes found or diameters not set.
        """
        if not self._mesh_info:
            self.compute_mesh_hashes()

        min_dt = None

        for mesh_hash, info in self._mesh_info.items():
            if info.smallest_diameter is None:
                self._logger.warning(f"Mesh {mesh_hash} missing diameter, skipping")
                continue

            dt = compute_cfl_timestep(
                smallest_diameter=info.smallest_diameter,
                max_speed=info.max_wave_speed,
                polynomial_order=info.polynomial_order,
            )

            if min_dt is None or dt < min_dt:
                min_dt = dt

        if min_dt is None:
            raise ValueError("Could not compute time step - no valid meshes")

        self._global_dt = min_dt
        self._logger.info(f"Global minimum dt = {min_dt:.6e}")
        return min_dt

    def update_mesh_diameter(self, mesh_hash: str, diameter: float) -> None:
        """Update smallest diameter for a mesh.

        Args:
            mesh_hash: Hash identifying the mesh.
            diameter: Smallest element diameter.
        """
        if mesh_hash in self._mesh_info:
            self._mesh_info[mesh_hash].smallest_diameter = diameter

    def get_parameter_file(self, sim_hash: str) -> Path:
        """Get parameter file path for a simulation hash.

        Args:
            sim_hash: Simulation hash (parameter file stem).

        Returns:
            Path to parameter file.
        """
        return self.parameter_dir / f"{sim_hash}.toml"

    def get_output_dir(self, sim_hash: str) -> Path:
        """Get output directory for a simulation.

        Args:
            sim_hash: Simulation hash.

        Returns:
            Path to output directory.
        """
        return self.simulations_dir / sim_hash

    def save_metadata(self) -> None:
        """Save batch metadata to file."""
        metadata = {
            "global_dt": self._global_dt,
            "meshes": {
                h: {
                    "polynomial_order": info.polynomial_order,
                    "max_wave_speed": info.max_wave_speed,
                    "smallest_diameter": info.smallest_diameter,
                }
                for h, info in self._mesh_info.items()
            },
        }

        meta_path = self.batch_dir / "batch_metadata.toml"
        with open(meta_path, "w") as f:
            toml.dump(metadata, f)

        self._logger.info(f"Saved batch metadata to {meta_path}")

    def load_metadata(self) -> bool:
        """Load batch metadata from file.

        Returns:
            True if metadata was loaded successfully.
        """
        meta_path = self.batch_dir / "batch_metadata.toml"
        if not meta_path.exists():
            return False

        with open(meta_path, "rb") as f:
            metadata = tomli.load(f)

        self._global_dt = metadata.get("global_dt")

        for mesh_hash, info in metadata.get("meshes", {}).items():
            if mesh_hash in self._mesh_info:
                self._mesh_info[mesh_hash].smallest_diameter = info.get(
                    "smallest_diameter"
                )

        self._logger.info(f"Loaded batch metadata from {meta_path}")
        return True

    def _load_config(self, path: Path) -> dict:
        """Load configuration from TOML file."""
        with open(path, "rb") as f:
            return tomli.load(f)

    def _compute_mesh_hash(self, config: dict) -> str:
        """Compute hash for mesh-relevant configuration.

        Only includes parameters that affect mesh geometry.
        """
        mesh_cfg = config.get("mesh", {})
        solver_cfg = config.get("solver", {})

        mesh_relevant = {
            "grid_size": mesh_cfg.get("grid_size"),
            "box_size": mesh_cfg.get("box_size"),
            "inclusion_center": mesh_cfg.get("inclusion_center"),
            "inclusion_scaling": mesh_cfg.get("inclusion_scaling"),
            "number_of_cubes": mesh_cfg.get("number_of_cubes"),
            "cube_centers": mesh_cfg.get("cube_centers"),
            "cube_widths": mesh_cfg.get("cube_widths"),
            "polynomial_order": solver_cfg.get("polynomial_order"),
        }

        hash_str = str(sorted(mesh_relevant.items()))
        return hashlib.sha1(hash_str.encode()).hexdigest()[:10]

    def generate_missing_meshes(
        self,
        geometry_type: GeometryType = GeometryType.ELLIPSOID,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Generate meshes for configurations that don't have them.

        Args:
            geometry_type: Type of geometry to generate.
            progress_callback: Optional callback for progress updates.
                Signature: (generated: int, total: int) -> None

        Returns:
            Number of meshes generated.
        """
        if not self._mesh_info:
            self.compute_mesh_hashes()

        generated = 0
        total = len(self._mesh_info)

        for mesh_hash, info in self._mesh_info.items():
            mesh_dir = self.mesh_dir / mesh_hash
            mesh_file = mesh_dir / "mesh.msh"

            if mesh_file.exists():
                self._logger.info(f"Mesh {mesh_hash} already exists")
                if info.smallest_diameter is None:
                    diameter = self._load_mesh_diameter(mesh_hash)
                    if diameter:
                        info.smallest_diameter = diameter
                generated += 1
                if progress_callback:
                    progress_callback(generated, total)
                continue

            self._logger.info(f"Generating mesh {mesh_hash}")

            config = self._load_config(info.param_file)
            mesh_path, diameter = generate_mesh_from_config(
                config=config,
                output_path=mesh_file,
                geometry_type=geometry_type,
            )

            info.smallest_diameter = diameter
            self._save_mesh_metadata(mesh_hash, diameter)
            generated += 1

            if progress_callback:
                progress_callback(generated, total)

        self._logger.info(f"Generated {generated} meshes")
        return generated

    def _save_mesh_metadata(self, mesh_hash: str, diameter: float) -> None:
        """Save mesh metadata to file."""
        mesh_dir = self.mesh_dir / mesh_hash
        mesh_dir.mkdir(parents=True, exist_ok=True)

        meta = {"smallest_diameter": diameter}
        meta_path = mesh_dir / "mesh_info.toml"

        with open(meta_path, "w") as f:
            toml.dump(meta, f)

    def _load_mesh_diameter(self, mesh_hash: str) -> float | None:
        """Load mesh diameter from metadata file."""
        meta_path = self.mesh_dir / mesh_hash / "mesh_info.toml"
        if not meta_path.exists():
            return None

        with open(meta_path, "rb") as f:
            meta = tomli.load(f)

        return meta.get("smallest_diameter")

    def get_mesh_file(self, mesh_hash: str) -> Path | None:
        """Get path to mesh file for a given hash.

        Args:
            mesh_hash: Mesh configuration hash.

        Returns:
            Path to mesh file, or None if not found.
        """
        mesh_file = self.mesh_dir / mesh_hash / "mesh.msh"
        if mesh_file.exists():
            return mesh_file
        return None

    def get_mesh_hash_for_simulation(self, sim_hash: str) -> str | None:
        """Get mesh hash for a simulation.

        Args:
            sim_hash: Simulation hash (parameter file stem).

        Returns:
            Mesh hash, or None if not found.
        """
        param_file = self.get_parameter_file(sim_hash)
        if not param_file.exists():
            return None

        config = self._load_config(param_file)
        return self._compute_mesh_hash(config)

    @property
    def global_dt(self) -> float | None:
        """Get global time step."""
        return self._global_dt

    @property
    def pending_simulations(self) -> list[str]:
        """Get list of pending simulation hashes."""
        return self._pending_simulations

    @property
    def mesh_info(self) -> dict[str, MeshInfo]:
        """Get mesh information dictionary."""
        return self._mesh_info
