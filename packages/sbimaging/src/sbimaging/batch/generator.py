"""Parameter file generation for batch simulations.

Generates simulation parameter files using Latin Hypercube Sampling
for efficient exploration of parameter spaces.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import toml
import tomli

from sbimaging.logging import get_logger

if TYPE_CHECKING:
    from sbimaging.config.simulation import SimulationConfig


@dataclass
class ParameterRange:
    """Range specification for a parameter.

    Attributes:
        min_val: Minimum value.
        max_val: Maximum value.
    """

    min_val: float
    max_val: float

    def sample(self, rng: np.random.Generator) -> float:
        """Sample uniformly from the range."""
        return float(rng.uniform(self.min_val, self.max_val))


@dataclass
class ParameterSpace:
    """Defines the parameter space for batch generation.

    Attributes:
        inclusion_density: Range for inclusion density.
        inclusion_speed: Range for inclusion wave speed.
        inclusion_scaling: Range for inclusion scaling factors.
        cube_width: Range for cube widths (if using cubes).
        cube_count: Range for number of cubes (min, max inclusive).
        domain_size: Size of the simulation domain.
        boundary_buffer: Minimum distance from domain boundary.
    """

    inclusion_density: ParameterRange = field(
        default_factory=lambda: ParameterRange(0.1, 2.0)
    )
    inclusion_speed: ParameterRange = field(
        default_factory=lambda: ParameterRange(0.1, 2.0)
    )
    inclusion_scaling: ParameterRange = field(
        default_factory=lambda: ParameterRange(0.03, 0.07)
    )
    cube_width: ParameterRange = field(
        default_factory=lambda: ParameterRange(0.05, 0.2)
    )
    cube_count: tuple[int, int] = (1, 3)
    domain_size: float = 1.0
    boundary_buffer: float = 0.05


class ParameterGenerator:
    """Generates parameter files for batch simulations.

    Uses Latin Hypercube Sampling for efficient parameter space coverage.

    Attributes:
        base_config: Base configuration dictionary.
        output_dir: Directory for generated parameter files.
        space: Parameter space specification.
        fixed_overrides: Fixed parameter overrides from SimulationConfig.
    """

    def __init__(
        self,
        base_config_path: Path,
        output_dir: Path,
        space: ParameterSpace | None = None,
        seed: int = 42,
        simulation_config: SimulationConfig | None = None,
    ):
        """Initialize parameter generator.

        Args:
            base_config_path: Path to base TOML configuration.
            output_dir: Directory for output parameter files.
            space: Parameter space specification.
            seed: Random seed for reproducibility.
            simulation_config: Optional SimulationConfig for fixed parameter overrides.
        """
        self._logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_config = self._load_config(base_config_path)
        self.space = space or ParameterSpace()
        self._rng = np.random.default_rng(seed)

        # Extract fixed overrides from SimulationConfig
        self.fixed_overrides: dict[str, Any] = {}
        if simulation_config is not None:
            self.fixed_overrides = simulation_config.get_fixed_overrides()

    def generate(self, num_samples: int) -> list[Path]:
        """Generate parameter files using Latin Hypercube Sampling.

        Args:
            num_samples: Number of parameter files to generate.

        Returns:
            List of paths to generated files.
        """
        self._logger.info(f"Generating {num_samples} parameter files")

        generated = []
        hashes_seen: set[str] = set()

        for _ in range(num_samples):
            config = self._sample_config()
            path = self._write_config(config, hashes_seen)
            if path:
                generated.append(path)

        self._logger.info(f"Generated {len(generated)} unique parameter files")
        return generated

    def generate_cube_samples(self, num_samples: int) -> list[Path]:
        """Generate parameter files with cube inclusions.

        Args:
            num_samples: Number of parameter files to generate.

        Returns:
            List of paths to generated files.
        """
        self._logger.info(f"Generating {num_samples} cube inclusion parameter files")

        generated = []
        hashes_seen: set[str] = set()

        for _ in range(num_samples):
            config = self._sample_cube_config()
            path = self._write_config(config, hashes_seen)
            if path:
                generated.append(path)

        self._logger.info(f"Generated {len(generated)} unique parameter files")
        return generated

    def _load_config(self, path: Path) -> dict:
        """Load base configuration from TOML file."""
        with open(path, "rb") as f:
            return tomli.load(f)

    def _sample_config(self) -> dict:
        """Sample a single configuration from parameter space."""
        config = _deep_copy_dict(self.base_config)

        # Apply fixed overrides first
        self._apply_fixed_overrides(config)

        # Then sample sweep parameters
        config["material"]["inclusion_density"] = self.space.inclusion_density.sample(
            self._rng
        )
        config["material"]["inclusion_wave_speed"] = self.space.inclusion_speed.sample(
            self._rng
        )

        scaling = [self.space.inclusion_scaling.sample(self._rng) for _ in range(3)]
        config["mesh"]["inclusion_scaling"] = scaling

        return config

    def _apply_fixed_overrides(self, config: dict) -> None:
        """Apply fixed parameter overrides to a config dictionary."""
        for section, values in self.fixed_overrides.items():
            if section not in config:
                config[section] = {}
            if isinstance(values, dict):
                for key, value in values.items():
                    config[section][key] = value
            else:
                config[section] = values

    def _sample_cube_config(self) -> dict:
        """Sample configuration with cube inclusions."""
        config = _deep_copy_dict(self.base_config)

        # Apply fixed overrides first
        self._apply_fixed_overrides(config)

        # Then sample sweep parameters
        config["material"]["inclusion_density"] = self.space.inclusion_density.sample(
            self._rng
        )
        config["material"]["inclusion_wave_speed"] = self.space.inclusion_speed.sample(
            self._rng
        )

        min_cubes, max_cubes = self.space.cube_count
        num_cubes = int(self._rng.integers(min_cubes, max_cubes + 1))

        widths = [self.space.cube_width.sample(self._rng) for _ in range(num_cubes)]

        centers = self._place_cubes(widths)

        config["mesh"]["number_of_cubes"] = num_cubes
        config["mesh"]["cube_centers"] = centers
        config["mesh"]["cube_widths"] = widths

        return config

    def _place_cubes(self, widths: list[float]) -> list[list[float]]:
        """Place cubes with non-overlapping constraint."""
        centers = []
        buffer = self.space.boundary_buffer
        domain = self.space.domain_size

        for i, w in enumerate(widths):
            half_w = w / 2
            min_coord = buffer + half_w
            max_coord = domain - buffer - half_w

            for _ in range(100):
                candidate = self._rng.uniform(min_coord, max_coord, size=3)

                valid = True
                for j, c in enumerate(centers):
                    w_j = widths[j]
                    min_dist = 0.5 * w * np.sqrt(3) + 0.5 * w_j * np.sqrt(3) + buffer
                    if np.linalg.norm(candidate - np.array(c)) < min_dist:
                        valid = False
                        break

                if valid:
                    centers.append(candidate.tolist())
                    break
            else:
                raise RuntimeError(
                    f"Could not place cube {i + 1}/{len(widths)} "
                    f"with width {w:.4f} after 100 attempts"
                )

        return centers

    def _write_config(self, config: dict, hashes_seen: set[str]) -> Path | None:
        """Write configuration to file with hash-based naming."""
        config_str = toml.dumps(config)
        hash_val = hashlib.sha1(config_str.encode()).hexdigest()[:10]

        if hash_val in hashes_seen:
            return None

        hashes_seen.add(hash_val)
        output_path = self.output_dir / f"{hash_val}.toml"

        with open(output_path, "w") as f:
            f.write(config_str)

        return output_path


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a dictionary."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = v.copy()
        else:
            result[k] = v
    return result
