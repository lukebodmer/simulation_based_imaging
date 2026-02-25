"""CLI for batch simulation management.

Commands:
    generate    Generate parameter files from a batch config
    run         Run simulations (all or a subset)
    status      Check batch completion status
    export      Export current GUI settings to batch config

Usage:
    python -m sbimaging.cli.batch generate --config batch.toml --output data/my_batch
    python -m sbimaging.cli.batch run --batch-dir data/my_batch
    python -m sbimaging.cli.batch run --batch-dir data/my_batch --start 0 --end 50
    python -m sbimaging.cli.batch status --batch-dir data/my_batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import toml
import tomli

from sbimaging.batch.executor import BatchExecutor
from sbimaging.batch.generator import ParameterGenerator, ParameterRange, ParameterSpace
from sbimaging.batch.planner import BatchPlanner
from sbimaging.config import get_base_config_path, load_preset
from sbimaging.config.simulation import (
    MeshConfig,
    OuterMaterialConfig,
    OutputConfig,
    ReceiverConfig,
    SimulationConfig,
    SolverConfig,
    SourceConfig,
)
from sbimaging.logging import configure_logging, get_logger

logger = get_logger(__name__)


def parse_batch_config(path: Path) -> dict:
    """Load and parse a batch configuration TOML file.

    Args:
        path: Path to batch config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "rb") as f:
        return tomli.load(f)


def config_to_parameter_space(config: dict) -> ParameterSpace:
    """Convert batch config to ParameterSpace.

    Args:
        config: Batch configuration dictionary.

    Returns:
        ParameterSpace for the generator.
    """
    sweep = config.get("sweep_parameters", {})
    material = sweep.get("material", {})
    geometry = sweep.get("geometry", {})
    cubes = sweep.get("cubes", {})

    wave_speed_range = material.get("wave_speed_range", [1.5, 4.0])
    density_range = material.get("density_range", [1.5, 4.0])
    scaling_x_range = geometry.get("scaling_x_range", [0.1, 0.3])
    cube_width_range = cubes.get("width_range", [0.05, 0.2])
    cube_quantity_range = cubes.get("quantity_range", [1, 3])
    boundary_buffer = geometry.get("boundary_buffer", 0.05)

    return ParameterSpace(
        inclusion_density=ParameterRange(density_range[0], density_range[1]),
        inclusion_speed=ParameterRange(wave_speed_range[0], wave_speed_range[1]),
        inclusion_scaling=ParameterRange(scaling_x_range[0], scaling_x_range[1]),
        cube_width=ParameterRange(cube_width_range[0], cube_width_range[1]),
        cube_count=(cube_quantity_range[0], cube_quantity_range[1]),
        boundary_buffer=boundary_buffer,
    )


def config_to_simulation_config(config: dict) -> SimulationConfig:
    """Convert batch config to SimulationConfig.

    Args:
        config: Batch configuration dictionary.

    Returns:
        SimulationConfig with fixed parameters.
    """
    fixed = config.get("fixed_parameters", {})
    batch = config.get("batch", {})

    sources_data = fixed.get("sources", {})
    source_count = sources_data.get("number", 6)

    # Build source centers - use provided or default
    default_centers = [
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 1.0],
        [0.5, 0.0, 0.5],
        [0.5, 1.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
    ]
    centers = sources_data.get("centers", default_centers[:source_count])

    # Apply uniform values if specified
    freq = sources_data.get("frequency", 3.0)
    amp = sources_data.get("amplitude", 1.0)
    radius = sources_data.get("radius", 0.05)

    sources = SourceConfig(
        number=source_count,
        centers=centers,
        radii=[radius] * source_count,
        amplitudes=[amp] * source_count,
        frequencies=[freq] * source_count,
    )

    outer = fixed.get("outer_material", {})
    outer_material = OuterMaterialConfig(
        wave_speed=outer.get("wave_speed", 2.0),
        density=outer.get("density", 2.0),
    )

    mesh_data = fixed.get("mesh", {})
    mesh = MeshConfig(
        grid_size=mesh_data.get("grid_size", 0.04),
        box_size=mesh_data.get("box_size", 1.0),
        inclusion_center=mesh_data.get("inclusion_center", [0.5, 0.5, 0.5]),
    )

    solver_data = fixed.get("solver", {})
    solver = SolverConfig(
        polynomial_order=solver_data.get("polynomial_order", 1),
        number_of_timesteps=solver_data.get("number_of_timesteps"),
        total_time=solver_data.get("total_time"),
    )

    receivers_data = fixed.get("receivers", {})
    receivers = ReceiverConfig(
        sensors_per_face=receivers_data.get("sensors_per_face", 25),
    )

    output_data = fixed.get("output_intervals", {})
    output = OutputConfig(
        image=output_data.get("image", 1000),
        data=output_data.get("data", 1000),
        points=output_data.get("points", 10),
        energy=output_data.get("energy", 500),
    )

    return SimulationConfig(
        sources=sources,
        outer_material=outer_material,
        mesh=mesh,
        solver=solver,
        receivers=receivers,
        output=output,
        batch_name=batch.get("name", "my_batch"),
        batch_description=batch.get("description", ""),
        num_samples=batch.get("num_samples", 100),
    )


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate parameter files from a batch config.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success).
    """
    config_path = Path(args.config)
    output_dir = Path(args.output)

    if args.preset:
        logger.info(f"Loading preset: {args.preset}")
        preset = load_preset(args.preset)
        parameter_space = preset.to_parameter_space()
        geometry_type = "ellipsoid"
        if preset.inclusion.is_sphere:
            geometry_type = "sphere"
        elif preset.inclusion.is_multi_cubes:
            geometry_type = "multi_cubes"
        elif preset.inclusion.is_cube_in_ellipsoid:
            geometry_type = "cube_in_ellipsoid"
        num_samples = args.num_samples or preset.default_num_samples
        simulation_config = preset.to_simulation_config()
    elif config_path.exists():
        logger.info(f"Loading batch config: {config_path}")
        config = parse_batch_config(config_path)
        parameter_space = config_to_parameter_space(config)
        simulation_config = config_to_simulation_config(config)
        geometry_type = config.get("batch", {}).get("inclusion_type", "ellipsoid")
        num_samples = args.num_samples or config.get("batch", {}).get(
            "num_samples", 100
        )
    else:
        logger.error(f"Config file not found: {config_path}")
        return 1

    base_config_path = get_base_config_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    param_dir = output_dir / "parameter_files"

    logger.info(f"Generating {num_samples} parameter files to {param_dir}")

    generator = ParameterGenerator(
        base_config_path=base_config_path,
        output_dir=param_dir,
        space=parameter_space,
        simulation_config=simulation_config,
    )

    if geometry_type in ("multi_cubes", "cube_in_ellipsoid"):
        files = generator.generate_cube_samples(num_samples)
    else:
        files = generator.generate(num_samples)

    logger.info(f"Generated {len(files)} parameter files")

    # Save the batch config to output directory for reference
    if config_path.exists():
        import shutil

        shutil.copy(config_path, output_dir / "batch_config.toml")
        logger.info(f"Copied batch config to {output_dir / 'batch_config.toml'}")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run simulations from a batch directory.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success).
    """
    batch_dir = Path(args.batch_dir)

    if not batch_dir.exists():
        logger.error(f"Batch directory not found: {batch_dir}")
        return 1

    planner = BatchPlanner(batch_dir)
    planner.compute_mesh_hashes()
    pending = planner.find_pending_simulations()

    if not pending:
        logger.info("No pending simulations")
        return 0

    # Handle subset selection
    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end or len(pending)
        pending = pending[start:end]
        logger.info(
            f"Running subset: indices {start} to {end} ({len(pending)} simulations)"
        )

    if args.index is not None:
        if args.index >= len(pending):
            logger.error(f"Index {args.index} out of range (max {len(pending) - 1})")
            return 1
        pending = [pending[args.index]]
        logger.info(f"Running single simulation at index {args.index}")

    # Check for existing metadata or generate meshes
    metadata_exists = planner.load_metadata()

    if not metadata_exists:
        logger.info("No metadata found - generating meshes and computing timestep")

        # Load batch config to get geometry type
        config_path = batch_dir / "batch_config.toml"
        geometry_type = "ellipsoid"
        if config_path.exists():
            config = parse_batch_config(config_path)
            geometry_type = config.get("batch", {}).get("inclusion_type", "ellipsoid")

        from sbimaging.meshing import GeometryType

        geometry_type_enum = GeometryType(geometry_type)

        def mesh_progress(generated: int, total: int):
            logger.info(f"Mesh generation: {generated}/{total}")

        planner.generate_missing_meshes(
            geometry_type=geometry_type_enum,
            progress_callback=mesh_progress,
        )

        global_dt = planner.compute_global_timestep()
        planner.save_metadata()
    else:
        global_dt = planner.global_dt

    if global_dt is None:
        logger.error("Could not determine global timestep")
        return 1

    logger.info(f"Using global_dt = {global_dt:.6e}")
    logger.info(f"Running {len(pending)} simulations")

    def mesh_resolver(sim_hash: str) -> Path | None:
        mesh_hash = planner.get_mesh_hash_for_simulation(sim_hash)
        if mesh_hash:
            return planner.get_mesh_file(mesh_hash)
        return None

    def progress_callback(remaining: int, completed: int, failed: int):
        total = completed + failed + remaining
        logger.info(f"Progress: {completed}/{total} completed, {failed} failed")

    executor = BatchExecutor(batch_dir)
    completed, failed = executor.run_all(
        pending=pending,
        global_dt=global_dt,
        mesh_file_resolver=mesh_resolver,
        progress_callback=progress_callback,
    )

    logger.info(f"Batch complete: {completed} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show batch status.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success).
    """
    batch_dir = Path(args.batch_dir)

    if not batch_dir.exists():
        logger.error(f"Batch directory not found: {batch_dir}")
        return 1

    planner = BatchPlanner(batch_dir)
    planner.compute_mesh_hashes()
    pending = planner.find_pending_simulations()
    completed = planner.count_completed_simulations()
    total = len(planner.discover_parameter_files())

    print(f"Batch: {batch_dir.name}")
    print(f"Total parameter files: {total}")
    print(f"Completed simulations: {completed}")
    print(f"Pending simulations: {len(pending)}")
    print(f"Unique mesh configurations: {len(planner.mesh_info)}")

    if planner.load_metadata() and planner.global_dt:
        print(f"Global timestep: {planner.global_dt:.6e}")

    if args.list_pending and pending:
        print("\nPending simulation hashes:")
        for i, h in enumerate(pending):
            print(f"  [{i}] {h}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export a preset to a batch config file.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success).
    """
    output_path = Path(args.output)

    if args.preset:
        preset = load_preset(args.preset)

        # Determine inclusion type
        inclusion_type = "ellipsoid"
        if preset.inclusion.is_sphere:
            inclusion_type = "sphere"
        elif preset.inclusion.is_multi_cubes:
            inclusion_type = "multi_cubes"
        elif preset.inclusion.is_cube_in_ellipsoid:
            inclusion_type = "cube_in_ellipsoid"

        config = {
            "batch": {
                "name": preset.name,
                "description": preset.description,
                "num_samples": preset.default_num_samples,
                "inclusion_type": inclusion_type,
            },
            "sweep_parameters": {
                "material": {
                    "wave_speed_range": list(preset.inclusion.wave_speed_range),
                    "density_range": list(preset.inclusion.density_range),
                },
                "geometry": {
                    "scaling_x_range": list(preset.inclusion.scaling_range[0]),
                    "scaling_y_range": list(preset.inclusion.scaling_range[1]),
                    "scaling_z_range": list(preset.inclusion.scaling_range[2]),
                    "allow_rotation": preset.inclusion.allow_rotation,
                    "allow_movement": preset.inclusion.allow_movement,
                    "boundary_buffer": preset.boundary_buffer,
                },
                "cubes": {
                    "quantity_range": list(preset.cubes.quantity_range),
                    "width_range": list(preset.cubes.width_range),
                },
            },
            "fixed_parameters": {
                "sources": {
                    "number": preset.sources.number if preset.sources else 6,
                    "frequency": preset.sources.frequency if preset.sources else 3.0,
                    "amplitude": preset.sources.amplitude if preset.sources else 1.0,
                    "radius": preset.sources.radius if preset.sources else 0.05,
                },
                "outer_material": {
                    "wave_speed": (
                        preset.outer_material.wave_speed
                        if preset.outer_material
                        else 2.0
                    ),
                    "density": preset.outer_material.density
                    if preset.outer_material
                    else 2.0,
                },
                "mesh": {
                    "grid_size": preset.mesh.grid_size if preset.mesh else 0.04,
                    "box_size": preset.mesh.box_size if preset.mesh else 1.0,
                    "inclusion_center": [0.5, 0.5, 0.5],
                },
                "solver": (
                    {
                        "polynomial_order": preset.solver.polynomial_order,
                        **(
                            {"total_time": preset.solver.total_time}
                            if preset.solver.total_time is not None
                            else {
                                "number_of_timesteps": preset.solver.number_of_timesteps
                                or 10000
                            }
                        ),
                    }
                    if preset.solver
                    else {"polynomial_order": 1, "number_of_timesteps": 10000}
                ),
                "receivers": {
                    "sensors_per_face": (
                        preset.receivers.sensors_per_face if preset.receivers else 25
                    ),
                },
                "output_intervals": {
                    "image": preset.output.image if preset.output else 1000,
                    "data": preset.output.data if preset.output else 1000,
                    "points": preset.output.points if preset.output else 10,
                    "energy": preset.output.energy if preset.output else 500,
                },
            },
        }
    else:
        # Create default config template
        config = {
            "batch": {
                "name": "my_batch",
                "description": "Batch simulation",
                "num_samples": 100,
                "inclusion_type": "ellipsoid",
            },
            "sweep_parameters": {
                "material": {
                    "wave_speed_range": [1.5, 4.0],
                    "density_range": [1.5, 4.0],
                },
                "geometry": {
                    "scaling_x_range": [0.1, 0.3],
                    "scaling_y_range": [0.1, 0.3],
                    "scaling_z_range": [0.1, 0.3],
                    "allow_rotation": False,
                    "allow_movement": False,
                    "boundary_buffer": 0.05,
                },
                "cubes": {
                    "quantity_range": [1, 3],
                    "width_range": [0.05, 0.2],
                },
            },
            "fixed_parameters": {
                "sources": {
                    "number": 6,
                    "frequency": 3.0,
                    "amplitude": 1.0,
                    "radius": 0.05,
                },
                "outer_material": {
                    "wave_speed": 2.0,
                    "density": 2.0,
                },
                "mesh": {
                    "grid_size": 0.04,
                    "box_size": 1.0,
                    "inclusion_center": [0.5, 0.5, 0.5],
                },
                "solver": {
                    "polynomial_order": 1,
                    "number_of_timesteps": 10000,
                },
                "receivers": {
                    "sensors_per_face": 25,
                },
                "output_intervals": {
                    "image": 1000,
                    "data": 1000,
                    "points": 10,
                    "energy": 500,
                },
            },
        }

    with open(output_path, "w") as f:
        toml.dump(config, f)

    logger.info(f"Exported batch config to {output_path}")
    return 0


def cmd_list_presets(args: argparse.Namespace) -> int:
    """List available presets.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success).
    """
    from sbimaging.config import list_presets

    presets = list_presets()
    print("Available presets:")
    for name in presets:
        preset = load_preset(name)
        print(f"  {name}: {preset.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for batch CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="sbimaging-batch",
        description="CLI for batch simulation management",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate parameter files from a batch config",
    )
    gen_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="batch_config.toml",
        help="Path to batch config TOML file",
    )
    gen_parser.add_argument(
        "--preset",
        "-p",
        type=str,
        help="Use a built-in preset instead of config file",
    )
    gen_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for batch data",
    )
    gen_parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        help="Number of samples to generate (overrides config)",
    )

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run simulations from a batch directory",
    )
    run_parser.add_argument(
        "--batch-dir",
        "-b",
        type=str,
        required=True,
        help="Path to batch directory",
    )
    run_parser.add_argument(
        "--start",
        "-s",
        type=int,
        help="Start index for subset (inclusive)",
    )
    run_parser.add_argument(
        "--end",
        "-e",
        type=int,
        help="End index for subset (exclusive)",
    )
    run_parser.add_argument(
        "--index",
        "-i",
        type=int,
        help="Run only the simulation at this index",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show batch status",
    )
    status_parser.add_argument(
        "--batch-dir",
        "-b",
        type=str,
        required=True,
        help="Path to batch directory",
    )
    status_parser.add_argument(
        "--list-pending",
        "-l",
        action="store_true",
        help="List all pending simulation hashes",
    )

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export a preset or template to a batch config file",
    )
    export_parser.add_argument(
        "--preset",
        "-p",
        type=str,
        help="Preset name to export (omit for default template)",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="batch_config.toml",
        help="Output file path",
    )

    # list-presets command
    subparsers.add_parser(
        "list-presets",
        help="List available presets",
    )

    args = parser.parse_args(argv)

    # Configure logging
    import logging

    configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_to_file=False,
    )

    # Dispatch to command handler
    if args.command == "generate":
        return cmd_generate(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "export":
        return cmd_export(args)
    elif args.command == "list-presets":
        return cmd_list_presets(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
