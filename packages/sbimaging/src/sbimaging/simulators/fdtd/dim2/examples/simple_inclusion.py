"""Example: 2D acoustic simulation with a pressure source and inclusion.

This example demonstrates a simple 2D linear acoustics FDTD simulation
with:
- A Gaussian pulse pressure source
- A triangular inclusion with different material properties
- Reflective (rigid wall) boundary conditions
- Sensors around the boundary recording pressure over time

The simulation shows wave propagation, scattering from the inclusion,
and reflections from the boundaries. The video includes a side-by-side
view of the pressure field and accumulating sensor data.
"""

from pathlib import Path
import numpy as np

from sbimaging import configure_logging, get_logger
from sbimaging.simulators.fdtd.dim2 import (
    Grid,
    Material,
    Simulation,
    BoundarySource,
    GaussianPulse,
    SensorArray,
    generate_boundary_sensors,
)
from sbimaging.simulators.fdtd.dim2.simulation import FrameRecorder
from sbimaging.simulators.fdtd.dim2.visualization import (
    TriangleOverlay,
    create_video_with_sensors,
)


def main():
    """Run the example simulation."""
    configure_logging()
    logger = get_logger(__name__)

    logger.info("2D Linear Acoustics FDTD - Simple Inclusion Example")
    logger.info("=" * 60)

    # Use normalized units for simplicity
    domain_size = 1.0
    nx, ny = 500, 500
    grid = Grid.from_domain_size(
        size_x=domain_size,
        size_y=domain_size,
        nx=nx,
        ny=ny,
    )
    logger.info(f"Grid: {grid}")

    # Simplified material values (relative densities and speeds)
    background_density = 1.0
    background_speed = 1.0

    material = Material.uniform(grid, density=background_density, wave_speed=background_speed)

    # Equilateral triangular inclusion, off-center (upper right quadrant)
    # Triangle with similar area to a circle of radius 0.1 (area ~ 0.031)
    # Equilateral triangle with side s has area = (sqrt(3)/4) * s^2
    # For area ~ 0.031, side ~ 0.27
    triangle_side = 0.27
    triangle_center = (0.65, 0.55)  # Off-center position
    # Height of equilateral triangle = (sqrt(3)/2) * side
    triangle_height = (np.sqrt(3) / 2) * triangle_side
    inclusion_vertices: list[tuple[float, float]] = [
        (triangle_center[0] - triangle_side / 2, triangle_center[1] - triangle_height / 3),  # Bottom left
        (triangle_center[0] + triangle_side / 2, triangle_center[1] - triangle_height / 3),  # Bottom right
        (triangle_center[0], triangle_center[1] + 2 * triangle_height / 3),                   # Top
    ]
    inclusion_density = 2.0
    inclusion_speed = 2.0

    material.set_triangular_inclusion(
        vertices=inclusion_vertices,
        density=inclusion_density,
        wave_speed=inclusion_speed,
    )
    logger.info(f"Material: {material}")
    logger.info(
        f"Triangular inclusion with vertices {inclusion_vertices}, "
        f"rho={inclusion_density} kg/m^3, c={inclusion_speed} m/s"
    )

    sim = Simulation(grid, material, courant_factor=0.9)

    # Boundary source on left wall, centered vertically
    source_position = 0.5  # Center of left wall (y-coordinate)
    source_width = 0.005  # Small width for approximate point source (a few grid cells)
    # Lower frequency for wider pulse (more timesteps per pulse, less numerical dispersion)
    source_frequency = 3.0
    source_amplitude = 1.0

    waveform = GaussianPulse(
        amplitude=source_amplitude,
        frequency=source_frequency,
    )

    # Apply source as boundary condition on left wall (like 3D DG)
    source = BoundarySource(
        boundary="left",
        position=source_position,
        width=source_width,
        waveform=waveform,
    )
    sim.add_boundary_source(source)
    logger.info(
        f"Boundary source on left wall at y={source_position}, "
        f"width={source_width}, frequency={source_frequency} Hz"
    )

    sensors_per_side = 10
    # Use offset_cells=2 to place sensors in fully symmetric interior region
    # (offset=1 places sensors where stencils touch boundary-zeroed values asymmetrically)
    sensor_locations = generate_boundary_sensors(grid, sensors_per_side=sensors_per_side, margin_fraction=0.05, offset_cells=2)
    sensors = SensorArray(grid, sensor_locations)
    sim.set_sensors(sensors)
    logger.info(f"Sensors: {sensors.num_sensors} sensors around the boundary")

    # Simulation time for wave to traverse domain multiple times
    # With c=1 and domain_size=1, one traversal takes ~1 time unit
    final_time = 4.0
    num_steps = int(final_time / sim.dt)
    frame_interval = max(1, num_steps // 200)

    logger.info(f"Simulation: {num_steps} steps, dt={sim.dt:.6g}s, t_final={final_time:.6g}s")
    logger.info(f"Recording every {frame_interval} steps")

    recorder = FrameRecorder(keep_in_memory=True)

    def callback(s: Simulation):
        recorder(s)
        if s.step % 100 == 0:
            max_p = np.abs(s.get_pressure()).max()
            logger.info(f"  Step {s.step}/{num_steps}, t={s.time:.6g}s, max|p|={max_p:.4g}")

    sim.run(num_steps, callback=callback, callback_interval=frame_interval)

    logger.info(f"Recorded {len(recorder.frames)} frames")
    logger.info(f"Sensor data shape: {sensors.get_data_matrix().shape}")

    output_dir = Path("data/fdtd_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_file = output_dir / "simple_inclusion.mp4"

    inclusion_overlay = TriangleOverlay(
        vertices=inclusion_vertices,
        linewidth=2.0,
    )

    # Subsample sensor data to match frame count
    full_sensor_data = sensors.get_data_matrix()
    sensor_indices = np.linspace(0, full_sensor_data.shape[1] - 1, len(recorder.frames), dtype=int)
    frame_sensor_data = full_sensor_data[:, sensor_indices]

    logger.info(f"Creating video: {video_file}")
    success = create_video_with_sensors(
        frames=recorder.frames,
        times=recorder.times,
        sensor_data=frame_sensor_data,
        output_file=video_file,
        x_coords=grid.x_coordinates(),
        y_coords=grid.y_coordinates(),
        sensor_locations=sensors.locations,
        vmin=-0.1,
        vmax=0.1,
        fps=15,
        dpi=150,
        figsize=(14, 5),
        overlays=[inclusion_overlay],
        sensors_per_side=sensors_per_side,
    )

    if success:
        logger.info(f"Video saved to: {video_file}")
    else:
        logger.error("Failed to create video")

    # Save sensor data
    sensor_file = output_dir / "sensor_data.npz"
    sensors.save(str(sensor_file))
    logger.info(f"Sensor data saved to: {sensor_file}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
