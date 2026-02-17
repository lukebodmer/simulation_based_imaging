"""Example: 1D acoustic simulation with a wide inclusion shifted right.

This example demonstrates a 1D linear acoustics FDTD simulation with:
- A Gaussian pulse pressure source from the left boundary
- A wide inclusion shifted to the right with high density and wave speed (3.5x)
- Reflective (rigid wall) boundary conditions
- Sensors at both ends recording pressure over time
"""

from pathlib import Path

import numpy as np

from sbimaging import configure_logging, get_logger
from sbimaging.simulators.fdtd.dim1 import (
    Grid,
    Material,
    Simulation,
    BoundarySource,
    GaussianPulse,
    SensorArray,
    generate_boundary_sensors,
)
from sbimaging.simulators.fdtd.dim1.simulation import FrameRecorder
from sbimaging.simulators.fdtd.dim1.visualization import create_video_with_sensors


def main():
    """Run the example simulation."""
    configure_logging()
    logger = get_logger(__name__)

    logger.info("1D Linear Acoustics FDTD - Right Inclusion Example")
    logger.info("=" * 60)

    # Domain setup - normalized units
    domain_size = 1.0
    nx = 500
    grid = Grid.from_domain_size(size_x=domain_size, nx=nx)
    logger.info(f"Grid: {grid}")

    # Background material
    background_density = 1.0
    background_speed = 1.0
    material = Material.uniform(grid, density=background_density, wave_speed=background_speed)

    # Wide inclusion shifted to the right with high density/wavespeed
    inclusion_center = 0.7
    inclusion_width = 0.25
    inclusion_x_min = inclusion_center - inclusion_width / 2
    inclusion_x_max = inclusion_center + inclusion_width / 2
    inclusion_density = 3.5
    inclusion_speed = 3.5

    material.set_inclusion(
        x_min=inclusion_x_min,
        x_max=inclusion_x_max,
        density=inclusion_density,
        wave_speed=inclusion_speed,
    )
    logger.info(f"Material: {material}")
    logger.info(
        f"Inclusion at x=[{inclusion_x_min:.2f}, {inclusion_x_max:.2f}], "
        f"rho={inclusion_density}, c={inclusion_speed}"
    )

    # Create simulation
    sim = Simulation(grid, material, courant_factor=0.9)

    # Boundary source on left wall
    source_frequency = 5.0
    source_amplitude = 1.0

    waveform = GaussianPulse(
        amplitude=source_amplitude,
        frequency=source_frequency,
    )

    source = BoundarySource(
        boundary="left",
        waveform=waveform,
    )
    sim.add_boundary_source(source)
    logger.info(f"Boundary source on left wall, frequency={source_frequency}")

    # Sensors at both ends
    sensor_locations = generate_boundary_sensors(grid, offset_cells=2)
    sensors = SensorArray(grid, sensor_locations)
    sim.set_sensors(sensors)
    logger.info(f"Sensors: {sensors.num_sensors} sensors at x={sensor_locations}")

    # Simulation time - enough for multiple reflections
    final_time = 4.0
    num_steps = int(final_time / sim.dt)
    frame_interval = max(1, num_steps // 300)

    logger.info(f"Simulation: {num_steps} steps, dt={sim.dt:.6g}s, t_final={final_time}s")
    logger.info(f"Recording every {frame_interval} steps")

    # Run simulation with frame recording
    recorder = FrameRecorder(keep_in_memory=True)

    def callback(s: Simulation):
        recorder(s)
        if s.step % 500 == 0:
            max_p = np.abs(s.get_pressure()).max()
            logger.info(f"  Step {s.step}/{num_steps}, t={s.time:.4f}s, max|p|={max_p:.4g}")

    sim.run(num_steps, callback=callback, callback_interval=frame_interval)

    logger.info(f"Recorded {len(recorder.frames)} frames")
    logger.info(f"Sensor data shape: {sensors.get_data_matrix().shape}")

    # Create output video
    output_dir = Path("data/fdtd_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_file = output_dir / "1d_right_inclusion.mp4"

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
        sensor_locations=sensor_locations,
        inclusion_region=(inclusion_x_min, inclusion_x_max),
        inclusion_alpha=0.6,  # More opaque to show higher density
        vmin=-1.0,
        vmax=1.0,
        fps=15,
        dpi=150,
        figsize=(14, 5),
    )

    if success:
        logger.info(f"Video saved to: {video_file}")
    else:
        logger.error("Failed to create video")

    # Save sensor data
    sensor_file = output_dir / "1d_right_sensor_data.npz"
    sensors.save(str(sensor_file))
    logger.info(f"Sensor data saved to: {sensor_file}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
