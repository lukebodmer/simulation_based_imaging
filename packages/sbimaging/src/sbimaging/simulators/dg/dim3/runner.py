"""Simulation runner for 3D DG acoustics.

Provides high-level interface for setting up and running simulations
from configuration files or programmatic setup.
"""

import shutil
import sys
import time
from pathlib import Path

from sbimaging.array.backend import to_numpy
from sbimaging.logging import get_logger
from sbimaging.simulators.dg.dim3.acoustics import AcousticsOperator, Source
from sbimaging.simulators.dg.dim3.config import SimulationConfig
from sbimaging.simulators.dg.dim3.mesh import MeshGeometry, MeshLoader
from sbimaging.simulators.dg.dim3.output import EnergyCalculator, SimulationOutput
from sbimaging.simulators.dg.dim3.reference_element import (
    ReferenceOperators,
    ReferenceTetrahedron,
)
from sbimaging.simulators.dg.dim3.sensors import SensorArray, generate_grid_sensors
from sbimaging.simulators.dg.dim3.time_stepping import (
    LowStorageRungeKutta,
    compute_cfl_timestep,
)


class SimulationRunner:
    """Runs 3D DG acoustic simulations.

    Orchestrates mesh loading, physics setup, time stepping,
    sensor evaluation, and output saving.

    Attributes:
        config: Simulation configuration.
        output_dir: Directory for output files.
    """

    def __init__(
        self,
        config: SimulationConfig,
        output_dir: Path,
        mesh_file: Path | None = None,
    ):
        """Initialize simulation runner.

        Args:
            config: Simulation configuration.
            output_dir: Directory for output files.
            mesh_file: Path to Gmsh mesh file (overrides config).
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self._mesh_file = mesh_file or (
            Path(config.mesh.msh_file) if config.mesh.msh_file else None
        )

        self._logger = get_logger(__name__)
        self._element: ReferenceTetrahedron | None = None
        self._operators: ReferenceOperators | None = None
        self._mesh: MeshGeometry | None = None
        self._physics: AcousticsOperator | None = None
        self._stepper: LowStorageRungeKutta | None = None
        self._sensors: SensorArray | None = None
        self._output: SimulationOutput | None = None
        self._energy_calc: EnergyCalculator | None = None

    def setup(self) -> None:
        """Set up all simulation components."""
        self._logger.info("Setting up simulation")
        self._setup_output_dir()
        self._setup_reference_element()
        self._setup_mesh()
        self._setup_physics()
        self._setup_time_stepper()
        self._setup_sensors()
        self._setup_output()
        self._transfer_to_gpu()
        self._logger.info("Simulation setup complete")

    def run(self) -> None:
        """Run simulation to completion."""
        if self._stepper is None:
            raise RuntimeError("Call setup() before run()")

        self._logger.info(f"Starting simulation: {self._stepper.num_steps} steps")
        self._output.start()

        if self._output.energy_interval > 0:
            self._save_energy()

        start_time = time.time()

        while self._stepper.current_step < self._stepper.num_steps:
            self._stepper.step()
            self._save_scheduled_outputs()
            self._log_progress(start_time)

        self._output.save_final_results()
        self._logger.info("Simulation completed")

    def _setup_output_dir(self) -> None:
        """Create output directory and copy config."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_reference_element(self) -> None:
        """Create reference element and operators."""
        p = self.config.solver.polynomial_order
        self._element = ReferenceTetrahedron(polynomial_order=p)
        self._operators = ReferenceOperators(self._element)
        self._logger.info(f"Created order {p} reference element")

    def _setup_mesh(self) -> None:
        """Load mesh from file."""
        if self._mesh_file is None:
            raise ValueError("No mesh file specified")

        loader = MeshLoader(self._mesh_file)
        self._mesh = loader.load(self._element, self._operators)

        mat = self.config.material
        self._speed, self._density = loader.get_material_properties(
            nodes_per_cell=self._element.nodes_per_cell,
            num_cells=self._mesh.num_cells,
            default_speed=mat.outer_wave_speed,
            default_density=mat.outer_density,
            inclusion_materials={"Cube": (mat.inclusion_wave_speed, mat.inclusion_density)},
        )

        self._logger.info(
            f"Loaded mesh: {self._mesh.num_cells} cells, "
            f"diameter={self._mesh.smallest_diameter:.6g}"
        )

    def _setup_physics(self) -> None:
        """Create acoustics operator and add sources."""
        self._physics = AcousticsOperator(
            mesh=self._mesh,
            operators=self._operators,
            speed=self._speed,
            density=self._density,
        )

        src_cfg = self.config.sources
        for center, radius, freq, amp in zip(
            src_cfg.centers,
            src_cfg.radii,
            src_cfg.frequencies,
            src_cfg.amplitudes,
        ):
            source = Source(
                center=tuple(center),
                radius=radius,
                frequency=freq,
                amplitude=amp,
            )
            self._physics.add_source(source)

        self._logger.info(f"Added {len(src_cfg.centers)} sources")

    def _setup_time_stepper(self) -> None:
        """Create time stepper with CFL-based dt."""
        max_speed = float(self._speed.max())
        dt = compute_cfl_timestep(
            smallest_diameter=self._mesh.smallest_diameter,
            max_speed=max_speed,
            polynomial_order=self.config.solver.polynomial_order,
            cfl_factor=self.config.solver.cfl_factor,
        )

        solver = self.config.solver
        if solver.total_time is not None:
            self._stepper = LowStorageRungeKutta(
                physics=self._physics,
                dt=dt,
                t_final=solver.total_time,
            )
        else:
            self._stepper = LowStorageRungeKutta(
                physics=self._physics,
                dt=dt,
                num_steps=solver.num_steps,
            )

    def _setup_sensors(self) -> None:
        """Set up sensor array for field evaluation."""
        rcv = self.config.receivers

        if rcv.pressure:
            locations = [tuple(p) for p in rcv.pressure]
        elif rcv.sensors_per_face:
            src_cfg = self.config.sources
            exclude = [
                (tuple(c), r)
                for c, r in zip(src_cfg.centers, src_cfg.radii)
            ]
            locations = generate_grid_sensors(
                box_size=self.config.mesh.box_size,
                sensors_per_face=rcv.sensors_per_face,
                exclude_regions=exclude,
            )
        else:
            locations = []

        if rcv.additional_sensors:
            locations.extend([tuple(s) for s in rcv.additional_sensors])

        if locations:
            self._sensors = SensorArray(
                mesh=self._mesh,
                operators=self._operators,
                locations=locations,
                polynomial_order=self.config.solver.polynomial_order,
            )
            self._logger.info(f"Created {self._sensors.num_sensors} sensors")

    def _setup_output(self) -> None:
        """Set up output handler."""
        out = self.config.output
        self._output = SimulationOutput(
            output_dir=self.output_dir,
            num_steps=self._stepper.num_steps,
            sensor_interval=out.sensor_interval,
            data_interval=out.data_interval,
            energy_interval=out.energy_interval,
        )

        if self._sensors and out.sensor_interval > 0:
            self._output.initialize_sensors(
                sensor_names=["pressure"],
                num_sensors=self._sensors.num_sensors,
            )

        if out.energy_interval > 0:
            self._energy_calc = EnergyCalculator(
                mass_matrix=self._operators.mass_matrix,
                jacobians=to_numpy(self._mesh.jacobians),
                density=self._density,
                speed=self._speed,
            )

    def _transfer_to_gpu(self) -> None:
        """Transfer data to GPU."""
        self._mesh.transfer_to_gpu()
        self._physics.transfer_to_gpu()

    def _save_scheduled_outputs(self) -> None:
        """Save outputs at scheduled intervals."""
        step = self._stepper.current_step

        if self._output.should_save_sensors(step) and self._sensors:
            values = self._sensors.evaluate(self._physics.p)
            self._output.save_sensor_reading("pressure", values)
            self._output.advance_sensor_index()

        if self._output.should_save_energy(step):
            self._save_energy()

        if self._output.should_save_data(step):
            self._output.save_snapshot(
                step=step,
                t=self._stepper.t,
                dt=self._stepper.dt,
                fields={
                    "p": self._physics.p,
                    "u": self._physics.u,
                    "v": self._physics.v,
                    "w": self._physics.w,
                },
            )

    def _save_energy(self) -> None:
        """Calculate and save energy."""
        if self._energy_calc:
            total, kinetic, potential = self._energy_calc.compute(
                self._physics.p,
                self._physics.u,
                self._physics.v,
                self._physics.w,
            )
            self._output.save_energy(total, kinetic, potential)

    def _log_progress(self, start_time: float) -> None:
        """Log simulation progress."""
        runtime = time.time() - start_time
        step = self._stepper.current_step
        t = self._stepper.t
        sys.stdout.write(f"\rStep: {step}, Time: {t:.6f}, Runtime: {runtime:.2f}s")
        sys.stdout.flush()


def run_simulation(
    config_path: Path,
    output_dir: Path,
    mesh_file: Path | None = None,
) -> None:
    """Run simulation from configuration file.

    Args:
        config_path: Path to TOML configuration file.
        output_dir: Directory for output files.
        mesh_file: Path to mesh file (overrides config).
    """
    config = SimulationConfig.from_toml(config_path)

    shutil.copy(config_path, output_dir / "config.toml")

    runner = SimulationRunner(
        config=config,
        output_dir=output_dir,
        mesh_file=mesh_file,
    )
    runner.setup()
    runner.run()
