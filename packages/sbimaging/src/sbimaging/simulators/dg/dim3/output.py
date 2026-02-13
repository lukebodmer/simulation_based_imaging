"""Output handling for simulation results.

Provides classes for saving simulation data, sensor readings,
energy history, and field snapshots.
"""

import pickle
import time
from pathlib import Path

import numpy as np

from sbimaging.array.backend import to_numpy, xp
from sbimaging.logging import get_logger


class SimulationOutput:
    """Manages output of simulation results.

    Handles periodic saving of sensor data, field snapshots,
    and energy calculations.

    Attributes:
        output_dir: Directory for output files.
        sensor_interval: Steps between sensor readings.
        data_interval: Steps between full data saves.
        energy_interval: Steps between energy calculations.
    """

    def __init__(
        self,
        output_dir: Path,
        num_steps: int,
        sensor_interval: int = 10,
        data_interval: int = 0,
        energy_interval: int = 0,
    ):
        """Initialize output handler.

        Args:
            output_dir: Directory for output files.
            num_steps: Total number of simulation steps.
            sensor_interval: Steps between sensor readings (0 to disable).
            data_interval: Steps between full data saves (0 to disable).
            energy_interval: Steps between energy calculations (0 to disable).
        """
        self.output_dir = Path(output_dir)
        self.sensor_interval = sensor_interval
        self.data_interval = data_interval
        self.energy_interval = energy_interval

        self._logger = get_logger(__name__)
        self._start_time = 0.0
        self._data_index = 0

        self._setup_directories()

        if sensor_interval > 0:
            self._num_sensor_readings = num_steps // sensor_interval
            self._sensor_index = 0
        else:
            self._num_sensor_readings = 0

        if energy_interval > 0:
            self._num_energy_readings = num_steps // energy_interval + 1
            self._energy_index = 0
            self.energy_data = np.zeros(self._num_energy_readings)
            self.kinetic_data = np.zeros(self._num_energy_readings)
            self.potential_data = np.zeros(self._num_energy_readings)
        else:
            self.energy_data = None
            self.kinetic_data = None
            self.potential_data = None

        self._sensor_data: dict[str, np.ndarray] = {}

    def initialize_sensors(
        self,
        sensor_names: list[str],
        num_sensors: int,
    ) -> None:
        """Initialize sensor data arrays.

        Args:
            sensor_names: Names of tracked sensor fields.
            num_sensors: Number of sensors.
        """
        for name in sensor_names:
            self._sensor_data[name] = np.zeros(
                (num_sensors, self._num_sensor_readings)
            )

    def start(self) -> None:
        """Mark simulation start time."""
        self._start_time = time.time()

    def should_save_sensors(self, step: int) -> bool:
        """Check if sensors should be saved at this step."""
        return self.sensor_interval > 0 and step % self.sensor_interval == 0

    def should_save_data(self, step: int) -> bool:
        """Check if full data should be saved at this step."""
        return self.data_interval > 0 and step % self.data_interval == 0

    def should_save_energy(self, step: int) -> bool:
        """Check if energy should be saved at this step."""
        return self.energy_interval > 0 and step % self.energy_interval == 0

    def save_sensor_reading(self, name: str, values: np.ndarray) -> None:
        """Save sensor reading.

        Args:
            name: Sensor field name.
            values: Array of sensor values.
        """
        if name in self._sensor_data:
            self._sensor_data[name][:, self._sensor_index] = to_numpy(values)

    def advance_sensor_index(self) -> None:
        """Increment sensor reading index."""
        self._sensor_index += 1

    def save_energy(
        self,
        total: float,
        kinetic: float,
        potential: float,
    ) -> None:
        """Save energy values.

        Args:
            total: Total energy.
            kinetic: Kinetic energy.
            potential: Potential energy.
        """
        if self.energy_data is not None:
            self.energy_data[self._energy_index] = float(total)
            self.kinetic_data[self._energy_index] = float(kinetic)
            self.potential_data[self._energy_index] = float(potential)
            self._energy_index += 1

    def save_snapshot(
        self,
        step: int,
        t: float,
        dt: float,
        fields: dict[str, np.ndarray],
        metadata: dict | None = None,
    ) -> None:
        """Save field snapshot to file.

        Args:
            step: Current time step.
            t: Current simulation time.
            dt: Time step size.
            fields: Dictionary of field arrays.
            metadata: Optional additional metadata.
        """
        data = {
            "step": step,
            "time": t,
            "dt": dt,
            "fields": {k: to_numpy(v) for k, v in fields.items()},
            "runtime": time.time() - self._start_time,
        }
        if metadata:
            data.update(metadata)

        filename = f"{self._data_index:08d}_t{step:08d}.pkl"
        filepath = self.output_dir / "data" / filename

        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._data_index += 1

    def save_final_results(self) -> None:
        """Save final sensor data and energy history."""
        if self._sensor_data:
            sensor_path = self.output_dir / "sensor_data.pkl"
            with open(sensor_path, "wb") as f:
                pickle.dump(self._sensor_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._logger.info(f"Saved sensor data to {sensor_path}")

        if self.energy_data is not None:
            energy_path = self.output_dir / "energy_data.pkl"
            energy_dict = {
                "total": self.energy_data[: self._energy_index],
                "kinetic": self.kinetic_data[: self._energy_index],
                "potential": self.potential_data[: self._energy_index],
            }
            with open(energy_path, "wb") as f:
                pickle.dump(energy_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._logger.info(f"Saved energy data to {energy_path}")

    def _setup_directories(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)


class EnergyCalculator:
    """Computes energy quantities from field data.

    Uses mass matrix integration for accurate energy calculation.
    """

    def __init__(
        self,
        mass_matrix: np.ndarray,
        jacobians: np.ndarray,
        density: np.ndarray,
        speed: np.ndarray,
    ):
        """Initialize energy calculator.

        Args:
            mass_matrix: Reference element mass matrix.
            jacobians: Jacobian determinants (first row).
            density: Material density (first row).
            speed: Wave speed (first row).
        """
        self._mass = xp.asarray(mass_matrix)
        self._j = xp.asarray(jacobians[0, :] if jacobians.ndim > 1 else jacobians)
        self._rho = xp.asarray(density[0, :] if density.ndim > 1 else density)
        self._c = xp.asarray(speed[0, :] if speed.ndim > 1 else speed)
        self._inv_bulk = 1.0 / (self._rho * self._c ** 2)

    def compute(
        self,
        p: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute total, kinetic, and potential energy.

        Args:
            p: Pressure field.
            u, v, w: Velocity fields.

        Returns:
            Tuple of (total, kinetic, potential) energy.
        """
        mp = self._mass @ p
        p_quad = xp.sum(p * mp, axis=0)
        potential = 0.5 * self._j * self._inv_bulk * p_quad

        mu = self._mass @ u
        mv = self._mass @ v
        mw = self._mass @ w
        kinetic_quad = (
            xp.sum(u * mu, axis=0) + xp.sum(v * mv, axis=0) + xp.sum(w * mw, axis=0)
        )
        kinetic = 0.5 * self._j * self._rho * kinetic_quad

        total = xp.sum(potential + kinetic)
        return float(total), float(xp.sum(kinetic)), float(xp.sum(potential))
