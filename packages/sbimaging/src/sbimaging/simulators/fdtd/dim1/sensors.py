"""Sensor arrays for 1D FDTD simulation.

Records pressure at specified locations over time.
"""

from pathlib import Path

import numpy as np

from sbimaging.array.backend import to_numpy
from sbimaging.simulators.fdtd.dim1.grid import Grid


class SensorArray:
    """Array of pressure sensors at fixed locations.

    Records pressure values at specified positions over time.

    Attributes:
        locations: List of sensor x-coordinates [m].
        num_sensors: Number of sensors.
        grid_indices: Grid indices for each sensor.
    """

    def __init__(self, grid: Grid, locations: list[float]):
        """Initialize sensor array.

        Args:
            grid: Computational grid.
            locations: List of x-coordinates for sensors [m].
        """
        self._grid = grid
        self.locations = list(locations)
        self.num_sensors = len(locations)

        # Find grid indices for each sensor
        self.grid_indices = [grid.index_at_position(x) for x in locations]

        # Storage for recorded data
        self._data: list[np.ndarray] = []
        self._times: list[float] = []

    def reset(self) -> None:
        """Clear all recorded data."""
        self._data = []
        self._times = []

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """Get pressure values at all sensor locations.

        Args:
            p: Current pressure field.

        Returns:
            Array of pressure values at sensor locations.
        """
        p_np = to_numpy(p)
        return np.array([p_np[i] for i in self.grid_indices])

    def record(self, p: np.ndarray, time: float) -> None:
        """Record current pressure at all sensors.

        Args:
            p: Current pressure field.
            time: Current simulation time [s].
        """
        values = self.evaluate(p)
        self._data.append(values)
        self._times.append(time)

    def get_data_matrix(self) -> np.ndarray:
        """Get all recorded data as a matrix.

        Returns:
            Array of shape (num_sensors, num_timesteps).
        """
        if not self._data:
            return np.array([]).reshape(self.num_sensors, 0)
        return np.column_stack(self._data)

    def get_times_array(self) -> np.ndarray:
        """Get array of recording times.

        Returns:
            1D array of times [s].
        """
        return np.array(self._times)

    def save(self, filepath: str | Path) -> None:
        """Save sensor data to file.

        Args:
            filepath: Output file path (.npz format).
        """
        filepath = Path(filepath)
        np.savez(
            filepath,
            data=self.get_data_matrix(),
            times=self.get_times_array(),
            locations=np.array(self.locations),
        )

    @classmethod
    def load(cls, filepath: str | Path, grid: Grid) -> "SensorArray":
        """Load sensor data from file.

        Args:
            filepath: Input file path (.npz format).
            grid: Computational grid.

        Returns:
            SensorArray with loaded data.
        """
        filepath = Path(filepath)
        data = np.load(filepath)

        locations = data["locations"].tolist()
        sensors = cls(grid, locations)

        sensors._data = [data["data"][:, i] for i in range(data["data"].shape[1])]
        sensors._times = data["times"].tolist()

        return sensors

    def __repr__(self) -> str:
        return f"SensorArray(num_sensors={self.num_sensors})"


def generate_boundary_sensors(
    grid: Grid,
    offset_cells: int = 2,
) -> list[float]:
    """Generate sensor locations at both ends of the domain.

    Args:
        grid: Computational grid.
        offset_cells: Number of cells inward from boundary.

    Returns:
        List of x-coordinates for sensors.
    """
    offset = offset_cells * grid.dx
    return [offset, grid.size_x - offset]
