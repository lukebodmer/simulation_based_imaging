"""Sensor arrays for 2D FDTD simulation.

Implements sensors positioned around the domain boundary to record
pressure values over time, similar to the 3D DG sensor system.
"""

import numpy as np

from sbimaging.array.backend import to_numpy
from sbimaging.simulators.fdtd.dim2.grid import Grid


class SensorArray:
    """Array of pressure sensors at fixed locations.

    Records pressure values at sensor locations over time.
    Sensors are typically placed around the domain boundary.

    Attributes:
        locations: Array of (x, y) sensor coordinates.
        num_sensors: Number of sensors.
        recordings: Recorded pressure values (num_sensors, num_timesteps).
        times: Recorded simulation times.
    """

    def __init__(
        self,
        grid: Grid,
        locations: list[tuple[float, float]],
    ):
        """Initialize sensor array.

        Args:
            grid: Computational grid for index lookup.
            locations: List of (x, y) sensor coordinates.
        """
        self._grid = grid
        self.locations = np.array(locations)
        self.num_sensors = len(locations)

        self._grid_indices = self._compute_grid_indices()

        self.recordings: list[np.ndarray] = []
        self.times: list[float] = []

    def _compute_grid_indices(self) -> list[tuple[int, int]]:
        """Compute grid indices for each sensor location."""
        indices = []
        for x, y in self.locations:
            i, j = self._grid.index_at_position(x, y)
            indices.append((i, j))
        return indices

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """Evaluate pressure at all sensor locations.

        Args:
            p: Pressure field array.

        Returns:
            Array of pressure values at sensor locations.
        """
        p_np = to_numpy(p)
        values = np.zeros(self.num_sensors)
        for idx, (i, j) in enumerate(self._grid_indices):
            values[idx] = p_np[i, j]
        return values

    def record(self, p: np.ndarray, time: float) -> None:
        """Record pressure at all sensors for current timestep.

        Args:
            p: Pressure field array.
            time: Current simulation time.
        """
        values = self.evaluate(p)
        self.recordings.append(values)
        self.times.append(time)

    def get_data_matrix(self) -> np.ndarray:
        """Get recorded data as a 2D matrix.

        Returns:
            Array of shape (num_sensors, num_timesteps).
        """
        if not self.recordings:
            return np.zeros((self.num_sensors, 0))
        return np.column_stack(self.recordings)

    def get_times_array(self) -> np.ndarray:
        """Get recorded times as array.

        Returns:
            Array of simulation times.
        """
        return np.array(self.times)

    def reset(self) -> None:
        """Clear all recorded data."""
        self.recordings = []
        self.times = []

    def save(self, filepath: str) -> None:
        """Save sensor data to file.

        Args:
            filepath: Path to save file (.npz format).
        """
        np.savez(
            filepath,
            locations=self.locations,
            data=self.get_data_matrix(),
            times=self.get_times_array(),
        )

    @classmethod
    def load(cls, filepath: str, grid: Grid) -> "SensorArray":
        """Load sensor data from file.

        Args:
            filepath: Path to load file (.npz format).
            grid: Computational grid.

        Returns:
            SensorArray with loaded data.
        """
        data = np.load(filepath)
        locations = [tuple(loc) for loc in data["locations"]]
        sensor_array = cls(grid, locations)

        data_matrix = data["data"]
        times = data["times"]

        for t_idx in range(data_matrix.shape[1]):
            sensor_array.recordings.append(data_matrix[:, t_idx])
            sensor_array.times.append(times[t_idx])

        return sensor_array


def generate_boundary_sensors(
    grid: Grid,
    sensors_per_side: int,
    margin_fraction: float = 0.1,
    offset_cells: int = 1,
) -> list[tuple[float, float]]:
    """Generate sensors near the domain boundary, offset inward.

    Places sensors on all four sides of the rectangular domain,
    offset inward by a specified number of grid cells. This ensures
    sensors measure non-zero pressure with reflective boundary conditions.

    Args:
        grid: Computational grid.
        sensors_per_side: Number of sensors on each side.
        margin_fraction: Fraction of side length to use as margin from corners.
        offset_cells: Number of grid cells to offset inward from boundary.

    Returns:
        List of (x, y) sensor coordinates.
    """
    sensors = []

    margin_x = grid.size_x * margin_fraction
    margin_y = grid.size_y * margin_fraction

    # Offset from boundary (in physical units)
    offset_x = offset_cells * grid.dx
    offset_y = offset_cells * grid.dy

    x_positions = np.linspace(margin_x, grid.size_x - margin_x, sensors_per_side)
    y_positions = np.linspace(margin_y, grid.size_y - margin_y, sensors_per_side)

    # Bottom edge (y = offset)
    for x in x_positions:
        sensors.append((x, offset_y))

    # Top edge (y = size_y - offset)
    for x in x_positions:
        sensors.append((x, grid.size_y - offset_y))

    # Left edge (x = offset)
    for y in y_positions:
        sensors.append((offset_x, y))

    # Right edge (x = size_x - offset)
    for y in y_positions:
        sensors.append((grid.size_x - offset_x, y))

    return sensors


def generate_boundary_sensors_uniform(
    grid: Grid,
    total_sensors: int,
    margin_fraction: float = 0.1,
) -> list[tuple[float, float]]:
    """Generate sensors uniformly distributed around domain perimeter.

    Distributes sensors evenly around the perimeter based on arc length.

    Args:
        grid: Computational grid.
        total_sensors: Total number of sensors around the boundary.
        margin_fraction: Fraction of domain to use as margin from corners.

    Returns:
        List of (x, y) sensor coordinates.
    """
    margin_x = grid.size_x * margin_fraction
    margin_y = grid.size_y * margin_fraction

    effective_width = grid.size_x - 2 * margin_x
    effective_height = grid.size_y - 2 * margin_y
    effective_perimeter = 2 * (effective_width + effective_height)

    sensors = []
    spacing = effective_perimeter / total_sensors

    for i in range(total_sensors):
        target_arc = i * spacing

        # Determine which edge and position
        if target_arc < effective_width:
            # Bottom edge
            x = margin_x + target_arc
            y = 0.0
        elif target_arc < effective_width + effective_height:
            # Right edge
            x = grid.size_x
            y = margin_y + (target_arc - effective_width)
        elif target_arc < 2 * effective_width + effective_height:
            # Top edge
            x = grid.size_x - margin_x - (target_arc - effective_width - effective_height)
            y = grid.size_y
        else:
            # Left edge
            x = 0.0
            y = grid.size_y - margin_y - (target_arc - 2 * effective_width - effective_height)

        sensors.append((x, y))

    return sensors
