"""Sensor evaluation for field interpolation at arbitrary points.

Provides efficient interpolation of DG fields at sensor locations
using precomputed basis function evaluations.
"""

import math

import gmsh
import numpy as np

from sbimaging.array.backend import to_gpu, to_numpy, xp
from sbimaging.simulators.dg.dim3.basis import evaluate_simplex_basis_3d
from sbimaging.simulators.dg.dim3.mesh import MeshGeometry
from sbimaging.simulators.dg.dim3.reference_element import ReferenceOperators


class SensorArray:
    """Evaluates DG fields at fixed sensor locations.

    Precomputes basis function evaluations for efficient repeated
    field interpolation at sensor points.

    Attributes:
        locations: Array of (x, y, z) sensor coordinates.
        num_sensors: Number of sensors.
    """

    def __init__(
        self,
        mesh: MeshGeometry,
        operators: ReferenceOperators,
        locations: list[tuple[float, float, float]],
        polynomial_order: int,
    ):
        """Initialize sensor array.

        Args:
            mesh: MeshGeometry for element lookup.
            operators: Reference operators with inverse Vandermonde.
            locations: List of (x, y, z) sensor coordinates.
            polynomial_order: Polynomial order of approximation.
        """
        self._mesh = mesh
        self._inv_v = operators._inv_vandermonde
        self._p = polynomial_order
        self.locations = np.array(locations)
        self.num_sensors = len(locations)

        if not gmsh.isInitialized():
            gmsh.initialize()

        self._element_offset, _ = gmsh.model.mesh.getElementsByType(4)
        self._precompute_interpolation()

    def evaluate(self, field: np.ndarray) -> np.ndarray:
        """Evaluate field at all sensor locations.

        Args:
            field: Field array with shape (nodes_per_cell, num_cells).

        Returns:
            Array of field values at sensor locations.
        """
        results = xp.zeros(self.num_sensors, dtype=float)

        for elem, sensor_list in self._sensors_by_element.items():
            values = field[:, elem]
            weights = self._inv_v @ values

            for idx, phi_vec in sensor_list:
                results[idx] = weights @ phi_vec

        return results

    def _precompute_interpolation(self) -> None:
        """Precompute element indices and basis vectors for each sensor."""
        self._sensor_cache = []

        for x, y, z in self.locations:
            elem = self._get_element(x, y, z)
            r, s, t = self._map_to_reference(x, y, z, elem)
            phi = self._compute_basis_vector(r, s, t)
            self._sensor_cache.append((elem, to_gpu(phi)))

        self._sensors_by_element: dict[int, list] = {}
        for idx, (elem, phi) in enumerate(self._sensor_cache):
            self._sensors_by_element.setdefault(elem, []).append((idx, phi))

        self._inv_v = to_gpu(self._inv_v)

    def _get_element(self, x: float, y: float, z: float) -> int:
        """Find element containing point."""
        elem = gmsh.model.mesh.getElementByCoordinates(x, y, z, 3)[0]
        return int(elem - self._element_offset[0])

    def _map_to_reference(
        self, x: float, y: float, z: float, cell: int
    ) -> tuple[float, float, float]:
        """Map physical coordinates to reference tetrahedron."""
        ctv = to_numpy(self._mesh.cell_to_vertices)
        vx = to_numpy(self._mesh.x_vertex)
        vy = to_numpy(self._mesh.y_vertex)
        vz = to_numpy(self._mesh.z_vertex)

        va, vb, vc, vd = ctv[cell, :4]

        j = np.array([
            [vx[vb] - vx[va], vx[vc] - vx[va], vx[vd] - vx[va]],
            [vy[vb] - vy[va], vy[vc] - vy[va], vy[vd] - vy[va]],
            [vz[vb] - vz[va], vz[vc] - vz[va], vz[vd] - vz[va]],
        ], dtype=float)

        b = np.array([
            2 * x + vx[va] - vx[vb] - vx[vc] - vx[vd],
            2 * y + vy[va] - vy[vb] - vy[vc] - vy[vd],
            2 * z + vz[va] - vz[vb] - vz[vc] - vz[vd],
        ], dtype=float)

        rst = np.linalg.solve(j, b)
        return tuple(rst)

    def _compute_basis_vector(
        self, r: float, s: float, t: float
    ) -> np.ndarray:
        """Compute basis function values at reference point."""
        n = self._p
        phi_list = []

        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    m = (
                        1
                        + (11 + 12 * n + 3 * n**2) * i / 6
                        + (2 * n + 3) * j / 2
                        + k
                        - (2 + n) * i**2 / 2
                        - i * j
                        - j**2 / 2
                        + i**3 / 6
                    )
                    m = math.ceil(m - 1)

                    phi = evaluate_simplex_basis_3d(
                        np.array([r]), np.array([s]), np.array([t]), i, j, k
                    )[0]

                    if len(phi_list) <= m:
                        phi_list.extend([0.0] * (m - len(phi_list) + 1))
                    phi_list[m] = phi

        return np.array(phi_list, dtype=float)


def generate_grid_sensors(
    box_size: float,
    sensors_per_face: int,
    exclude_regions: list[tuple[tuple[float, float, float], float]] | None = None,
) -> list[tuple[float, float, float]]:
    """Generate sensor grid on domain boundary faces.

    Args:
        box_size: Size of cubic domain.
        sensors_per_face: Number of sensors along each dimension per face.
        exclude_regions: List of (center, radius) tuples for excluded regions.

    Returns:
        List of (x, y, z) sensor coordinates.
    """
    sensors = []
    margin = box_size / (2 * sensors_per_face)
    coords = np.linspace(margin, box_size - margin, sensors_per_face)

    for x in coords:
        for y in coords:
            sensors.append((x, y, 0.0))
            sensors.append((x, y, box_size))

    for x in coords:
        for z in coords:
            sensors.append((x, 0.0, z))
            sensors.append((x, box_size, z))

    for y in coords:
        for z in coords:
            sensors.append((0.0, y, z))
            sensors.append((box_size, y, z))

    if exclude_regions:
        filtered = []
        for sx, sy, sz in sensors:
            excluded = False
            for (cx, cy, cz), r in exclude_regions:
                dist2 = (sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2
                if dist2 < r ** 2:
                    excluded = True
                    break
            if not excluded:
                filtered.append((sx, sy, sz))
        sensors = filtered

    return sensors
