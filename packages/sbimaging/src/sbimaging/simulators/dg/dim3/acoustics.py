"""Linear acoustics physics for 3D DG methods.

Implements the velocity-pressure formulation of linear acoustics
with upwind numerical flux for heterogeneous media.
"""

import numpy as np

from sbimaging.array.backend import to_gpu, to_numpy, xp
from sbimaging.simulators.dg.dim3.mesh import MeshGeometry
from sbimaging.simulators.dg.dim3.reference_element import ReferenceOperators


class AcousticsOperator:
    """Computes right-hand side for linear acoustics equations.

    Uses velocity-pressure formulation with upwind flux for
    heterogeneous media. Supports multiple pressure sources.

    Attributes:
        mesh: MeshGeometry with geometric data.
        operators: Reference operators for differentiation.
        p: Pressure field (nodes_per_cell, num_cells).
        u, v, w: Velocity fields (nodes_per_cell, num_cells).
    """

    def __init__(
        self,
        mesh: MeshGeometry,
        operators: ReferenceOperators,
        speed: np.ndarray,
        density: np.ndarray,
    ):
        """Initialize acoustics operator.

        Args:
            mesh: MeshGeometry with all geometric quantities.
            operators: Reference operators (lift, differentiation).
            speed: Wave speed at each node (nodes_per_cell, num_cells).
            density: Material density at each node (nodes_per_cell, num_cells).
        """
        self.mesh = mesh
        self._operators = operators
        self._speed = speed
        self._density = density

        npc = mesh._element.nodes_per_cell
        k = mesh.num_cells

        self.p = xp.zeros((npc, k), order="F")
        self.u = xp.zeros((npc, k), order="F")
        self.v = xp.zeros((npc, k), order="F")
        self.w = xp.zeros((npc, k), order="F")

        self._sources: list[Source] = []
        self._precompute_operators()

    def add_source(self, source: "Source") -> None:
        """Add a pressure source.

        Args:
            source: Source object defining location and waveform.
        """
        source.locate_nodes(self.mesh)
        self._sources.append(source)

    def transfer_to_gpu(self) -> None:
        """Transfer operator data to GPU."""
        self._lift = to_gpu(self._lift)
        self._face_scale = to_gpu(self._face_scale)
        self._inv_rho = to_gpu(self._inv_rho)
        self._bulk = to_gpu(self._bulk)
        self._dx = to_gpu(self._dx)
        self._dy = to_gpu(self._dy)
        self._dz = to_gpu(self._dz)
        self._speed = to_gpu(self._speed)
        self._density = to_gpu(self._density)

        self.p = to_gpu(to_numpy(self.p))
        self.u = to_gpu(to_numpy(self.u))
        self.v = to_gpu(to_numpy(self.v))
        self.w = to_gpu(to_numpy(self.w))

        self._precompute_flux_terms()
        self._precompute_indices()

    def compute_rhs(
        self,
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        w: np.ndarray | None = None,
        p: np.ndarray | None = None,
        time: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute right-hand side of acoustics equations.

        Args:
            u, v, w: Velocity fields (use stored if None).
            p: Pressure field (use stored if None).
            time: Current simulation time.

        Returns:
            Tuple of (rhs_u, rhs_v, rhs_w, rhs_p) arrays.
        """
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if p is None:
            p = self.p

        dudx = xp.einsum("ijk,jk->ik", self._dx, u, optimize="optimal")
        dvdy = xp.einsum("ijk,jk->ik", self._dy, v, optimize="optimal")
        dwdz = xp.einsum("ijk,jk->ik", self._dz, w, optimize="optimal")
        dpdx = xp.einsum("ijk,jk->ik", self._dx, p, optimize="optimal")
        dpdy = xp.einsum("ijk,jk->ik", self._dy, p, optimize="optimal")
        dpdz = xp.einsum("ijk,jk->ik", self._dz, p, optimize="optimal")

        self._apply_boundary_conditions(u, v, w, p, time)

        ndotum = self.mesh.nx * self._u_m + self.mesh.ny * self._v_m + self.mesh.nz * self._w_m
        ndotup = self.mesh.nx * self._u_p + self.mesh.ny * self._v_p + self.mesh.nz * self._w_p

        flux_p, flux_u, flux_v, flux_w = self._compute_upwind_flux(ndotum, ndotup)

        rhs_p = -self._bulk * (dudx + dvdy + dwdz) - self._lift @ (self._face_scale * flux_p)
        rhs_u = -self._inv_rho * dpdx - self._lift @ (self._face_scale * flux_u)
        rhs_v = -self._inv_rho * dpdy - self._lift @ (self._face_scale * flux_v)
        rhs_w = -self._inv_rho * dpdz - self._lift @ (self._face_scale * flux_w)

        return rhs_u, rhs_v, rhs_w, rhs_p

    def _precompute_operators(self) -> None:
        """Precompute spatial derivative and material operators."""
        self._lift = self._operators.lift
        self._face_scale = self.mesh.surface_to_volume_jacobian

        self._inv_rho = 1.0 / self._density
        self._bulk = self._density * (self._speed ** 2)

        dr = self._operators.diff_r
        ds = self._operators.diff_s
        dt = self._operators.diff_t

        npc = self.mesh._element.nodes_per_cell
        k = self.mesh.num_cells

        self._dx = np.empty((npc, npc, k))
        self._dy = np.empty((npc, npc, k))
        self._dz = np.empty((npc, npc, k))

        drdx = to_numpy(self.mesh.drdx)
        drdy = to_numpy(self.mesh.drdy)
        drdz = to_numpy(self.mesh.drdz)
        dsdx = to_numpy(self.mesh.dsdx)
        dsdy = to_numpy(self.mesh.dsdy)
        dsdz = to_numpy(self.mesh.dsdz)
        dtdx = to_numpy(self.mesh.dtdx)
        dtdy = to_numpy(self.mesh.dtdy)
        dtdz = to_numpy(self.mesh.dtdz)

        for cell in range(k):
            self._dx[:, :, cell] = (
                drdx[:, cell, None] * dr
                + dsdx[:, cell, None] * ds
                + dtdx[:, cell, None] * dt
            )
            self._dy[:, :, cell] = (
                drdy[:, cell, None] * dr
                + dsdy[:, cell, None] * ds
                + dtdy[:, cell, None] * dt
            )
            self._dz[:, :, cell] = (
                drdz[:, cell, None] * dr
                + dsdz[:, cell, None] * ds
                + dtdz[:, cell, None] * dt
            )

        self._precompute_flux_terms()
        self._precompute_indices()

    def _precompute_flux_terms(self) -> None:
        """Precompute material properties at face nodes for flux computation."""
        ext = self.mesh.exterior_face_node_indices
        intr = self.mesh.interior_face_node_indices
        npf = self.mesh._element.nodes_per_face
        nf = self.mesh._element.num_faces
        k = self.mesh.num_cells

        rho_flat = self._density.ravel("F")
        c_flat = self._speed.ravel("F")

        self._rho_p = rho_flat[ext].reshape((npf * nf, k), order="F")
        self._rho_m = rho_flat[intr].reshape((npf * nf, k), order="F")
        self._c_p = c_flat[ext].reshape((npf * nf, k), order="F")
        self._c_m = c_flat[intr].reshape((npf * nf, k), order="F")

        self._flux_denom = self._c_m * self._rho_m + self._c_p * self._rho_p
        self._z_p = self._c_p * self._rho_p
        self._k_m = self._c_m ** 2 * self._rho_m

    def _precompute_indices(self) -> None:
        """Precompute index arrays for boundary conditions."""
        self._boundary_flat = self.mesh.boundary_face_node_indices.ravel(order="F")
        self._interior_flat = self.mesh.interior_face_node_indices.ravel(order="F")
        self._exterior_flat = self.mesh.exterior_face_node_indices.ravel(order="F")

        self._nx_flat = self.mesh.nx.ravel(order="F")
        self._ny_flat = self.mesh.ny.ravel(order="F")
        self._nz_flat = self.mesh.nz.ravel(order="F")

        for source in self._sources:
            source.precompute_boundary_indices(self._boundary_flat)

    def _apply_boundary_conditions(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        p: np.ndarray,
        time: float,
    ) -> None:
        """Apply boundary conditions to get interior/exterior traces."""
        u_flat = u.ravel("F")
        v_flat = v.ravel("F")
        w_flat = w.ravel("F")
        p_flat = p.ravel("F")

        u_m = u_flat[self._interior_flat]
        v_m = v_flat[self._interior_flat]
        w_m = w_flat[self._interior_flat]
        p_m = p_flat[self._interior_flat]

        u_p = u_flat[self._exterior_flat].copy()
        v_p = v_flat[self._exterior_flat].copy()
        w_p = w_flat[self._exterior_flat].copy()
        p_p = p_flat[self._exterior_flat].copy()

        self._apply_reflecting_bc(u_m, v_m, w_m, p_m, u_p, v_p, w_p, p_p)
        self._apply_source_bc(u_m, v_m, w_m, p_m, u_p, v_p, w_p, p_p, time)

        npf = self.mesh._element.nodes_per_face
        nf = self.mesh._element.num_faces
        k = self.mesh.num_cells

        self._u_m = u_m.reshape((npf * nf, k), order="F")
        self._v_m = v_m.reshape((npf * nf, k), order="F")
        self._w_m = w_m.reshape((npf * nf, k), order="F")
        self._p_m = p_m.reshape((npf * nf, k), order="F")
        self._u_p = u_p.reshape((npf * nf, k), order="F")
        self._v_p = v_p.reshape((npf * nf, k), order="F")
        self._w_p = w_p.reshape((npf * nf, k), order="F")
        self._p_p = p_p.reshape((npf * nf, k), order="F")

    def _apply_reflecting_bc(
        self,
        u_m: np.ndarray,
        v_m: np.ndarray,
        w_m: np.ndarray,
        p_m: np.ndarray,
        u_p: np.ndarray,
        v_p: np.ndarray,
        w_p: np.ndarray,
        p_p: np.ndarray,
    ) -> None:
        """Apply perfectly reflecting boundary condition."""
        bnd = self._boundary_flat
        nx, ny, nz = self._nx_flat, self._ny_flat, self._nz_flat

        ndotum = nx[bnd] * u_m[bnd] + ny[bnd] * v_m[bnd] + nz[bnd] * w_m[bnd]

        u_p[bnd] = u_m[bnd] - 2.0 * ndotum * nx[bnd]
        v_p[bnd] = v_m[bnd] - 2.0 * ndotum * ny[bnd]
        w_p[bnd] = w_m[bnd] - 2.0 * ndotum * nz[bnd]
        p_p[bnd] = p_m[bnd]

    def _apply_source_bc(
        self,
        u_m: np.ndarray,
        v_m: np.ndarray,
        w_m: np.ndarray,
        p_m: np.ndarray,
        u_p: np.ndarray,
        v_p: np.ndarray,
        w_p: np.ndarray,
        p_p: np.ndarray,
        time: float,
    ) -> None:
        """Apply source boundary conditions."""
        nx, ny, nz = self._nx_flat, self._ny_flat, self._nz_flat

        for source in self._sources:
            p_source = source.get_pressure(time)
            src_nodes = source.boundary_node_indices

            if src_nodes.size == 0:
                continue

            z = self._rho_p.ravel("F")[src_nodes] * self._c_p.ravel("F")[src_nodes]

            pm = p_m[src_nodes]
            um, vm, wm = u_m[src_nodes], v_m[src_nodes], w_m[src_nodes]
            nx_s, ny_s, nz_s = nx[src_nodes], ny[src_nodes], nz[src_nodes]

            delta_un = (pm - p_source) / z

            p_p[src_nodes] = p_source
            u_p[src_nodes] = um + delta_un * nx_s
            v_p[src_nodes] = vm + delta_un * ny_s
            w_p[src_nodes] = wm + delta_un * nz_s

    def _compute_upwind_flux(
        self,
        ndotum: np.ndarray,
        ndotup: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute upwind numerical flux for heterogeneous media."""
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz

        normal_vel_jump = ndotup - ndotum
        pressure_jump = self._p_p - self._p_m

        num = -self._z_p * normal_vel_jump + pressure_jump
        common = num / self._flux_denom

        flux_p = -self._k_m * common
        flux_u = nx * self._c_m * common
        flux_v = ny * self._c_m * common
        flux_w = nz * self._c_m * common

        return flux_p, flux_u, flux_v, flux_w


class Source:
    """Pressure source on mesh boundary.

    Attributes:
        center: (x, y, z) coordinates of source center.
        radius: Radius of circular source region.
        frequency: Source frequency in Hz.
        amplitude: Source pressure amplitude.
    """

    def __init__(
        self,
        center: tuple[float, float, float],
        radius: float,
        frequency: float,
        amplitude: float,
    ):
        """Initialize pressure source.

        Args:
            center: (x, y, z) center coordinates on boundary.
            radius: Radius of source region.
            frequency: Source frequency in Hz.
            amplitude: Pressure amplitude.
        """
        self.center = center
        self.radius = radius
        self.frequency = frequency
        self.amplitude = amplitude

        self.node_indices: np.ndarray = np.array([], dtype=int)
        self.boundary_node_indices: np.ndarray = np.array([], dtype=int)

    def locate_nodes(self, mesh: MeshGeometry) -> None:
        """Find mesh nodes within source region.

        Args:
            mesh: MeshGeometry to search for source nodes.
        """
        ext = to_numpy(mesh.exterior_face_node_indices)
        bnd = to_numpy(mesh.boundary_face_node_indices)
        npf = mesh._element.nodes_per_face
        tol = 1e-6

        x_b = to_numpy(mesh.x).ravel(order="F")[ext][bnd]
        y_b = to_numpy(mesh.y).ravel(order="F")[ext][bnd]
        z_b = to_numpy(mesh.z).ravel(order="F")[ext][bnd]

        cx, cy, cz = self.center
        r = self.radius

        if abs(cx) < tol or abs(cx - 1.0) < tol:
            in_source = ((y_b - cy) ** 2 + (z_b - cz) ** 2 < r ** 2 + tol) & (
                np.abs(x_b - cx) < tol
            )
        elif abs(cy) < tol or abs(cy - 1.0) < tol:
            in_source = ((x_b - cx) ** 2 + (z_b - cz) ** 2 < r ** 2 + tol) & (
                np.abs(y_b - cy) < tol
            )
        elif abs(cz) < tol or abs(cz - 1.0) < tol:
            in_source = ((x_b - cx) ** 2 + (y_b - cy) ** 2 < r ** 2 + tol) & (
                np.abs(z_b - cz) < tol
            )
        else:
            raise ValueError(
                f"Source center at ({cx}, {cy}, {cz}) is not on a boundary plane."
            )

        faces = np.where(in_source)[0] // npf
        unique_faces, counts = np.unique(faces, return_counts=True)
        included_faces = unique_faces[counts == npf]

        base = included_faces * npf
        offsets = np.arange(npf)
        full_ranges = base[:, np.newaxis] + offsets
        self.node_indices = full_ranges.ravel()

    def precompute_boundary_indices(self, boundary_flat: np.ndarray) -> None:
        """Precompute boundary node indices for source.

        Args:
            boundary_flat: Flattened boundary node indices.
        """
        self.boundary_node_indices = to_numpy(boundary_flat)[self.node_indices]
        self.boundary_node_indices = to_gpu(self.boundary_node_indices)

    def get_pressure(self, time: float) -> float:
        """Get source pressure at given time.

        Uses Gaussian pulse waveform.

        Args:
            time: Current simulation time.

        Returns:
            Source pressure value.
        """
        f = self.frequency
        t0 = 1 / (2 * f)
        sigma = t0 / 4
        pulse = np.exp(-((time - t0) ** 2 / (2 * sigma ** 2)))
        return self.amplitude * pulse
