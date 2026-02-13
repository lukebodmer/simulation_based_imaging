"""Mesh loading and geometry computations for 3D DG methods.

Loads tetrahedral meshes from Gmsh files and computes all geometric
quantities needed for DG simulation: mapping coefficients, normals,
connectivity, and face node mappings.
"""

import pickle
from pathlib import Path

import gmsh
import numpy as np

from sbimaging.array.backend import to_gpu, xp
from sbimaging.logging import get_logger
from sbimaging.simulators.dg.dim3.reference_element import (
    ReferenceOperators,
    ReferenceTetrahedron,
)


class MeshGeometry:
    """Geometric data for a tetrahedral mesh.

    Stores vertex coordinates, cell-to-vertex mappings, connectivity,
    and all derived geometric quantities like Jacobians and normals.

    Attributes:
        num_vertices: Number of mesh vertices.
        num_cells: Number of tetrahedral cells.
        vertex_coordinates: (num_vertices, 3) array of vertex positions.
        x, y, z: Nodal coordinates (nodes_per_cell, num_cells).
        cell_to_vertices: (num_cells, 4) vertex indices per cell.
        cell_to_cells: (num_cells, 4) neighbor cell indices.
        cell_to_faces: (num_cells, 4) neighbor face indices.
        jacobians: Jacobian determinants at each node.
        drdx, drdy, drdz: Metric coefficients for r derivative.
        dsdx, dsdy, dsdz: Metric coefficients for s derivative.
        dtdx, dtdy, dtdz: Metric coefficients for t derivative.
        nx, ny, nz: Outward unit normals at face nodes.
        surface_jacobians: Surface Jacobians at face nodes.
        interior_face_node_indices: Global indices for interior face nodes.
        exterior_face_node_indices: Global indices for exterior face nodes.
        boundary_face_node_indices: Indices of nodes on domain boundary.
    """

    def __init__(
        self,
        reference_element: ReferenceTetrahedron,
        reference_operators: ReferenceOperators,
        vertex_coords: np.ndarray,
        cell_to_vertices: np.ndarray,
        smallest_diameter: float,
    ):
        """Initialize mesh geometry from vertex and cell data.

        Args:
            reference_element: Reference tetrahedron with nodes and faces.
            reference_operators: Precomputed reference operators.
            vertex_coords: (num_vertices, 3) vertex coordinates.
            cell_to_vertices: (num_cells, 4) vertex indices per cell.
            smallest_diameter: Smallest cell inscribed diameter.
        """
        self._element = reference_element
        self._operators = reference_operators
        self.vertex_coordinates = vertex_coords
        self.cell_to_vertices = cell_to_vertices
        self.smallest_diameter = smallest_diameter

        self.num_vertices = vertex_coords.shape[0]
        self.num_cells = cell_to_vertices.shape[0]

        self.x_vertex = vertex_coords[:, 0]
        self.y_vertex = vertex_coords[:, 1]
        self.z_vertex = vertex_coords[:, 2]

        logger = get_logger(__name__)
        logger.info("Building connectivity matrices")
        self._build_connectivity()

        logger.info("Computing mapped nodal coordinates")
        self._compute_nodal_coordinates()

        logger.info("Computing mapping coefficients")
        self._compute_mapping_coefficients()

        logger.info("Computing face normals")
        self._compute_face_normals()

        logger.info("Computing face node mappings")
        self._compute_face_node_mappings()

        logger.info("Finding boundary nodes")
        self._find_boundary_nodes()

        logger.info("Computing surface-to-volume Jacobian")
        self._compute_surface_to_volume_jacobian()

    def transfer_to_gpu(self) -> None:
        """Transfer all arrays to GPU memory."""
        logger = get_logger(__name__)
        logger.info("Transferring mesh arrays to GPU")

        self.vertex_coordinates = to_gpu(self.vertex_coordinates)
        self.x_vertex = to_gpu(self.x_vertex)
        self.y_vertex = to_gpu(self.y_vertex)
        self.z_vertex = to_gpu(self.z_vertex)
        self.x = to_gpu(self.x)
        self.y = to_gpu(self.y)
        self.z = to_gpu(self.z)

        self.cell_to_vertices = to_gpu(self.cell_to_vertices)
        self.cell_to_cells = to_gpu(self.cell_to_cells)
        self.cell_to_faces = to_gpu(self.cell_to_faces)

        self.jacobians = to_gpu(self.jacobians)
        self.drdx = to_gpu(self.drdx)
        self.drdy = to_gpu(self.drdy)
        self.drdz = to_gpu(self.drdz)
        self.dsdx = to_gpu(self.dsdx)
        self.dsdy = to_gpu(self.dsdy)
        self.dsdz = to_gpu(self.dsdz)
        self.dtdx = to_gpu(self.dtdx)
        self.dtdy = to_gpu(self.dtdy)
        self.dtdz = to_gpu(self.dtdz)

        self.nx = to_gpu(self.nx)
        self.ny = to_gpu(self.ny)
        self.nz = to_gpu(self.nz)
        self.surface_jacobians = to_gpu(self.surface_jacobians)

        self.interior_face_node_indices = to_gpu(self.interior_face_node_indices)
        self.exterior_face_node_indices = to_gpu(self.exterior_face_node_indices)
        self.boundary_face_node_indices = to_gpu(self.boundary_face_node_indices)
        self.boundary_node_indices = to_gpu(self.boundary_node_indices)
        self.surface_to_volume_jacobian = to_gpu(self.surface_to_volume_jacobian)

    def _build_connectivity(self) -> None:
        """Build cell-to-cell and cell-to-face connectivity."""
        num_faces = 4
        k = self.num_cells
        ctv = self.cell_to_vertices

        face_vertices = np.vstack([
            ctv[:, [0, 1, 2]],
            ctv[:, [0, 1, 3]],
            ctv[:, [1, 2, 3]],
            ctv[:, [0, 2, 3]],
        ])
        face_vertices = np.sort(face_vertices, axis=1)

        nv = self.num_vertices
        face_hashes = (
            face_vertices[:, 0] * nv * nv
            + face_vertices[:, 1] * nv
            + face_vertices[:, 2]
            + 1
        )

        vertex_ids = np.arange(num_faces * k)
        ctc = np.tile(np.arange(k)[:, np.newaxis], num_faces)
        ctf = np.tile(np.arange(num_faces), (k, 1))

        mapping_table = np.column_stack([
            face_hashes,
            vertex_ids,
            np.ravel(ctc, order="F"),
            np.ravel(ctf, order="F"),
        ])

        sorted_table = mapping_table[np.lexsort((mapping_table[:, 1], mapping_table[:, 0]))]
        matches = np.where(sorted_table[:-1, 0] == sorted_table[1:, 0])[0]

        match_l = np.vstack([sorted_table[matches], sorted_table[matches + 1]])
        match_r = np.vstack([sorted_table[matches + 1], sorted_table[matches]])

        ctc_flat = np.ravel(ctc, order="F")
        ctf_flat = np.ravel(ctf, order="F")
        ctc_flat[match_l[:, 1].astype(int)] = match_r[:, 2]
        ctf_flat[match_l[:, 1].astype(int)] = match_r[:, 3]

        self.cell_to_cells = ctc_flat.reshape(ctc.shape, order="F")
        self.cell_to_faces = ctf_flat.reshape(ctf.shape, order="F")

    def _compute_nodal_coordinates(self) -> None:
        """Map reference coordinates to physical coordinates."""
        ctv = self.cell_to_vertices
        vx = self.x_vertex.reshape(-1, 1)
        vy = self.y_vertex.reshape(-1, 1)
        vz = self.z_vertex.reshape(-1, 1)
        r = self._element.r
        s = self._element.s
        t = self._element.t

        va, vb, vc, vd = ctv[:, 0], ctv[:, 1], ctv[:, 2], ctv[:, 3]

        self.x = (
            0.5
            * (
                -(1 + r + s + t) * vx[va]
                + (1 + r) * vx[vb]
                + (1 + s) * vx[vc]
                + (1 + t) * vx[vd]
            )
        ).T
        self.y = (
            0.5
            * (
                -(1 + r + s + t) * vy[va]
                + (1 + r) * vy[vb]
                + (1 + s) * vy[vc]
                + (1 + t) * vy[vd]
            )
        ).T
        self.z = (
            0.5
            * (
                -(1 + r + s + t) * vz[va]
                + (1 + r) * vz[vb]
                + (1 + s) * vz[vc]
                + (1 + t) * vz[vd]
            )
        ).T

    def _compute_mapping_coefficients(self) -> None:
        """Compute Jacobian and inverse metric coefficients."""
        dr = self._operators.diff_r
        ds = self._operators.diff_s
        dt = self._operators.diff_t

        xr = dr @ self.x
        xs = ds @ self.x
        xt = dt @ self.x
        yr = dr @ self.y
        ys = ds @ self.y
        yt = dt @ self.y
        zr = dr @ self.z
        zs = ds @ self.z
        zt = dt @ self.z

        j = xr * (ys * zt - zs * yt) - yr * (xs * zt - zs * xt) + zr * (xs * yt - ys * xt)
        self.jacobians = j

        self.drdx = (ys * zt - zs * yt) / j
        self.drdy = -(xs * zt - zs * xt) / j
        self.drdz = (xs * yt - ys * xt) / j
        self.dsdx = -(yr * zt - zr * yt) / j
        self.dsdy = (xr * zt - zr * xt) / j
        self.dsdz = -(xr * yt - yr * xt) / j
        self.dtdx = (yr * zs - zr * ys) / j
        self.dtdy = -(xr * zs - zr * xs) / j
        self.dtdz = (xr * ys - yr * xs) / j

    def _compute_face_normals(self) -> None:
        """Compute outward unit normals and surface Jacobians at face nodes."""
        nfp = self._element.nodes_per_face
        k = self.num_cells
        num_faces = self._element.num_faces
        face_idx = self._element.face_node_indices

        face_drdx = self.drdx[face_idx, :]
        face_dsdx = self.dsdx[face_idx, :]
        face_dtdx = self.dtdx[face_idx, :]
        face_drdy = self.drdy[face_idx, :]
        face_dsdy = self.dsdy[face_idx, :]
        face_dtdy = self.dtdy[face_idx, :]
        face_drdz = self.drdz[face_idx, :]
        face_dsdz = self.dsdz[face_idx, :]
        face_dtdz = self.dtdz[face_idx, :]

        nx = np.zeros((num_faces * nfp, k))
        ny = np.zeros((num_faces * nfp, k))
        nz = np.zeros((num_faces * nfp, k))

        f0 = np.arange(0, nfp)
        f1 = np.arange(nfp, 2 * nfp)
        f2 = np.arange(2 * nfp, 3 * nfp)
        f3 = np.arange(3 * nfp, 4 * nfp)

        # Face 0: t = -1
        nx[f0, :] = -face_dtdx[f0, :]
        ny[f0, :] = -face_dtdy[f0, :]
        nz[f0, :] = -face_dtdz[f0, :]

        # Face 1: s = -1
        nx[f1, :] = -face_dsdx[f1, :]
        ny[f1, :] = -face_dsdy[f1, :]
        nz[f1, :] = -face_dsdz[f1, :]

        # Face 2: r + s + t = -1
        nx[f2, :] = face_drdx[f2, :] + face_dsdx[f2, :] + face_dtdx[f2, :]
        ny[f2, :] = face_drdy[f2, :] + face_dsdy[f2, :] + face_dtdy[f2, :]
        nz[f2, :] = face_drdz[f2, :] + face_dsdz[f2, :] + face_dtdz[f2, :]

        # Face 3: r = -1
        nx[f3, :] = -face_drdx[f3, :]
        ny[f3, :] = -face_drdy[f3, :]
        nz[f3, :] = -face_drdz[f3, :]

        sj = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= sj
        ny /= sj
        nz /= sj
        sj *= self.jacobians[face_idx, :]

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.surface_jacobians = sj

    def _compute_face_node_mappings(self) -> None:
        """Compute mappings between interior and exterior face nodes."""
        np_cell = self._element.nodes_per_cell
        nfp = self._element.nodes_per_face
        num_faces = self._element.num_faces
        tolerance = 1e-6
        ctc = self.cell_to_cells
        ctf = self.cell_to_faces
        k = self.num_cells

        node_ids = np.arange(k * np_cell).reshape(np_cell, k, order="F")
        interior = np.zeros((nfp, num_faces, k), dtype=int)
        exterior = np.zeros((nfp, num_faces, k), dtype=int)

        face_idx = self._element.face_node_indices.reshape(4, -1).T

        for cell in range(k):
            for face in range(num_faces):
                interior[:, face, cell] = node_ids[face_idx[:, face], cell]

        for cell in range(k):
            for face in range(num_faces):
                adj_cell = ctc[cell, face]
                adj_face = ctf[cell, face]

                int_ids = interior[:, face, cell]
                ext_ids = interior[:, adj_face, adj_cell]

                x_int = np.ravel(self.x, order="F")[int_ids][:, None]
                y_int = np.ravel(self.y, order="F")[int_ids][:, None]
                z_int = np.ravel(self.z, order="F")[int_ids][:, None]
                x_ext = np.ravel(self.x, order="F")[ext_ids][:, None]
                y_ext = np.ravel(self.y, order="F")[ext_ids][:, None]
                z_ext = np.ravel(self.z, order="F")[ext_ids][:, None]

                dist2 = (
                    (x_int - x_ext.T) ** 2
                    + (y_int - y_ext.T) ** 2
                    + (z_int - z_ext.T) ** 2
                )

                int_idx, ext_idx = np.where(np.abs(dist2) < tolerance)
                exterior[int_idx, face, cell] = interior[ext_idx, adj_face, adj_cell]

        self.exterior_face_node_indices = exterior.ravel(order="F")
        self.interior_face_node_indices = interior.ravel(order="F")

    def _find_boundary_nodes(self) -> None:
        """Identify nodes on domain boundary."""
        self.boundary_face_node_indices = np.where(
            self.exterior_face_node_indices == self.interior_face_node_indices
        )[0]
        self.boundary_node_indices = self.interior_face_node_indices[
            self.boundary_face_node_indices
        ]

    def _compute_surface_to_volume_jacobian(self) -> None:
        """Compute ratio of surface to volume Jacobian."""
        face_idx = self._element.face_node_indices
        self.surface_to_volume_jacobian = (
            self.surface_jacobians / self.jacobians[face_idx, :]
        )

    def save_to_pickle(self, path: Path) -> None:
        """Save mesh geometry data to pickle file.

        Saves all arrays needed for visualization without the reference
        element/operators (which can be reconstructed from polynomial order).

        Args:
            path: Path to save pickle file.
        """
        from sbimaging.array.backend import to_numpy

        data = {
            "num_vertices": self.num_vertices,
            "num_cells": self.num_cells,
            "smallest_diameter": self.smallest_diameter,
            "vertex_coordinates": to_numpy(self.vertex_coordinates),
            "cell_to_vertices": to_numpy(self.cell_to_vertices),
            "x": to_numpy(self.x),
            "y": to_numpy(self.y),
            "z": to_numpy(self.z),
            "cell_to_cells": to_numpy(self.cell_to_cells),
            "cell_to_faces": to_numpy(self.cell_to_faces),
            "jacobians": to_numpy(self.jacobians),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_from_pickle(cls, path: Path) -> dict:
        """Load mesh geometry data from pickle file.

        Returns a dictionary with mesh data for visualization.
        Does not return a full MeshGeometry object (would need reference element).

        Args:
            path: Path to pickle file.

        Returns:
            Dictionary with mesh data arrays.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


class MeshLoader:
    """Loads meshes from Gmsh files.

    Handles initialization of Gmsh, extraction of vertices and cells,
    and optional material property assignment.
    """

    def __init__(self, mesh_file: Path):
        """Initialize mesh loader.

        Args:
            mesh_file: Path to Gmsh .msh file.

        Raises:
            FileNotFoundError: If mesh file does not exist.
        """
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        self._mesh_file = mesh_file
        self._logger = get_logger(__name__)

        if not gmsh.isInitialized():
            # interruptible=False prevents gmsh from setting up signal handlers,
            # which is required when running in a thread pool (not main thread)
            gmsh.initialize(interruptible=False)

        gmsh.option.setNumber("General.Terminal", 0)
        self._logger.info(f"Loading mesh from {mesh_file}")
        gmsh.open(str(mesh_file))

    def load(
        self,
        reference_element: ReferenceTetrahedron,
        reference_operators: ReferenceOperators,
    ) -> MeshGeometry:
        """Load mesh and create geometry object.

        Args:
            reference_element: Reference tetrahedron.
            reference_operators: Reference operators.

        Returns:
            MeshGeometry object with all computed quantities.
        """
        vertex_coords, cell_to_vertices = self._extract_mesh_data()
        smallest_diameter = self._get_smallest_diameter()

        return MeshGeometry(
            reference_element=reference_element,
            reference_operators=reference_operators,
            vertex_coords=vertex_coords,
            cell_to_vertices=cell_to_vertices,
            smallest_diameter=smallest_diameter,
        )

    def get_material_properties(
        self,
        nodes_per_cell: int,
        num_cells: int,
        default_speed: float = 1.0,
        default_density: float = 1.0,
        inclusion_materials: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract material properties from physical groups.

        Args:
            nodes_per_cell: Number of nodes per cell.
            num_cells: Number of cells.
            default_speed: Default wave speed.
            default_density: Default material density.
            inclusion_materials: Dict mapping physical group name prefixes
                to (speed, density) tuples.

        Returns:
            Tuple of (speed, density) arrays with shape (nodes_per_cell, num_cells).
        """
        model = gmsh.model

        material_map = {"BackgroundMaterial": (default_speed, default_density)}
        if inclusion_materials:
            dim = 3
            for dim, tag in model.getPhysicalGroups(dim):
                name = model.getPhysicalName(dim, tag)
                for prefix, (speed, density) in inclusion_materials.items():
                    if name.startswith(prefix):
                        material_map[name] = (speed, density)

        speed = np.full((nodes_per_cell, num_cells), default_speed)
        density = np.full((nodes_per_cell, num_cells), default_density)

        dim = 3
        for dim, tag in model.getPhysicalGroups(dim):
            name = model.getPhysicalName(dim, tag)
            if name not in material_map:
                continue

            mat_speed, mat_density = material_map[name]
            entities = model.getEntitiesForPhysicalGroup(dim, tag)

            for entity in entities:
                _, elem_tags, _ = model.mesh.getElements(dim, entity)
                offset, _ = gmsh.model.mesh.getElementsByType(4)
                indices = np.array(elem_tags).flatten() - offset[0]
                speed[:, indices] = mat_speed
                density[:, indices] = mat_density

        return speed, density

    def _extract_mesh_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract vertex coordinates and cell connectivity from Gmsh."""
        _, coords, _ = gmsh.model.mesh.getNodes(4)
        vertex_coords = coords.reshape(-1, 3)

        node_tags, _, _ = gmsh.model.mesh.getNodesByElementType(4)
        cell_to_vertices = node_tags.reshape(-1, 4).astype(int) - 1

        self._logger.info(
            f"Loaded mesh: {vertex_coords.shape[0]} vertices, "
            f"{cell_to_vertices.shape[0]} cells"
        )

        return vertex_coords, cell_to_vertices

    def _get_smallest_diameter(self) -> float:
        """Get smallest cell inscribed diameter."""
        _, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
        radii = gmsh.model.mesh.getElementQualities(elem_tags[0], "innerRadius")
        return float(np.min(radii) * 2)

    def close(self) -> None:
        """Finalize Gmsh."""
        gmsh.finalize()
