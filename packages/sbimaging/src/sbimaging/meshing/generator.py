"""Mesh generation using Gmsh.

Generates 3D tetrahedral meshes for wave simulation with support
for various inclusion geometries: ellipsoids, spheres, cubes, and
composite geometries.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import gmsh
import numpy as np

from sbimaging.logging import get_logger


class GeometryType(Enum):
    """Types of inclusion geometry."""

    ELLIPSOID = "ellipsoid"
    SPHERE = "sphere"
    MULTI_CUBE = "multi_cubes"
    CUBE_IN_ELLIPSOID = "cube_in_ellipsoid"


@dataclass
class MeshGeneratorConfig:
    """Configuration for mesh generation.

    Attributes:
        output_path: Path to write the generated mesh file.
        grid_size: Target element size.
        box_size: Domain cube side length.
        geometry_type: Type of inclusion geometry.
        source_centers: List of source center coordinates.
        source_radii: List of source radii.
        inclusion_center: Center of inclusion region.
        inclusion_scaling: Scaling factors for inclusion (semi-axes).
        inclusion_orientation: Direction of semi-major axis.
        number_of_cubes: Number of cube inclusions.
        cube_centers: Centers of cube inclusions.
        cube_widths: Widths of cube inclusions.
        grid_variation: Fractional variation in element size.
    """

    output_path: Path
    grid_size: float = 0.04
    box_size: float = 1.0
    geometry_type: GeometryType = GeometryType.ELLIPSOID
    source_centers: list[list[float]] | None = None
    source_radii: list[float] | None = None
    inclusion_center: list[float] | None = None
    inclusion_scaling: list[float] | None = None
    inclusion_orientation: list[float] | None = None
    number_of_cubes: int = 0
    cube_centers: list[list[float]] | None = None
    cube_widths: list[float] | None = None
    grid_variation: float = 0.03

    def __post_init__(self):
        if self.source_centers is None:
            self.source_centers = [[0.5, 0.5, 0.0]]
        if self.source_radii is None:
            self.source_radii = [0.05]
        if self.inclusion_center is None:
            self.inclusion_center = [0.5, 0.5, 0.5]
        if self.inclusion_scaling is None:
            self.inclusion_scaling = [0.2, 0.2, 0.2]
        if self.inclusion_orientation is None:
            self.inclusion_orientation = [1.0, 0.0, 0.0]
        if self.cube_centers is None:
            self.cube_centers = []
        if self.cube_widths is None:
            self.cube_widths = []


class MeshGenerator:
    """Generates 3D tetrahedral meshes using Gmsh.

    Supports multiple geometry types including ellipsoids, spheres,
    cubes, and composite geometries with embedded inclusions.
    """

    def __init__(self, config: MeshGeneratorConfig):
        """Initialize mesh generator.

        Args:
            config: Mesh generation configuration.
        """
        self.config = config
        self._logger = get_logger(__name__)
        self._smallest_diameter: float | None = None

    def generate(self) -> Path:
        """Generate mesh and write to output file.

        Returns:
            Path to the generated mesh file.
        """
        self._logger.info(f"Generating {self.config.geometry_type.value} mesh")

        self._initialize_gmsh()

        if self.config.geometry_type == GeometryType.ELLIPSOID:
            self._generate_ellipsoid()
        elif self.config.geometry_type == GeometryType.SPHERE:
            self._generate_sphere()
        elif self.config.geometry_type == GeometryType.MULTI_CUBE:
            self._generate_multi_cube()
        elif self.config.geometry_type == GeometryType.CUBE_IN_ELLIPSOID:
            self._generate_cube_in_ellipsoid()
        else:
            raise ValueError(f"Unknown geometry type: {self.config.geometry_type}")

        self._finalize_mesh()
        self._compute_smallest_diameter()

        gmsh.finalize()

        self._logger.info(f"Mesh written to {self.config.output_path}")
        return self.config.output_path

    @property
    def smallest_diameter(self) -> float | None:
        """Get smallest element diameter (available after generation)."""
        return self._smallest_diameter

    def _initialize_gmsh(self):
        """Initialize Gmsh with appropriate settings."""
        if gmsh.isInitialized():
            gmsh.finalize()

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        variation = self.config.grid_variation
        min_size = self.config.grid_size * (1 - variation)
        max_size = self.config.grid_size * (1 + variation)

        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)

        gmsh.model.add("geometry")

    def _finalize_mesh(self):
        """Generate and write the mesh."""
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(self.config.output_path))

    def _compute_smallest_diameter(self):
        """Compute smallest element inner radius."""
        _, ele_tags, _ = gmsh.model.mesh.getElements(dim=3)
        if ele_tags and len(ele_tags[0]) > 0:
            radii = gmsh.model.mesh.getElementQualities(ele_tags[0], "innerRadius")
            self._smallest_diameter = float(np.min(radii)) * 2
        else:
            self._smallest_diameter = None

    def _create_domain_box(self) -> int:
        """Create the outer domain box."""
        return gmsh.model.occ.addBox(
            0, 0, 0,
            self.config.box_size,
            self.config.box_size,
            self.config.box_size,
        )

    def _add_source_disks(self) -> list[tuple[int, int]]:
        """Create source disks on boundary faces.

        Returns:
            List of (dim, tag) tuples for the disk entities.
        """
        occ = gmsh.model.occ
        disks = []
        tol = 1e-8
        box_size = self.config.box_size

        for (sx, sy, sz), r in zip(
            self.config.source_centers, self.config.source_radii
        ):
            tag = occ.addDisk(float(sx), float(sy), float(sz), float(r), float(r))

            if abs(sx - 0.0) < tol or abs(sx - box_size) < tol:
                occ.rotate([(2, tag)], sx, sy, sz, 0, 1, 0, np.pi / 2)
            elif abs(sy - 0.0) < tol or abs(sy - box_size) < tol:
                occ.rotate([(2, tag)], sx, sy, sz, 1, 0, 0, -np.pi / 2)

            disks.append((2, tag))

        return disks

    def _create_rotation_matrix(self) -> np.ndarray:
        """Create rotation matrix aligning x-axis with inclusion orientation."""
        v = np.array(self.config.inclusion_orientation, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.eye(3)

        v = v / norm
        x_axis = np.array([1.0, 0.0, 0.0])

        axis = np.cross(x_axis, v)
        angle = np.linalg.norm(axis)

        if angle < 1e-10:
            return np.eye(3)

        axis = axis / angle
        ux, uy, uz = axis
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos = 1 - cos_theta

        return np.array([
            [
                cos_theta + ux**2 * one_minus_cos,
                ux * uy * one_minus_cos - uz * sin_theta,
                ux * uz * one_minus_cos + uy * sin_theta,
            ],
            [
                uy * ux * one_minus_cos + uz * sin_theta,
                cos_theta + uy**2 * one_minus_cos,
                uy * uz * one_minus_cos - ux * sin_theta,
            ],
            [
                uz * ux * one_minus_cos - uy * sin_theta,
                uz * uy * one_minus_cos + ux * sin_theta,
                cos_theta + uz**2 * one_minus_cos,
            ],
        ])

    def _create_affine_transform(self) -> list[float]:
        """Create affine transformation matrix for ellipsoid."""
        a, b, c = self.config.inclusion_scaling
        S = np.diag([a, b, c])
        R = self._create_rotation_matrix()
        A = R @ S

        transform = np.eye(4)
        transform[:3, :3] = A
        transform[:3, 3] = self.config.inclusion_center

        return transform.flatten().tolist()

    def _generate_ellipsoid(self):
        """Generate mesh with ellipsoidal inclusion."""
        occ = gmsh.model.occ
        model = gmsh.model

        domain_box = self._create_domain_box()
        source_disks = self._add_source_disks()

        sphere_tag = occ.addSphere(0, 0, 0, 1.0)
        transform = self._create_affine_transform()
        occ.affineTransform([(3, sphere_tag)], transform)

        out_dim_tags, _ = occ.fragment(
            [(3, domain_box), (3, sphere_tag)], source_disks
        )

        occ.synchronize()
        model.mesh.setSize(model.getEntities(0), self.config.grid_size)

        for dim, tag in out_dim_tags:
            if dim == 3:
                model.addPhysicalGroup(3, [tag], tag)

    def _generate_sphere(self):
        """Generate mesh with spherical inclusion."""
        self.config.inclusion_scaling = [
            self.config.inclusion_scaling[0],
            self.config.inclusion_scaling[0],
            self.config.inclusion_scaling[0],
        ]
        self._generate_ellipsoid()

    def _generate_multi_cube(self):
        """Generate mesh with multiple cube inclusions."""
        occ = gmsh.model.occ
        model = gmsh.model

        domain_box = self._create_domain_box()

        cube_tags = []
        for i in range(self.config.number_of_cubes):
            cx, cy, cz = self.config.cube_centers[i]
            w = self.config.cube_widths[i]

            x0 = cx - w / 2
            y0 = cy - w / 2
            z0 = cz - w / 2
            tag = occ.addBox(x0, y0, z0, w, w, w)
            cube_tags.append((3, tag))

        source_disks = self._add_source_disks()

        solids = [(3, domain_box)] + cube_tags
        out_dim_tags, _ = occ.fragment(solids, source_disks)

        occ.synchronize()

        cube_centers = np.array(self.config.cube_centers)
        tol = 1e-6

        for dim, tag in out_dim_tags:
            if dim == 3:
                xmin, ymin, zmin, xmax, ymax, zmax = occ.getBoundingBox(dim, tag)
                center = np.array([
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                ])

                matched = False
                for i, cc in enumerate(cube_centers):
                    if np.all(np.abs(center - cc) < tol):
                        phys = model.addPhysicalGroup(3, [tag])
                        model.setPhysicalName(3, phys, f"Cube{i}")
                        matched = True
                        break

                if not matched:
                    phys = model.addPhysicalGroup(3, [tag])
                    model.setPhysicalName(3, phys, "BackgroundMaterial")

        model.mesh.setSize(model.getEntities(0), self.config.grid_size)

    def _generate_cube_in_ellipsoid(self):
        """Generate mesh with cube inside ellipsoid inside domain."""
        occ = gmsh.model.occ
        model = gmsh.model

        domain_box = occ.addBox(
            0, 0, 0,
            self.config.box_size,
            self.config.box_size,
            self.config.box_size,
        )

        sphere_tag = occ.addSphere(0, 0, 0, 1.0)
        transform = self._create_affine_transform()
        occ.affineTransform([(3, sphere_tag)], transform)

        cx, cy, cz = self.config.cube_centers[0]
        w = self.config.cube_widths[0]
        x0, y0, z0 = cx - w / 2, cy - w / 2, cz - w / 2
        cube_tag = occ.addBox(x0, y0, z0, w, w, w)

        solids = [(3, domain_box), (3, sphere_tag), (3, cube_tag)]
        out_dim_tags, _ = occ.fragment(solids, [])

        occ.synchronize()

        volumes = []
        for dim, tag in out_dim_tags:
            if dim == 3:
                xmin, ymin, zmin, xmax, ymax, zmax = occ.getBoundingBox(3, tag)
                diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
                volumes.append((tag, diag))

        volumes.sort(key=lambda x: x[1])
        cube_vol_tag, ellipsoid_vol_tag, background_vol_tag = [v[0] for v in volumes]

        pg_cube = model.addPhysicalGroup(3, [cube_vol_tag])
        model.setPhysicalName(3, pg_cube, "Cube")

        pg_ellipsoid = model.addPhysicalGroup(3, [ellipsoid_vol_tag])
        model.setPhysicalName(3, pg_ellipsoid, "Ellipsoid")

        pg_background = model.addPhysicalGroup(3, [background_vol_tag])
        model.setPhysicalName(3, pg_background, "BackgroundMaterial")

        model.mesh.setSize(model.getEntities(0), self.config.grid_size)


def generate_mesh_from_config(
    config: dict,
    output_path: Path,
    geometry_type: GeometryType = GeometryType.ELLIPSOID,
) -> tuple[Path, float]:
    """Generate mesh from simulation configuration dictionary.

    Args:
        config: Simulation configuration dictionary (from TOML).
        output_path: Path to write the mesh file.
        geometry_type: Type of geometry to generate.

    Returns:
        Tuple of (mesh_path, smallest_diameter).
    """
    mesh_cfg = config.get("mesh", {})
    sources_cfg = config.get("sources", {})

    gen_config = MeshGeneratorConfig(
        output_path=output_path,
        grid_size=mesh_cfg.get("grid_size", 0.04),
        box_size=mesh_cfg.get("box_size", 1.0),
        geometry_type=geometry_type,
        source_centers=sources_cfg.get("centers"),
        source_radii=sources_cfg.get("radii"),
        inclusion_center=mesh_cfg.get("inclusion_center"),
        inclusion_scaling=mesh_cfg.get("inclusion_scaling"),
        inclusion_orientation=mesh_cfg.get("inclusion_semi_major_axis_direction"),
        number_of_cubes=mesh_cfg.get("number_of_cubes", 0),
        cube_centers=mesh_cfg.get("cube_centers"),
        cube_widths=mesh_cfg.get("cube_widths"),
    )

    generator = MeshGenerator(gen_config)
    mesh_path = generator.generate()

    return mesh_path, generator.smallest_diameter
