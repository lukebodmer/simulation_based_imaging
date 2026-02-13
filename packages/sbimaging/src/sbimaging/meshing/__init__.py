"""Mesh generation utilities.

Provides Gmsh-based mesh generation for 3D wave simulations
with support for various inclusion geometries.
"""

from sbimaging.meshing.generator import (
    GeometryType,
    MeshGenerator,
    MeshGeneratorConfig,
    generate_mesh_from_config,
)

__all__ = [
    "GeometryType",
    "MeshGenerator",
    "MeshGeneratorConfig",
    "generate_mesh_from_config",
]
