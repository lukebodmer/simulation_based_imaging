# Plan: Save Mesh Data as Pickle Files

## Goal
Save processed mesh data as `.pkl` files to avoid reprocessing the Gmsh `.msh` file each time we want to visualize a simulation.

## Current State
- Meshes are stored at `/data/simulations/{batch_name}/meshes/{mesh_hash}/mesh.msh`
- Each time visualization loads, it would need to reprocess the `.msh` file through `MeshLoader` and `MeshGeometry`
- The `MeshGeometry` class computes many derived quantities (connectivity, Jacobians, normals, etc.) which is expensive

## Implementation Steps

### 1. Add save/load methods to MeshGeometry

**File:** `packages/sbimaging/src/sbimaging/simulators/dg/dim3/mesh.py`

Add import at top:
```python
import pickle
```

Add these methods to `MeshGeometry` class:

```python
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
```

### 2. Save mesh.pkl during batch execution

**File:** `packages/sbimaging/src/sbimaging/batch/executor.py`

Modify `_run_single()` method to save mesh data after loading. Add after mesh loading (around line 150):

```python
# Save mesh as pickle for visualization
mesh_pkl_path = output_dir / "mesh.pkl"
if not mesh_pkl_path.exists():
    runner._mesh.save_to_pickle(mesh_pkl_path)
```

Or alternatively, modify the runner's `_setup_mesh()` to optionally save.

### 3. Alternative: Save during mesh generation

**File:** `packages/sbimaging/src/sbimaging/batch/planner.py`

In `generate_missing_meshes()`, after generating each mesh, also save a `.pkl` version:

```python
# After line 312 (after generate_mesh_from_config):
# Load mesh and save as pickle for visualization
from sbimaging.simulators.dg.dim3.mesh import MeshLoader
from sbimaging.simulators.dg.dim3.reference_element import (
    ReferenceOperators,
    ReferenceTetrahedron,
)

elem = ReferenceTetrahedron(info.polynomial_order)
ops = ReferenceOperators(elem)
loader = MeshLoader(mesh_path)
mesh_geom = loader.load(elem, ops)
mesh_geom.save_to_pickle(mesh_dir / "mesh.pkl")
loader.close()
```

### 4. Update visualization to find mesh correctly

**File:** `control_panel/apps/simulation_runner/pages/visualize_batch.py`

The `_load_mesh_data()` method currently looks in `batch_metadata.toml` for mesh hash, but this isn't stored there. Fix the lookup:

```python
def _load_mesh_data(self, batch_name: str, sim_hash: str):
    """Load mesh data for the simulation."""
    from sbimaging.batch.planner import BatchPlanner

    batch_dir = DEFAULT_DATA_DIR / batch_name
    planner = BatchPlanner(batch_dir)
    mesh_hash = planner.get_mesh_hash_for_simulation(sim_hash)

    if mesh_hash is None:
        logger.error(f"No mesh hash found for simulation {sim_hash}")
        return

    mesh_file = batch_dir / "meshes" / mesh_hash / "mesh.pkl"
    if mesh_file.exists():
        with open(mesh_file, "rb") as f:
            self.mesh_data = pickle.load(f)
        logger.info(f"Loaded mesh data from {mesh_file}")
    else:
        logger.error(f"Mesh pickle not found: {mesh_file}")
```

### 5. Also save speed/density in mesh pickle

For wave speed visualization, we need material properties. Modify save to include them:

In `SimulationRunner._setup_mesh()` or `BatchExecutor._run_single()`, after getting speed/density:

```python
# Save mesh with material properties
mesh_data = {
    # ... existing mesh data ...
    "speed": to_numpy(self._speed),
    "density": to_numpy(self._density),
    "speed_per_cell": to_numpy(self._speed[0, :]),  # One value per cell for visualization
}
```

## Files to Modify

1. `packages/sbimaging/src/sbimaging/simulators/dg/dim3/mesh.py` - Add save/load methods
2. `packages/sbimaging/src/sbimaging/batch/executor.py` - Save mesh.pkl after running simulation
3. `control_panel/apps/simulation_runner/pages/visualize_batch.py` - Fix mesh lookup logic

## Testing

After implementation:
1. Run a small batch of simulations
2. Verify `mesh.pkl` files appear in the mesh directories
3. Open visualization page and select a simulation
4. Confirm wave speed and simulation views render correctly
