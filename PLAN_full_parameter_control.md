# Plan: Full Simulation Parameter Control in Trame Control Panel

## Problem Statement

The old `parameters.toml` file from `old_wave_map_code/` exposes many configurable parameters that are not available in the current Trame control panel:

### Missing from UI (from old config):
| Category | Parameters |
|----------|-----------|
| **Sources** | number, centers, radii, amplitudes, frequencies |
| **Material (outer)** | outer_wave_speed, outer_density |
| **Mesh** | grid_size, box_size, inclusion_center |
| **Solver** | polynomial_order, number_of_timesteps |
| **Receivers** | sensors_per_face |
| **Output Intervals** | image, data, points, energy |

### Current UI exposes only:
- **Material (inclusion)**: wave_speed, density ranges
- **Geometry**: scaling ranges, rotation, movement, boundary_buffer
- **Cube params**: quantity, width ranges
- **Batch meta**: name, description, num_samples, output directory

## User Decisions

- **Sources**: Full per-source control (each source can have unique frequency, amplitude, radius, center)
- **Timesteps**: Direct count (user specifies exact number_of_timesteps)

## Design Approach

Rather than maintaining separate config files + UI, create a **unified SimulationConfig dataclass** that represents all simulation parameters. The control panel becomes the single source of truth, with presets loading into this unified model.

### Key Design Decisions:

1. **Replace `ParameterSpace` with `SimulationConfig`** - A comprehensive dataclass holding all simulation parameters, not just ranges

2. **Parameters organized into "sweep" vs "fixed"** categories:
   - **Sweep parameters**: Vary per sample (material properties, inclusion geometry)
   - **Fixed parameters**: Same across batch (sources, mesh settings, solver, receivers, outputs)

3. **UI organized into collapsible sections** - Users can expand/collapse sections they care about

4. **Smart defaults** - Copy defaults from `base_parameters.toml` so users only change what matters

## Implementation Plan

### Phase 1: Create Unified Configuration Model

**File: `packages/sbimaging/src/sbimaging/config/simulation.py`** (new)

```python
@dataclass
class SourceConfig:
    """Source configuration with defaults from base_parameters.toml."""
    number: int = 6
    centers: list[list[float]] = field(default_factory=lambda: [...])  # 6 face centers
    radii: list[float] = field(default_factory=lambda: [0.05] * 6)
    amplitudes: list[float] = field(default_factory=lambda: [1.0] * 6)
    frequencies: list[float] = field(default_factory=lambda: [3.0] * 6)

@dataclass
class MaterialConfig:
    """Both inclusion and outer material properties."""
    inclusion_density: ParameterRange
    inclusion_wave_speed: ParameterRange
    outer_density: float = 2.0
    outer_wave_speed: float = 2.0

@dataclass
class MeshConfig:
    """Mesh generation settings."""
    grid_size: float = 0.04
    box_size: float = 1.0
    inclusion_center: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

@dataclass
class SolverConfig:
    """DG solver settings."""
    polynomial_order: int = 1
    number_of_timesteps: int = 10000

@dataclass
class ReceiverConfig:
    """Sensor/receiver configuration."""
    sensors_per_face: int = 25

@dataclass
class OutputConfig:
    """Output interval settings."""
    image: int = 1000
    data: int = 1000
    points: int = 10
    energy: int = 500

@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    sources: SourceConfig
    material: MaterialConfig
    mesh: MeshConfig
    solver: SolverConfig
    receivers: ReceiverConfig
    output: OutputConfig
    # Inclusion geometry (existing)
    inclusion_type: str
    inclusion_scaling: ...
    # etc.

    def to_toml_dict(self) -> dict:
        """Convert to TOML-compatible dictionary."""
        ...
```

### Phase 2: Update Trame UI

**File: `control_panel/apps/simulation_runner/app.py`**

Add new state variables and UI sections:

```python
# Sources (full control - each source individually configurable)
self.state.source_count = 6
self.state.source_centers = [[0.5,0.5,0.0], [0.5,0.5,1.0], ...]  # Per-source
self.state.source_frequencies = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]   # Per-source
self.state.source_amplitudes = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]    # Per-source
self.state.source_radii = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]   # Per-source

# Outer material (fixed, not swept)
self.state.outer_wave_speed = 2.0
self.state.outer_density = 2.0

# Mesh
self.state.grid_size = 0.04
self.state.box_size = 1.0

# Solver
self.state.polynomial_order = 1
self.state.number_of_timesteps = 10000

# Receivers
self.state.sensors_per_face = 25

# Output intervals
self.state.output_image_interval = 1000
self.state.output_data_interval = 1000
self.state.output_points_interval = 10
self.state.output_energy_interval = 500
```

UI will use **v3.VExpansionPanels** for collapsible sections:
- "Batch Settings" (expanded by default)
- "Material Properties" (expanded by default)
- "Inclusion Geometry" (expanded by default)
- "Sources" (collapsed by default) - Full per-source control with data table
- "Mesh Settings" (collapsed by default)
- "Solver" (collapsed by default) - Direct timestep count
- "Receivers" (collapsed by default)
- "Output Intervals" (collapsed by default)

**Sources Section Design:**
Since each source needs individual control, use a data table/list with editable rows:
- Row per source: center (x,y,z), frequency, amplitude, radius
- Add/remove source buttons
- Quick-set button to apply same value to all sources

### Phase 3: Update Parameter Generator

**File: `packages/sbimaging/src/sbimaging/batch/generator.py`**

Modify `ParameterGenerator` to accept `SimulationConfig`:
- Use fixed params directly from config
- Sample only the "sweep" parameters (material ranges, scaling ranges)
- Generate TOML files with all parameters

### Phase 4: Update Preset System

**File: `packages/sbimaging/src/sbimaging/config/preset.py`**

Extend `ConfigPreset` to include all new parameters so presets can set defaults for sources, mesh, solver, etc.

## Files to Modify

| File | Changes |
|------|---------|
| `packages/sbimaging/src/sbimaging/config/simulation.py` | NEW - Unified config model |
| `packages/sbimaging/src/sbimaging/config/__init__.py` | Export new classes |
| `packages/sbimaging/src/sbimaging/batch/generator.py` | Accept SimulationConfig, generate full TOML |
| `control_panel/apps/simulation_runner/app.py` | Add state vars + UI sections |
| `packages/sbimaging/src/sbimaging/config/preset.py` | Extend ConfigPreset |
| `packages/sbimaging/src/sbimaging/config/presets/*.toml` | Update preset files with new sections |

## Verification

1. **Run control panel**: `python start_server.py --trame simulation_runner`
2. **Verify all parameters visible**: Expand each section, confirm all fields present
3. **Test preset loading**: Select a preset, verify all fields populate
4. **Generate batch**: Create a small batch (2-3 samples), inspect generated TOML files
5. **Run simulation**: Execute batch, verify simulation runs with configured parameters
