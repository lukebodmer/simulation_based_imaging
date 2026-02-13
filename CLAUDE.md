# Claude Code Guidelines

## Project Overview

This is a simulation-based imaging platform with a hybrid architecture:

- `packages/sbimaging/` - Python library for numerical solvers and inverse models
- `backend/` - Django REST API (serves precomputed data for public site)
- `frontend/` - React + Vite + TypeScript (public-facing educational website)
- `control_panel/` - Trame apps (local control panel for running experiments)

**Hybrid Model**: React frontend is for public showcase (hostable remotely). Trame control panel is for local experimentation (requires GPU, direct sbimaging access).

## Development Environment

Use Nix for development. The `flake.nix` provides all dependencies and makes `sbimaging` available in editable mode automatically.

```bash
nix develop
```

Do not use `pip install` when in the nix shell - dependencies are managed through `flake.nix`.

## Code Style

### Python

**Formatter/Linter**: Ruff (configured in `ruff.toml`)
```bash
ruff check --fix .
ruff format .
```

**Type Checking**: pyright
```bash
pyright
```

**Key rules**:
- Line length: 88 characters
- Double quotes for strings
- Imports sorted with isort rules (first-party: `sbimaging`)

### TypeScript

- ESLint + Prettier
- Strict mode enabled

## Code Principles

1. **Small, focused functions** - Each function does one thing with a descriptive name
2. **No nested function definitions** - Define functions at module level
3. **Object-oriented design** - Classes with clear, single responsibilities (Unix philosophy)
4. **No utility grab-bags** - Never create generic "utils" modules; create purpose-specific classes instead
5. **No code duplication** - Extract shared logic into well-named classes/functions
6. **Minimal dependencies** - Add dependencies only when clearly needed

## Project Structure Conventions

### sbimaging Package

```
packages/sbimaging/src/sbimaging/
├── simulators/
│   ├── fdtd/          # Method first
│   │   ├── dim1/      # Dimension as detail
│   │   └── dim2/
│   └── dg/
│       └── dim3/
├── inverse_models/
│   ├── gp/            # Gaussian processes
│   └── nn/            # Neural networks
├── meshing/
└── array/             # GPU/CPU backend abstraction
```

### GPU/CPU Backend

Always use the array backend for GPU-compatible code:
```python
from sbimaging.array import xp  # cupy if available, else numpy
from sbimaging.array.backend import to_numpy, to_gpu, GPU_AVAILABLE
```

### Logging

Use the unified logging system for all output:
```python
from sbimaging import get_logger

logger = get_logger(__name__)
logger.info("Simulation started")
logger.debug("Detailed debug info")
logger.error("Something went wrong")
```

Configure at application startup:
```python
from sbimaging import configure_logging
import logging

configure_logging(
    log_dir="data/logs",      # Default location
    level=logging.DEBUG,      # Set verbosity
    log_to_file=True,         # Write to file
)
```

Logs go to both terminal and `data/logs/sbi_YYYYMMDD_HHMMSS.log`.

### Frontend

- Components in `src/components/`
- Pages in `src/pages/`
- Zustand stores in `src/stores/`
- CSS Modules in `src/styles/` (colocated with components is also acceptable)

## Running Tests

```bash
pytest packages/sbimaging/tests -v
```

## Starting Development Servers

```bash
python start_server.py                        # All servers
python start_server.py --no-frontend          # Django only
python start_server.py --trame result_viewer  # Include control panel app
```

## File Naming

- Python: `snake_case.py`
- TypeScript/React: `PascalCase.tsx` for components, `camelCase.ts` for utilities
- CSS Modules: `ComponentName.module.css`

## Adding Dependencies

- Python: Add to `flake.nix` under `buildInputs` or `propagatedBuildInputs`
- Frontend: Use `npm install` in the `frontend/` directory

## Data Storage

Simulation data is stored on `/data/sbi/` (12TB HDD), symlinked to `data/` in the project root. Training inverse models requires running hundreds or thousands of simulations.

- `data/simulations/` - Raw simulation outputs
- `data/precomputed/` - Processed results for web display
- `data/meshes/` - Generated meshes

Always use the `data/` symlink in code, never hardcode `/data/sbi/`.
