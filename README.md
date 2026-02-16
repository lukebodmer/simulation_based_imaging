# Simulation Based Imaging

An interactive platform for exploring simulation-based imaging techniques, featuring numerical solvers (FDTD, Discontinuous Galerkin), inverse models (Gaussian Processes, Neural Networks), and web-based visualizations.

## Project Structure

```
simulation_based_imaging/
├── packages/
│   └── sbimaging/                 # Standalone Python package (pip installable)
│       ├── src/
│       │   └── sbimaging/
│       │       ├── simulators/
│       │       │   ├── fdtd/      # Finite-difference time-domain
│       │       │   │   ├── dim1/
│       │       │   │   └── dim2/
│       │       │   └── dg/        # Discontinuous Galerkin
│       │       │       └── dim3/
│       │       ├── inverse_models/
│       │       │   ├── gp/        # Gaussian processes
│       │       │   ├── nn/        # Neural networks (CNN, MLP)
│       │       │   └── dim2/      # 2D-specific k-space utilities
│       │       ├── meshing/       # Mesh generation utilities (gmsh)
│       │       └── array/         # GPU/CPU array backend abstraction
│       ├── tests/
│       ├── pyproject.toml
│       └── README.md
│
├── backend/                       # Django REST API
│   ├── config/                    # Django settings
│   ├── api/                       # API endpoints
│   ├── manage.py
│   └── pyproject.toml
│
├── frontend/                      # React + Vite + TypeScript
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── stores/                # Zustand stores
│   │   ├── styles/                # CSS modules
│   │   └── api/                   # API client
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── control_panel/                 # Trame control panel (simulation management)
│   ├── apps/
│   │   ├── simulation_planner/    # Plan and queue simulations
│   │   ├── simulation_runner/     # Run simulations with progress
│   │   ├── model_trainer/         # Train inverse models
│   │   └── result_viewer/         # View and explore results
│   └── pyproject.toml
│
├── data/                          # Symlink to /data/sbi (git-ignored)
│   ├── simulations/               # Raw simulation outputs
│   ├── precomputed/               # Processed results for web display
│   └── meshes/                    # Generated meshes
│
├── old_wave_map_code/             # Legacy code (reference only, delete when done)
│
├── flake.nix                      # Nix development environment
├── flake.lock
├── pyproject.toml                 # Workspace root (for tooling config)
├── ruff.toml                      # Ruff linter/formatter config
├── start_server.py                # Start all development servers
└── README.md
```

## Architecture

### Python Package: sbimaging

A standalone library for numerical simulations and inverse modeling. Can be used independently of the web interface.

**GPU/CPU Fallback**: Uses CuPy when available, automatically falls back to NumPy on CPU-only systems. Import the array backend:

```python
from sbimaging.array import xp  # xp is cupy or numpy
```

**2D Inverse Models**: Train neural networks to recover inclusion geometry from sensor data:

```python
from sbimaging.inverse_models.dim2 import (
    DataLoader2D,
    train_2d_inverse_model,
    predict_and_visualize,
    inclusion_to_kspace,
    kspace_to_image,
)

# Train a model on batch simulation data
results = train_2d_inverse_model(
    batch_dir="/data/2d-simulations",
    output_path="models/2d_inverse.pkl",
    architecture="mlp",  # or "cnn"
    grid_size=64,
    epochs=500,
)

# Predict and visualize
pred_image, true_image = predict_and_visualize(
    model_path="models/2d_inverse.pkl",
    batch_dir="/data/2d-simulations",
    sim_id="sim_00050",
)
```

The 2D inverse model uses k-space (frequency domain) representation:
1. Sensor data (40 sensors x ~3000 timesteps) is flattened as input
2. Inclusion geometry is converted to a 2D k-space grid (64x64 complex = 8192 coefficients)
3. Neural network learns the mapping: sensor data → k-space coefficients
4. Inverse FFT recovers the real-space inclusion image

**Logging**: Unified logging to both terminal and log files:

```python
from sbimaging import get_logger, configure_logging

configure_logging(level=logging.DEBUG)  # Optional, call once at startup
logger = get_logger(__name__)
logger.info("Simulation started")
```

Logs are written to `data/logs/sbi_YYYYMMDD_HHMMSS.log`.

**Key Dependencies**:
- numpy, scipy (numerical computing)
- cupy (GPU acceleration, optional)
- gmsh (mesh generation)
- pyvista (visualization utilities)

### Backend: Django REST API

Serves precomputed simulation data and static content for the public-facing website.

**Key Dependencies**:
- Django, Django REST Framework
- sbimaging (local package)

### Frontend: React + Vite + TypeScript

Public-facing educational website showcasing simulation-based imaging techniques. Displays precomputed results and interactive learning materials.

**Key Dependencies**:
- React 18, TypeScript
- Vite (build tool)
- Zustand (state management)
- CSS Modules (styling)

### Control Panel: Trame Apps

Trame-based control panel for directly driving the sbimaging library. This is the operational interface for running experiments locally.

**Capabilities**:
- Plan simulation parameters and batch runs
- Execute simulations with real-time progress
- Train inverse models (GP, neural networks)
- Explore and visualize results with PyVista

**Key Dependencies**:
- trame, trame-vuetify, trame-vtk
- pyvista
- sbimaging (local package)

**Architecture Note**: The hybrid model separates concerns:
- **React frontend** = Public showcase (can be hosted remotely)
- **Trame control panel** = Local experimentation (requires GPU, large storage)

## Data Storage

Simulation data is stored on a separate 12TB HDD mounted at `/data/`. Training inverse models requires hundreds or thousands of simulations, each producing substantial output.

```bash
# Create the data directory and symlink
sudo mkdir -p /data/sbi
sudo chown $USER:$USER /data/sbi
ln -s /data/sbi data
```

The `data/` symlink in the project root points to `/data/sbi/` and is git-ignored.

## Development Setup

### Prerequisites

- Nix (recommended) or Python 3.12+ with pip
- Node.js 20+
- CUDA toolkit (optional, for GPU support)
- `/data/` mount point for simulation storage

### Using Nix (Recommended)

```bash
# Enter development shell (sbimaging is automatically available in editable mode)
nix develop

# Install frontend dependencies
cd frontend && npm install
```

### Without Nix

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install packages
pip install -e packages/sbimaging
pip install -e backend
pip install -e visualizations

# Install frontend dependencies
cd frontend && npm install
```

## Running the Project

### Development (All Servers)

```bash
# Start all servers with one command
python start_server.py

# Or with options
python start_server.py --django-port 8000 --frontend-port 5173
python start_server.py --no-frontend  # Django only
python start_server.py --trame result_viewer --trame-port 8080  # Include control panel
```

### Development (Individual Servers)

```bash
# Terminal 1: Django backend
cd backend && python manage.py runserver

# Terminal 2: React frontend
cd frontend && npm run dev

# Terminal 3: Trame control panel
cd control_panel && python -m apps.result_viewer
```

### Running Tests

```bash
# Python tests
pytest packages/sbimaging/tests -v

# Frontend tests
cd frontend && npm test
```

## Code Style

### Python

- **Formatter/Linter**: Ruff (configured in `ruff.toml`)
- **Type Checking**: pyright
- **Style**: PEP 8, with preferences defined in `ruff.toml`

Auto-format on save is configured via LSP. Run manually:

```bash
ruff check --fix .
ruff format .
```

### TypeScript

- **Formatter**: Prettier
- **Linter**: ESLint
- **Style**: Configured in `.eslintrc.cjs` and `.prettierrc`

### Code Principles

1. **Small, focused functions**: Each function does one thing with a descriptive name
2. **No nested function definitions**: Define functions at module level
3. **Object-oriented design**: Classes with clear, single responsibilities (Unix philosophy)
4. **No utility grab-bags**: Avoid generic "utils" modules; create purpose-specific classes
5. **Minimal dependencies**: Add dependencies only when clearly needed
6. **Progressive enhancement**: Start simple, add complexity only when required

## Deployment

### Backend (Django)

Deploy to any Python-compatible server (Digital Ocean, AWS, etc.):

```bash
pip install packages/sbimaging
pip install backend
cd backend && gunicorn config.wsgi:application
```

### Frontend (React)

Build and serve static files:

```bash
cd frontend && npm run build
# Serve dist/ via nginx, CDN, or static hosting
```

### Control Panel (Trame)

The control panel is designed for local use with GPU access. For remote deployment, consider running on a GPU-enabled server with appropriate access controls.

```bash
pip install packages/sbimaging control_panel
python -m apps.result_viewer --server --port 8080
```

## License

MIT
