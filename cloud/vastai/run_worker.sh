#!/bin/bash
# Vast.ai worker script
#
# This script runs inside a Vast.ai instance.
# Upload your batch data, then run this script.
#
# Usage:
#   1. Upload batch data to the instance
#   2. Run: bash run_worker.sh <batch_dir> [worker_id] [total_workers]
#
# Example:
#   bash run_worker.sh /workspace/test_cloud_batch 0 1

set -e

BATCH_DIR=${1:?Usage: $0 <batch_dir> [worker_id] [total_workers]}
WORKER_ID=${2:-0}
TOTAL_WORKERS=${3:-1}

echo "=== Vast.ai Simulation Worker ==="
echo "Batch dir: $BATCH_DIR"
echo "Worker: $WORKER_ID / $TOTAL_WORKERS"

# Install dependencies if needed
if ! python -c "import sbimaging" 2>/dev/null; then
    echo "Installing sbimaging..."
    cd /workspace/packages/sbimaging
    pip install -e . -q
fi

# Ensure gmsh and toml are installed
pip install gmsh toml tomli -q

# Run the worker
cd /workspace
python -c "
import sys
sys.path.insert(0, '/workspace/packages/sbimaging/src')

from pathlib import Path
from sbimaging.batch.planner import BatchPlanner
from sbimaging.batch.executor import BatchExecutor
from sbimaging.logging import configure_logging
import logging

configure_logging(level=logging.INFO, log_to_file=False)
logger = logging.getLogger(__name__)

batch_dir = Path('$BATCH_DIR')
worker_id = $WORKER_ID
total_workers = $TOTAL_WORKERS

# Plan
planner = BatchPlanner(batch_dir)
planner.compute_mesh_hashes()
all_pending = planner.find_pending_simulations()

if not all_pending:
    print('No pending simulations')
    sys.exit(0)

# Get this worker's subset
my_pending = [h for i, h in enumerate(all_pending) if i % total_workers == worker_id]
print(f'Total pending: {len(all_pending)}')
print(f'This worker: {len(my_pending)} simulations')

if not my_pending:
    print('No simulations for this worker')
    sys.exit(0)

# Load or generate metadata
if not planner.load_metadata():
    print('Generating meshes...')
    import tomli
    config_path = batch_dir / 'batch_config.toml'
    geometry_type = 'ellipsoid'
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = tomli.load(f)
        geometry_type = config.get('batch', {}).get('inclusion_type', 'ellipsoid')

    from sbimaging.meshing import GeometryType
    planner.generate_missing_meshes(geometry_type=GeometryType(geometry_type))
    global_dt = planner.compute_global_timestep()
    planner.save_metadata()
else:
    global_dt = planner.global_dt

print(f'Global dt: {global_dt:.6e}')

# Run
def mesh_resolver(sim_hash):
    mesh_hash = planner.get_mesh_hash_for_simulation(sim_hash)
    if mesh_hash:
        return planner.get_mesh_file(mesh_hash)
    return None

def progress(remaining, completed, failed):
    total = completed + failed + remaining
    print(f'Progress: {completed}/{total} completed, {failed} failed')

executor = BatchExecutor(batch_dir)
completed, failed = executor.run_all(
    pending=my_pending,
    global_dt=global_dt,
    mesh_file_resolver=mesh_resolver,
    progress_callback=progress,
)

print(f'Done: {completed} completed, {failed} failed')
"

echo "=== Worker complete ==="
