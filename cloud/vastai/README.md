# Vast.ai Cloud Simulation Guide

Run large batches of simulations on Vast.ai GPU instances.

## Overview

Vast.ai provides affordable GPU rental for running simulation batches. This guide covers renting instances, uploading data, running parallel workers, and downloading results.

## Prerequisites

1. **Vast.ai account**: Sign up at https://vast.ai and add credits
2. **Vast.ai CLI**: Install via Nix flake:
   ```bash
   nix run github:dialohq/vast-cli.nix -- help
   ```
3. **API key**: Get from https://vast.ai/console/account/ and set it:
   ```bash
   nix run github:dialohq/vast-cli.nix -- set api-key YOUR_API_KEY
   ```
4. **SSH key**: Ensure your SSH key is added to your Vast.ai account

## Quick Start

### 1. Generate batch parameter files locally

```bash
cd /path/to/simulation_based_imaging

# Generate parameter files from a preset
python packages/sbimaging/src/sbimaging/cli/batch.py generate \
    --preset multi_cube_1000 \
    --output data/simulations/my_batch

# Create batch_config.toml (required for geometry type detection)
cat > data/simulations/my_batch/batch_config.toml << EOF
[batch]
name = "my_batch"
inclusion_type = "multi_cubes"
EOF
```

### 2. Find and rent a GPU instance

```bash
# Search for available instances (sorted by price)
nix run github:dialohq/vast-cli.nix -- search offers \
    'gpu_name=RTX_3090 num_gpus=4 cuda_vers>=12.0 disk_space>=100' -o 'dph+'

# Rent an instance (use ID from search results)
# IMPORTANT: Use pytorch 2.5.1+ image for Python 3.11 compatibility
nix run github:dialohq/vast-cli.nix -- create instance <INSTANCE_ID> \
    --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
    --disk 100
```

### 3. Check instance status

```bash
nix run github:dialohq/vast-cli.nix -- show instances
```

Note the SSH address and port (e.g., `ssh3.vast.ai:24248`).

### 4. Upload code and data

```bash
# Set variables for convenience
SSH_KEY=~/.ssh/your_key
SSH_HOST=root@ssh3.vast.ai
SSH_PORT=24248

# Create directories on instance
ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST "mkdir -p /workspace/packages /workspace/data/simulations"

# Upload sbimaging package (exclude large/unnecessary directories)
rsync -avz --progress \
    --exclude 'inverse_models/gp/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    -e "ssh -p $SSH_PORT -i $SSH_KEY" \
    packages/sbimaging $SSH_HOST:/workspace/packages/

# Upload worker scripts
scp -r -P $SSH_PORT -i $SSH_KEY cloud $SSH_HOST:/workspace/

# Upload batch data
scp -r -P $SSH_PORT -i $SSH_KEY \
    data/simulations/my_batch $SSH_HOST:/workspace/data/simulations/
```

### 5. Install dependencies on instance

SSH into the instance:
```bash
ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST
```

Install required libraries:
```bash
apt-get update && apt-get install -y \
    libglu1-mesa libgl1-mesa-glx \
    libxrender1 libxcursor1 libxft2 libxinerama1 libfontconfig1
```

### 6. Run simulations

For a 4-GPU instance, run 4 parallel workers:

```bash
cd /workspace

CUDA_VISIBLE_DEVICES=0 nohup bash cloud/vastai/run_worker.sh data/simulations/my_batch 0 4 > worker0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash cloud/vastai/run_worker.sh data/simulations/my_batch 1 4 > worker1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash cloud/vastai/run_worker.sh data/simulations/my_batch 2 4 > worker2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash cloud/vastai/run_worker.sh data/simulations/my_batch 3 4 > worker3.log 2>&1 &
```

The worker arguments are: `<batch_dir> <worker_id> <total_workers>`
- Worker 0 runs simulations 0, 4, 8, 12, ...
- Worker 1 runs simulations 1, 5, 9, 13, ...
- etc.

### 7. Monitor progress

```bash
# Watch all logs
tail -f worker*.log

# Count completed simulations
ls data/simulations/my_batch/simulations/ 2>/dev/null | wc -l
```

### 8. Download results

From your local machine:
```bash
scp -r -P $SSH_PORT -i $SSH_KEY \
    $SSH_HOST:/workspace/data/simulations/my_batch/simulations \
    data/simulations/my_batch/

# Also download meshes and metadata
scp -r -P $SSH_PORT -i $SSH_KEY \
    $SSH_HOST:/workspace/data/simulations/my_batch/meshes \
    data/simulations/my_batch/
scp -P $SSH_PORT -i $SSH_KEY \
    $SSH_HOST:/workspace/data/simulations/my_batch/batch_metadata.toml \
    data/simulations/my_batch/
```

### 9. Destroy instance

```bash
nix run github:dialohq/vast-cli.nix -- destroy instance <INSTANCE_ID>
```

## Cost Estimates

| GPU Config | $/hr | Sims/hr (approx) | Time for 1000 sims |
|------------|------|------------------|-------------------|
| 1x RTX 3090 | $0.15 | ~12 | ~83 hours |
| 4x RTX 3090 | $0.55 | ~48 | ~21 hours |
| 4x RTX 4090 | $1.00 | ~80 | ~12 hours |

Actual speed depends on mesh complexity and timesteps.

## Troubleshooting

### SSH Permission Denied
If SSH fails but SCP worked, try specifying the key explicitly:
```bash
ssh -p $SSH_PORT -i ~/.ssh/your_specific_key root@sshX.vast.ai
```

### Missing Libraries (OSError: libXXX.so not found)
Install the missing library:
```bash
apt-get update && apt-get install -y libXXX1
```

Common ones needed: `libglu1-mesa`, `libgl1-mesa-glx`, `libxrender1`, `libfontconfig1`

### Python Version Mismatch
sbimaging requires Python 3.11+. Use the `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` image or newer.

### Wrong Geometry Type (ellipsoid instead of cubes)
Ensure `batch_config.toml` exists with the correct `inclusion_type`:
```toml
[batch]
inclusion_type = "multi_cubes"
```

## Worker Script Reference

The `run_worker.sh` script:
1. Installs sbimaging if not present
2. Discovers pending simulations
3. Divides work across workers using modulo assignment
4. Generates meshes (first worker generates all, others wait)
5. Runs simulations sequentially on assigned GPU

Each worker uses `CUDA_VISIBLE_DEVICES` to target a specific GPU, allowing parallel execution across multiple GPUs.
