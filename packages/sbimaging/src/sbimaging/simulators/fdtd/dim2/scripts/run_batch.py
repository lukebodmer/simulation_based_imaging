#!/usr/bin/env python
"""Run a batch of 2D FDTD simulations for training inverse models.

Usage:
    python -m sbimaging.simulators.fdtd.dim2.scripts.run_batch [OPTIONS]

Options:
    --batch-dir PATH    Directory for batch data (default: /data/2d-simulations)
    --num-simulations N Number of simulations to run (default: 500)
    --seed N            Random seed for reproducibility
    --no-resume         Don't skip completed simulations
"""

import argparse
from pathlib import Path

from sbimaging.simulators.fdtd.dim2.batch import run_2d_batch


def main():
    parser = argparse.ArgumentParser(
        description="Run a batch of 2D FDTD simulations"
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("/data/2d-simulations"),
        help="Directory for batch data (default: /data/2d-simulations)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=500,
        help="Number of simulations to run (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip completed simulations",
    )

    args = parser.parse_args()

    completed, failed = run_2d_batch(
        batch_dir=args.batch_dir,
        num_simulations=args.num_simulations,
        seed=args.seed,
        resume=not args.no_resume,
    )

    print(f"\nBatch complete: {completed} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
