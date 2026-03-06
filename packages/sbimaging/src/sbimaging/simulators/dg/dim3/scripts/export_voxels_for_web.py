#!/usr/bin/env python
"""Export voxel data for web-based 3D visualization.

Converts k-space predictions and ground truth to voxel grids and exports
as compressed JSON for use with Three.js/React Three Fiber.
"""

import argparse
import gzip
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

# Default paths
DEFAULT_DATA_DIR = Path("/data/simulations")
DEFAULT_OUTPUT_DIR = Path("/home/lj/projects/simulation_based_imaging/frontend/public/voxels")


def to_numpy(array):
    """Convert CuPy array to NumPy if needed."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def kspace_to_voxels(data: np.ndarray) -> np.ndarray | None:
    """Convert k-space data to real-space voxels.

    Args:
        data: Flat array of k-space coefficients [real_part, imag_part].

    Returns:
        3D voxel array or None if conversion fails.
    """
    try:
        data = to_numpy(data)

        if data.ndim == 1:
            n_half = len(data) // 2
            cube_root = round(n_half ** (1 / 3))

            if cube_root**3 == n_half:
                real_part = data[:n_half].reshape((cube_root,) * 3)
                imag_part = data[n_half:].reshape((cube_root,) * 3)
                kspace = real_part + 1j * imag_part
            else:
                print(f"Cannot reshape {n_half} to cubic grid")
                return None
        else:
            kspace = data

        # Convert to real-space image via inverse FFT
        voxel_data = np.fft.ifftn(np.fft.ifftshift(kspace))
        voxel_data = np.real(voxel_data)
        return voxel_data.astype(np.float32)

    except Exception as e:
        print(f"Failed to convert k-space to voxels: {e}")
        return None


def normalize_voxels(voxels: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalize voxel data to 0-1 range and return original min/max.

    Args:
        voxels: 3D voxel array.

    Returns:
        Tuple of (normalized_voxels, original_min, original_max).
    """
    vmin = float(np.min(voxels))
    vmax = float(np.max(voxels))

    if vmax - vmin > 1e-10:
        normalized = (voxels - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(voxels)

    return normalized, vmin, vmax


def export_sample(
    sim_hash: str,
    ground_truth_path: Path,
    prediction_path: Path,
    output_path: Path,
    sample_index: int,
    prediction_scale: float = 1.0,
) -> dict | None:
    """Export a single ground truth / prediction pair.

    Args:
        sim_hash: Simulation hash identifier.
        ground_truth_path: Path to model_output.pkl.
        prediction_path: Path to prediction .pkl file.
        output_path: Output directory for JSON files.
        sample_index: Index of this sample (0-4).
        prediction_scale: Scale factor to boost prediction values before normalization.

    Returns:
        Metadata dict or None if export fails.
    """
    # Load ground truth
    if not ground_truth_path.exists():
        print(f"  Ground truth not found: {ground_truth_path}")
        return None

    with open(ground_truth_path, "rb") as f:
        ground_truth_kspace = pickle.load(f)

    # Load prediction
    if not prediction_path.exists():
        print(f"  Prediction not found: {prediction_path}")
        return None

    with open(prediction_path, "rb") as f:
        prediction_kspace = pickle.load(f)

    # Convert to voxels
    gt_voxels = kspace_to_voxels(ground_truth_kspace)
    pred_voxels = kspace_to_voxels(prediction_kspace)

    if gt_voxels is None or pred_voxels is None:
        print(f"  Failed to convert k-space to voxels")
        return None

    # Apply scale factor to prediction
    if prediction_scale != 1.0:
        pred_voxels = pred_voxels * prediction_scale

    # Compute shared normalization from ground truth
    gt_norm, vmin, vmax = normalize_voxels(gt_voxels)

    # Apply same normalization to prediction
    if vmax - vmin > 1e-10:
        pred_norm = (pred_voxels - vmin) / (vmax - vmin)
        pred_norm = np.clip(pred_norm, 0, 1)
    else:
        pred_norm = np.zeros_like(pred_voxels)

    # Get grid dimensions
    grid_size = gt_voxels.shape[0]

    # Create output data structure
    # Quantize to uint8 to reduce file size (256 levels is enough for visualization)
    gt_quantized = (gt_norm * 255).astype(np.uint8)
    pred_quantized = (pred_norm * 255).astype(np.uint8)

    output_data = {
        "simHash": sim_hash,
        "gridSize": grid_size,
        "valueRange": {"min": vmin, "max": vmax},
        "groundTruth": gt_quantized.flatten().tolist(),
        "prediction": pred_quantized.flatten().tolist(),
    }

    # Write compressed JSON
    output_file = output_path / f"sample_{sample_index:02d}.json.gz"
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f, separators=(",", ":"))

    file_size_kb = output_file.stat().st_size / 1024
    print(f"  Exported {output_file.name} ({file_size_kb:.1f} KB)")

    return {
        "index": sample_index,
        "simHash": sim_hash,
        "gridSize": grid_size,
        "valueRange": {"min": vmin, "max": vmax},
        "file": f"sample_{sample_index:02d}.json.gz",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export voxel data for web 3D visualization"
    )
    parser.add_argument(
        "batch_name",
        type=str,
        help="Name of the simulation batch (e.g., 'multi_cube_500')",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to export (default: 5)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model subfolder name within predictions/ (e.g., 'larger cnn')",
    )
    parser.add_argument(
        "-s",
        "--prediction-scale",
        type=float,
        default=1.0,
        help="Scale factor to boost prediction values (default: 1.0)",
    )

    args = parser.parse_args()

    batch_dir = Path(args.data_dir) / args.batch_name
    output_dir = Path(args.output_dir)

    # Validate batch directory
    if not batch_dir.exists():
        print(f"Error: Batch directory not found: {batch_dir}")
        return 1

    predictions_dir = batch_dir / "predictions"
    simulations_dir = batch_dir / "simulations"

    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        return 1

    # If model is specified, use that subfolder
    if args.model:
        predictions_dir = predictions_dir / args.model
        if not predictions_dir.exists():
            print(f"Error: Model predictions not found: {predictions_dir}")
            print("Available models:")
            for subdir in sorted((batch_dir / "predictions").iterdir()):
                if subdir.is_dir():
                    print(f"  - {subdir.name}")
            return 1

    # Clear and recreate output directory
    if output_dir.exists():
        print(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find prediction files
    prediction_files = sorted(predictions_dir.glob("*.pkl"))

    if len(prediction_files) == 0:
        print(f"Error: No prediction files found in {predictions_dir}")
        return 1

    print(f"Found {len(prediction_files)} predictions in {batch_dir.name}")
    print(f"Exporting {args.num_samples} samples to {output_dir}")

    # Export samples
    metadata = []
    exported_count = 0

    for i, pred_file in enumerate(prediction_files):
        if exported_count >= args.num_samples:
            break

        sim_hash = pred_file.stem
        ground_truth_path = simulations_dir / sim_hash / "model_output.pkl"

        print(f"Processing {sim_hash}...")

        result = export_sample(
            sim_hash=sim_hash,
            ground_truth_path=ground_truth_path,
            prediction_path=pred_file,
            output_path=output_dir,
            sample_index=exported_count,
            prediction_scale=args.prediction_scale,
        )

        if result is not None:
            metadata.append(result)
            exported_count += 1

    if exported_count == 0:
        print("Error: No samples were exported successfully")
        return 1

    # Write metadata file
    metadata_file = output_dir / "metadata.json"
    metadata_content = {
        "batchName": args.batch_name,
        "numSamples": len(metadata),
        "samples": metadata,
    }
    if args.model:
        metadata_content["modelName"] = args.model
    with open(metadata_file, "w") as f:
        json.dump(metadata_content, f, indent=2)

    print(f"\nExported {exported_count} samples")
    print(f"Metadata written to {metadata_file}")

    return 0


if __name__ == "__main__":
    exit(main())
