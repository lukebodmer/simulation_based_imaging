#!/usr/bin/env python
"""Train GP and NN models at multiple noise levels and generate comparison table.

Trains 3 GP models and 3 NN models (0%, 5%, 10% noise) and outputs
a summary table suitable for dissertation inclusion.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sbimaging import configure_logging
from sbimaging.inverse_models.base import train_test_split_by_index
from sbimaging.inverse_models.dim1 import (
    DataLoader1D,
    GaussianProcess1D,
    NeuralNetwork1D,
    params_to_density_profile,
)
from sbimaging.inverse_models.dim1.train import add_gaussian_noise
from sbimaging.logging import get_logger


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    confidence: float = 0.90,
) -> dict:
    """Evaluate a model on test data.

    Args:
        model: Trained GP or NN model.
        X_test: Test input features.
        y_test: Test ground truth outputs.
        confidence: Confidence level for GP intervals.

    Returns:
        Dictionary with MSE, MAE, and optionally coverage/width for GP.
    """
    is_gp = isinstance(model, GaussianProcess1D)

    if is_gp:
        pred_result = model.predict_with_uncertainty(X_test, confidence=confidence)
        pred_mean = pred_result["mean"]
        pred_lower = pred_result["lower"]
        pred_upper = pred_result["upper"]
    else:
        pred_mean = model.predict(X_test)
        pred_lower = None
        pred_upper = None

    mse = float(np.mean((pred_mean - y_test) ** 2))
    mae = float(np.mean(np.abs(pred_mean - y_test)))

    result = {"mse": mse, "mae": mae}

    if pred_lower is not None and pred_upper is not None:
        in_interval = (y_test >= pred_lower) & (y_test <= pred_upper)
        result["coverage"] = float(np.mean(in_interval))
        result["ci_width"] = float(np.mean(pred_upper - pred_lower))

    return result


def train_all_models(
    batch_dir: Path,
    output_dir: Path,
    grid_size: int = 100,
    test_fraction: float = 0.1,
    trim_timesteps: int = 500,
    downsample_factor: int = 4,
    nn_epochs: int = 500,
    noise_levels: list[float] | None = None,
) -> dict:
    """Train GP and NN models at multiple noise levels.

    Args:
        batch_dir: Path to batch simulation directory.
        output_dir: Directory to save trained models.
        grid_size: Density profile resolution.
        test_fraction: Fraction of data for testing.
        trim_timesteps: Initial timesteps to trim.
        downsample_factor: Sensor data downsample factor.
        nn_epochs: Number of NN training epochs.
        noise_levels: List of noise percentages to test.

    Returns:
        Dictionary with all results.
    """
    logger = get_logger(__name__)

    if noise_levels is None:
        noise_levels = [0.0, 5.0, 10.0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    logger.info(f"Loading data from {batch_dir}")
    loader = DataLoader1D(
        batch_dir=batch_dir,
        grid_size=grid_size,
        trim_timesteps=trim_timesteps,
        downsample_factor=downsample_factor,
    )
    X_clean, y, sample_ids = loader.load()
    logger.info(f"Loaded {len(sample_ids)} samples, input shape: {X_clean.shape}")

    # Split indices (use same split for all models)
    n_samples = len(sample_ids)
    n_test = int(n_samples * test_fraction)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train_clean = X_clean[train_indices]
    X_test_clean = X_clean[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    logger.info(f"Train: {len(train_indices)}, Test: {len(test_indices)}")

    results = {}

    for noise_pct in noise_levels:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training models with {noise_pct}% noise")
        logger.info(f"{'='*60}")

        # Add noise to training data
        if noise_pct > 0:
            rng_noise = np.random.default_rng(int(noise_pct * 100))
            X_train = add_gaussian_noise(X_train_clean, noise_pct, rng_noise)
            # Test data also gets noise (simulating noisy measurements)
            X_test = add_gaussian_noise(X_test_clean, noise_pct, rng_noise)
        else:
            X_train = X_train_clean.copy()
            X_test = X_test_clean.copy()

        noise_key = f"{int(noise_pct)}pct"

        # Train GP model
        logger.info(f"Training GP model ({noise_pct}% noise)...")
        gp_model = GaussianProcess1D(
            name=f"gp_{noise_key}",
            isotropic=True,
            nugget_est=True,
            num_initial_values=10,
        )

        # GP uses its own train method
        gp_model._model = None
        gp_model.train_indices = [sample_ids[i] for i in train_indices]
        gp_model.test_indices = [sample_ids[i] for i in test_indices]

        # Remove constant columns and train
        y_train_filtered, gp_model._mask, gp_model._constant_values = (
            gp_model._remove_constant_columns(y_train)
        )

        task = gp_model._rgasp.create_task(
            X_train,
            y_train_filtered,
            isotropic=True,
            num_initial_values=10,
            nugget_est=True,
        )
        gp_model._model = gp_model._rgasp.train_ppgasp(task)

        gp_path = output_dir / f"gp_model_{noise_key}.pkl"
        gp_model.save(gp_path)
        logger.info(f"GP model saved to {gp_path}")

        # Evaluate GP
        gp_metrics = evaluate_model(gp_model, X_test, y_test)
        logger.info(f"GP MSE: {gp_metrics['mse']:.6e}, MAE: {gp_metrics['mae']:.6e}")
        logger.info(
            f"GP Coverage: {gp_metrics['coverage']:.2%}, CI Width: {gp_metrics['ci_width']:.4f}"
        )

        # Train NN model
        logger.info(f"Training NN model ({noise_pct}% noise)...")
        nn_model = NeuralNetwork1D(name=f"nn_{noise_key}", large=False)
        nn_model._input_dim = X_train.shape[1]
        nn_model._output_dim = y_train.shape[1]

        nn_model.train(
            X=np.vstack([X_train, X_test]),
            y=np.vstack([y_train, y_test]),
            test_fraction=test_fraction,
            epochs=nn_epochs,
            batch_size=16,
            learning_rate=1e-4,
            sample_ids=[sample_ids[i] for i in train_indices]
            + [sample_ids[i] for i in test_indices],
        )

        nn_path = output_dir / f"nn_model_{noise_key}.pkl"
        nn_model.save(nn_path)
        logger.info(f"NN model saved to {nn_path}")

        # Evaluate NN
        nn_metrics = evaluate_model(nn_model, X_test, y_test)
        logger.info(f"NN MSE: {nn_metrics['mse']:.6e}, MAE: {nn_metrics['mae']:.6e}")

        results[noise_pct] = {
            "gp": gp_metrics,
            "nn": nn_metrics,
        }

    return results


def format_latex_table(results: dict) -> str:
    """Format results as a LaTeX table.

    Args:
        results: Dictionary with results by noise level.

    Returns:
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of Gaussian Process and Neural Network inverse models at different noise levels. Coverage indicates the fraction of true density values falling within the 90\% credible interval.}",
        r"\label{tab:1d-model-comparison}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Noise & Model & MSE & MAE & 90\% CI Coverage & CI Width \\",
        r"\midrule",
    ]

    for noise_pct in sorted(results.keys()):
        gp = results[noise_pct]["gp"]
        nn = results[noise_pct]["nn"]

        noise_str = f"{int(noise_pct)}\\%"

        # GP row
        lines.append(
            f"{noise_str} & GP & {gp['mse']:.3e} & {gp['mae']:.3e} & "
            f"{gp['coverage']*100:.1f}\\% & {gp['ci_width']:.3f} \\\\"
        )

        # NN row
        lines.append(
            f" & NN & {nn['mse']:.3e} & {nn['mae']:.3e} & --- & --- \\\\"
        )

        # Add midrule between noise levels (except after last)
        if noise_pct != max(results.keys()):
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def format_org_table(results: dict) -> str:
    """Format results as an org-mode table.

    Args:
        results: Dictionary with results by noise level.

    Returns:
        Org-mode table string.
    """
    lines = [
        "#+CAPTION: Comparison of Gaussian Process and Neural Network inverse models at different noise levels. Coverage indicates the fraction of true density values falling within the 90% credible interval.",
        "#+NAME: tab:1d-model-comparison",
        "#+ATTR_LATEX: :environment longtable :align llcccc :booktabs t",
        "| Noise | Model | MSE | MAE | 90% CI Coverage | CI Width |",
        "|-------+-------+-----+-----+-----------------+----------|",
    ]

    for noise_pct in sorted(results.keys()):
        gp = results[noise_pct]["gp"]
        nn = results[noise_pct]["nn"]

        noise_str = f"{int(noise_pct)}%"

        # GP row
        lines.append(
            f"| {noise_str} | GP | {gp['mse']:.3e} | {gp['mae']:.3e} | "
            f"{gp['coverage']*100:.1f}% | {gp['ci_width']:.3f} |"
        )

        # NN row
        lines.append(
            f"| | NN | {nn['mse']:.3e} | {nn['mae']:.3e} | --- | --- |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Train GP and NN models at multiple noise levels"
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="/data/1d-simulations",
        help="Path to batch simulation directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/noise_comparison",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Density profile resolution",
    )
    parser.add_argument(
        "--nn-epochs",
        type=int,
        default=500,
        help="Number of NN training epochs",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--table-file",
        type=str,
        default=None,
        help="Path to save formatted table (org-mode)",
    )

    args = parser.parse_args()

    configure_logging()
    logger = get_logger(__name__)

    results = train_all_models(
        batch_dir=Path(args.batch_dir),
        output_dir=Path(args.output_dir),
        grid_size=args.grid_size,
        nn_epochs=args.nn_epochs,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- Org-mode Table ---\n")
    org_table = format_org_table(results)
    print(org_table)

    print("\n--- LaTeX Table ---\n")
    latex_table = format_latex_table(results)
    print(latex_table)

    # Save results
    if args.results_file:
        results_path = Path(args.results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    if args.table_file:
        table_path = Path(args.table_file)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(org_table)
        logger.info(f"Table saved to {table_path}")


if __name__ == "__main__":
    main()
