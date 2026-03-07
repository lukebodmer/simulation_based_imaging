#!/usr/bin/env python
"""Visualize MLP and CNN neural network architectures for 3D DG inverse models.

Uses PlotNeuralNet (https://github.com/HarisIqbal88/PlotNeuralNet) to create
publication-quality 3D architecture diagrams.

Usage:
    python visualize_nn_architecture.py [--output OUTPUT_DIR]

Requirements:
    - pdflatex (TeX Live)
    - pdf2svg or pdftoppm for conversion
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add plotnn to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "plotnn"))

from pycore.tikzeng import (
    to_begin,
    to_connection,
    to_Conv,
    to_end,
    to_generate,
    to_head,
    to_Pool,
    to_SoftMax,
)

# Re-export for use by other modules
__all__ = [
    "generate_mlp_arch",
    "generate_cnn_arch",
    "generate_cnn2d_arch",
    "compile_tex",
    "to_generate",
]


def to_cor_custom():
    """Custom color definitions for our architectures."""
    return r"""
\def\ConvColor{rgb:blue,5;white,3}
\def\ConvReluColor{rgb:blue,5;red,3;white,2}
\def\PoolColor{rgb:cyan,3;white,2}
\def\FcColor{rgb:green,5;white,3}
\def\FcReluColor{rgb:green,5;red,2;white,2}
\def\InputColor{rgb:yellow,5;white,3}
\def\OutputColor{rgb:orange,5;white,3}
\def\ResColor{rgb:magenta,4;white,3}
\def\NormColor{rgb:gray,3;white,5}
"""


def to_Fc(
    name,
    n_units=512,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=20,
    depth=2,
    caption=" ",
    fill=r"\FcColor",
):
    """Fully connected layer block."""
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + r""",
        zlabel="""
        + str(n_units)
        + """,
        fill="""
        + fill
        + """,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_text(name, text, offset="(0,0,0)", to="(0,0,0)"):
    """Add text annotation."""
    return (
        r"""
\node[shift={"""
        + offset
        + """}] at """
        + to
        + """ ("""
        + name
        + """) {"""
        + text
        + """};
"""
    )


def generate_mlp_arch(
    input_dim: int = 86400,
    output_dim: int = 4096,
    hidden_layers: list[int] | None = None,
):
    """Generate MLP architecture specification."""
    if hidden_layers is None:
        hidden_layers = [4096, 2048, 1024]

    arch = [
        to_head(str(SCRIPT_DIR / "plotnn")),
        to_cor_custom(),
        to_begin(),
        # Input layer
        to_Fc(
            "input",
            n_units=input_dim,
            offset="(0,0,0)",
            to="(0,0,0)",
            width=1,
            height=35,
            depth=2,
            caption="Input",
            fill=r"\InputColor",
        ),
    ]

    prev_name = "input"
    for i, units in enumerate(hidden_layers):
        name = f"fc{i + 1}"
        # Scale height based on units (log scale)
        import math

        height = max(8, int(20 * math.log10(units) / math.log10(4096)))

        arch.append(
            to_Fc(
                name,
                n_units=units,
                offset="(2,0,0)",
                to=f"({prev_name}-east)",
                width=2,
                height=height,
                depth=2,
                caption=f"FC{i + 1}\\\\BN+ReLU",
                fill=r"\FcColor",
            )
        )
        arch.append(to_connection(prev_name, name))
        prev_name = name

    # Output layer
    arch.append(
        to_Fc(
            "output",
            n_units=output_dim,
            offset="(2,0,0)",
            to=f"({prev_name}-east)",
            width=1,
            height=25,
            depth=2,
            caption="Output",
            fill=r"\OutputColor",
        )
    )
    arch.append(to_connection(prev_name, "output"))
    arch.append(to_end())

    return arch


def generate_cnn_arch(
    input_dim: int = 86400,
    output_dim: int = 4096,
    conv_channels: list[int] | None = None,
    pool_size: int = 16,
    regressor_hidden: int = 512,
    use_residual: bool = True,
):
    """Generate CNN architecture specification."""
    if conv_channels is None:
        conv_channels = [32, 64]

    arch = [
        to_head(str(SCRIPT_DIR / "plotnn")),
        to_cor_custom(),
        to_begin(),
        # Input (1D signal)
        to_Conv(
            "input",
            s_filer=input_dim,
            n_filer=1,
            offset="(0,0,0)",
            to="(0,0,0)",
            width=1,
            height=40,
            depth=3,
            caption="Input\\\\1×" + str(input_dim),
        ),
    ]

    prev_name = "input"
    seq_len = input_dim

    for i, ch in enumerate(conv_channels):
        seq_len = (seq_len + 1) // 2  # stride 2

        # Conv layer
        conv_name = f"conv{i + 1}"
        arch.append(
            to_Conv(
                conv_name,
                s_filer=seq_len,
                n_filer=ch,
                offset="(2,0,0)",
                to=f"({prev_name}-east)",
                width=max(2, ch // 16),
                height=max(10, 40 - i * 8),
                depth=max(5, 20 - i * 4),
                caption=f"Conv{i + 1}",
            )
        )
        arch.append(to_connection(prev_name, conv_name))
        prev_name = conv_name

        # Residual block (optional)
        if use_residual:
            res_name = f"res{i + 1}"
            arch.append(
                to_Conv(
                    res_name,
                    s_filer=seq_len,
                    n_filer=ch,
                    offset="(1,0,0)",
                    to=f"({conv_name}-east)",
                    width=max(2, ch // 16),
                    height=max(10, 40 - i * 8),
                    depth=max(5, 20 - i * 4),
                    caption=f"Res{i + 1}",
                )
            )
            arch.append(to_connection(conv_name, res_name))
            prev_name = res_name

    # Adaptive pool
    arch.append(
        to_Pool(
            "pool",
            offset="(2,0,0)",
            to=f"({prev_name}-east)",
            width=2,
            height=15,
            depth=10,
            caption=f"AdaptPool\\\\{pool_size}",
        )
    )
    arch.append(to_connection(prev_name, "pool"))

    # FC regressor
    arch.append(
        to_Fc(
            "fc1",
            n_units=regressor_hidden,
            offset="(2,0,0)",
            to="(pool-east)",
            width=3,
            height=18,
            depth=2,
            caption="FC\\\\LN+GELU",
            fill=r"\FcColor",
        )
    )
    arch.append(to_connection("pool", "fc1"))

    # Output
    arch.append(
        to_Fc(
            "output",
            n_units=output_dim,
            offset="(2,0,0)",
            to="(fc1-east)",
            width=2,
            height=25,
            depth=2,
            caption="Output",
            fill=r"\OutputColor",
        )
    )
    arch.append(to_connection("fc1", "output"))
    arch.append(to_end())

    return arch


def generate_cnn2d_arch(
    num_sensors: int = 600,
    timesteps: int = 144,
    output_dim: int = 4096,
    conv_channels: list[int] | None = None,
    pool_size: tuple[int, int] = (4, 8),
    regressor_hidden: int = 512,
    use_residual: bool = True,
):
    """Generate 2D CNN architecture specification."""
    if conv_channels is None:
        conv_channels = [32, 64]

    arch = [
        to_head(str(SCRIPT_DIR / "plotnn")),
        to_cor_custom(),
        to_begin(),
        # Input (2D: sensors × time)
        to_Conv(
            "input",
            s_filer=timesteps,
            n_filer=1,
            offset="(0,0,0)",
            to="(0,0,0)",
            width=1,
            height=35,
            depth=25,
            caption=f"Input\\\\{num_sensors}×{timesteps}",
        ),
    ]

    prev_name = "input"
    h, w = num_sensors, timesteps

    for i, ch in enumerate(conv_channels):
        h = (h + 1) // 2  # stride 2
        w = (w + 1) // 2

        # Conv layer
        conv_name = f"conv{i + 1}"
        arch.append(
            to_Conv(
                conv_name,
                s_filer=w,
                n_filer=ch,
                offset="(2,0,0)",
                to=f"({prev_name}-east)",
                width=max(2, ch // 16),
                height=max(10, 35 - i * 8),
                depth=max(8, 25 - i * 5),
                caption=f"Conv2D{i + 1}",
            )
        )
        arch.append(to_connection(prev_name, conv_name))
        prev_name = conv_name

        # Residual block (optional)
        if use_residual:
            res_name = f"res{i + 1}"
            arch.append(
                to_Conv(
                    res_name,
                    s_filer=w,
                    n_filer=ch,
                    offset="(1,0,0)",
                    to=f"({conv_name}-east)",
                    width=max(2, ch // 16),
                    height=max(10, 35 - i * 8),
                    depth=max(8, 25 - i * 5),
                    caption=f"Res2D{i + 1}",
                )
            )
            arch.append(to_connection(conv_name, res_name))
            prev_name = res_name

    # Adaptive pool
    arch.append(
        to_Pool(
            "pool",
            offset="(2,0,0)",
            to=f"({prev_name}-east)",
            width=2,
            height=12,
            depth=15,
            caption=f"AdaptPool\\\\{pool_size[0]}×{pool_size[1]}",
        )
    )
    arch.append(to_connection(prev_name, "pool"))

    # FC regressor
    arch.append(
        to_Fc(
            "fc1",
            n_units=regressor_hidden,
            offset="(2,0,0)",
            to="(pool-east)",
            width=3,
            height=18,
            depth=2,
            caption="FC\\\\LN+GELU",
            fill=r"\FcColor",
        )
    )
    arch.append(to_connection("pool", "fc1"))

    # Output
    arch.append(
        to_Fc(
            "output",
            n_units=output_dim,
            offset="(2,0,0)",
            to="(fc1-east)",
            width=2,
            height=25,
            depth=2,
            caption="Output",
            fill=r"\OutputColor",
        )
    )
    arch.append(to_connection("fc1", "output"))
    arch.append(to_end())

    return arch


def compile_tex(tex_path: Path, output_dir: Path):
    """Compile .tex to PDF and convert to PNG."""
    # Compile to PDF
    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory",
            str(output_dir),
            str(tex_path),
        ],
        capture_output=True,
        cwd=output_dir,
    )

    pdf_path = output_dir / tex_path.with_suffix(".pdf").name
    if not pdf_path.exists():
        print(f"LaTeX compilation failed for {tex_path.name}")
        stdout = (
            result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        )
        print(stdout[-2000:] if len(stdout) > 2000 else stdout)
        return None

    # Convert to PNG using pdftocairo (from poppler-utils)
    png_base = output_dir / tex_path.stem
    result = subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-r",
            "300",
            "-singlefile",
            str(pdf_path),
            str(png_base),
        ],
        capture_output=True,
    )

    final_png = png_base.with_suffix(".png")
    if final_png.exists():
        return final_png

    # Fallback: try ImageMagick (requires ghostscript)
    subprocess.run(
        ["magick", "-density", "300", str(pdf_path), "-quality", "95", str(final_png)],
        capture_output=True,
    )

    return final_png if final_png.exists() else pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NN architectures using PlotNeuralNet"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument(
        "--input-dim", type=int, default=86400, help="Input dimension (for 1D CNN/MLP)"
    )
    parser.add_argument("--output-dim", type=int, default=4096, help="Output dimension")
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[4096, 2048, 1024],
        help="MLP hidden layers",
    )
    parser.add_argument(
        "--conv-channels",
        type=int,
        nargs="+",
        default=[32, 64],
        help="CNN conv channels",
    )
    parser.add_argument(
        "--pool-size", type=int, default=16, help="1D CNN adaptive pool size"
    )
    parser.add_argument(
        "--regressor-hidden", type=int, default=512, help="CNN regressor hidden size"
    )
    parser.add_argument(
        "--num-sensors", type=int, default=600, help="Number of sensors (for 2D CNN)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=144, help="Timesteps per sensor (for 2D CNN)"
    )
    parser.add_argument(
        "--pool-2d", type=int, nargs=2, default=[4, 8], help="2D CNN pool size (H W)"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Generate MLP
    print("Generating MLP architecture...")
    mlp_arch = generate_mlp_arch(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_layers=args.hidden_layers,
    )
    mlp_tex = args.output / "mlp_architecture.tex"
    to_generate(mlp_arch, str(mlp_tex))
    mlp_out = compile_tex(mlp_tex, args.output)
    if mlp_out:
        print(f"Saved: {mlp_out}")

    # Generate 1D CNN
    print("Generating 1D CNN architecture...")
    cnn_arch = generate_cnn_arch(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        conv_channels=args.conv_channels,
        pool_size=args.pool_size,
        regressor_hidden=args.regressor_hidden,
    )
    cnn_tex = args.output / "cnn1d_architecture.tex"
    to_generate(cnn_arch, str(cnn_tex))
    cnn_out = compile_tex(cnn_tex, args.output)
    if cnn_out:
        print(f"Saved: {cnn_out}")

    # Generate 2D CNN
    print("Generating 2D CNN architecture...")
    cnn2d_arch = generate_cnn2d_arch(
        num_sensors=args.num_sensors,
        timesteps=args.timesteps,
        output_dim=args.output_dim,
        conv_channels=args.conv_channels,
        pool_size=tuple(args.pool_2d),
        regressor_hidden=args.regressor_hidden,
    )
    cnn2d_tex = args.output / "cnn2d_architecture.tex"
    to_generate(cnn2d_arch, str(cnn2d_tex))
    cnn2d_out = compile_tex(cnn2d_tex, args.output)
    if cnn2d_out:
        print(f"Saved: {cnn2d_out}")

    # Cleanup aux files
    for ext in [".aux", ".log"]:
        for f in args.output.glob(f"*{ext}"):
            f.unlink()

    print("Done!")


if __name__ == "__main__":
    main()
