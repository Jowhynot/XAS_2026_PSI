from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_plot_style() -> None:
    """Configure LaTeX-style plotting close to the existing concentration figure."""
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.facecolor": "#EAEAEA",
            "figure.facecolor": "#EAEAEA",
        }
    )


def load_mcr_component(path: Path, component_index: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=1)
    energy = data[:, 0]
    component_col = component_index + 1
    if component_col >= data.shape[1]:
        raise ValueError(
            f"Component index {component_index} is out of bounds for file {path.name}. "
            f"Available component indices: 0..{data.shape[1] - 2}"
        )
    mu = data[:, component_col]
    return energy, mu


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=1)
    return data[:, 0], data[:, 1]


def plot_overlay(
    out_path: Path,
    ref_path: Path,
    mcr_path: Path,
    component_index: int,
    energy_min: float,
    energy_max: float,
    mu_min: float,
    mu_max: float,
) -> None:
    e_ref, mu_ref = load_reference(ref_path)
    e_mcr, mu_mcr = load_mcr_component(mcr_path, component_index)

    mask_ref = (e_ref >= energy_min) & (e_ref <= energy_max)
    mask_mcr = (e_mcr >= energy_min) & (e_mcr <= energy_max)

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=150)
    ax.plot(e_ref[mask_ref], mu_ref[mask_ref], lw=1.4, label=ref_path.name)
    ax.plot(e_mcr[mask_mcr], mu_mcr[mask_mcr], lw=1.4, label=mcr_path.name)

    ax.set_xlim(energy_min, energy_max)
    ax.set_ylim(mu_min, mu_max)
    ax.set_xlabel(r"$E\,/\,\mathrm{eV}$")
    ax.set_ylabel(r"$\mu\,/\,-$")
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot zoomed TPO/TPOR XANES overlays for MCR components and references."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory containing the TPO files and reference files.",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="TPO",
        choices=["TPO", "TPOR", "TPR"],
        help="Dataset prefix used inside 'TPO files' for *_mcr_spectra_3.dat and outputs.",
    )
    parser.add_argument(
        "--component-index",
        type=int,
        default=0,
        help="MCR component index for PdO/Pd0 overlays (0-based).",
    )
    parser.add_argument("--energy-min", type=float, default=24380.0)
    parser.add_argument("--energy-max", type=float, default=25200.0)
    parser.add_argument("--mu-min", type=float, default=0.8)
    parser.add_argument("--mu-max", type=float, default=1.08)
    args = parser.parse_args()

    configure_plot_style()

    base = args.base_dir
    mcr_path = base / "TPO files" / f"{args.dataset_prefix}_mcr_spectra_3.dat"

    out_1 = base / "TPO files" / f"{args.dataset_prefix}_fitted_PdO_latex_zoom.png"
    ref_1 = base / "PdO Reference_PdK_mu_timeseries_nAverage300.dat"
    plot_overlay(
        out_path=out_1,
        ref_path=ref_1,
        mcr_path=mcr_path,
        component_index=args.component_index,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
    )

    out_2 = base / "TPO files" / f"{args.dataset_prefix}_fitted_Pd0_latex_zoom.png"
    ref_2 = base / "PdO Reference_PdK_mu_ref_timeseries_nAverage300.dat"
    plot_overlay(
        out_path=out_2,
        ref_path=ref_2,
        mcr_path=mcr_path,
        component_index=args.component_index,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
    )

    out_3 = base / "TPO files" / f"{args.dataset_prefix}_fitted_3rd_comp_latex_zoom.png"
    ref_3 = base / "Intial H2 5wt% Pd_Al2O3_PdK_mu_ref_timeseries_nAverage300.dat"
    plot_overlay(
        out_path=out_3,
        ref_path=ref_3,
        mcr_path=mcr_path,
        component_index=1,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
    )

    print(f"Saved: {out_1}")
    print(f"Saved: {out_2}")
    print(f"Saved: {out_3}")


if __name__ == "__main__":
    main()
