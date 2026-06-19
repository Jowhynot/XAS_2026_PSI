from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_plot_style() -> None:
	"""Use true LaTeX rendering with Computer Modern font."""
	use_tex = shutil.which("latex") is not None
	if not use_tex:
		raise RuntimeError(
			"LaTeX executable not found. Install a TeX distribution (e.g. texlive) "
			"to render with the exact LaTeX font."
		)

	plt.rcParams.update(
		{
			"text.usetex": True,
			"font.family": "serif",
			"font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman"],
			"text.latex.preamble": "",
			"axes.labelsize": 11,
			"xtick.labelsize": 10,
			"ytick.labelsize": 10,
			"legend.fontsize": 9,
		}
	)


def load_concentration_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	data = np.loadtxt(path, skiprows=1)
	spectra = data[:, 0]
	components = data[:, 1:4]
	lof = data[:, 4]
	return spectra, components, lof


def find_switch_index(components: np.ndarray) -> int:
	# Detect the TPO->TPR boundary from the largest point-to-point component jump.
	jump_sizes = np.linalg.norm(np.diff(components, axis=0), axis=1)
	return int(np.argmax(jump_sizes) + 1)


def build_temperature_axis_piecewise(
	spectra: np.ndarray,
	switch_idx: int,
	start_temp_c: float,
	end_temp_c: float,
) -> tuple[np.ndarray, float, float]:
	"""Build piecewise linear T axis: heat up to switch, then cool down."""
	if switch_idx <= 0 or switch_idx >= len(spectra):
		raise ValueError("Switch index must split the dataset into two non-empty segments.")

	heat_count = switch_idx
	cool_count = len(spectra) - switch_idx

	t_heat = np.linspace(start_temp_c, end_temp_c, heat_count)
	t_cool = np.linspace(end_temp_c, start_temp_c, cool_count)
	temperatures = np.concatenate([t_heat, t_cool])

	delta_t = abs(end_temp_c - start_temp_c)
	total_segment_seconds = (delta_t / 10.0) * 60.0
	sec_per_spectrum_heat = total_segment_seconds / max(heat_count - 1, 1)
	sec_per_spectrum_cool = total_segment_seconds / max(cool_count - 1, 1)

	return temperatures, sec_per_spectrum_heat, sec_per_spectrum_cool


def build_tick_positions_and_labels(
	n_points: int,
	switch_idx: int,
	start_temp_c: float,
	end_temp_c: float,
) -> tuple[np.ndarray, list[str]]:
	"""Create x ticks so displayed temperature runs 30 -> 350 -> 30."""
	pos = [0]
	labels = [f"{start_temp_c:.0f}"]

	if switch_idx > 0:
		pos.extend([int(round(0.25 * switch_idx)), int(round(0.5 * switch_idx)), int(round(0.75 * switch_idx))])
		labels.extend(
			[
				f"{(start_temp_c + 0.25 * (end_temp_c - start_temp_c)):.0f}",
				f"{(start_temp_c + 0.50 * (end_temp_c - start_temp_c)):.0f}",
				f"{(start_temp_c + 0.75 * (end_temp_c - start_temp_c)):.0f}",
			]
		)

	pos.append(switch_idx)
	labels.append(f"{end_temp_c:.0f}")

	tail = max(n_points - 1 - switch_idx, 1)
	pos.extend(
		[
			switch_idx + int(round(0.25 * tail)),
			switch_idx + int(round(0.5 * tail)),
			switch_idx + int(round(0.75 * tail)),
			n_points - 1,
		]
	)
	labels.extend(
		[
			f"{(end_temp_c - 0.25 * (end_temp_c - start_temp_c)):.0f}",
			f"{(end_temp_c - 0.50 * (end_temp_c - start_temp_c)):.0f}",
			f"{(end_temp_c - 0.75 * (end_temp_c - start_temp_c)):.0f}",
			f"{start_temp_c:.0f}",
		]
	)

	# Remove duplicates and keep order.
	seen: set[int] = set()
	uniq_pos: list[int] = []
	uniq_labels: list[str] = []
	for p, l in zip(pos, labels):
		if p not in seen:
			seen.add(p)
			uniq_pos.append(p)
			uniq_labels.append(l)

	return np.asarray(uniq_pos), uniq_labels


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot MCR concentration profiles with LaTeX-style formatting."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("TPO files/TPOR_mcr_concetrations_3.dat"),
		help="Path to concentration data file.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("TPO files/TPOR_concentration_profile_latex.pdf"),
		help="Output figure path (PDF/PNG/SVG supported by matplotlib).",
	)
	parser.add_argument(
		"--start-temp-c", type=float, default=30.0, help="Start temperature in degree C."
	)
	parser.add_argument("--end-temp-c", type=float, default=350.0, help="Peak temperature in degree C.")
	args = parser.parse_args()

	configure_plot_style()

	spectra, components, _lof = load_concentration_file(args.input)
	switch_idx = find_switch_index(components)
	temperatures, sec_per_spectrum_heat, sec_per_spectrum_cool = build_temperature_axis_piecewise(
		spectra=spectra,
		switch_idx=switch_idx,
		start_temp_c=args.start_temp_c,
		end_temp_c=args.end_temp_c,
	)
	switch_temp = temperatures[switch_idx]
	x_index = np.arange(len(spectra))

	# A4 with geometry margins L/R=3.5 cm gives text width 14 cm.
	# Figure height is one-third of text height: (29.7 - 2.5 - 2.5)/3 = 8.23 cm.
	fig_w_cm = 14.0
	fig_h_cm = (29.7 - 2.5 - 2.5) / 3.0
	fig, ax = plt.subplots(figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54))

	component_styles = [
		(0, "blue", "PdO"),
		(1, "purple", "Pd"),
		(2, "red", "PdH"),
	]
	for idx, color, name in component_styles:
		ax.plot(x_index, components[:, idx], lw=1.8, color=color, label=name)

	ax.axvline(switch_idx, color="0.5", linestyle="--", lw=1.2, label="TPO/TPR switch")

	ax.set_xlabel(r"$T\,/\,^\circ\mathrm{C}$")
	ax.set_ylabel(r"$x_j\,/\,-$")
	ax.set_xlim(0, len(spectra) - 1)
	ax.set_ylim(0.0, 1.08)

	tick_positions, tick_labels = build_tick_positions_and_labels(
		n_points=len(spectra),
		switch_idx=switch_idx,
		start_temp_c=args.start_temp_c,
		end_temp_c=args.end_temp_c,
	)
	ax.set_xticks(tick_positions)
	ax.set_xticklabels(tick_labels)

	left_mid_x = 0.5 * switch_idx
	right_mid_x = 0.5 * (switch_idx + (len(spectra) - 1))
	ax.text(left_mid_x, 1.035, "TPO", ha="center", va="center")
	ax.text(right_mid_x, 1.035, "TPR", ha="center", va="center")

	handles, labels = ax.get_legend_handles_labels()
	fig.legend(
		handles,
		labels,
		loc="upper center",
		bbox_to_anchor=(0.5, 0.995),
		ncol=4,
		frameon=True,
		fancybox=False,
		edgecolor="black",
		facecolor="white",
	)

	fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
	args.output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(args.output, dpi=300)

	# Also write PNG for quick viewing.
	png_output = args.output.with_suffix(".png")
	fig.savefig(png_output, dpi=300)
	plt.close(fig)

	print(f"Saved: {args.output}")
	print(f"Saved: {png_output}")
	print(f"Detected switch at spectrum #{int(spectra[switch_idx])} (T={switch_temp:.2f} C)")
	print(f"Estimated time/spectrum (heating): {sec_per_spectrum_heat:.3f} s")
	print(f"Estimated time/spectrum (cooling): {sec_per_spectrum_cool:.3f} s")


if __name__ == "__main__":
	main()
