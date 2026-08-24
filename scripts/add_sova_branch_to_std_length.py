#!/usr/bin/env python3
"""Add the logged SOVA-TW-GRPO traces to the existing all-vector PDF.

The source page, axes, original curves, labels, and panel geometry are retained
verbatim. Only the two original five-entry legends are replaced by matching
six-entry legends, and the raw plus EMA-smoothed SOVA traces are overlaid.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
MPL392 = WORKSPACE / "tmp" / "pydeps" / "mpl392"
if MPL392.exists():
    sys.path.insert(0, str(MPL392))

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject


PAGE_W = 1728.0
PAGE_H = 576.0
AXES = (
    (83.0, 95.6736, 748.915, 469.5264),
    (945.4025, 95.6736, 748.915, 469.5264),
)
LEFT_YLIM = (-0.044008577, 0.924180114)
RIGHT_YLIM = (50.790625, 386.771875)
SOVA_COLOR = "#93A3A9"
EMA_ALPHA = 0.05
RIGHT_EMA_ALPHA = 0.10
MAX_PLOTTED_STEP = 500

LABELS = (
    "GRPO(fixed reward)",
    "GRPO(soft reward)",
    "GRPO(single-choice)",
    "TW-GRPO(fixed reward)",
    "TW-GRPO",
    "SOVA-TW-GRPO",
)
COLORS = ("#7DAEE0", "#E9C46A", "#299D8F", "#B395BD", "#EA8379", SOVA_COLOR)
LEGEND_ALPHAS = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--desktop-copy", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def ema(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        result.append((1.0 - alpha) * result[-1] + alpha * value)
    return result


def require_finite(name: str, values: list[float]) -> None:
    if not values:
        raise ValueError(f"{name} is empty")
    bad = [i for i, value in enumerate(values) if not math.isfinite(value)]
    if bad:
        raise ValueError(f"{name} contains non-finite values at {bad[:10]}")


def load_metrics(path: Path) -> tuple[list[int], list[float], list[float], dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    series = payload.get("trainer_series", [])
    by_step = {
        int(row["step"]): row
        for row in series
        if "step" in row and 1 <= int(row["step"]) <= MAX_PLOTTED_STEP
    }
    expected_steps = list(range(1, MAX_PLOTTED_STEP + 1))
    if not series:
        raise ValueError(
            "Exact trainer_series with explicit step, reward_std, and "
            "completion_length fields is required"
        )
    missing = [
        step
        for step in expected_steps
        if step not in by_step
        or "completion_length" not in by_step[step]
        or "reward_std" not in by_step[step]
    ]
    if missing:
        raise ValueError(
            "trainer_series must contain reward_std and completion_length for "
            f"every step 1..{MAX_PLOTTED_STEP}; missing {missing[:20]}"
        )
    steps = expected_steps
    reward = [float(by_step[step]["reward_std"]) for step in steps]
    lengths = [float(by_step[step]["completion_length"]) for step in steps]

    require_finite("reward_std", reward)
    require_finite("completion_length", lengths)

    if min(reward) < LEFT_YLIM[0] or max(reward) > LEFT_YLIM[1]:
        raise ValueError(
            f"SOVA reward_std range {min(reward):.6g}..{max(reward):.6g} "
            f"falls outside frozen left axis {LEFT_YLIM}"
        )
    if min(lengths) < RIGHT_YLIM[0] or max(lengths) > RIGHT_YLIM[1]:
        raise ValueError(
            f"SOVA completion_length range {min(lengths):.6g}..{max(lengths):.6g} "
            f"falls outside frozen right axis {RIGHT_YLIM}"
        )

    return steps, reward, lengths, payload


def figure_axes(fig: plt.Figure):
    for x, y, w, h in AXES:
        ax = fig.add_axes([x / PAGE_W, y / PAGE_H, w / PAGE_W, h / PAGE_H])
        ax.patch.set_alpha(0)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        yield ax


def make_curve_overlay(
    path: Path, steps: list[int], reward: list[float], lengths: list[float]
) -> None:
    fig = plt.figure(figsize=(24, 8))
    fig.patch.set_alpha(0)
    left, right = tuple(figure_axes(fig))

    left.set_xlim(0, MAX_PLOTTED_STEP)
    left.set_ylim(*LEFT_YLIM)
    left.plot(steps, reward, color=SOVA_COLOR, linewidth=5, alpha=0.1)
    left.plot(steps, ema(reward), color=SOVA_COLOR, linewidth=5, alpha=1.0)

    right.set_xlim(0, MAX_PLOTTED_STEP)
    right.set_ylim(*RIGHT_YLIM)
    right.plot(steps, lengths, color=SOVA_COLOR, linewidth=5, alpha=0.1)
    right.plot(
        steps,
        ema(lengths, RIGHT_EMA_ALPHA),
        color=SOVA_COLOR,
        linewidth=5,
        alpha=1.0,
    )

    fig.savefig(path, transparent=True)
    plt.close(fig)


def make_legend_overlay(path: Path) -> None:
    fig = plt.figure(figsize=(24, 8))
    fig.patch.set_alpha(0)
    handles = [
        Line2D([], [], color=color, linewidth=5, alpha=alpha, label=label)
        for label, color, alpha in zip(LABELS, COLORS, LEGEND_ALPHAS)
    ]
    for ax in figure_axes(fig):
        ax.legend(
            handles=handles,
            fontsize=20,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
        )
    fig.savefig(path, transparent=True)
    plt.close(fig)


def strip_original_legends(page) -> None:
    data = page.get_contents().get_data()
    left_start_marker = b"/A6 gs 1 g 0 j 0.8 G 1 g\n\n502.53045"
    right_axes_marker = b"/A1 gs 1 g 0 j 0 w 0 G 1 g\n\n945.4025"
    right_start_marker = b"/A6 gs 1 g 0 j 0.8 G 1 g\n\n1364.93295"

    if data.count(left_start_marker) != 1:
        raise ValueError("Could not uniquely identify the left legend")
    if data.count(right_axes_marker) != 1:
        raise ValueError("Could not uniquely identify the right panel start")
    if data.count(right_start_marker) != 1:
        raise ValueError("Could not uniquely identify the right legend")

    left_start = data.index(left_start_marker)
    right_axes = data.index(right_axes_marker, left_start)
    right_start = data.index(right_start_marker, right_axes)
    stripped = data[:left_start] + data[right_axes:right_start] + b"Q\n"
    stream = DecodedStreamObject()
    stream.set_data(stripped)
    page.replace_contents(stream)


def build_pdf(source: Path, curve_overlay: Path, legend_overlay: Path, output: Path) -> None:
    source_reader = PdfReader(source)
    writer = PdfWriter(clone_from=source)
    writer.add_metadata(dict(source_reader.metadata or {}))
    page = writer.pages[0]
    strip_original_legends(page)
    page.merge_page(PdfReader(curve_overlay).pages[0])
    page.merge_page(PdfReader(legend_overlay).pages[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)


def main() -> None:
    args = parse_args()
    if matplotlib.__version__ != "3.9.2":
        raise RuntimeError(f"Expected Matplotlib 3.9.2, found {matplotlib.__version__}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    steps, reward, lengths, payload = load_metrics(args.metrics)
    curve_overlay = args.work_dir / "sova_curves_overlay.pdf"
    legend_overlay = args.work_dir / "six_entry_legends_overlay.pdf"
    make_curve_overlay(curve_overlay, steps, reward, lengths)
    make_legend_overlay(legend_overlay)
    build_pdf(args.source, curve_overlay, legend_overlay, args.output)
    if args.desktop_copy:
        args.desktop_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.desktop_copy)

    summary = {
        "matplotlib": matplotlib.__version__,
        "source": str(args.source.resolve()),
        "metrics": str(args.metrics.resolve()),
        "output": str(args.output.resolve()),
        "desktop_copy": str(args.desktop_copy.resolve()) if args.desktop_copy else None,
        "plotted_points": len(reward),
        "x_range": [min(steps), max(steps)],
        "reward_std_range": [min(reward), max(reward)],
        "completion_length_range": [min(lengths), max(lengths)],
        "reward_std_ema_alpha": EMA_ALPHA,
        "completion_length_ema_alpha": RIGHT_EMA_ALPHA,
        "debug_step_count": payload.get("debug_step_count"),
        "trainer_row_count": payload.get("trainer_row_count"),
        "trainer_state_source": payload.get("trainer_state_source"),
        "max_abs_reward_std_error": payload.get("max_abs_reward_std_error"),
    }
    (args.work_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
