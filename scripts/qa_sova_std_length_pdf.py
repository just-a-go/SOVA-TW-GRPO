#!/usr/bin/env python3
"""Structural and rendered QA for the SOVA-augmented std_length PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from pypdf import PdfReader


PAGE_SIZE = (1728.0, 576.0)
ORIGINAL_LABELS = (
    "GRPO(fixed reward)",
    "GRPO(soft reward)",
    "GRPO(single-choice)",
    "TW-GRPO(fixed reward)",
    "TW-GRPO",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--pdftoppm", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(pdftoppm: Path, pdf: Path, prefix: Path) -> Path:
    subprocess.run(
        [str(pdftoppm), "-png", "-r", "300", "-singlefile", str(pdf), str(prefix)],
        check=True,
    )
    return prefix.with_suffix(".png")


def main() -> None:
    args = parse_args()
    args.render_dir.mkdir(parents=True, exist_ok=True)
    source_doc = fitz.open(args.source)
    final_doc = fitz.open(args.final)
    if len(source_doc) != 1 or len(final_doc) != 1:
        raise AssertionError("Both PDFs must have exactly one page")
    if tuple(source_doc[0].rect)[2:] != PAGE_SIZE:
        raise AssertionError(f"Source page size changed from expected {PAGE_SIZE}")
    if tuple(final_doc[0].rect)[2:] != PAGE_SIZE:
        raise AssertionError(f"Final page size is not {PAGE_SIZE}")

    source_text = source_doc[0].get_text()
    final_text = final_doc[0].get_text()
    source_lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    final_lines = [line.strip() for line in final_text.splitlines() if line.strip()]
    for label in ORIGINAL_LABELS:
        if source_lines.count(label) != 2:
            raise AssertionError(f"Unexpected source count for {label!r}")
        if final_lines.count(label) != 2:
            raise AssertionError(f"Final count changed for {label!r}")
    if final_lines.count("SOVA-TW-GRPO") != 2:
        raise AssertionError("SOVA-TW-GRPO must appear exactly once per panel legend")

    if source_doc[0].get_images(full=True):
        raise AssertionError("Source unexpectedly contains raster images")
    if final_doc[0].get_images(full=True):
        raise AssertionError("Final PDF must remain all-vector")

    source_png = render(args.pdftoppm, args.source, args.render_dir / "source_300dpi")
    final_png = render(args.pdftoppm, args.final, args.render_dir / "final_300dpi")
    source_image = Image.open(source_png)
    final_image = Image.open(final_png)
    if source_image.size != final_image.size or source_image.size != (7200, 2400):
        raise AssertionError(f"Unexpected render sizes: {source_image.size}, {final_image.size}")

    # The only permitted visual changes are inside the two plot rectangles
    # (which also contain the legends). Labels, ticks, panel letters, margins,
    # and all other page content must remain pixel-identical.
    source_pixels = np.asarray(source_image.convert("RGB"))
    final_pixels = np.asarray(final_image.convert("RGB"))
    outside = np.ones(source_pixels.shape[:2], dtype=bool)
    scale = 300.0 / 72.0
    for x, y, w, h in ((83.0, 10.8, 748.915, 469.5264), (945.4025, 10.8, 748.915, 469.5264)):
        x0 = max(0, int(math.floor(x * scale)) - 2)
        x1 = min(outside.shape[1], int(math.ceil((x + w) * scale)) + 2)
        y0 = max(0, int(math.floor(y * scale)) - 2)
        y1 = min(outside.shape[0], int(math.ceil((y + h) * scale)) + 2)
        outside[y0:y1, x0:x1] = False
    outside_changed = np.any(source_pixels != final_pixels, axis=2) & outside
    outside_changed_count = int(outside_changed.sum())
    if outside_changed_count:
        raise AssertionError(
            f"Detected {outside_changed_count} changed pixels outside the two plots"
        )

    metrics = json.loads(args.metrics.read_text(encoding="utf-8-sig"))
    series = metrics.get("trainer_series", [])
    if not series:
        raise AssertionError("Exact trainer_series is required")
    by_step = {int(row["step"]): row for row in series if "step" in row}
    missing = [
        step
        for step in range(1, 501)
        if step not in by_step
        or "reward_std" not in by_step[step]
        or "completion_length" not in by_step[step]
    ]
    if missing:
        raise AssertionError(f"Missing exact trainer metrics for steps: {missing[:20]}")

    raw_reward = [float(by_step[step]["reward_std"]) for step in range(1, 501)]
    raw_length = [float(by_step[step]["completion_length"]) for step in range(1, 501)]

    def ema(values: list[float], alpha: float) -> list[float]:
        result = [values[0]]
        for value in values[1:]:
            result.append((1.0 - alpha) * result[-1] + alpha * value)
        return result

    smooth_reward = ema(raw_reward, 0.05)
    smooth_length = ema(raw_length, 0.10)

    # Vector-level checks for the two raw and two EMA SOVA curve paths.
    reader = PdfReader(args.final)
    stream = reader.pages[0].get_contents().get_data().decode("latin-1")
    color_operator = "0.57647059 0.63921569 0.6627451 RG"
    if stream.count(color_operator) < 6:
        raise AssertionError("Expected SOVA curve and legend strokes are missing")
    if "5 w" not in stream:
        raise AssertionError("Expected 5 pt SOVA line width is missing")

    extgstates = reader.pages[0]["/Resources"]["/ExtGState"]
    stroke_alphas = {
        round(float(ref.get_object().get("/CA", 1.0)), 6)
        for ref in extgstates.values()
    }
    if not {0.1, 1.0}.issubset(stroke_alphas):
        raise AssertionError(f"Missing raw/smoothed stroke alpha: {sorted(stroke_alphas)}")

    # Associate the graphics state and every encoded vertex with the expected
    # raw or EMA SOVA trace. Matplotlib may simplify collinear vertices, but
    # each retained vertex must still map exactly to a logged integer step.
    sova_rgb = (147 / 255, 163 / 255, 169 / 255)
    sova_paths = [
        drawing
        for drawing in final_doc[0].get_drawings(extended=True)
        if drawing.get("color")
        and max(abs(drawing["color"][i] - sova_rgb[i]) for i in range(3)) < 1e-4
        and len(drawing.get("items", [])) > 20
    ]
    if len(sova_paths) != 4:
        raise AssertionError(f"Expected four SOVA curve paths, found {len(sova_paths)}")

    path_kinds = set()
    max_step_error = 0.0
    max_y_error_points = 0.0
    for drawing in sova_paths:
        if abs(float(drawing.get("width", 0.0)) - 5.0) > 1e-6:
            raise AssertionError(f"SOVA path has wrong width: {drawing.get('width')}")
        opacity = float(drawing.get("stroke_opacity", -1.0))
        if min(abs(opacity - 0.1), abs(opacity - 1.0)) > 1e-5:
            raise AssertionError(f"SOVA path has wrong opacity: {opacity}")

        is_left = drawing["rect"].x0 < 900
        is_smooth = abs(opacity - 1.0) <= 1e-5
        kind = ("reward" if is_left else "length", "ema" if is_smooth else "raw")
        path_kinds.add(kind)

        axis_x = 83.0 if is_left else 945.4025
        axis_y = 95.6736
        axis_w = 748.915
        axis_h = 469.5264
        ymin, ymax = (
            (-0.044008577, 0.924180114)
            if is_left
            else (50.790625, 386.771875)
        )
        values = (
            smooth_reward
            if kind == ("reward", "ema")
            else raw_reward
            if kind == ("reward", "raw")
            else smooth_length
            if kind == ("length", "ema")
            else raw_length
        )

        vertices = []
        for item in drawing.get("items", []):
            if item[0] == "l":
                vertices.extend((item[1], item[2]))
        unique_vertices = []
        for point in vertices:
            if not unique_vertices or point != unique_vertices[-1]:
                unique_vertices.append(point)
        if len(unique_vertices) < 20:
            raise AssertionError(f"SOVA path is unexpectedly short: {kind}")

        seen_steps = []
        for point in unique_vertices:
            step_float = (point.x - axis_x) / axis_w * 500.0
            step = int(round(step_float))
            step_error = abs(step_float - step)
            max_step_error = max(max_step_error, step_error)
            if step_error > 1e-3 or not 1 <= step <= 500:
                raise AssertionError(
                    f"SOVA {kind} vertex does not map to an integer step: {step_float}"
                )
            expected_y = PAGE_SIZE[1] - (
                axis_y + (values[step - 1] - ymin) / (ymax - ymin) * axis_h
            )
            y_error = abs(point.y - expected_y)
            max_y_error_points = max(max_y_error_points, y_error)
            if y_error > 1e-3:
                raise AssertionError(
                    f"SOVA {kind} vertex at step {step} has y error {y_error} pt"
                )
            seen_steps.append(step)
        if seen_steps[0] != 1 or seen_steps[-1] != 500:
            raise AssertionError(f"SOVA {kind} does not span steps 1..500")

    expected_kinds = {
        ("reward", "raw"),
        ("reward", "ema"),
        ("length", "raw"),
        ("length", "ema"),
    }
    if path_kinds != expected_kinds:
        raise AssertionError(f"Unexpected SOVA path set: {sorted(path_kinds)}")

    report = {
        "source_sha256": sha256(args.source),
        "final_sha256": sha256(args.final),
        "pages": len(final_doc),
        "page_points": list(PAGE_SIZE),
        "render_pixels_300dpi": list(final_image.size),
        "all_vector": True,
        "changed_pixels_outside_plots": outside_changed_count,
        "sova_legend_occurrences": final_lines.count("SOVA-TW-GRPO"),
        "sova_color_operator_occurrences": stream.count(color_operator),
        "stroke_alphas": sorted(stroke_alphas),
        "verified_sova_paths": sorted("/".join(kind) for kind in path_kinds),
        "max_step_coordinate_error": max_step_error,
        "max_y_coordinate_error_points": max_y_error_points,
        "reward_std_ema_alpha": 0.05,
        "completion_length_ema_alpha": 0.10,
        "source_render": str(source_png.resolve()),
        "final_render": str(final_png.resolve()),
    }
    report_path = args.render_dir / "qa_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
