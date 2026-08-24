#!/usr/bin/env python3
"""Convert the MVBench TVQA frame directories into evaluator-ready MP4 files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any


EXPECTED_ROWS = 4000
EXPECTED_TVQA_ROWS = 200
EXPECTED_TVQA_VIDEOS = 166


class PreparationError(RuntimeError):
    """Raised when the TVQA videos cannot be prepared safely."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read {path}: {exc}") from exc

    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise PreparationError(f"{path} must contain a JSON list of objects")
    if len(rows) != EXPECTED_ROWS:
        raise PreparationError(
            f"{path} has {len(rows)} rows, expected {EXPECTED_ROWS}"
        )
    return rows


def required_stems(rows: list[dict[str, Any]]) -> list[str]:
    tvqa_rows = [row for row in rows if "/tvqa/" in str(row.get("video", ""))]
    if len(tvqa_rows) != EXPECTED_TVQA_ROWS:
        raise PreparationError(
            f"found {len(tvqa_rows)} TVQA rows, expected {EXPECTED_TVQA_ROWS}"
        )

    stems: set[str] = set()
    for row in tvqa_rows:
        path_stem = Path(str(row["video"])).stem
        stem = row.get("ori_video") or path_stem
        if (
            not isinstance(stem, str)
            or not stem
            or stem in {".", ".."}
            or Path(stem).name != stem
        ):
            raise PreparationError(f"unsafe TVQA video stem: {stem!r}")
        if stem != path_stem:
            raise PreparationError(
                f"TVQA ori_video/path mismatch: {stem!r} != {path_stem!r}"
            )
        stems.add(stem)

    if len(stems) != EXPECTED_TVQA_VIDEOS:
        raise PreparationError(
            f"found {len(stems)} unique TVQA videos, expected "
            f"{EXPECTED_TVQA_VIDEOS}"
        )
    return sorted(stems)


def frame_sequence(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise PreparationError(f"frame directory is missing: {directory}")

    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        raise PreparationError(f"frame directory is empty: {directory}")

    suffixes = {path.suffix for path in images}
    if len(suffixes) != 1:
        raise PreparationError(f"mixed frame extensions in {directory}: {suffixes}")

    widths = {len(path.stem) for path in images}
    try:
        numbers = sorted(int(path.stem) for path in images)
    except ValueError as exc:
        raise PreparationError(f"non-numeric frame name in {directory}") from exc
    if len(widths) != 1 or numbers != list(range(numbers[0], numbers[-1] + 1)):
        raise PreparationError(f"non-contiguous frame sequence in {directory}")

    width = next(iter(widths))
    suffix = next(iter(suffixes))
    return [directory / f"{number:0{width}d}{suffix}" for number in numbers]


def validate_video(
    path: Path, expected_frames: int, fps: Fraction, av_module: ModuleType
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise PreparationError(f"video is missing or empty: {path}")

    try:
        with av_module.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise PreparationError(f"expected one video stream in {path}")
            stream = streams[0]
            actual_fps = Fraction(stream.average_rate) if stream.average_rate else None
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            actual_frames = sum(1 for _ in container.decode(stream))
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError(f"cannot decode {path}: {exc}") from exc

    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise PreparationError(f"invalid video dimensions for {path}: {width}x{height}")
    if actual_frames != expected_frames:
        raise PreparationError(
            f"{path} has {actual_frames} frames, expected {expected_frames}"
        )
    if actual_fps != fps:
        raise PreparationError(f"{path} has frame rate {actual_fps}, expected {fps}")


def convert_video(
    frame_directory: Path,
    output_path: Path,
    fps: Fraction,
    av_module: ModuleType,
    image_module: ModuleType,
    preset: str,
) -> str:
    frames = frame_sequence(frame_directory)
    frame_count = len(frames)
    if output_path.exists():
        validate_video(output_path, frame_count, fps, av_module)
        return "verified"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.mp4")
    if temporary.exists():
        raise PreparationError(f"temporary output already exists: {temporary}")

    try:
        with image_module.open(frames[0]) as first_image:
            source_size = first_image.size
        output_size = (source_size[0] - source_size[0] % 2, source_size[1] - source_size[1] % 2)
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise PreparationError(f"invalid source dimensions in {frame_directory}")

        with av_module.open(str(temporary), mode="w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = output_size
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": preset}

            for frame_path in frames:
                with image_module.open(frame_path) as image:
                    if image.size != source_size:
                        raise PreparationError(
                            f"inconsistent frame dimensions in {frame_directory}"
                        )
                    image = image.convert("RGB")
                    if image.size != output_size:
                        image = image.resize(output_size, image_module.Resampling.LANCZOS)
                    video_frame = av_module.VideoFrame.from_image(image)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        validate_video(temporary, frame_count, fps, av_module)
        os.replace(temporary, output_path)
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError(f"PyAV failed for {frame_directory}: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return "created"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--fps", type=Fraction, default=Fraction(3, 1))
    parser.add_argument(
        "--preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium"),
        default="ultrafast",
        help="x264 encoding preset (default: ultrafast)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        print("Preparation failed: --fps must be positive", file=sys.stderr)
        return 2

    try:
        import av
        from PIL import Image

        av.codec.Codec("libx264", "w")
    except Exception as exc:
        print(f"Preparation failed: PyAV, Pillow, and libx264 are required: {exc}", file=sys.stderr)
        return 3

    try:
        stems = required_stems(load_rows(args.annotations))
        counts = {"created": 0, "verified": 0}
        for index, stem in enumerate(stems, start=1):
            status = convert_video(
                frame_directory=args.frame_root / stem,
                output_path=args.video_root / f"{stem}.mp4",
                fps=args.fps,
                av_module=av,
                image_module=Image,
                preset=args.preset,
            )
            counts[status] += 1
            if index == 1 or index % 10 == 0 or index == len(stems):
                print(
                    f"TVQA videos: {index}/{len(stems)} "
                    f"(created={counts['created']}, verified={counts['verified']})",
                    flush=True,
                )
    except PreparationError as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 4

    print(
        f"TVQA preparation complete: {len(stems)}/{len(stems)} videos",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
