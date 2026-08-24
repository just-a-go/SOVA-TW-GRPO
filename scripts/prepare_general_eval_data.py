#!/usr/bin/env python3
"""Prepare and validate annotation files for the general video evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SAMPLES = {
    "eval_nextgqa": 5553,
    "eval_mmvu": 625,
    "eval_mvbench": 4000,
    "eval_tempcompass": 7540,
    "eval_videomme": 2700,
}
SUPPORTED_PROBLEM_TYPES = {
    "multiple choice",
    "numerical",
    "OCR",
    "free-form",
    "regression",
}


class PreparationError(RuntimeError):
    """Raised when a dataset cannot be made evaluator-ready."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read {path}: {exc}") from exc

    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise PreparationError(f"{path} must contain a JSON list of objects")
    return rows


def validate_structure(dataset: str, rows: list[dict[str, Any]]) -> None:
    expected = EXPECTED_SAMPLES[dataset]
    if len(rows) != expected:
        raise PreparationError(
            f"{dataset} has {len(rows)} rows, expected {expected}"
        )

    invalid_rows: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row.get("video"), str) or not row["video"]:
            invalid_rows.append(f"row {index}: invalid video")
        if not isinstance(row.get("problem"), str):
            invalid_rows.append(f"row {index}: invalid problem")
        if not isinstance(row.get("solution"), str):
            invalid_rows.append(f"row {index}: invalid solution")

        problem_type = row.get("problem_type")
        if problem_type not in SUPPORTED_PROBLEM_TYPES:
            invalid_rows.append(f"row {index}: unsupported problem_type={problem_type!r}")
        if problem_type == "multiple choice":
            options = row.get("options")
            if not isinstance(options, list) or not options or not all(
                isinstance(option, str) for option in options
            ):
                invalid_rows.append(f"row {index}: invalid multiple-choice options")

    if invalid_rows:
        examples = "\n".join(f"  - {message}" for message in invalid_rows[:8])
        raise PreparationError(
            f"{dataset} has {len(invalid_rows)} evaluator-incompatible fields. "
            f"Examples:\n{examples}"
        )


def rows_have_valid_videos(rows: list[dict[str, Any]]) -> bool:
    return all(Path(str(row["video"])).is_file() for row in rows)


def rows_match_except_video(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right):
        left_fields = {key: value for key, value in left_row.items() if key != "video"}
        right_fields = {key: value for key, value in right_row.items() if key != "video"}
        if left_fields != right_fields:
            return False
    return True


def iter_fallback_files(
    roots: Iterable[Path], wanted_names: set[str]
) -> dict[str, list[Path]]:
    matches: dict[str, list[Path]] = defaultdict(list)
    if not wanted_names:
        return matches

    for root in roots:
        if not root.is_dir():
            continue
        for directory, _, filenames in os.walk(root):
            for filename in wanted_names.intersection(filenames):
                matches[filename].append(Path(directory, filename).resolve())
    return matches


def suffix_score(candidate: Path, original_path: str) -> int:
    original_parts = Path(original_path.lstrip("/")).parts
    candidate_parts = candidate.parts
    score = 0
    for width in range(2, min(len(original_parts), len(candidate_parts)) + 1):
        if candidate_parts[-width:] == original_parts[-width:]:
            score = width
    return score


def choose_fallback(candidates: list[Path], original_path: str) -> Path | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        candidate = candidates[0]
        return candidate if suffix_score(candidate, original_path) >= 2 else None

    scored = [(suffix_score(candidate, original_path), candidate) for candidate in candidates]
    best_score = max(score for score, _ in scored)
    best = [candidate for score, candidate in scored if score == best_score]
    if best_score >= 2 and len(best) == 1:
        return best[0]
    return None


def normalize_video_paths(
    rows: list[dict[str, Any]], video_root: Path, fallback_roots: list[Path]
) -> tuple[list[dict[str, Any]], list[str]]:
    prepared = [dict(row) for row in rows]
    unresolved: list[tuple[int, str]] = []

    for index, row in enumerate(prepared):
        original = str(row["video"])
        original_path = Path(original)
        primary_path = video_root / original.lstrip("/")

        if original_path.is_absolute() and original_path.is_file():
            row["video"] = str(original_path.resolve())
        elif primary_path.is_file():
            row["video"] = str(primary_path.resolve())
        else:
            unresolved.append((index, original))

    wanted_names = {Path(original).name for _, original in unresolved}
    fallback_matches = iter_fallback_files(fallback_roots, wanted_names)
    still_missing: list[str] = []

    for index, original in unresolved:
        match = choose_fallback(fallback_matches.get(Path(original).name, []), original)
        if match is None:
            still_missing.append(original)
        else:
            prepared[index]["video"] = str(match)

    return prepared, still_missing


def atomic_write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_dataset(
    dataset: str,
    source_root: Path,
    target_root: Path,
    video_root: Path,
    fallback_roots: list[Path],
) -> str:
    target_path = target_root / f"{dataset}.json"
    source_path = source_root / f"{dataset}.json"
    if not source_path.is_file():
        raise PreparationError(f"source annotation is missing: {source_path}")

    source_rows = load_rows(source_path)
    validate_structure(dataset, source_rows)

    if target_path.is_file():
        existing_rows = load_rows(target_path)
        try:
            validate_structure(dataset, existing_rows)
        except PreparationError:
            pass
        else:
            if rows_match_except_video(existing_rows, source_rows) and rows_have_valid_videos(
                existing_rows
            ):
                return "already valid"

    prepared_rows, missing = normalize_video_paths(
        source_rows, video_root, fallback_roots
    )
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:8])
        raise PreparationError(
            f"{dataset} is missing {len(missing)} video files after path resolution. "
            f"Examples:\n{examples}"
        )

    atomic_write_json(target_path, prepared_rows)
    return "prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument(
        "--fallback-root", type=Path, action="append", default=[], dest="fallback_roots"
    )
    parser.add_argument(
        "datasets", nargs="+", choices=tuple(EXPECTED_SAMPLES), metavar="DATASET"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        for dataset in args.datasets:
            status = prepare_dataset(
                dataset=dataset,
                source_root=args.source_root,
                target_root=args.target_root,
                video_root=args.video_root,
                fallback_roots=args.fallback_roots,
            )
            print(f"{dataset}: {status} ({EXPECTED_SAMPLES[dataset]} samples)")
    except PreparationError as exc:
        print(f"Data preflight failed: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
