"""Reproducible configuration loading.

All pipeline behaviour that should be stable across runs (paths, split
boundaries, blocklists, ...) lives in configs/data.yaml as data, not as
constants scattered through code. This module is the single place that reads
it, so every script and test sees the identical, versioned configuration.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


@dataclass(frozen=True)
class SplitBounds:
    train_day_min: int
    train_day_max: int
    val_day_min: int
    val_day_max: int
    test_day_min: int
    test_day_max: int

    def __post_init__(self) -> None:
        # Fail fast on a misconfigured split rather than silently producing
        # overlapping or unordered data downstream.
        if not (self.train_day_min <= self.train_day_max < self.val_day_min <= self.val_day_max
                < self.test_day_min <= self.test_day_max):
            raise ValueError(f"Split boundaries are not strictly ordered / non-overlapping: {self}")


@dataclass(frozen=True)
class Config:
    random_seed: int
    raw_dir: pathlib.Path
    processed_dir: pathlib.Path
    features_dir: pathlib.Path
    experiments_dir: pathlib.Path
    reports_dir: pathlib.Path
    raw_files: dict = field(default_factory=dict)
    day_seconds: int = 86400
    split: SplitBounds = None
    identity_from_sep: str = "-"
    identity_to_sep: str = "_"
    feature_blocklist: tuple = ()
    freq_encode_columns: tuple = ()
    d_columns_to_clip: tuple = ()
    downcast_float64_to_float32: bool = True
    downcast_int64_to_int32: bool = True
    expected_counts: dict = field(default_factory=dict)

    def raw_path(self, key: str) -> pathlib.Path:
        return self.raw_dir / self.raw_files[key]


def load_config(path: pathlib.Path | str = DEFAULT_CONFIG_PATH) -> Config:
    path = pathlib.Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    paths = raw["paths"]

    def resolve(p: str) -> pathlib.Path:
        pp = pathlib.Path(p)
        return pp if pp.is_absolute() else (PROJECT_ROOT / pp)

    split = SplitBounds(**raw["split"])

    return Config(
        random_seed=raw["random_seed"],
        raw_dir=resolve(paths["raw_dir"]),
        processed_dir=resolve(paths["processed_dir"]),
        features_dir=resolve(paths["features_dir"]),
        experiments_dir=resolve(paths["experiments_dir"]),
        reports_dir=resolve(paths["reports_dir"]),
        raw_files=raw["raw_files"],
        day_seconds=raw["day_seconds"],
        split=split,
        identity_from_sep=raw["identity_column_normalize"]["from_sep"],
        identity_to_sep=raw["identity_column_normalize"]["to_sep"],
        feature_blocklist=tuple(raw["feature_blocklist"]),
        freq_encode_columns=tuple(raw["freq_encode_columns"]),
        d_columns_to_clip=tuple(raw["d_columns_to_clip"]),
        downcast_float64_to_float32=raw["downcast"]["float64_to_float32"],
        downcast_int64_to_int32=raw["downcast"]["int64_to_int32"],
        expected_counts=raw.get("expected_counts", {}),
    )
