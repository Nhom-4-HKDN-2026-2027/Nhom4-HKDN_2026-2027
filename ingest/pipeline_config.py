"""Non-secret configuration shared by CLI and notebooks (Python 3.11+)."""

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineConfig:
    window_start_utc: str = "2024-11-14T00:00:00Z"
    window_end_utc: str = "2026-11-14T00:00:00Z"
    alive_cutoff_utc: str = "2026-11-01T00:00:00Z"
    sample_start_utc: str = "2025-11-15T00:00:00Z"
    sample_end_utc: str = "2025-11-22T00:00:00Z"
    max_missing_2yr_pct: float = 60.0
    max_gt500_count: int = 0
    include_boundary_gaps: bool = False
    weather_model: str = "era5"
    raw_dir: str = "data/raw/verified_2026-11-16"
    sample_dir: str = "data/sample"

    def __post_init__(self):
        bounds = [
            pd.Timestamp(getattr(self, name))
            for name in (
                "window_start_utc",
                "window_end_utc",
                "alive_cutoff_utc",
                "sample_start_utc",
                "sample_end_utc",
            )
        ]
        if any(
            t.tzinfo is None or t.utcoffset().total_seconds() != 0 or t != t.floor("h")
            for t in bounds
        ):
            raise ValueError("Config timestamps must be whole hours in UTC")
        start, end, cutoff, sample_start, sample_end = bounds
        if not start <= sample_start < sample_end <= end or not start <= cutoff < end:
            raise ValueError("Sample and cutoff must lie within the evaluation window")
        if start != start.floor("D") or end != end.floor("D"):
            raise ValueError("Weather date requests require midnight window bounds")
        if not 0 <= self.max_missing_2yr_pct <= 100 or self.max_gt500_count < 0:
            raise ValueError("Invalid quality thresholds")
        if not isinstance(self.include_boundary_gaps, bool):
            raise ValueError("include_boundary_gaps must be boolean")
        if self.weather_model not in {"era5", "era5_land", "ecmwf_ifs"}:
            raise ValueError("Select one supported weather model")

    def metrics_kwargs(self):
        return {
            name: getattr(self, name)
            for name in (
                "window_start_utc",
                "window_end_utc",
                "alive_cutoff_utc",
                "max_missing_2yr_pct",
                "max_gt500_count",
                "include_boundary_gaps",
            )
        }

    def provenance(self):
        return {k: v for k, v in asdict(self).items() if k not in {"raw_dir", "sample_dir"}}

    def path(self, name: str) -> Path:
        path = Path(getattr(self, name))
        return path if path.is_absolute() else PROJECT_ROOT / path


DEFAULT_CONFIG = PipelineConfig()


def load_config(path: Path) -> PipelineConfig:
    """Read only explicitly selected non-secret TOML; reject misspelled keys."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"pipeline"}:
        raise ValueError("Config must contain only a [pipeline] table")
    unknown = set(payload["pipeline"]) - {field.name for field in fields(PipelineConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return PipelineConfig(**payload["pipeline"])
