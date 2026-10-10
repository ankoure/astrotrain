"""Synthetic data shaped like MBTA LAMP subway on-time performance files."""

from pathlib import Path

import numpy as np
import pandas as pd


def stations(start: int, count: int) -> list[str]:
    return [f"place-s{i}" for i in range(start, start + count)]


def lamp_frame(
    service_date: int,
    route_id: str,
    stop_names: list[str],
    n_trips: int,
    seed: int = 0,
) -> pd.DataFrame:
    """One service day of trips, each stopping at every station in order."""
    rng = np.random.default_rng(seed)
    rows = []
    for trip in range(n_trips):
        for seq, station in enumerate(stop_names):
            rows.append(
                {
                    "service_date": service_date,
                    "trip_id": f"{service_date}-{route_id}-{trip}",
                    "stop_sequence": seq,
                    "parent_station": station,
                    "route_id": route_id,
                    "direction_id": trip % 2 == 0,
                    "start_time": 6 * 3600 + trip * 600,
                    "stop_count": len(stop_names),
                    "travel_time_seconds": rng.normal(120, 20),
                    "dwell_time_seconds": rng.normal(30, 5),
                    "headway_trunk_seconds": rng.normal(400, 50),
                    "headway_branch_seconds": rng.normal(600, 50),
                    "scheduled_travel_time": 120.0,
                    "scheduled_headway_trunk": 400.0,
                    "scheduled_headway_branch": 600.0,
                }
            )
    return pd.DataFrame(rows)


def write_daily_files(directory: Path, n_days: int = 5) -> list[str]:
    """
    Write one parquet file per day with 4 trips of 8 stops. The station list shifts
    each day and later days switch to a route the first days never have, so some
    values only show up partway through the dataset.
    """
    directory.mkdir(parents=True, exist_ok=True)
    routes = ["Red", "Orange", "Blue", "Green-B", "Green-B"]
    files = []
    for day in range(n_days):
        df = lamp_frame(
            service_date=int(f"202509{day + 1:02d}"),
            route_id=routes[day],
            stop_names=stations(day * 2, 8),
            n_trips=4,
            seed=day,
        )
        path = directory / f"2025-09-{day + 1:02d}.parquet"
        df.to_parquet(path)
        files.append(str(path))
    return files
