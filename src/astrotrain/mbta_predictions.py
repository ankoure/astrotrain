"""
Head-to-head with the MBTA's own real-time predictions: what the MBTA predicted,
at the same moment, for the same train arriving at the same stop.

The MBTA's predictions come from its GTFS-realtime TripUpdates feed as archived
(every poll kept) by the us-rail-peformance-archiver project: one trip_updates
parquet file per UTC day, under year=/month=/day=/data.parquet.

Our predictions are made as a train leaves a stop. For each, this takes the
MBTA's latest prediction for that trip and target stop made at or before that
moment (never after), so both sides had the same chance to know things.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from astrotrain.evaluate import MBTA_ACCURACY_BINS, score_like_mbta

SUBWAY_ROUTES = [
    "Red",
    "Orange",
    "Blue",
    "Mattapan",
    "Green-B",
    "Green-C",
    "Green-D",
    "Green-E",
]

# The archive polls every 30s but sometimes misses polls for up to an hour; an
# older prediction isn't what the MBTA was showing at that moment
MAX_PREDICTION_AGE_SECONDS = 120

_COLUMNS = {
    "feed_timestamp": "made",
    "trip_update.trip.trip_id": "trip_id",
    "trip_update.stop_time_update.stop_id": "stop_id",
    "trip_update.stop_time_update.arrival.time": "arrival",
    "trip_update.stop_time_update.departure.time": "departure",
}


def archive_files(feed_dir: Path, service_dates) -> List[Path]:
    """
    The archive's daily files covering these service dates (YYYYMMDD). Files are
    split by UTC day, so a service day's evening is in the next day's file.
    """
    days = pd.to_datetime(pd.Series(service_dates).astype(str), format="%Y%m%d")
    wanted = sorted(set(days) | set(days + pd.Timedelta(days=1)))
    paths = [
        Path(feed_dir) / f"year={d.year}/month={d.month}/day={d.day}/data.parquet"
        for d in wanted
    ]
    return [p for p in paths if p.exists()]


def load_mbta_predictions(files: List[Path]) -> pd.DataFrame:
    """
    Every archived subway prediction: when it was made, the trip and stop it's
    for, and the predicted time (arrival, or departure where there's no arrival,
    as at terminals), all in unix seconds
    """
    parts = []
    for path in files:
        table = pq.read_table(
            path,
            columns=list(_COLUMNS),
            filters=[("trip_update.trip.route_id", "in", SUBWAY_ROUTES)],
        )
        df = table.to_pandas().rename(columns=_COLUMNS)
        df["predicted_at"] = df["arrival"].fillna(df["departure"])
        parts.append(df[["made", "trip_id", "stop_id", "predicted_at"]])
    predictions = pd.concat(parts, ignore_index=True).dropna()
    predictions["made"] = predictions["made"].astype("float64")
    return predictions.sort_values("made", kind="stable").reset_index(drop=True)


def attach_mbta_predictions(
    results: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """
    Add "mbta": the MBTA's predicted seconds from departure_time to arriving at
    the target stop, from its latest prediction for that trip and stop made at
    or before departure_time and at most MAX_PREDICTION_AGE_SECONDS old. Missing
    where there was none.
    """
    order = np.argsort(results["departure_time"].to_numpy(), kind="stable")
    ours = results.iloc[order].reset_index(names="_row")
    ours = ours.dropna(subset=["departure_time"])
    matched = pd.merge_asof(
        ours.astype({"departure_time": "float64"}),
        predictions,
        left_on="departure_time",
        right_on="made",
        by=["trip_id", "stop_id"],
        direction="backward",
        tolerance=float(MAX_PREDICTION_AGE_SECONDS),
    )
    mbta = pd.Series(np.nan, index=results.index)
    mbta.iloc[matched["_row"].to_numpy()] = (
        matched["predicted_at"] - matched["departure_time"]
    ).to_numpy()
    return results.assign(mbta=mbta.to_numpy())


def head_to_head(results: pd.DataFrame) -> dict:
    """
    Our predictions against the MBTA's on the targets where the MBTA had one:
    mean absolute error by stops ahead and by route, and percent accurate by the
    MBTA's own rules by route and lead-time group (each side grouped by its own
    prediction, as the MBTA groups its own).
    """
    both = results.dropna(subset=["mbta"])
    errors = both.assign(
        ours=(both["predicted"] - both["actual"]).abs(),
        mbta_error=(both["mbta"] - both["actual"]).abs(),
    )

    def mae(by):
        table = pd.concat([errors.assign(**{by: "all"}), errors.sort_values(by)])
        table = table.groupby(by, sort=False).agg(
            targets=("ours", "size"), ours=("ours", "mean"), mbta=("mbta_error", "mean")
        )
        table["ours_vs_mbta"] = (table["ours"] / table["mbta"] - 1).map(
            "{:+.0%}".format
        )
        return table.round(1)

    keys = ["route_id", "bin"]
    ours = score_like_mbta(both, "predicted").groupby(keys)["accurate"].mean()
    theirs = score_like_mbta(both, "mbta").groupby(keys)["accurate"].mean()
    accuracy = (100 * pd.DataFrame({"ours": ours, "mbta": theirs})).reset_index()
    accuracy["bin"] = pd.Categorical(
        accuracy["bin"], [b[0] for b in MBTA_ACCURACY_BINS], ordered=True
    )
    accuracy = accuracy.sort_values(keys).set_index(keys)
    accuracy["ours_minus_mbta"] = accuracy["ours"] - accuracy["mbta"]

    return {
        "coverage": len(both) / len(results) if len(results) else 0.0,
        "by_stops_ahead": mae("stops_ahead"),
        "by_route": mae("route_id"),
        "accuracy": accuracy.round(1),
    }
