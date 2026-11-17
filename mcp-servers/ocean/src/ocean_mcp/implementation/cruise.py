"""Plan stations to sample a moving feature within a ship's range.

Given a forecast track of the feature's centre (time, lon, lat), a port, a
ship speed and the days available, place a line of stations across the
feature at the position it is forecast to have when the ship arrives, then
check the whole schedule (steam out, stations, steam home) fits. A plan the
scientist edits (stations moved) is re-timed with ``schedule``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import OceanToolError
from .sections import along_track_km

KNOT_KM_PER_HOUR = 1.852


def _leg_hours(
    a: tuple[float, float], b: tuple[float, float], speed_knots: float
) -> float:
    km = float(along_track_km(np.array([a[0], b[0]]), np.array([a[1], b[1]]))[-1])
    return km / (speed_knots * KNOT_KM_PER_HOUR)


def schedule(
    stations: list[dict[str, Any]],
    *,
    port: tuple[float, float],
    depart: pd.Timestamp,
    speed_knots: float,
    station_hours: float,
    days_available: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Time every station in order and test the plan against the days available."""
    if not stations:
        raise OceanToolError("bad_input", "a plan needs at least one station")
    rows = []
    clock = depart
    here = port
    total_km = 0.0
    for number, station in enumerate(stations, start=1):
        there = (float(station["longitude"]), float(station["latitude"]))
        hours = _leg_hours(here, there, speed_knots)
        total_km += hours * speed_knots * KNOT_KM_PER_HOUR
        arrive = clock + pd.Timedelta(hours=hours)
        leave = arrive + pd.Timedelta(hours=station_hours)
        rows.append(
            {
                "station": station.get("name") or f"S{number:02d}",
                "longitude": there[0],
                "latitude": there[1],
                "arrive": arrive,
                "leave": leave,
                "steam_hours": hours,
            }
        )
        clock, here = leave, there
    home_hours = _leg_hours(here, port, speed_knots)
    total_km += home_hours * speed_knots * KNOT_KM_PER_HOUR
    back = clock + pd.Timedelta(hours=home_hours)
    used_days = (back - depart).total_seconds() / 86400
    numbers = {
        "stations": len(rows),
        "steaming_km": total_km,
        "return": back.isoformat(),
        "days_used": used_days,
        "days_available": days_available,
        "fits": used_days <= days_available,
        "spare_hours": (days_available - used_days) * 24,
    }
    return pd.DataFrame(rows), numbers


def plan_across_feature(
    track: pd.DataFrame,
    *,
    port: tuple[float, float],
    depart: pd.Timestamp,
    speed_knots: float,
    station_hours: float,
    days_available: float,
    stations: int,
    line_length_km: float,
    bearing_degrees: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A line of ``stations`` across the feature at its forecast arrival position.

    The arrival time is found by iterating: steam to the feature's forecast
    position, look up where the feature is at that time, and repeat until
    the target moves less than 1 km.
    """
    track = track.sort_values("time")
    times = pd.to_datetime(track["time"], utc=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    t_num = (times - epoch).dt.total_seconds().to_numpy()

    def centre_at(when: pd.Timestamp) -> tuple[float, float]:
        x = (when - epoch).total_seconds()
        if x < t_num[0] or x > t_num[-1]:
            raise OceanToolError(
                "out_of_range",
                f"arrival {when.isoformat()} is outside the forecast track "
                f"({times.iloc[0].isoformat()} to {times.iloc[-1].isoformat()})",
            )
        return float(np.interp(x, t_num, track["longitude"])), float(
            np.interp(x, t_num, track["latitude"])
        )

    target = centre_at(max(depart, times.iloc[0]))
    for _ in range(20):
        arrival = depart + pd.Timedelta(hours=_leg_hours(port, target, speed_knots))
        moved = centre_at(arrival)
        shift = float(
            along_track_km(
                np.array([target[0], moved[0]]), np.array([target[1], moved[1]])
            )[-1]
        )
        target = moved
        if shift < 1.0:
            break
    # Stations along a line through the centre at the stated bearing.
    offsets = np.linspace(-line_length_km / 2, line_length_km / 2, stations)
    b = np.deg2rad(bearing_degrees)
    dlat = offsets * np.cos(b) / 111.2
    dlon = offsets * np.sin(b) / (111.2 * np.cos(np.deg2rad(target[1])))
    plan = [
        {"name": f"S{i + 1:02d}", "longitude": target[0] + x, "latitude": target[1] + y}
        for i, (x, y) in enumerate(zip(dlon, dlat, strict=True))
    ]
    table, numbers = schedule(
        plan,
        port=port,
        depart=depart,
        speed_knots=speed_knots,
        station_hours=station_hours,
        days_available=days_available,
    )
    numbers["feature_position_at_arrival"] = {
        "longitude": target[0],
        "latitude": target[1],
        "time": arrival.isoformat(),
    }
    return table, numbers
