"""IBTrACS best tracks: provisional rows take JTWC values, and say so."""

from __future__ import annotations

import pandas as pd

from weather_mcp.implementation import fetch

CSV = (
    "SID,SEASON,NAME,ISO_TIME,LAT,LON,WMO_WIND,WMO_PRES,USA_WIND,USA_PRES\n"
    ",Year,,,degrees_north,degrees_east,kts,mb,kts,mb\n"
    "2026310N18149,2026,CEMPAKA,2026-11-06 06:00:00,18.0,149.0, , ,25,1007\n"
    "2026310N18149,2026,CEMPAKA,2026-11-09 18:00:00,24.6,136.6,85,965,93,963\n"
    "2026310N18149,2026,CEMPAKA,2026-11-10 00:00:00,25.0,138.0, , , , \n"
)


class _Response:
    text = CSV

    def raise_for_status(self) -> None:
        return None


def test_provisional_rows_take_jtwc_values_with_their_agency(monkeypatch) -> None:
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Response())
    table, _ = fetch.ibtracs("WP", "2026310N18149", None, None)
    assert table["wind_kt"].tolist()[:2] == [25.0, 85.0]
    assert table["wind_agency"].tolist() == ["usa", "wmo", ""]
    assert table["mslp_agency"].tolist() == ["usa", "wmo", ""]
    assert table["usa_wind_kt"].iloc[1] == 93.0
    assert pd.isna(table["wmo_wind_kt"].iloc[0])
    assert pd.isna(table["wind_kt"].iloc[2])


def test_track_score_names_the_best_track_agencies() -> None:
    from weather_mcp.implementation import verification

    observed = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-11-06 06:00", "2026-11-07 06:00"], utc=True),
            "latitude": [18.0, 20.0],
            "longitude": [149.0, 142.8],
            "mslp_hpa": [1007.0, 999.0],
            "wind_kt": [25.0, 40.0],
            "mslp_agency": ["usa", "usa"],
            "wind_agency": ["usa", "wmo"],
        }
    )
    forecasts = pd.DataFrame(
        {
            "model": ["X"],
            "time": pd.to_datetime(["2026-11-06 18:00"], utc=True),
            "lead_hours": [12],
            "latitude": [19.0],
            "longitude": [146.0],
            "mslp_hpa": [1000.0],
            "wind_kt": [30.0],
        }
    )
    _, numbers = verification.track_score(forecasts, observed)
    assert numbers["best_track_mslp_agency"] == ["usa"]
    assert numbers["best_track_wind_agency"] == ["usa", "wmo"]
    assert any("mixes agencies" in n for n in numbers["notes"])
