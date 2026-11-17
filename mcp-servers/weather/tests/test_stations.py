"""GHCN-hourly station fetch: names resolved, failed quality codes removed."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from weather_mcp.implementation import stations
from weather_mcp.implementation.common import WeatherToolError

LIST = (
    "GHCN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME,GSN,(US)HCN_(US)CRN,WMO_ID,ICAO,ISO_CODE\n"
    "UKI0000EGLL,51.4775,-0.4614,25.3,,HEATHROW,,,,EGLL,GB\n"
    "GMMU0010637,50.05,8.58,112.0,,FRANKFURT-MAIN,,,10637,,DE\n"
)


def _parquet() -> bytes:
    frame = pd.DataFrame(
        {
            "Station_name": ["HEATHROW"] * 3,
            "DATE": [
                "2026-10-01T00:00:00",
                "2026-10-01T01:00:00",
                "2026-10-01T02:00:00",
            ],
            "LATITUDE": [51.4775] * 3,
            "LONGITUDE": [-0.4614] * 3,
            "temperature": [10.0, 99.0, 11.0],
            "temperature_Quality_Code": ["1", "L", ""],
            "dew_point_temperature": [8.0, 8.5, 9.0],
            "dew_point_temperature_Quality_Code": ["1", "1", "7"],
            "sea_level_pressure": [1013.0, None, 1012.0],
        }
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


class _Response:
    def __init__(self, status: int, content: bytes) -> None:
        self.status_code = status
        self.content = content
        self.text = content.decode(errors="replace")
        self.ok = status < 400

    def raise_for_status(self) -> None:
        if not self.ok:
            raise stations.requests.HTTPError(str(self.status_code))


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def get(url: str, timeout: float) -> _Response:
        seen.append(url)
        if url == stations.GHCNH_STATIONS:
            return _Response(200, LIST.encode())
        if "UKI0000EGLL_2026" in url:
            return _Response(200, _parquet())
        return _Response(404, b"")

    monkeypatch.setattr(stations.requests, "get", get)
    return seen


def test_names_resolve_by_icao_wmo_isd_and_id(network: list[str]) -> None:
    assert stations.resolve(["EGLL", "10637", "10637099999", "UKI0000EGLL"]) == {
        "EGLL": "UKI0000EGLL",
        "10637": "GMMU0010637",
        "10637099999": "GMMU0010637",
        "UKI0000EGLL": "UKI0000EGLL",
    }


def test_unknown_names_are_refused_together(network: list[str]) -> None:
    with pytest.raises(WeatherToolError) as caught:
        stations.resolve(["EGLL", "ZZZZ", "03772099999"])
    assert caught.value.reason == "bad_input"
    assert "ZZZZ" in str(caught.value) and "03772099999" in str(caught.value)


def test_failed_quality_codes_are_removed_and_counted(network: list[str]) -> None:
    table, source, done = stations.ghcnh_stations(["EGLL"], "2026-10-01", "2026-10-01")
    assert table["t2m_c"].tolist()[0] == 10.0
    assert pd.isna(table["t2m_c"].tolist()[1])  # flagged L
    assert pd.isna(table["td2m_c"].tolist()[2])  # flagged 7
    assert done["removed_by_quality_code"] == {"t2m_c": 1, "td2m_c": 1}
    assert done["stations"] == {"EGLL": "UKI0000EGLL"}
    assert "GHCNh" in source.name


def test_a_page_that_is_not_parquet_is_a_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stations.requests,
        "get",
        lambda url, timeout: _Response(200, b"<html>moved</html>"),
    )
    with pytest.raises(WeatherToolError) as caught:
        stations.ghcnh_stations(["UKI0000EGLL"], "2026-10-01", "2026-10-02")
    assert caught.value.reason == "backend_unavailable"
