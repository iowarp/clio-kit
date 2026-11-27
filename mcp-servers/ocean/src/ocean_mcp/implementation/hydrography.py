"""TEOS-10 conversions, water-mass assignment and per-profile metrics.

All thermodynamics is gsw (TEOS-10). Mixed layer depth and stratification
use a criterion the caller states; nothing is assumed silently, and the
criterion is returned with the result and written to provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Literal

import gsw
import numpy as np
import pandas as pd

from .common import OceanToolError


def add_teos10(table: pd.DataFrame) -> pd.DataFrame:
    """Add SA, CT, pt0, sigma0 (and depth if absent) to a profile table.

    Needs in-situ ``temperature`` (ITS-90) and practical ``salinity``.
    """
    for column in ("temperature", "salinity"):
        if column not in table:
            raise OceanToolError("bad_input", f"TEOS-10 needs a '{column}' column")
    out = table.copy()
    p = out["pressure"].to_numpy(float)
    lat = out["latitude"].to_numpy(float)
    lon = out["longitude"].to_numpy(float)
    sp = out["salinity"].to_numpy(float)
    t = out["temperature"].to_numpy(float)
    sa = gsw.SA_from_SP(sp, p, lon, lat)
    ct = gsw.CT_from_t(sa, t, p)
    out["SA"] = sa
    out["CT"] = ct
    out["pt0"] = gsw.pt0_from_t(sa, t, p)
    out["sigma0"] = gsw.sigma0(sa, ct)
    if "depth" not in out:
        out["depth"] = -gsw.z_from_p(p, lat)
    return out


def buoyancy_frequency(table: pd.DataFrame) -> pd.DataFrame:
    """N² per profile at mid-pressures (gsw.Nsquared), as its own table."""
    if "SA" not in table:
        table = add_teos10(table)
    rows = []
    for profile, group in table.groupby("profile", sort=False):
        group = group.sort_values("pressure").dropna(subset=["SA", "CT"])
        if len(group) < 3:
            continue
        lat = float(group["latitude"].iloc[0])
        n2, p_mid = gsw.Nsquared(
            group["SA"].to_numpy(),
            group["CT"].to_numpy(),
            group["pressure"].to_numpy(),
            lat,
        )
        rows.append(
            pd.DataFrame(
                {
                    "profile": profile,
                    "time": group["time"].iloc[0],
                    "latitude": lat,
                    "longitude": float(group["longitude"].iloc[0]),
                    "pressure": np.asarray(p_mid).ravel(),
                    "N2": np.asarray(n2).ravel(),
                }
            )
        )
    if not rows:
        raise OceanToolError("no_data", "no profile has three valid levels for N²")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- water masses


@dataclass(frozen=True)
class WaterMass:
    name: str
    theta: tuple[float, float]  # potential temperature range, °C
    salinity: tuple[float, float]  # practical salinity range
    pressure: tuple[float, float]  # dbar band where the mass is expected

    def contains(self, theta: float, sal: float, pres: float) -> bool:
        return (
            self.theta[0] <= theta <= self.theta[1]
            and self.salinity[0] <= sal <= self.salinity[1]
            and self.pressure[0] <= pres <= self.pressure[1]
        )

    def distance(self, theta: float, sal: float) -> float:
        """Distance from the box centre in units of the box half-widths."""
        tc, th = sum(self.theta) / 2, (self.theta[1] - self.theta[0]) / 2 or 1.0
        sc, sh = (
            sum(self.salinity) / 2,
            (self.salinity[1] - self.salinity[0]) / 2 or 1.0,
        )
        return float(np.hypot((theta - tc) / th, (sal - sc) / sh))


def builtin_definitions() -> dict[str, dict[str, Any]]:
    """Water-mass definitions shipped with the server, each with its citation."""
    text = (
        resources.files("ocean_mcp")
        .joinpath("data/water_masses.json")
        .read_text("utf-8")
    )
    return json.loads(text)


def parse_definition(
    definition: str | dict[str, Any],
) -> tuple[str, list[WaterMass], str]:
    """A built-in name or an inline definition -> (name, masses, citation)."""
    if isinstance(definition, str):
        library = builtin_definitions()
        if definition not in library:
            raise OceanToolError(
                "bad_input",
                f"unknown definition {definition!r}; built-in: {', '.join(sorted(library))}, "
                "or pass your own {'name', 'citation', 'masses': [...]}",
            )
        spec = library[definition]
        spec = {"name": definition, **spec}
    else:
        spec = definition
    try:
        masses = [
            WaterMass(
                name=str(m["name"]),
                theta=(float(m["theta"][0]), float(m["theta"][1])),
                salinity=(float(m["salinity"][0]), float(m["salinity"][1])),
                pressure=(
                    float(m.get("pressure", [0, 12000])[0]),
                    float(m.get("pressure", [0, 12000])[1]),
                ),
            )
            for m in spec["masses"]
        ]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise OceanToolError(
            "bad_input",
            "a definition needs masses with name, theta [lo, hi], salinity [lo, hi] "
            f"and optional pressure [lo, hi]: {exc}",
        ) from exc
    return (
        str(spec.get("name", "custom")),
        masses,
        str(spec.get("citation", "stated by caller")),
    )


def assign_water_masses(
    table: pd.DataFrame, masses: list[WaterMass]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Label every sample with the water mass whose box holds it.

    A sample in more than one box takes the one whose centre is nearest in
    normalised θ-S; a sample in none is ``unassigned`` (mixtures and surface
    water outside every stated range).
    """
    if "pt0" not in table:
        table = add_teos10(table)
    labels = []
    for theta, sal, pres in zip(
        table["pt0"].to_numpy(),
        table["salinity"].to_numpy(),
        table["pressure"].to_numpy(),
        strict=True,
    ):
        if not (np.isfinite(theta) and np.isfinite(sal)):
            labels.append("missing")
            continue
        inside = [m for m in masses if m.contains(theta, sal, pres)]
        if not inside:
            labels.append("unassigned")
        else:
            labels.append(min(inside, key=lambda m: m.distance(theta, sal)).name)
    out = table.copy()
    out["water_mass"] = labels
    counts = out["water_mass"].value_counts().to_dict()
    return out, {str(k): int(v) for k, v in counts.items()}


def water_mass_cores(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Where each assigned mass sits: pressure, latitude span and mean θ-S."""
    cores = []
    for name, group in table[
        ~table["water_mass"].isin(["unassigned", "missing"])
    ].groupby("water_mass"):
        cores.append(
            {
                "water_mass": name,
                "samples": len(group),
                "profiles": int(group["profile"].nunique()),
                "pressure_dbar": [
                    float(group["pressure"].quantile(0.1)),
                    float(group["pressure"].quantile(0.9)),
                ],
                "latitude": [
                    float(group["latitude"].min()),
                    float(group["latitude"].max()),
                ],
                "mean_theta": float(group["pt0"].mean()),
                "mean_salinity": float(group["salinity"].mean()),
            }
        )
    return sorted(cores, key=lambda c: c["pressure_dbar"][0])


# ---------------------------------------------------------------- profile metrics

MldCriterion = Literal["density_threshold", "temperature_threshold"]


def mixed_layer_depth(
    group: pd.DataFrame,
    criterion: MldCriterion,
    threshold: float,
    reference_pressure: float,
) -> float:
    """MLD (dbar) of one profile: first level where the property departs from
    its value at the reference pressure by more than ``threshold``
    (de Boyer Montégut et al. 2004 use 0.03 kg m-3 or 0.2 °C from 10 dbar)."""
    group = group.sort_values("pressure")
    column = "sigma0" if criterion == "density_threshold" else "CT"
    group = group.dropna(subset=[column])
    if group.empty or group["pressure"].min() > reference_pressure + 5:
        return float("nan")
    p = group["pressure"].to_numpy()
    v = group[column].to_numpy()
    ref = float(np.interp(reference_pressure, p, v))
    below = p > reference_pressure
    departed = np.abs(v - ref) > threshold
    hits = np.nonzero(below & departed)[0]
    if hits.size == 0:
        return float("nan")
    k = hits[0]
    if k == 0:
        return float(p[0])
    # Interpolate between the last level inside and the first outside.
    dv0, dv1 = abs(v[k - 1] - ref), abs(v[k] - ref)
    frac = (threshold - dv0) / (dv1 - dv0) if dv1 != dv0 else 0.0
    return float(p[k - 1] + frac * (p[k] - p[k - 1]))


def profile_metrics(
    table: pd.DataFrame,
    *,
    criterion: MldCriterion,
    threshold: float,
    reference_pressure: float,
    stratification_range: tuple[float, float],
) -> pd.DataFrame:
    """One row per profile: MLD, mean N² over a band, surface properties."""
    if "SA" not in table:
        table = add_teos10(table)
    n2 = buoyancy_frequency(table)
    lo, hi = stratification_range
    band = (
        n2[(n2["pressure"] >= lo) & (n2["pressure"] <= hi)]
        .groupby("profile")["N2"]
        .mean()
    )
    rows = []
    for profile, group in table.groupby("profile", sort=False):
        rows.append(
            {
                "profile": profile,
                "time": group["time"].iloc[0],
                "latitude": float(group["latitude"].iloc[0]),
                "longitude": float(group["longitude"].iloc[0]),
                "mld_dbar": mixed_layer_depth(
                    group, criterion, threshold, reference_pressure
                ),
                "n2_band_mean": float(band.get(profile, np.nan)),
                "surface_CT": float(group.sort_values("pressure")["CT"].iloc[0]),
                "surface_SA": float(group.sort_values("pressure")["SA"].iloc[0]),
                "max_pressure": float(group["pressure"].max()),
            }
        )
    return pd.DataFrame(rows)


def interpolate_standard_levels(
    table: pd.DataFrame, variables: list[str], levels: np.ndarray
) -> pd.DataFrame:
    """Each profile linearly interpolated onto standard pressures (no extrapolation)."""
    rows = []
    for profile, group in table.groupby("profile", sort=False):
        group = group.sort_values("pressure").drop_duplicates("pressure")
        record: dict[str, Any] = {
            "profile": [profile] * len(levels),
            "time": [group["time"].iloc[0]] * len(levels),
            "latitude": [float(group["latitude"].iloc[0])] * len(levels),
            "longitude": [float(group["longitude"].iloc[0])] * len(levels),
            "pressure": levels,
        }
        p = group["pressure"].to_numpy()
        for variable in variables:
            valid = group[variable].notna().to_numpy()
            if valid.sum() < 2:
                record[variable] = np.full(len(levels), np.nan)
                continue
            record[variable] = np.interp(
                levels,
                p[valid],
                group[variable].to_numpy()[valid],
                left=np.nan,
                right=np.nan,
            )
        rows.append(pd.DataFrame(record))
    return pd.concat(rows, ignore_index=True)


def anomaly_profiles(
    table: pd.DataFrame,
    reference: pd.DataFrame,
    variables: list[str],
    levels: np.ndarray,
) -> pd.DataFrame:
    """Profiles minus a reference mean profile, both on standard levels.

    ``reference`` is any profile table (a climatology, the profiles outside an
    event, ...); its level means are subtracted.
    """
    on_levels = interpolate_standard_levels(table, variables, levels)
    ref = (
        interpolate_standard_levels(reference, variables, levels)
        .groupby("pressure")[variables]
        .mean()
    )
    out = on_levels.copy()
    for variable in variables:
        out[f"{variable}_anomaly"] = out[variable] - out["pressure"].map(ref[variable])
    return out


def classify_vertical_structure(anomaly: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Vertical type of each anomaly profile by this server's stated rule.

    ``column-wide`` when more than 80 % of levels are warm, otherwise
    ``surface-intensified`` when the peak anomaly is at or above 50 dbar and
    ``subsurface`` when it is deeper. Published typologies (for example Zhang
    et al. 2023) use more classes; the rule is returned with the result so a
    reader can compare.
    """
    column = f"{variable}_anomaly"
    rows = []
    for profile, group in anomaly.groupby("profile", sort=False):
        group = group.dropna(subset=[column])
        if group.empty:
            continue
        p = group["pressure"].to_numpy()
        a = group[column].to_numpy()
        peak = int(np.argmax(a))
        surface = a[p <= 20].mean() if (p <= 20).any() else a[0]
        positive = a > 0
        if positive.mean() > 0.8:
            kind = "column-wide"
        elif p[peak] <= 50:
            kind = "surface-intensified"
        else:
            kind = "subsurface"
        rows.append(
            {
                "profile": profile,
                "peak_anomaly": float(a[peak]),
                "peak_pressure": float(p[peak]),
                "surface_anomaly": float(surface),
                "positive_fraction": float(positive.mean()),
                "vertical_type": kind,
            }
        )
    return pd.DataFrame(rows)
