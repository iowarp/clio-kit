from __future__ import annotations

import numpy as np
import pytest

from fdmnes_mcp import spectrum
from fdmnes_mcp.errors import FdmnesError

from .conftest import FIXTURES


def _edge(shift: float = 0.0, step: float = 2.0):
    e = np.arange(-40.0, 80.0, 0.5)
    mu = 0.1 + 0.001 * e + step / (1 + np.exp(-(e - shift) / 1.5))
    return e, mu


def test_normalize_gives_unit_step_at_the_edge():
    e, mu = _edge(shift=3.0)
    n = spectrum.normalize(e, mu)
    assert n["normalized"] and abs(n["e0_eV"] - 3.0) < 0.6
    assert abs(n["edge_step"] - 2.0) < 0.1
    assert abs(np.mean(n["mu_norm"][e > 50]) - 1.0) < 0.1


def test_a_sharp_pre_edge_peak_is_not_taken_for_the_edge():
    """A pre-edge peak on a fine grid rises faster than the edge (magnetite's tetrahedral Fe)."""
    e = np.concatenate(
        [np.arange(-30.0, -10.0, 1.0), np.arange(-10.0, 20.0, 0.2), np.arange(20.0, 81.0, 1.0)]
    )
    edge = 0.03 / (1 + np.exp(-(e - 12.0) / 2.0))
    pre_peak = 0.008 * np.exp(-(((e - 0.5) / 0.4) ** 2))
    wiggle = 0.006 * np.sin((e - 25.0) / 6.0) * (e > 25)
    mu = 0.0002 + edge + pre_peak + wiggle
    n = spectrum.normalize(e, mu)
    assert abs(n["e0_eV"] - 12.0) < 2.0
    assert abs(n["edge_step"] - 0.03) < 0.006


def test_short_range_is_reported_not_normalized():
    e, mu = _edge()
    keep = (e > -5) & (e < 10)
    n = spectrum.normalize(e[keep], mu[keep])
    assert not n["normalized"] and "extend the energy range" in n["reason"]


def test_reads_fdmnes_conv_file():
    e, _mu, header = spectrum.read_two_columns(FIXTURES / "cu_conv_fdmnes.txt")
    assert header.startswith("Energy") and len(e) == 101 and e[0] == -10.0


def test_compare_aligns_edges_and_scores_identical_shapes():
    e1, mu1 = _edge(0.0)
    e2, mu2 = _edge(0.0, step=5.0)
    e2 = e2 + 8979.0
    summary, cols = spectrum.compare([("calc", e1, mu1), ("exp", e2, mu2)], align=True)
    exp = summary["spectra"][1]
    assert abs(exp["shift_applied_eV"] + 8979.0) < 0.6
    assert exp["r_factor_vs_first"] < 1e-3 and exp["correlation_vs_first"] > 0.999
    assert set(cols) == {"energy_eV", "calc", "exp"}


def test_compare_without_overlap_says_to_align():
    e1, mu1 = _edge()
    with pytest.raises(FdmnesError, match="align true"):
        spectrum.compare([("a", e1, mu1), ("b", e1 + 9000, mu1)], align=False)
