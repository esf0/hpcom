"""T12.1: OFDMConfig validation and derived-property tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hpcom.ofdm import OFDMConfig

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 5, 6, 7, 10, 12])
def test_n_subcarriers_must_be_power_of_two(n):
    with pytest.raises(ValueError, match="power of 2"):
        OFDMConfig(n_subcarriers=n)


def test_n_subcarriers_min_is_four():
    with pytest.raises(ValueError, match="≥ 4"):
        OFDMConfig(n_subcarriers=2)


@pytest.mark.parametrize("cp", [-1, 64, 65])
def test_cp_len_must_be_in_range(cp):
    with pytest.raises(ValueError, match="cp_len"):
        OFDMConfig(n_subcarriers=64, cp_len=cp)


@pytest.mark.parametrize("g", [-1, 32, 33])
def test_n_guard_must_be_below_half_n(g):
    with pytest.raises(ValueError, match="n_guard"):
        OFDMConfig(n_subcarriers=64, n_guard=g)


def test_pilot_indices_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        OFDMConfig(n_subcarriers=64, pilot_subcarrier_indices=(3, 5, 3))


def test_pilot_index_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        OFDMConfig(n_subcarriers=64, pilot_subcarrier_indices=(64,))


def test_pilots_disjoint_from_guard():
    # n_guard=4 over n=64 reserves indices 28..35.
    with pytest.raises(ValueError, match="guard"):
        OFDMConfig(n_subcarriers=64, n_guard=4, pilot_subcarrier_indices=(30,))


# ---------------------------------------------------------------------------
# Derived sets
# ---------------------------------------------------------------------------

def test_no_guard_no_pilots_means_all_data():
    cfg = OFDMConfig(n_subcarriers=64)
    assert cfg.n_guard == 0
    assert cfg.n_pilots == 0
    assert cfg.n_data == 64
    np.testing.assert_array_equal(
        cfg.data_subcarrier_indices(), np.arange(64, dtype=np.intp)
    )
    assert cfg.guard_subcarrier_indices().size == 0


def test_guard_subcarrier_indices_straddle_nyquist():
    cfg = OFDMConfig(n_subcarriers=64, n_guard=4)
    guards = cfg.guard_subcarrier_indices()
    np.testing.assert_array_equal(guards, np.arange(28, 36, dtype=np.intp))
    assert cfg.n_data == 64 - 8


def test_pilots_and_guards_disjoint_data_excludes_both():
    cfg = OFDMConfig(
        n_subcarriers=64, n_guard=4, pilot_subcarrier_indices=(3, 10, 50)
    )
    data = cfg.data_subcarrier_indices()
    pilots = set(cfg.pilot_subcarrier_indices)
    guards = set(cfg.guard_subcarrier_indices().tolist())
    data_set = set(int(i) for i in data)
    assert data_set.isdisjoint(pilots)
    assert data_set.isdisjoint(guards)
    assert data.size == cfg.n_data
    assert data.size == 64 - 8 - 3


def test_pilot_indices_canonicalised_sorted():
    cfg = OFDMConfig(
        n_subcarriers=32, pilot_subcarrier_indices=(7, 1, 4)
    )
    assert cfg.pilot_subcarrier_indices == (1, 4, 7)


def test_samples_per_ofdm_symbol_includes_cp():
    cfg = OFDMConfig(n_subcarriers=64, cp_len=16)
    assert cfg.samples_per_ofdm_symbol == 80


# ---------------------------------------------------------------------------
# Frozen / replaceable
# ---------------------------------------------------------------------------

def test_dataclass_is_frozen():
    cfg = OFDMConfig(n_subcarriers=64)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_subcarriers = 128


def test_replace_preserves_invariants():
    cfg = OFDMConfig(n_subcarriers=64, cp_len=16, n_guard=4)
    cfg2 = dataclasses.replace(cfg, cp_len=8)
    assert cfg2.n_subcarriers == 64
    assert cfg2.cp_len == 8
    assert cfg2.n_guard == 4
