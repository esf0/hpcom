"""MulticoreFiber: model, coupling layouts, linear system, integration with Link/EDFA/DBP."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import EDFA, DigitalBackPropagation, Fiber, Link, Simulation, SingleCarrierTransmitter
from hpcom.backends.base import multicore_linear_system
from hpcom.multicore import (
    MulticoreFiber,
    hexagonal_coupling,
    linear_array_coupling,
    ring_coupling,
)

BASE = Fiber.ssmf(40.0, n_steps=20, polarization="dual")


def _mcf(coupling=None, **kw) -> MulticoreFiber:
    return MulticoreFiber.from_fiber(
        BASE, ring_coupling(4, 1e-4) if coupling is None else coupling, **kw
    )


def test_coupling_layouts():
    lin = linear_array_coupling(3, 2.0)
    np.testing.assert_array_equal(lin, [[0, 2, 0], [2, 0, 2], [0, 2, 0]])
    ring = ring_coupling(5, 1.0)
    assert np.all(ring.sum(axis=1) == 2) and np.all(ring == ring.T)
    hexa = hexagonal_coupling(1.0)
    assert hexa.shape == (7, 7)
    assert hexa[0].sum() == 6 and np.all(hexa[1:].sum(axis=1) == 3)  # centre 6, ring 3
    with pytest.raises(ValueError):
        ring_coupling(2, 1.0)
    with pytest.raises(ValueError):
        linear_array_coupling(1, 1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coupling": np.zeros((1, 1))}, "M >= 2"),
        ({"coupling": np.zeros((2, 3))}, "M >= 2"),
        ({"coupling": np.array([[0.0, 1.0], [2.0, 0.0]])}, "symmetric"),
        ({"coupling": np.array([[1.0, 1.0], [1.0, 0.0]])}, "zero diagonal"),
        ({"coupling": np.array([[0.0, np.nan], [np.nan, 0.0]])}, "finite"),
        ({"core_beta0_offsets_per_m": [0.0, 1.0]}, "core_beta0_offsets_per_m must be 4"),
        ({"core_beta1_offsets_s_per_m": [0.0] * 5}, "core_beta1_offsets_s_per_m must be 4"),
    ],
)
def test_invalid_multicore_fibers_are_rejected(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        _mcf(**kwargs)


def test_model_basics_and_json_round_trip():
    f = _mcf(
        core_beta0_offsets_per_m=[0, 1e-5, 0, -1e-5], core_beta1_offsets_s_per_m=[0, 1e-15, 0, 0]
    )
    assert f.n_cores == 4 and BASE.n_cores == 1
    assert (f.length_km, f.n_steps, f.polarization, f.scheme) == (40.0, 20, "dual", BASE.scheme)
    assert MulticoreFiber.model_validate_json(f.model_dump_json()) == f
    core = f.core(1)
    assert type(core) is Fiber and core.beta1_s_per_m == pytest.approx(1e-15)
    with pytest.raises(IndexError):
        f.core(4)
    assert isinstance(f.with_steps(n_steps=3), MulticoreFiber)


@pytest.mark.parametrize("skew", [False, True])
def test_linear_system_diagonalises_the_coupled_operator(skew):
    """V diag(λ) V† = K(ω) = diag(k(ω) + δβ − δβ₁ω) + κ for every frequency."""
    n, dt = 64, 1e-12
    f = _mcf(
        core_beta0_offsets_per_m=[0, 1e-4, -2e-4, 3e-5],
        core_beta1_offsets_s_per_m=[0, 1e-15, -2e-15, 5e-16] if skew else None,
    )
    vectors, exponent = multicore_linear_system(n, dt, f)
    assert vectors.shape == ((n, 4, 4) if skew else (1, 4, 4))
    alpha = f.alpha_db_per_km * np.log(10) / 10 / 1e3
    lam = ((exponent + alpha / 2) / 1j).real
    k = np.broadcast_to(vectors, (n, 4, 4)) @ (
        lam[..., None] * np.conj(np.swapaxes(np.broadcast_to(vectors, (n, 4, 4)), -1, -2))
    )
    omega = 2 * np.pi * np.fft.fftfreq(n, dt)
    kw = f.beta2_s2_per_m * omega**2 / 2
    for i in (0, 7, n - 1):
        diag = kw[i] + np.array([0, 1e-4, -2e-4, 3e-5])
        if skew:
            diag = diag - omega[i] * np.array([0, 1e-15, -2e-15, 5e-16])
        np.testing.assert_allclose(
            k[i], f.coupling_per_m + np.diag(diag), atol=1e-12 * abs(kw[i]) + 1e-18
        )


def test_link_with_edfas_and_backpropagation():
    mcf = _mcf(hexagonal_coupling(1e-4))
    link = Link.from_spans(2, mcf, EDFA(gain_db=mcf.loss_db, noise_figure_db=None))
    x = np.random.default_rng(0).standard_normal((7, 2, 512)) * 0.05 + 0j
    out = link.propagate(x, 1e-11)
    assert out.shape == x.shape
    np.testing.assert_allclose(link.backpropagate(out, 1e-11), x, atol=1e-12)
    noisy = Link.from_spans(1, mcf, EDFA(gain_db=mcf.loss_db))
    a = noisy.propagate(np.zeros_like(x), 1e-11, rng=np.random.default_rng(1))
    corr = abs(np.mean(a[0] * np.conj(a[1]))) / np.mean(np.abs(a) ** 2)
    assert corr < 0.05  # independent ASE per core


def test_dbp_of_a_skewed_multicore_link_at_a_channel_offset():
    """Linear DBP of a shifted-down channel = full-field DBP then shift (exact), with skew."""
    n, dt = 2048, 1e-12
    mcf = _mcf(
        linear_array_coupling(3, 5e-4),
        core_beta1_offsets_s_per_m=[0.0, 3e-15, -2e-15],
        core_beta0_offsets_per_m=[0.0, 1e-4, -1e-4],
    ).replace(beta3_s3_per_m=1.2e-40)
    link = Link(elements=(mcf,))
    t = np.arange(n) * dt
    f_k = 150 / (n * dt)
    rng = np.random.default_rng(2)
    channel = rng.standard_normal((3, 2, n)) + 0j
    spec = np.fft.fft(channel, axis=-1)
    spec[..., np.abs(np.fft.fftfreq(n)) > 0.05] = 0
    channel = np.fft.ifft(spec, axis=-1) * 0.05
    full = channel * np.exp(2j * np.pi * f_k * t)
    dbp = DigitalBackPropagation(link=link, steps_per_span=4, nonlinearity_scale=0.0)
    expected = dbp.compensate(full, dt) * np.exp(-2j * np.pi * f_k * t)
    np.testing.assert_allclose(
        dbp.compensate(channel, dt, frequency_offset_hz=f_k), expected, atol=1e-12
    )


def test_simulation_rejects_multicore_links_with_a_clear_message():
    tx = SingleCarrierTransmitter(
        modulation_order=4, symbol_rate_hz=32e9, n_symbols=64, polarization="dual"
    )
    with pytest.raises(ValidationError, match="4]-core fibers"):
        Simulation(
            transmitter=tx,
            link=Link(elements=(_mcf(),)),
            receiver=tx.matched_receiver(),
        )
