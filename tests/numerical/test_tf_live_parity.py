"""TF wrapper live parity (Test E).

Validates that ``Fiber(backend="tensorflow").propagate(...)`` produces
byte-identical output to a *direct* call into the vendored
``tf_fiber_propogate`` / ``tf_manakov_fiber_propogate`` kernels with manually
converted parameters. This catches bugs in the adapter layer (unit conversions
for α, β₂; sign of length for backward; n_steps clipping) that the existing
golden-npz parity test (``test_tf_parity.py``) cannot, because the golden npz
was produced by the wrapper itself.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from hpcom.backends.tensorflow._vendor.propagation import (  # noqa: E402
    tf_fiber_propogate,
    tf_manakov_fiber_propogate,
)
from hpcom.fiber import Fiber  # noqa: E402


def _alpha_np_per_km(alpha_db_per_km: float) -> float:
    return float(alpha_db_per_km * np.log(10.0) / 10.0)


def _seeded_complex_signal(n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)


def test_tf_wrapper_matches_direct_vendor_call_scalar():
    """SSMF, scalar (single-pol) — wrapper output must equal direct call exactly."""
    length_km = 80.0
    n_steps = 32
    alpha_db_per_km = 0.2
    beta2_s2_per_m = -2.16e-26
    gamma = 1.27
    n_samples = 2048
    dt_s = 1.0 / 80e9  # fs = 80 GHz

    signal = _seeded_complex_signal(n_samples, seed=11)

    fiber = Fiber(
        length_km=length_km,
        alpha_db_per_km=alpha_db_per_km,
        beta2_s2_per_m=beta2_s2_per_m,
        gamma_per_w_per_km=gamma,
        n_steps=n_steps,
        backend="tensorflow",
        polarization="single",
    )
    wrapper_out = fiber.propagate(signal, dt_s=dt_s)

    # Direct vendor call with the same converted parameters.
    t_span = dt_s * n_samples
    sig_tf = tf.constant(signal, dtype=tf.complex128)
    direct_tf = tf_fiber_propogate(
        sig_tf,
        t_span,
        length_km,
        n_steps,
        gamma,
        beta2_s2_per_m * 1e3,  # s²/m → s²/km
        alpha=_alpha_np_per_km(alpha_db_per_km),
        beta3=0.0,
    )
    direct_out = np.asarray(direct_tf, dtype=np.complex128)

    np.testing.assert_array_equal(wrapper_out, direct_out)


def test_tf_wrapper_matches_direct_vendor_call_manakov():
    """Manakov (dual-pol) parity for the wrapper."""
    length_km = 80.0
    n_steps = 16
    alpha_db_per_km = 0.2
    beta2_s2_per_m = -2.16e-26
    gamma = 1.27
    n_samples = 1024
    dt_s = 1.0 / 80e9

    sig_x = _seeded_complex_signal(n_samples, seed=21)
    sig_y = _seeded_complex_signal(n_samples, seed=22)
    field = np.stack([sig_x, sig_y], axis=0)

    fiber = Fiber(
        length_km=length_km,
        alpha_db_per_km=alpha_db_per_km,
        beta2_s2_per_m=beta2_s2_per_m,
        gamma_per_w_per_km=gamma,
        n_steps=n_steps,
        backend="tensorflow",
        polarization="dual",
    )
    wrapper_out = fiber.propagate(field, dt_s=dt_s)
    assert wrapper_out.shape == (2, n_samples)

    t_span = dt_s * n_samples
    sig_x_tf = tf.constant(sig_x, dtype=tf.complex128)
    sig_y_tf = tf.constant(sig_y, dtype=tf.complex128)
    x_out, y_out = tf_manakov_fiber_propogate(
        sig_x_tf,
        sig_y_tf,
        t_span,
        length_km,
        n_steps,
        gamma,
        beta2_s2_per_m * 1e3,
        alpha=_alpha_np_per_km(alpha_db_per_km),
        beta3=0.0,
    )
    direct = np.stack(
        [np.asarray(x_out, dtype=np.complex128), np.asarray(y_out, dtype=np.complex128)],
        axis=0,
    )
    np.testing.assert_array_equal(wrapper_out, direct)


def test_tf_wrapper_backward_uses_negative_length():
    """`direction="backward"` should be equivalent to forward propagation with
    negated length — that's the contract the vendor kernel exposes."""
    length_km = 50.0
    n_steps = 16
    alpha_db_per_km = 0.0  # lossless so backward is the exact inverse, easier to reason
    beta2_s2_per_m = -2.16e-26
    gamma = 0.0  # linear so direction sign maps cleanly
    n_samples = 1024
    dt_s = 1.0 / 80e9

    signal = _seeded_complex_signal(n_samples, seed=33)

    fiber = Fiber(
        length_km=length_km,
        alpha_db_per_km=alpha_db_per_km,
        beta2_s2_per_m=beta2_s2_per_m,
        gamma_per_w_per_km=gamma,
        n_steps=n_steps,
        backend="tensorflow",
        polarization="single",
    )
    wrapper_back = fiber.propagate(signal, dt_s=dt_s, direction="backward")

    t_span = dt_s * n_samples
    sig_tf = tf.constant(signal, dtype=tf.complex128)
    direct_back_tf = tf_fiber_propogate(
        sig_tf,
        t_span,
        -length_km,
        n_steps,
        gamma,
        beta2_s2_per_m * 1e3,
        alpha=0.0,
        beta3=0.0,
    )
    np.testing.assert_array_equal(
        wrapper_back, np.asarray(direct_back_tf, dtype=np.complex128)
    )
