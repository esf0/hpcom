"""The Scheme model: validation, shorthand, wiring into Fiber/DBP/Link, CPU fallback."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import hpcom.backends as backends
from hpcom import EDFA, DigitalBackPropagation, Fiber, Link, Scheme
from hpcom.backends import CPUFallbackWarning, Propagator
from hpcom.backends.base import split_step_sequence
from hpcom.backends.numpy.ssfm import NumpySSFMPropagator


def test_default_scheme_is_the_symmetric_split_step():
    s = Scheme()
    assert (s.method, s.nonlinear_position, s.kerr_length, s.kerr_filter_bandwidth_hz) == (
        "ssfm",
        0.5,
        "step",
        None,
    )
    assert s.order == 2 and s.is_split_step and s.is_default
    assert Fiber(length_km=1.0).scheme == s


def test_method_name_shorthand_and_json_round_trip():
    fiber = Fiber(length_km=1.0, scheme="ssfm")
    assert fiber.scheme == Scheme()
    custom = Fiber(
        length_km=1.0,
        scheme=Scheme(
            nonlinear_position=0.2, kerr_length="effective", kerr_filter_bandwidth_hz=5e9
        ),
    )
    assert Fiber.model_validate_json(custom.model_dump_json()) == custom
    with pytest.raises(ValidationError, match="unknown scheme"):
        Fiber(length_km=1.0, scheme="leapfrog")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"nonlinear_position": 1.5}, "less than or equal"),
        ({"kerr_filter_shape": "rectangular"}, "needs kerr_filter_bandwidth_hz"),
        ({"kerr_filter_bandwidth_hz": 0.0}, "greater than 0"),
        ({"kerr_length": "average"}, "kerr_length"),
    ],
)
def test_invalid_schemes_are_rejected(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        Scheme(**kwargs)


def test_order_reflects_the_kerr_position():
    assert Scheme(nonlinear_position=0.5, kerr_length="effective").order == 2
    assert Scheme(nonlinear_position=0.0).order == 1
    assert not Scheme(nonlinear_position=0.3).is_default


def test_sequence_places_the_kerr_step_at_the_same_z_in_both_directions():
    """Forward: D(rh) N D((1−r)h); backward: D(−(1−r)h) N D(−rh) — the exact inverse."""
    fiber = Fiber(length_km=10.0, n_steps=2, scheme=Scheme(nonlinear_position=0.2))
    lin_f, kerr_f = split_step_sequence(fiber, "forward")
    lin_b, kerr_b = split_step_sequence(fiber, "backward")
    np.testing.assert_allclose(lin_f, [1e3, 4e3 + 1e3, 4e3])
    np.testing.assert_allclose(kerr_f, [5e3, 5e3])
    np.testing.assert_allclose(lin_b, -lin_f[::-1])
    np.testing.assert_allclose(kerr_b, -kerr_f[::-1])


def test_effective_kerr_length_formula():
    alpha = 0.2 * np.log(10) / 10 / 1e3
    h = 80e3
    for r in (0.0, 0.3, 0.5, 1.0):
        fiber = Fiber(
            length_km=80.0,
            n_steps=1,
            scheme=Scheme(nonlinear_position=r, kerr_length="effective"),
        )
        _, kerr = split_step_sequence(fiber, "forward")
        expected = (np.exp(alpha * r * h) - np.exp(-alpha * (1 - r) * h)) / alpha
        np.testing.assert_allclose(kerr, [expected], rtol=1e-14)
    lossless = Fiber(
        length_km=80.0, alpha_db_per_km=0.0, n_steps=4, scheme=Scheme(kerr_length="effective")
    )
    np.testing.assert_allclose(split_step_sequence(lossless, "forward")[1], 20e3)


def test_dbp_and_link_pass_the_scheme_to_the_backward_fibers():
    fiber = Fiber.ssmf(80.0, n_steps=10)
    link = Link.from_spans(2, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))
    scheme = Scheme(nonlinear_position=0.1, kerr_length="effective")
    dbp = DigitalBackPropagation(link=link, steps_per_span=1, scheme=scheme)
    assert dbp._backward_fiber(fiber, 0.0).scheme == scheme
    assert (
        DigitalBackPropagation(link=link, steps_per_span=1)._backward_fiber(fiber, 0.0).scheme
        == Scheme()
    )
    x = np.random.default_rng(0).standard_normal(256) * 0.03 + 0j
    np.testing.assert_allclose(
        dbp.compensate(x, 1e-11), link.backpropagate(x, 1e-11, steps_per_span=1, scheme=scheme)
    )


class _PlainOnly(NumpySSFMPropagator):
    """A backend that implements no scheme at all (to exercise the fallback)."""

    name = "plain-only"
    supported_methods: frozenset[str] = frozenset()


def test_unsupported_scheme_falls_back_to_numpy_with_a_warning(monkeypatch):
    monkeypatch.setitem(backends._factories, "plain-only", _PlainOnly)
    monkeypatch.delitem(backends._instances, "plain-only", raising=False)
    fiber = Fiber.ssmf(10.0, n_steps=4)
    x = np.random.default_rng(1).standard_normal(128) * 0.05 + 0j
    with pytest.warns(CPUFallbackWarning, match="running this span on the NumPy"):
        out = fiber.replace(backend="plain-only").propagate(x, 1e-11)
    np.testing.assert_array_equal(out, fiber.propagate(x, 1e-11))
    assert isinstance(backends.get_propagator("plain-only"), Propagator)


@pytest.mark.parametrize("method", ["ssfm4-yoshida", "ssfm4-suzuki", "rk4ip"])
def test_fourth_order_methods(method):
    s = Scheme(method=method)
    assert s.order == 4
    assert s.is_split_step == (method != "rk4ip")
    assert (
        Fiber.model_validate_json(Fiber(length_km=1.0, scheme=method).model_dump_json()).scheme == s
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "rk4ip", "kerr_length": "effective"},
        {"method": "rk4ip", "kerr_filter_bandwidth_hz": 1e9},
        {"method": "rk4ip", "nonlinear_position": 0.2},
        {"method": "ssfm4-yoshida", "nonlinear_position": 0.2},
    ],
)
def test_options_that_do_not_apply_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        Scheme(**kwargs)


def test_composition_weights_are_symmetric_and_sum_to_one():
    from hpcom.backends.base import COMPOSITION_WEIGHTS

    for weights in COMPOSITION_WEIGHTS.values():
        assert sum(weights) == pytest.approx(1.0, abs=1e-15)
        assert tuple(reversed(weights)) == weights
    w1, w0, _ = COMPOSITION_WEIGHTS["ssfm4-yoshida"]
    assert w1 == pytest.approx(1 / (2 - 2 ** (1 / 3))) and w0 < 0
    # Fourth-order condition for symmetric compositions of a second-order step: Σ w³ = 0.
    for name in ("ssfm4-yoshida", "ssfm4-suzuki"):
        assert sum(w**3 for w in COMPOSITION_WEIGHTS[name]) == pytest.approx(0.0, abs=1e-14)


def test_composition_sequence_has_one_kerr_step_per_stage():
    fiber = Fiber(length_km=10.0, n_steps=2, scheme="ssfm4-suzuki")
    linear, kerr = split_step_sequence(fiber, "forward")
    assert kerr.size == 10 and linear.size == 11
    assert kerr.sum() == pytest.approx(10e3) and linear.sum() == pytest.approx(10e3)
    assert np.sum(kerr < 0) == 2  # the negative middle stage of each step
