"""Link composition, summary quantities, propagation and digital back-propagation."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import EDFA, Fiber, Link


class _Attenuator:
    """A duck-typed third-party component (not an HpcomModel) with an inverse."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def apply(self, field, dt_s, *, rng=None):
        return field * self.factor

    def apply_inverse(self, field, dt_s):
        return field / self.factor


def test_from_spans_alternates_fiber_and_amplifier():
    f, a = Fiber.ssmf(80.0), EDFA(gain_db=16.0)
    link = Link.from_spans(3, f, a)
    assert link.elements == (f, a, f, a, f, a)
    assert link.strict and link.n_spans == 3 and len(link) == 6
    assert list(link) == list(link.elements)


def test_from_spans_rejects_zero_spans():
    with pytest.raises(ValueError, match="n_spans"):
        Link.from_spans(0, Fiber.ssmf(80.0), EDFA(gain_db=16.0))


def test_empty_link_is_rejected():
    with pytest.raises(ValidationError, match="at least 1"):
        Link(elements=())


def test_strict_link_requires_alternation():
    f, a = Fiber.ssmf(80.0), EDFA(gain_db=16.0)
    with pytest.raises(ValidationError, match="element 1 to be a Amplifier"):
        Link(elements=(f, f), strict=True)
    with pytest.raises(ValidationError, match="element 0 to be a Fiber"):
        Link(elements=(a, f), strict=True)
    Link(elements=(f, a, f), strict=True)
    Link(elements=(f, f))  # fine when not strict


def test_elements_must_implement_apply():
    with pytest.raises(ValidationError):
        Link(elements=(object(),))


def test_summary_quantities():
    f1 = Fiber.ssmf(80.0, dispersion_slope_ps_per_nm2_per_km=0.057)
    f2 = Fiber(length_km=50.0, beta2_s2_per_m=1e-26)
    link = Link(elements=(f1, EDFA(gain_db=16.0), f2))
    assert link.total_length_km == 130.0
    assert link.total_dispersion_s2 == pytest.approx(f1.beta2_s2_per_m * 80e3 + 1e-26 * 50e3)
    assert link.total_beta3_s3 == pytest.approx(f1.beta3_s3_per_m * 80e3)
    assert link.polarizations == {"single"}


def test_propagate_applies_elements_in_order():
    f = Fiber(length_km=10.0, alpha_db_per_km=0.0, gamma_per_w_per_km=0.0, beta2_s2_per_m=0.0)
    link = Link(elements=(f, _Attenuator(0.5), EDFA(gain_db=0.0)))
    x = np.ones(16, complex)
    np.testing.assert_allclose(link.propagate(x, 1e-11), 0.5 * x)


def test_propagate_is_reproducible_with_an_rng():
    link = Link.from_spans(3, Fiber.ssmf(20.0, n_steps=4), EDFA(gain_db=4.0))
    x = np.ones(64, complex) * 1e-2
    a = link.propagate(x, 1e-11, rng=np.random.default_rng(5))
    b = link.propagate(x, 1e-11, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


def test_backpropagate_is_the_exact_inverse_of_a_noiseless_link():
    f = Fiber.ssmf(80.0, n_steps=40)
    link = Link(elements=(f, EDFA(gain_db=16.0, noise_figure_db=0.0), f, _Attenuator(0.7)))
    x = np.random.default_rng(1).standard_normal(1024) * 3e-2 + 0j
    forward = x
    for el in link.elements:  # noiseless forward pass
        forward = (
            el.propagate(forward, 1e-12)
            if isinstance(el, Fiber)
            else (
                forward * np.sqrt(el.gain_linear)
                if isinstance(el, EDFA)
                else el.apply(forward, 1e-12)
            )
        )
    np.testing.assert_allclose(link.backpropagate(forward, 1e-12), x, rtol=0, atol=1e-13)


def test_backpropagation_with_zero_nonlinearity_is_linear_cd_compensation():
    f = Fiber.ssmf(80.0, n_steps=40)
    link = Link.from_spans(2, f, EDFA(gain_db=16.0))
    x = np.random.default_rng(2).standard_normal(512) * 1e-2 + 0j
    linear = Link.from_spans(2, f.replace(backend="numpy-linear"), EDFA(gain_db=16.0))
    via_dbp = link.backpropagate(x, 1e-12, nonlinearity_scale=0.0)
    via_linear = linear.backpropagate(x, 1e-12)
    np.testing.assert_allclose(via_dbp, via_linear, rtol=0, atol=1e-13)


def test_backpropagation_overrides_steps_and_backend():
    f = Fiber.ssmf(80.0, n_steps=400)
    link = Link(elements=(f,))
    x = np.random.default_rng(3).standard_normal(512) * 0.1 + 0j
    y = f.propagate(x, 1e-12)
    coarse = link.backpropagate(y, 1e-12, steps_per_span=4)
    fine = link.backpropagate(y, 1e-12)
    assert np.linalg.norm(fine - x) < 1e-10 * np.linalg.norm(x)
    assert np.linalg.norm(coarse - x) > 1e-6 * np.linalg.norm(x)  # fewer steps → step error
    np.testing.assert_allclose(link.backpropagate(y, 1e-12, backend="numpy"), fine, atol=0)


def test_backpropagation_needs_invertible_components():
    class NoInverse:
        def apply(self, field, dt_s, *, rng=None):
            return field

    link = Link(elements=(Fiber.ssmf(1.0), NoInverse()))
    with pytest.raises(TypeError, match="cannot be back-propagated"):
        link.backpropagate(np.ones(8, complex), 1e-12)
