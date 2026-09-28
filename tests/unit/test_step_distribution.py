"""Split-step step placement: boundaries, signed step sequences, validation."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hpcom import DigitalBackPropagation, Fiber, Link
from hpcom.backends.base import merged_linear_steps, step_boundaries_m, step_sizes_m

ALPHA = 0.2 * np.log(10) / 10 / 1e3  # 0.2 dB/km in 1/m


def _log(k: float = 1.0, **kw) -> Fiber:
    kw.setdefault("n_steps", 7)
    return Fiber(length_km=80.0, step_distribution="logarithmic", step_log_factor=k, **kw)


@pytest.mark.parametrize(
    "fiber", [Fiber(length_km=80.0, n_steps=7), _log(1.0), _log(1 / 3), _log(2.0)]
)
def test_boundaries_span_the_fiber_monotonically(fiber):
    z = step_boundaries_m(fiber)
    assert z.shape == (8,)
    assert z[0] == 0.0 and z[-1] == 80e3
    assert np.all(np.diff(z) > 0)
    np.testing.assert_allclose(fiber.step_boundaries_km, z / 1e3)


def test_uniform_steps_are_equal():
    np.testing.assert_allclose(np.diff(step_boundaries_m(Fiber(length_km=80.0, n_steps=8))), 10e3)


@pytest.mark.parametrize("k", [1.0, 1 / 3, 0.5, 2.0])
def test_logarithmic_steps_carry_equal_shares_of_the_weighted_length(k):
    """Every step has the same ∫ e^{−kαz} dz; k = 1 is equal effective length (nonlinear phase)."""
    z = step_boundaries_m(_log(k, n_steps=9))
    shares = (np.exp(-k * ALPHA * z[:-1]) - np.exp(-k * ALPHA * z[1:])) / (k * ALPHA)
    np.testing.assert_allclose(shares, shares.mean(), rtol=1e-12)
    assert np.all(np.diff(np.diff(z)) > 0)  # steps lengthen as the power decays


def test_logarithmic_steps_reduce_to_uniform_without_loss_or_for_small_k():
    lossless = Fiber(
        length_km=80.0, n_steps=5, alpha_db_per_km=0.0, step_distribution="logarithmic"
    )
    np.testing.assert_array_equal(step_boundaries_m(lossless), np.arange(6) * 16e3)
    np.testing.assert_allclose(
        step_boundaries_m(_log(1e-9, n_steps=5)), np.arange(6) * 16e3, rtol=1e-8
    )


def test_one_step_is_the_whole_span_for_every_placement():
    for fiber in (Fiber(length_km=80.0, n_steps=1), _log(1.0, n_steps=1)):
        np.testing.assert_allclose(step_boundaries_m(fiber), [0.0, 80e3])


@pytest.mark.parametrize("fiber", [Fiber(length_km=80.0, n_steps=7), _log(1.0), _log(1 / 3)])
def test_backward_steps_are_the_forward_steps_reversed_and_negated(fiber):
    forward, backward = step_sizes_m(fiber, "forward"), step_sizes_m(fiber, "backward")
    np.testing.assert_allclose(forward, np.diff(step_boundaries_m(fiber)))
    np.testing.assert_array_equal(backward, -forward[::-1])
    assert forward.sum() == pytest.approx(80e3, rel=1e-14)


def test_uniform_step_sizes_are_bitwise_l_over_k():
    np.testing.assert_array_equal(
        step_sizes_m(Fiber(length_km=80.0, n_steps=3), "forward"), np.full(3, 80e3 / 3)
    )


def test_merged_linear_steps_join_adjacent_half_steps():
    h = np.array([1.0, 2.0, 4.0])
    np.testing.assert_array_equal(merged_linear_steps(h), [0.5, 1.5, 3.0, 2.0])
    np.testing.assert_array_equal(merged_linear_steps(np.array([5.0])), [2.5, 2.5])


def test_step_placement_validation():
    with pytest.raises(ValidationError, match="only to step_distribution"):
        Fiber(length_km=80.0, step_log_factor=0.5)
    with pytest.raises(ValidationError):
        _log(0.0)
    with pytest.raises(ValidationError):
        Fiber(length_km=80.0, step_distribution="adaptive")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="needs step_distribution"):
        DigitalBackPropagation(
            link=Link(elements=(Fiber(length_km=80.0),)), steps_per_span=2, step_log_factor=0.5
        )


def test_with_steps_semantics():
    fiber = _log(1 / 3, n_steps=10)
    assert fiber.with_steps() is fiber
    assert fiber.with_steps(n_steps=4).step_log_factor == 1 / 3  # placement kept
    uniform = fiber.with_steps(distribution="uniform")
    assert (uniform.step_distribution, uniform.step_log_factor) == ("uniform", 1.0)
    assert fiber.with_steps(distribution="logarithmic").step_log_factor == 1.0
    assert fiber.with_steps(log_factor=0.5).step_log_factor == 0.5
    with pytest.raises(ValidationError):
        Fiber(length_km=80.0).with_steps(log_factor=0.5)


def test_step_placement_round_trips_through_json():
    fiber = _log(1 / 3, polarization="dual")
    assert Fiber.model_validate_json(fiber.model_dump_json()) == fiber
