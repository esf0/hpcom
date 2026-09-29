"""Shared type aliases and Protocols for hpcom.

Kept tiny and dependency-free so any module can import from here without
risk of a cycle.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt

#: A complex-valued NumPy array. Time is always the last axis; a dual-
#: polarisation field has shape ``(..., 2, n_samples)`` (X, Y).
ComplexArray: TypeAlias = npt.NDArray[np.complexfloating]

#: Direction of propagation through a span.
Direction: TypeAlias = Literal["forward", "backward"]


#: Polarization mode of an optical fiber.
Polarization: TypeAlias = Literal["single", "dual"]

#: Placement of the split-step steps along a span (see ``docs/numerics.rst``).
StepDistribution: TypeAlias = Literal["uniform", "logarithmic"]


class SchemeLike(Protocol):
    """Numerical scheme of a span as read by propagators (see :class:`hpcom.numerics.Scheme`)."""

    @property
    def method(self) -> str: ...
    @property
    def nonlinear_position(self) -> float: ...
    @property
    def kerr_length(self) -> str: ...
    @property
    def kerr_filter_bandwidth_hz(self) -> float | None: ...
    @property
    def kerr_filter_shape(self) -> str: ...
    @property
    def local_error(self) -> float | None: ...


class LossSpectrumLike(Protocol):
    """Frequency-dependent loss (see :class:`hpcom.fiber.LossSpectrum`)."""

    def at(self, wavelength_m: npt.ArrayLike) -> npt.NDArray[np.float64]: ...


class RamanLike(Protocol):
    """Delayed Raman response (see :class:`hpcom.raman.RamanResponse`)."""

    @property
    def f_r(self) -> float: ...
    def transfer(self, omega_rad_per_s: npt.ArrayLike) -> npt.NDArray[np.complex128]: ...


@runtime_checkable
class FiberLike(Protocol):
    """Minimal fiber interface a :class:`Propagator` depends on.

    Defined as a Protocol so backends do not need to import the concrete
    :class:`hpcom.fiber.Fiber` (which would create an import cycle). Members
    are declared as read-only properties so frozen models and dataclasses
    satisfy the Protocol structurally. Units are in the member names.

    ``polarization`` selects the model — scalar NLSE (``"single"``) or
    Manakov (``"dual"``) — from the fiber config rather than from the input
    shape, which keeps batched pipelines unambiguous. The generalised-NLSE
    members (``higher_order_betas`` … ``effective_shock_time_s``) are ``()``,
    ``None``, ``None`` and ``0.0`` for the plain NLSE.
    """

    @property
    def length_km(self) -> float: ...
    @property
    def alpha_db_per_km(self) -> float: ...
    @property
    def beta1_s_per_m(self) -> float: ...
    @property
    def beta2_s2_per_m(self) -> float: ...
    @property
    def beta3_s3_per_m(self) -> float: ...
    @property
    def gamma_per_w_per_km(self) -> float: ...
    @property
    def n_steps(self) -> int: ...
    @property
    def step_distribution(self) -> StepDistribution: ...
    @property
    def step_log_factor(self) -> float: ...
    @property
    def scheme(self) -> SchemeLike: ...
    @property
    def n_cores(self) -> int: ...
    @property
    def polarization(self) -> Polarization: ...
    @property
    def higher_order_betas(self) -> tuple[float, ...]: ...
    @property
    def wavelength_m(self) -> float: ...
    @property
    def loss_spectrum(self) -> LossSpectrumLike | None: ...
    @property
    def raman(self) -> RamanLike | None: ...
    @property
    def effective_shock_time_s(self) -> float: ...


@runtime_checkable
class LinkComponent(Protocol):
    """Anything that can be placed in a :class:`hpcom.link.Link`.

    Concrete components (:class:`hpcom.fiber.Fiber`, :class:`hpcom.amplifier.Amplifier`,
    future filters / DCMs / couplers) implement :meth:`apply` and are composed
    in order by ``Link.propagate``.
    """

    def apply(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray: ...


def _is_power_of_two(n: int) -> bool:
    """Return True iff n is an integer power of two (n >= 2)."""
    return n >= 2 and (n & (n - 1)) == 0
