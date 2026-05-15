"""Shared type aliases and Protocols for hpcom.

Kept tiny and dependency-free so any module can import from here without
risk of a cycle.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt

#: A complex-valued NumPy array (1D = single polarization, 2D = dual pol).
ComplexArray: TypeAlias = npt.NDArray[np.complexfloating]

#: Direction of propagation through a span.
Direction: TypeAlias = Literal["forward", "backward"]


#: Polarization mode of an optical fiber.
Polarization: TypeAlias = Literal["single", "dual"]


@runtime_checkable
class FiberLike(Protocol):
    """Minimal fiber interface a :class:`Propagator` depends on.

    Defined as a Protocol so backends do not need to import the concrete
    :class:`hpcom.fiber.Fiber` (which would create an import cycle). Members
    are declared as read-only properties so frozen dataclasses satisfy the
    Protocol structurally.

    ``polarization`` lets backends dispatch scalar SSFM vs. Manakov from the
    fiber config rather than from input shape, which keeps batched / WDM
    pipelines unambiguous.
    """

    @property
    def length_km(self) -> float: ...
    @property
    def alpha_db_per_km(self) -> float: ...
    @property
    def beta2_s2_per_m(self) -> float: ...
    @property
    def gamma_per_w_per_km(self) -> float: ...
    @property
    def n_steps(self) -> int: ...
    @property
    def polarization(self) -> Polarization: ...


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
