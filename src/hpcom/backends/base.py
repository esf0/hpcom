"""Abstract base class for fiber propagation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hpcom._typing import ComplexArray, Direction, FiberLike


class Propagator(ABC):
    """Backend-agnostic interface for propagating an optical field through a fiber span.

    Implementations live under ``hpcom.backends.*`` and are registered via
    :func:`hpcom.backends.register_propagator`. A :class:`hpcom.fiber.Fiber`
    delegates its :meth:`~hpcom.fiber.Fiber.propagate` call to the backend
    selected by name.

    Backend instances may carry state (e.g. a compiled TF graph) and are
    expected to be cheap to call repeatedly.
    """

    #: Stable identifier used in the registry. Subclasses should set this.
    name: str = ""

    @abstractmethod
    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        """Propagate ``field`` through one fiber span.

        Parameters
        ----------
        field
            Complex envelope, time-domain. Last axis is samples; leading axes
            (if any) carry polarization or batch dims.
        fiber
            Span parameters. See :class:`hpcom._typing.FiberLike`.
        dt_s
            Sample period of ``field`` in seconds.
        direction
            ``"forward"`` for normal propagation; ``"backward"`` for
            reverse-direction (digital back-propagation).

        Returns
        -------
        ComplexArray
            Propagated field, same shape and dtype as the input.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
