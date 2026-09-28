"""Composable optical-link chain.

A :class:`Link` is an ordered tuple of :class:`~hpcom._typing.LinkComponent`
instances (built-ins: :class:`~hpcom.fiber.Fiber`,
:class:`~hpcom.amplifier.Amplifier`). New components — filters, DCMs,
couplers — slot in by implementing ``apply(field, dt_s, *, rng)``; to take
part in :meth:`Link.backpropagate` they also implement
``apply_inverse(field, dt_s)``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from pydantic import Field, model_validator

from hpcom._model import HpcomModel, Polymorphic
from hpcom._typing import ComplexArray, LinkComponent
from hpcom.amplifier import Amplifier
from hpcom.fiber import Fiber


class Link(HpcomModel):
    """An ordered chain of link components, applied left to right.

    ``strict=True`` additionally requires the conventional
    ``Fiber, Amplifier, Fiber, Amplifier, …`` alternation.
    """

    elements: tuple[Polymorphic[LinkComponent], ...] = Field(min_length=1)
    strict: bool = False

    @model_validator(mode="after")
    def _validate(self) -> Link:
        for i, el in enumerate(self.elements):
            if not callable(getattr(el, "apply", None)):
                raise TypeError(
                    f"Link element {i} ({type(el).__name__}) does not implement "
                    "apply(field, dt_s, *, rng)"
                )
        if self.strict:
            for i, el in enumerate(self.elements):
                expected = Fiber if i % 2 == 0 else Amplifier
                if not isinstance(el, expected):
                    raise ValueError(
                        f"strict Link expects element {i} to be a {expected.__name__}, "
                        f"got {type(el).__name__}"
                    )
        return self

    @classmethod
    def from_spans(
        cls,
        n_spans: int,
        fiber: Fiber,
        amplifier: Amplifier,
        *,
        strict: bool = True,
    ) -> Link:
        """Uniform link of ``n_spans`` × (``fiber`` → ``amplifier``)."""
        if n_spans < 1:
            raise ValueError(f"n_spans must be >= 1, got {n_spans}")
        return cls(elements=(fiber, amplifier) * n_spans, strict=strict)

    # ---- summary quantities ---------------------------------------------

    @property
    def fibers(self) -> tuple[Fiber, ...]:
        return tuple(el for el in self.elements if isinstance(el, Fiber))

    @property
    def n_spans(self) -> int:
        """Number of :class:`Fiber` elements."""
        return len(self.fibers)

    @property
    def total_length_km(self) -> float:
        return float(sum(f.length_km for f in self.fibers))

    @property
    def total_dispersion_s2(self) -> float:
        """Accumulated ``Σ β₂·L`` in s² — pass to a receiver's ``cd_compensation_s2``."""
        return float(sum(f.accumulated_dispersion_s2 for f in self.fibers))

    @property
    def total_beta3_s3(self) -> float:
        """Accumulated ``Σ β₃·L`` in s³ — pass to a receiver's ``cd_compensation_s3``."""
        return float(sum(f.accumulated_beta3_s3 for f in self.fibers))

    @property
    def polarizations(self) -> set[str]:
        """Set of polarisation models used by the fibers of this link."""
        return {f.polarization for f in self.fibers}

    # ---- propagation ----------------------------------------------------

    def propagate(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        """Apply every element in order; ``rng`` feeds the stochastic ones (ASE)."""
        out = field
        for el in self.elements:
            out = el.apply(out, dt_s, rng=rng)
        return out

    def backpropagate(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        steps_per_span: int | None = None,
        nonlinearity_scale: float = 1.0,
        backend: str | None = None,
    ) -> ComplexArray:
        """Digital back-propagation: undo the link deterministically, last element first.

        Each fiber is propagated backward (``direction="backward"``) and each
        component's ``apply_inverse`` is applied (``field/√G`` for amplifiers;
        ASE cannot be undone). With the defaults this is the exact inverse of
        a noiseless :meth:`propagate`.

        Parameters
        ----------
        steps_per_span
            Split-step count used for every fiber (``None`` = the fiber's own
            ``n_steps``). DBP is commonly run with far fewer steps than the
            forward model.
        nonlinearity_scale
            Multiplies every fiber's γ (the "optimised γ" of practical DBP).
            ``0`` gives linear chromatic-dispersion compensation.
        backend
            Propagator for the backward fibers (``None`` = each fiber's own).
        """
        out = field
        for el in reversed(self.elements):
            if isinstance(el, Fiber):
                changes: dict[str, object] = {
                    "gamma_per_w_per_km": el.gamma_per_w_per_km * nonlinearity_scale
                }
                if steps_per_span is not None:
                    changes["n_steps"] = steps_per_span
                if backend is not None:
                    changes["backend"] = backend
                out = el.replace(**changes).apply_inverse(out, dt_s)
            elif callable(inverse := getattr(el, "apply_inverse", None)):
                out = inverse(out, dt_s)
            else:
                raise TypeError(
                    f"{type(el).__name__} has no apply_inverse(field, dt_s); "
                    "it cannot be back-propagated"
                )
        return out

    def __iter__(self) -> Iterator[LinkComponent]:  # type: ignore[override]
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)
