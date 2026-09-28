"""Composable optical-link chain.

A :class:`Link` is an ordered tuple of :class:`hpcom._typing.LinkComponent`
instances. Built-ins today: :class:`hpcom.fiber.Fiber`,
:class:`hpcom.amplifier.Amplifier`. Future filters / DCMs / couplers slot in
by implementing the same ``apply()`` contract — no edits to ``Link``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from hpcom._typing import ComplexArray, LinkComponent
from hpcom.amplifier import Amplifier
from hpcom.fiber import Fiber


class Link(BaseModel):
    """An ordered chain of optical link components.

    The chain is applied left-to-right by :meth:`propagate`. ``strict`` mode
    requires alternation of fiber and amplifier starting with a fiber
    (the conventional EDFA-compensated SSMF link).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    elements: tuple[LinkComponent, ...]
    strict: bool = False

    @model_validator(mode="after")
    def _validate(self) -> Link:
        if not self.elements:
            raise ValueError("Link must have at least one element")
        for i, el in enumerate(self.elements):
            if not hasattr(el, "apply"):
                raise TypeError(
                    f"Link element {i} of type {type(el).__name__} does not implement "
                    "LinkComponent.apply(field, dt_s, *, rng)"
                )
        if self.strict:
            for i, el in enumerate(self.elements):
                expected = Fiber if i % 2 == 0 else Amplifier
                if not isinstance(el, expected):
                    raise ValueError(
                        f"strict Link expects element {i} to be {expected.__name__}, "
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
        """Build a uniform N-span link as ``[Fiber, Amp, Fiber, Amp, ...]``."""
        if n_spans < 1:
            raise ValueError(f"n_spans must be >= 1, got {n_spans}")
        elements: list[LinkComponent] = []
        for _ in range(n_spans):
            elements.append(fiber)
            elements.append(amplifier)
        return cls(elements=tuple(elements), strict=strict)

    @property
    def n_spans(self) -> int:
        return sum(1 for el in self.elements if isinstance(el, Fiber))

    @property
    def total_dispersion_s2(self) -> float:
        """Sum of ``β₂ · length_m`` over every fiber span (used for CD compensation)."""
        return sum(
            el.beta2_s2_per_m * el.length_km * 1e3 for el in self.elements if isinstance(el, Fiber)
        )

    def propagate(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        """Propagate ``field`` through every element in order, returning the output field."""
        out = field
        for el in self.elements:
            out = el.apply(out, dt_s, rng=rng)
        return out

    def __iter__(self) -> Iterator[LinkComponent]:  # type: ignore[override]  # pragma: no cover — convenience
        return iter(self.elements)
