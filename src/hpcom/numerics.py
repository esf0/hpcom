r"""Numerical schemes for fiber propagation (formulas and references: ``docs/numerics.rst``).

A :class:`Scheme` says *how* each step of a span is integrated; *where* the
steps go is ``Fiber.n_steps`` / ``step_distribution`` / ``step_log_factor``.
The default, ``Scheme()``, is the symmetric (Strang) split-step method with
the Kerr phase at the step midpoint — second order, and identical to pre-v1
hpcom, ``ssfm-gpu`` and OptiCommPy's ``ssfm``.

``method``

* ``"ssfm"`` — one split step per step: linear :math:`r h`, Kerr, linear
  :math:`(1-r)h`, with :math:`r` = ``nonlinear_position``. Second order for
  :math:`r = 1/2`, first order otherwise.
* ``"ssfm4-yoshida"`` — the symmetric step composed as a triple jump
  :math:`S(w_1h)S(w_0h)S(w_1h)`, :math:`w_1 = 1/(2-2^{1/3})`,
  :math:`w_0 = 1-2w_1` (Yoshida 1990): fourth order, 3 Kerr steps per step.
* ``"ssfm4-suzuki"`` — the fractal composition
  :math:`S(ph)^2S((1-4p)h)S(ph)^2`, :math:`p = 1/(4-4^{1/3})` (Suzuki 1990):
  fourth order with a smaller error constant, 5 Kerr steps per step.
* ``"rk4ip"`` — fourth-order Runge–Kutta in the interaction picture
  (Hult 2007): 4 Kerr evaluations and 8 FFTs per step. Not time-reversible,
  so backward propagation is accurate to fourth order, not exact.

Kerr-step options (split-step methods only)

* ``kerr_length="effective"`` — the Kerr phase uses the length that makes a
  loss-only step exact for pure self-phase modulation,
  :math:`c(h) = (e^{\alpha r h} - e^{-\alpha(1-r)h})/\alpha`
  (:math:`2\sinh(\alpha h/2)/\alpha` at :math:`r = 1/2`) instead of :math:`h`.
  Keeps the order; matters at coarse steps (DBP).
* ``kerr_filter_bandwidth_hz`` — low-pass filter the power before the Kerr
  phase (filtered DBP, after Du & Lowery 2010). This changes the equation
  being solved; use it for back-propagation, not for forward simulation.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import Field, model_validator

from hpcom._model import HpcomModel

__all__ = ["KerrFilterShape", "KerrLength", "Scheme", "SchemeMethod"]

#: Integration methods (see the module docstring).
SchemeMethod = Literal["ssfm", "ssfm4-yoshida", "ssfm4-suzuki", "rk4ip"]
#: Length used in the Kerr phase of a split step.
KerrLength = Literal["step", "effective"]
#: Shape of the intensity filter of a filtered Kerr step.
KerrFilterShape = Literal["gaussian", "rectangular"]

_SPLIT_STEP: frozenset[str] = frozenset({"ssfm", "ssfm4-yoshida", "ssfm4-suzuki"})


class Scheme(HpcomModel):
    """How each split-step step is integrated (see the module docstring).

    A bare method name is accepted wherever a scheme is expected:
    ``Fiber(..., scheme="rk4ip")``.

    Attributes
    ----------
    method
        :data:`SchemeMethod`.
    nonlinear_position
        :math:`r \\in [0, 1]`: the Kerr step of step :math:`k` acts at
        :math:`z_{k-1} + r h_k` in the fiber's own coordinate, in both
        propagation directions (``"ssfm"`` only).
    kerr_length
        ``"step"`` (Kerr phase :math:`\\gamma h P`) or ``"effective"``.
    kerr_filter_bandwidth_hz
        Cut-off frequency :math:`f_c` of the intensity filter, or ``None``.
    kerr_filter_shape
        ``"gaussian"``, :math:`H(f) = e^{-\\frac{\\ln 2}{2}(f/f_c)^2}`
        (:math:`|H(f_c)|^2 = 1/2`), or ``"rectangular"``,
        :math:`H(f) = [\\,|f| \\le f_c\\,]`.
    """

    method: SchemeMethod = "ssfm"
    nonlinear_position: float = Field(default=0.5, ge=0.0, le=1.0)
    kerr_length: KerrLength = "step"
    kerr_filter_bandwidth_hz: float | None = Field(default=None, gt=0)
    kerr_filter_shape: KerrFilterShape = "gaussian"

    @model_validator(mode="before")
    @classmethod
    def _from_name(cls, data: Any) -> Any:
        if isinstance(data, str):
            if data not in get_args(SchemeMethod):
                raise ValueError(f"unknown scheme {data!r}; one of {get_args(SchemeMethod)}")
            return {"method": data}
        return data

    @model_validator(mode="after")
    def _validate_combination(self) -> Scheme:
        if self.nonlinear_position != 0.5 and self.method != "ssfm":
            raise ValueError(
                f'nonlinear_position applies only to method="ssfm" ({self.method} '
                "composes symmetric steps)"
            )
        if self.method not in _SPLIT_STEP:
            if self.kerr_length != "step":
                raise ValueError(f'kerr_length="{self.kerr_length}" needs a split-step method')
            if self.kerr_filter_bandwidth_hz is not None:
                raise ValueError("a Kerr intensity filter needs a split-step method")
        if self.kerr_filter_bandwidth_hz is None and self.kerr_filter_shape != "gaussian":
            raise ValueError("kerr_filter_shape needs kerr_filter_bandwidth_hz")
        return self

    @property
    def is_split_step(self) -> bool:
        """``True`` for the split-step methods (exactly invertible, :math:`N` is a phase)."""
        return self.method in _SPLIT_STEP

    @property
    def order(self) -> int:
        """Global order of accuracy in the step for the unfiltered equation."""
        if self.method == "ssfm":
            return 2 if self.nonlinear_position == 0.5 else 1
        return 4

    @property
    def is_default(self) -> bool:
        """``True`` for the plain symmetric split step (the fast uniform path applies)."""
        return self == _DEFAULT


_DEFAULT = Scheme()
