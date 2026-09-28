"""Fiber span: parameters + propagation through a configured backend.

A :class:`Fiber` is pure configuration (a frozen model), so one instance can
be shared across simulations and threads. Numerical work is delegated to the
:class:`~hpcom.backends.base.Propagator` named by :attr:`Fiber.backend`
(see :mod:`hpcom.backends`); the equation solved is documented in
:doc:`/physics`.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from hpcom._model import HpcomModel
from hpcom._typing import ComplexArray, Direction, Polarization, StepDistribution
from hpcom.backends import get_propagator
from hpcom.backends.base import step_boundaries_m
from hpcom.units import dispersion_slope_to_beta3, dispersion_to_beta2


class Fiber(HpcomModel):
    r"""A single span of optical fiber.

    Units are SI except where the field name says otherwise.

    Attributes
    ----------
    length_km
        Span length ``L`` in km.
    alpha_db_per_km
        Power attenuation in dB/km (``P(L) = P(0)·10^(−α_dB L/10)``).
    beta1_s_per_m
        Group delay per metre β₁ in s/m *relative to the simulation's
        retarded-time frame* — 0 in normal use (the frame moves with the
        carrier's group velocity). A nonzero value delays the field by
        ``β₁L``; digital back-propagation uses it to evaluate the dispersion
        of a WDM channel at the channel's own frequency.
    beta2_s2_per_m
        Group-velocity dispersion β₂ in s²/m (negative = anomalous; SSMF ≈
        −2.17e-26). Use :func:`hpcom.units.dispersion_to_beta2` to convert
        from ps/(nm·km).
    beta3_s3_per_m
        Third-order dispersion β₃ in s³/m (SSMF ≈ 1.3e-40).
    gamma_per_w_per_km
        Kerr nonlinearity γ in 1/(W·km). For ``polarization="dual"`` the
        Manakov equation uses (8/9)·γ.
    n_steps
        Number of nonlinear steps ``K`` of the split-step solver for this
        span. Ignored by the ``"numpy-linear"`` backend.
    step_distribution
        Placement of the ``K`` steps: ``"uniform"`` (``h = L/K``) or
        ``"logarithmic"`` — short steps where the power is high, each step
        carrying the same share of :math:`\int e^{-k\alpha z}dz` (see
        :attr:`step_boundaries_km` and ``docs/numerics.rst``).
    step_log_factor
        :math:`k` of the logarithmic placement: ``1`` equal nonlinear phase
        per step (Bosco et al. 2000), ``1/3`` constant local error of the
        symmetric scheme (Zhang & Hayee 2008); ``→ 0`` tends to uniform. Only
        allowed with ``step_distribution="logarithmic"``.
    polarization
        ``"single"`` → scalar NLSE on fields of shape ``(..., n)``;
        ``"dual"`` → Manakov equation on fields of shape ``(..., 2, n)``.
    backend
        Registered propagator name (``"numpy"``, ``"tensorflow"``,
        ``"numpy-linear"``, …); see :mod:`hpcom.backends`.
    """

    length_km: float = Field(gt=0)
    alpha_db_per_km: float = Field(default=0.2, ge=0)
    beta1_s_per_m: float = 0.0
    beta2_s2_per_m: float = -2.16e-26
    beta3_s3_per_m: float = 0.0
    gamma_per_w_per_km: float = Field(default=1.27, ge=0)
    n_steps: int = Field(default=100, ge=1)
    step_distribution: StepDistribution = "uniform"
    step_log_factor: float = Field(default=1.0, gt=0)
    polarization: Polarization = "single"
    backend: str = "numpy"

    @classmethod
    def ssmf(
        cls,
        length_km: float,
        *,
        n_steps: int = 100,
        polarization: Polarization = "single",
        backend: str = "numpy",
        dispersion_ps_per_nm_per_km: float = 17.0,
        dispersion_slope_ps_per_nm2_per_km: float = 0.0,
        wavelength_m: float = 1550e-9,
    ) -> Fiber:
        """Standard single-mode fiber (ITU-T G.652) at ``wavelength_m``.

        α = 0.2 dB/km, γ = 1.27 1/(W·km), D = 17 ps/(nm·km). The dispersion
        slope defaults to 0 (β₃ = 0, as in the pre-v1 channel model); pass
        ``dispersion_slope_ps_per_nm2_per_km=0.057`` for the typical G.652
        value.
        """
        return cls(
            length_km=length_km,
            alpha_db_per_km=0.2,
            beta2_s2_per_m=dispersion_to_beta2(dispersion_ps_per_nm_per_km, wavelength_m),
            beta3_s3_per_m=(
                dispersion_slope_to_beta3(
                    dispersion_slope_ps_per_nm2_per_km, dispersion_ps_per_nm_per_km, wavelength_m
                )
                if dispersion_slope_ps_per_nm2_per_km
                else 0.0
            ),
            gamma_per_w_per_km=1.27,
            n_steps=n_steps,
            polarization=polarization,
            backend=backend,
        )

    # ---- derived quantities ---------------------------------------------

    @model_validator(mode="after")
    def _validate_steps(self) -> Fiber:
        if self.step_distribution == "uniform" and self.step_log_factor != 1.0:
            raise ValueError('step_log_factor applies only to step_distribution="logarithmic"')
        return self

    @property
    def step_boundaries_km(self) -> tuple[float, ...]:
        """Step boundaries :math:`0 = z_0 < … < z_K = L` in km (``step_distribution``)."""
        return tuple(float(z) / 1e3 for z in step_boundaries_m(self))

    @property
    def length_m(self) -> float:
        return self.length_km * 1e3

    @property
    def loss_db(self) -> float:
        """Span loss ``α_dB · L`` in dB (the EDFA gain that exactly compensates it)."""
        return self.alpha_db_per_km * self.length_km

    @property
    def effective_length_km(self) -> float:
        """Nonlinear effective length ``L_eff = (1 − e^{−αL}) / α`` in km."""
        alpha_per_km = self.alpha_db_per_km * np.log(10.0) / 10.0
        if alpha_per_km == 0:
            return self.length_km
        return float((1.0 - np.exp(-alpha_per_km * self.length_km)) / alpha_per_km)

    @property
    def accumulated_dispersion_s2(self) -> float:
        """``β₂ · L`` in s² (what a CD compensator must undo for this span)."""
        return self.beta2_s2_per_m * self.length_m

    @property
    def accumulated_beta3_s3(self) -> float:
        """``β₃ · L`` in s³."""
        return self.beta3_s3_per_m * self.length_m

    # ---- propagation ----------------------------------------------------

    def with_steps(
        self,
        n_steps: int | None = None,
        distribution: StepDistribution | None = None,
        log_factor: float | None = None,
    ) -> Fiber:
        """Copy with another step count or placement; ``None`` keeps this fiber's.

        Setting ``distribution`` resets ``step_log_factor`` to ``log_factor``
        (default 1); ``log_factor`` alone needs a logarithmic fiber.
        """
        changes: dict[str, object] = {}
        if n_steps is not None:
            changes["n_steps"] = n_steps
        if distribution is not None:
            changes["step_distribution"] = distribution
            changes["step_log_factor"] = 1.0 if log_factor is None else log_factor
        elif log_factor is not None:
            changes["step_log_factor"] = log_factor
        return self.replace(**changes) if changes else self

    def with_backend(self, backend: str) -> Fiber:
        """Return a copy of this fiber that propagates on ``backend``."""
        return self.replace(backend=backend)

    def propagate(
        self,
        field: ComplexArray,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        """Propagate ``field`` (sample period ``dt_s``) through this span."""
        return get_propagator(self.backend).propagate(field, self, dt_s=dt_s, direction=direction)

    def apply(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        """:class:`~hpcom._typing.LinkComponent` interface: forward :meth:`propagate`."""
        del rng  # deterministic
        return self.propagate(field, dt_s, direction="forward")

    def apply_inverse(self, field: ComplexArray, dt_s: float) -> ComplexArray:
        """Exact inverse of :meth:`apply` (backward propagation), for back-propagation."""
        return self.propagate(field, dt_s, direction="backward")
