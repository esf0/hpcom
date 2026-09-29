"""Fiber span: parameters + propagation through a configured backend.

A :class:`Fiber` is pure configuration (a frozen model), so one instance can
be shared across simulations and threads. Numerical work is delegated to the
:class:`~hpcom.backends.base.Propagator` named by :attr:`Fiber.backend`
(see :mod:`hpcom.backends`); the equation solved is documented in
:doc:`/physics`.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hpcom._model import FloatArrayField, HpcomModel
from hpcom._typing import ComplexArray, Direction, Polarization, StepDistribution
from hpcom.backends import get_propagator
from hpcom.backends.base import CPUFallbackWarning, step_boundaries_m
from hpcom.numerics import FWM_PHASE_RAD, Scheme, fwm_aware_first_step_m
from hpcom.raman import RamanResponse
from hpcom.units import SPEED_OF_LIGHT_M_PER_S, dispersion_slope_to_beta3, dispersion_to_beta2

__all__ = ["Fiber", "LossSpectrum"]


class LossSpectrum(HpcomModel):
    r"""Tabulated fiber loss :math:`\alpha_\mathrm{dB}(\lambda)`, for frequency-dependent loss.

    Linearly interpolated in wavelength, and constant beyond the first and
    last points. Attach it with :meth:`Fiber.with_loss_spectrum`, which also
    sets ``alpha_db_per_km`` to the loss at the fiber's carrier.

    Attributes
    ----------
    wavelengths_m
        Vacuum wavelengths, strictly increasing (at least two).
    alpha_db_per_km
        Power loss at each wavelength, dB/km (≥ 0).
    """

    wavelengths_m: FloatArrayField
    alpha_db_per_km: FloatArrayField

    @model_validator(mode="after")
    def _validate_table(self) -> LossSpectrum:
        lam, alpha = self.wavelengths_m, self.alpha_db_per_km
        if lam.ndim != 1 or lam.shape != alpha.shape or lam.size < 2:
            raise ValueError(
                "wavelengths_m and alpha_db_per_km must be 1-D of the same length (at least 2), "
                f"got {lam.shape} and {alpha.shape}"
            )
        if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(alpha))):
            raise ValueError("the loss spectrum must be finite")
        if lam[0] <= 0 or np.any(np.diff(lam) <= 0):
            raise ValueError("wavelengths_m must be positive and strictly increasing")
        if np.any(alpha < 0):
            raise ValueError("alpha_db_per_km must be >= 0")
        return self

    def at(self, wavelength_m: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Loss in dB/km at ``wavelength_m`` (``inf`` gives the last value)."""
        lam = np.asarray(wavelength_m, dtype=np.float64)
        return np.asarray(np.interp(lam, self.wavelengths_m, self.alpha_db_per_km), np.float64)


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
    scheme
        How each step is integrated — a :class:`~hpcom.numerics.Scheme` or a
        method name. Default: the symmetric split step with the Kerr phase at
        the step midpoint (second order). See :mod:`hpcom.numerics` and
        ``docs/numerics.rst``.
    step_log_factor
        :math:`k` of the logarithmic placement: ``1`` equal nonlinear phase
        per step (Bosco et al. 2000), ``1/3`` constant local error of the
        symmetric scheme (Zhang & Hayee 2008); ``→ 0`` tends to uniform. Only
        allowed with ``step_distribution="logarithmic"``.
    polarization
        ``"single"`` → scalar NLSE on fields of shape ``(..., n)``;
        ``"dual"`` → Manakov equation on fields of shape ``(..., 2, n)``.
    higher_order_betas
        Dispersion beyond β₃: ``(β₄, β₅, …)`` in sⁿ/m, continuing the
        Taylor expansion of the propagation constant about the carrier
        (generalised NLSE; ``docs/physics.rst``).
    wavelength_m
        Carrier (reference) wavelength λ₀: the zero of the simulation's
        baseband frequency grid, at which the β's and γ are given. Used by
        ``loss_spectrum`` and self-steepening.
    loss_spectrum
        Frequency-dependent loss (:class:`LossSpectrum`); ``alpha_db_per_km``
        must equal its value at λ₀ (:meth:`with_loss_spectrum` sets both).
        ``None``: the loss ``alpha_db_per_km`` at every frequency.
    raman
        Delayed Raman response (:class:`~hpcom.raman.RamanResponse`): the
        Kerr term becomes :math:`i\gamma A\,[(1-f_R)|A|^2 + f_R h_R*|A|^2]`.
        Every scheme supports it; single polarisation only (the
        polarisation-averaged vector Raman term is not implemented).
    self_steepening
        Include the shock term :math:`(1 + i\tau_\mathrm{shock}\partial_t)`
        on the Kerr term. The Kerr step is then no longer solvable exactly,
        so it needs ``scheme="rk4ip"``.
    shock_time_s
        :math:`\tau_\mathrm{shock}`; ``None``: :math:`1/\omega_0 = \lambda_0/2\pi c`.
        Only with ``self_steepening``.
    pmd_ps_per_sqrt_km
        PMD coefficient :math:`D_\mathrm{PMD}` in ps/√km (0: none; SSMF ≈
        0.05–0.1). Random birefringence by the coarse-step Manakov-PMD model:
        the span's mean differential group delay is :math:`D_\mathrm{PMD}\sqrt L`
        (:func:`~hpcom.backends.base.pmd_realization`). Dual polarisation,
        single core and a fixed-step scheme only.
    pmd_sections
        Birefringence sections per span (independent random axes); ≳ 50 for
        Maxwellian DGD statistics.
    pmd_seed
        Seed of this span's realisation; :meth:`hpcom.Link.from_spans` gives
        every span its own.
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
    scheme: Scheme = Scheme()
    polarization: Polarization = "single"
    higher_order_betas: tuple[float, ...] = ()
    wavelength_m: float = Field(default=1550e-9, gt=0)
    loss_spectrum: LossSpectrum | None = None
    raman: RamanResponse | None = None
    self_steepening: bool = False
    shock_time_s: float | None = Field(default=None, ge=0)
    pmd_ps_per_sqrt_km: float = Field(default=0.0, ge=0)
    pmd_sections: int = Field(default=100, ge=1)
    pmd_seed: int = Field(default=0, ge=0)
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
            wavelength_m=wavelength_m,
            backend=backend,
        )

    # ---- derived quantities ---------------------------------------------

    @model_validator(mode="after")
    def _validate_steps(self) -> Fiber:
        if self.step_distribution == "uniform" and self.step_log_factor != 1.0:
            raise ValueError('step_log_factor applies only to step_distribution="logarithmic"')
        return self

    @model_validator(mode="after")
    def _validate_gnlse(self) -> Fiber:
        if not all(math.isfinite(b) for b in self.higher_order_betas):
            raise ValueError("higher_order_betas must be finite")
        if self.raman is not None and self.polarization == "dual":
            raise ValueError(
                "Raman scattering is implemented for single polarisation (the scalar "
                "generalised NLSE); the polarisation-averaged vector Raman term of the "
                "Manakov equation is not"
            )
        if self.shock_time_s is not None and not self.self_steepening:
            raise ValueError("shock_time_s needs self_steepening=True")
        if self.self_steepening and self.scheme.method != "rk4ip":
            raise ValueError(
                "self_steepening makes the Kerr step non-integrable: use scheme='rk4ip' "
                f"(got {self.scheme.method!r})"
            )
        if self.pmd_ps_per_sqrt_km > 0:
            if self.polarization != "dual":
                raise ValueError("PMD needs polarization='dual' (the Manakov-PMD equation)")
            if self.n_cores > 1:
                raise ValueError("PMD is not implemented for multicore fibers")
            if self.scheme.method == "ssfm-local-error":
                raise ValueError(
                    "PMD needs a fixed-step scheme; the adaptive local-error method is not supported"
                )
        if self.loss_spectrum is not None:
            carrier = float(self.loss_spectrum.at(self.wavelength_m))
            if not math.isclose(carrier, self.alpha_db_per_km, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"alpha_db_per_km ({self.alpha_db_per_km}) must equal the loss spectrum at "
                    f"wavelength_m ({carrier} dB/km); use Fiber.with_loss_spectrum"
                )
        return self

    @property
    def mean_dgd_s(self) -> float:
        """Mean differential group delay of the span, :math:`D_\\mathrm{PMD}\\sqrt L`, in s."""
        return self.pmd_ps_per_sqrt_km * 1e-12 * math.sqrt(self.length_km)

    @property
    def carrier_frequency_hz(self) -> float:
        """Optical frequency ν₀ = c/λ₀ of the grid's 0 Hz."""
        return SPEED_OF_LIGHT_M_PER_S / self.wavelength_m

    @property
    def dispersion_betas(self) -> tuple[float, ...]:
        """``(β₁, β₂, β₃, β₄, …)`` in sⁿ/m."""
        return (
            self.beta1_s_per_m,
            self.beta2_s2_per_m,
            self.beta3_s3_per_m,
            *self.higher_order_betas,
        )

    @property
    def effective_shock_time_s(self) -> float:
        r""":math:`\tau_\mathrm{shock}` in use (0 without self-steepening)."""
        if not self.self_steepening:
            return 0.0
        if self.shock_time_s is not None:
            return float(self.shock_time_s)
        return 1.0 / (2.0 * math.pi * self.carrier_frequency_hz)

    @property
    def n_cores(self) -> int:
        """Number of cores (1; see :class:`~hpcom.multicore.MulticoreFiber`)."""
        return 1

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

    def with_first_step(
        self,
        first_step_km: float,
        distribution: StepDistribution | None = None,
        log_factor: float | None = None,
    ) -> Fiber:
        r"""Copy with the fewest steps whose first step is at most ``first_step_km``.

        Placement as in :meth:`with_steps` (``None`` keeps this fiber's). For
        uniform steps :math:`K = \lceil L/h_1 \rceil`; for logarithmic
        steps with factor :math:`k`,

        .. math:: K = \left\lceil \frac{1-e^{-k\alpha L}}{1-e^{-k\alpha h_1}} \right\rceil,

        so the steps then grow as :math:`h \propto e^{k\alpha z}`
        (``docs/numerics.rst``).
        """
        if not first_step_km > 0:
            raise ValueError(f"first_step_km must be positive, got {first_step_km}")
        placed = self.with_steps(distribution=distribution, log_factor=log_factor)
        length, h1 = placed.length_km * 1e3, first_step_km * 1e3
        rate = placed.step_log_factor * placed.alpha_db_per_km * math.log(10) / 10 / 1e3
        if h1 >= length:
            n_steps = 1
        elif placed.step_distribution == "uniform" or rate == 0.0:
            n_steps = math.ceil(length / h1 * (1 - 1e-12))
        else:
            n_steps = math.ceil(-math.expm1(-rate * length) / -math.expm1(-rate * h1) * (1 - 1e-12))
        return placed.replace(n_steps=max(n_steps, 1))

    def with_fwm_aware_steps(
        self,
        bandwidth_hz: float,
        rule: Literal["cle", "nlp"] = "cle",
        phase_rad: float | None = None,
    ) -> Fiber:
        """Steps by the FWM-aware rule of Musetti, Serena & Bononi (2018).

        First step :func:`~hpcom.numerics.fwm_aware_first_step_m` for the
        signal bandwidth ``bandwidth_hz`` (the whole WDM comb), then growing
        as constant local error (``"cle"``: logarithmic, :math:`k = 1/3`) or
        equal nonlinear phase (``"nlp"``: :math:`k = 1`). ``phase_rad``
        defaults to the paper's value for ≈ 0.01 dB SNR error
        (:data:`~hpcom.numerics.FWM_PHASE_RAD`: 20 rad CLE, 4 rad NLP).
        """
        phi = FWM_PHASE_RAD[rule] if phase_rad is None else phase_rad
        h1_m = fwm_aware_first_step_m(phi, bandwidth_hz, self.beta2_s2_per_m)
        return self.with_first_step(
            min(h1_m, self.length_m) / 1e3,
            distribution="logarithmic",
            log_factor=1.0 / 3.0 if rule == "cle" else 1.0,
        )

    def with_pmd(
        self, pmd_ps_per_sqrt_km: float, *, sections: int | None = None, seed: int | None = None
    ) -> Fiber:
        """Copy with polarisation-mode dispersion (``sections``/``seed`` default: this fiber's)."""
        changes: dict[str, object] = {"pmd_ps_per_sqrt_km": pmd_ps_per_sqrt_km}
        if sections is not None:
            changes["pmd_sections"] = sections
        if seed is not None:
            changes["pmd_seed"] = seed
        return self.replace(**changes)

    def with_loss_spectrum(self, spectrum: LossSpectrum | None) -> Fiber:
        """Copy with frequency-dependent loss; ``alpha_db_per_km`` becomes its value at λ₀.

        ``None`` removes the spectrum and keeps ``alpha_db_per_km``.
        """
        if spectrum is None:
            return self.replace(loss_spectrum=None)
        return self.replace(
            loss_spectrum=spectrum, alpha_db_per_km=float(spectrum.at(self.wavelength_m))
        )

    def dispersion_phase_per_m(self, frequency_offset_hz: float) -> float:
        r"""Phase :math:`D(\omega_k) = \sum_n \beta_n(-\omega_k)^n/n!` per metre of a tone at ``f_k``.

        The propagation constant, relative to the moving frame, of a CW
        tone at baseband frequency :math:`f_k` (:math:`\omega_k = 2\pi f_k`);
        see :meth:`recentred`.
        """
        w = 2.0 * math.pi * frequency_offset_hz
        return math.fsum(b * (-w) ** n / math.factorial(n) for n, b in _enumerate_betas(self))

    def recentred(self, frequency_offset_hz: float) -> Fiber:
        r"""This fiber as seen by a grid centred on baseband frequency ``f_k``.

        A field shifted down by :math:`f_k` (a WDM channel moved to 0 Hz)
        propagates in this fiber exactly as it does in the returned one,
        apart from the constant phase :meth:`dispersion_phase_per_m`, which
        commutes with every step. The Taylor series is re-expanded about
        :math:`\omega_k`,

        .. math:: \beta_m' = \sum_{n \ge m} \beta_n\,\frac{(-\omega_k)^{n-m}}{(n-m)!},

        the carrier moves to :math:`\nu_0 - f_k` (with the loss there, for a
        loss spectrum) and, with self-steepening, :math:`\gamma' = \gamma(1 -
        \tau\omega_k)`, :math:`\tau' = \tau/(1 - \tau\omega_k)`.
        """
        w = 2.0 * math.pi * frequency_offset_hz
        if w == 0.0:
            return self
        if self.pmd_ps_per_sqrt_km > 0:
            raise ValueError(
                "a fiber with PMD cannot be recentred on another frequency (its birefringence "
                "is not a Taylor series); back-propagate it full-field or without PMD"
            )
        betas = self.dispersion_betas
        shifted = [
            math.fsum(
                betas[n - 1] * (-w) ** (n - m) / math.factorial(n - m)
                for n in range(m, len(betas) + 1)
            )
            for m in range(1, len(betas) + 1)
        ]
        wavelength = SPEED_OF_LIGHT_M_PER_S / (self.carrier_frequency_hz - frequency_offset_hz)
        changes: dict[str, object] = {
            "beta1_s_per_m": shifted[0],
            "beta2_s2_per_m": shifted[1],
            "beta3_s3_per_m": shifted[2],
            "higher_order_betas": tuple(shifted[3:]),
            "wavelength_m": wavelength,
        }
        if self.loss_spectrum is not None:
            changes["alpha_db_per_km"] = float(self.loss_spectrum.at(wavelength))
        if self.self_steepening:
            factor = 1.0 - self.effective_shock_time_s * w
            changes["gamma_per_w_per_km"] = self.gamma_per_w_per_km * factor
            if self.shock_time_s is not None:
                changes["shock_time_s"] = self.shock_time_s / factor
        return self.replace(**changes)

    def with_backend(self, backend: str) -> Fiber:
        """Return a copy of this fiber that propagates on ``backend``."""
        return self.replace(backend=backend)

    def propagate(
        self,
        field: ComplexArray,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        """Propagate ``field`` (sample period ``dt_s``) through this span.

        If :attr:`backend` does not implement :attr:`scheme`, the span runs on
        the NumPy (CPU) backend and a :class:`~hpcom.backends.CPUFallbackWarning`
        is emitted.

        Raises
        ------
        FloatingPointError
            A finite field came out non-finite: the explicit scheme went
            unstable (steps too long for the field — typically RK4IP with
            strong self-steepening).
        """
        propagator = get_propagator(self.backend)
        if self.scheme.method not in propagator.supported_methods:
            warnings.warn(
                f"backend {self.backend!r} does not implement scheme "
                f"{self.scheme.method!r}; running this span on the NumPy (CPU) backend",
                CPUFallbackWarning,
                stacklevel=2,
            )
            propagator = get_propagator("numpy")
        out = propagator.propagate(field, self, dt_s=dt_s, direction=direction)
        if not np.all(np.isfinite(out)) and np.all(np.isfinite(field)):
            raise FloatingPointError(
                f"propagation through a {self.length_km:g} km span produced non-finite values: "
                f"scheme {self.scheme.method!r} is unstable with n_steps={self.n_steps} for "
                "this field; increase n_steps"
            )
        return out

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


def _enumerate_betas(fiber: Fiber) -> list[tuple[int, float]]:
    """``[(n, β_n)]`` for n = 1, 2, …"""
    return list(enumerate(fiber.dispersion_betas, start=1))
