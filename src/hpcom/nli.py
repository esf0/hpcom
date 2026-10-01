r"""Nonlinear interference (NLI) models: GN, EGN and the first-order perturbation.

Every model here is first order in the Kerr nonlinearity: the field that
reaches the receiver is the linear solution plus the nonlinear interference
it generates, :math:`A = A_\mathrm{lin} + \delta A`, and the received
signal-to-noise ratio is

.. math:: \mathrm{SNR} = \frac{P}{P_\mathrm{ASE} + \eta P^3},

with :math:`P` the launch power per channel and :math:`\eta` the NLI
coefficient of the channel under test (CUT), counted, like the ASE, in its
matched filter.

* :func:`gn_psd`, :func:`nli_coefficient` (``model="gn"``) — the Gaussian-noise
  (GN) reference formula [Poggiolini2014]_, integrated numerically for any
  link of :class:`~hpcom.Fiber` spans and :class:`~hpcom.EDFA` amplifiers
  (coherent accumulation over the spans) and any channel plan of
  root-raised-cosine channels. It treats the transmitted signal as Gaussian
  noise and so overestimates the NLI of QAM.
* ``model="egn"`` — the GN result minus the closed-form EGN correction of
  [Poggiolini2015]_ for the non-Gaussian modulation (dual polarisation,
  asymptotic in the number of spans).
* :func:`first_order_nli` — the first-order perturbation evaluated
  numerically on the actual transmitted signal and scored by the actual
  receiver: exact to first order for any modulation, pulse and link (the
  reference the GN and EGN models approximate).
* :func:`predict_snr` — SNR with ASE, and the optimum launch power
  :math:`P_\mathrm{opt} = (P_\mathrm{ASE}/2\eta)^{1/3}`.

All are referred to the link input: for a link whose amplifiers restore
the launch power they equal the values at the receiver.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field

from hpcom._model import HpcomModel
from hpcom._typing import ComplexArray
from hpcom.amplifier import EDFA
from hpcom.backends.base import (
    attenuation_per_m,
    effective_gamma_per_m,
    linear_operator_exponent,
)
from hpcom.fiber import Fiber
from hpcom.link import Link
from hpcom.transmitter import Transmitter
from hpcom.units import w_to_dbm

__all__ = [
    "FirstOrderNLI",
    "NLIModel",
    "SNRPrediction",
    "first_order_nli",
    "gn_psd",
    "modulation_phi",
    "nli_coefficient",
    "predict_snr",
]

NLIModel = Literal["gn", "egn"]


# ---------------------------------------------------------------------------
# Link and channel description
# ---------------------------------------------------------------------------


class _Profile(HpcomModel):
    """The fibers of a link as seen by the NLI models (arrays, one entry per fiber).

    ``power`` is the signal power at each fiber's input relative to the
    launch; ``dispersion`` the accumulated :math:`\\sum\\beta_2L` before it.
    """

    gamma: tuple[float, ...]  # 1/(W·m), without the Manakov 8/9
    alpha: tuple[float, ...]  # power attenuation, 1/m
    beta2: tuple[float, ...]  # s²/m
    length: tuple[float, ...]  # m
    power: tuple[float, ...]
    dispersion: tuple[float, ...]  # s²
    amplifier_gains: tuple[float, ...]  # linear power gain of each EDFA
    amplifier_power_after: tuple[float, ...]  # signal power after it, relative to the launch
    amplifier_ase_psd: tuple[float, ...]  # W/Hz per polarisation (0: noiseless)


def _profile(link: Link, frequency_offset_hz: float = 0.0) -> _Profile:
    gamma, alpha, beta2, length, power, dispersion = [], [], [], [], [], []
    gains, after, ase = [], [], []
    p, b = 1.0, 0.0
    for el in link.elements:
        if isinstance(el, EDFA):
            p *= el.gain_linear
            gains.append(el.gain_linear)
            after.append(p)
            ase.append(0.0 if el.is_noiseless else el.ase_psd_per_polarization_w_per_hz)
            continue
        if type(el) is not Fiber:
            raise TypeError(
                f"the NLI models support links of Fiber and EDFA elements, got {type(el).__name__}"
            )
        unsupported = [
            name
            for name, present in (
                ("a Raman response", el.raman is not None),
                ("self-steepening", el.self_steepening),
                ("PMD", el.pmd_ps_per_sqrt_km > 0),
                ("nonlinear gain", el.nonlinear_gain_per_w_per_km != 0.0),
                ("dispersion beyond β3", any(el.higher_order_betas)),
            )
            if present
        ]
        if unsupported:
            raise ValueError(f"the NLI models do not include {', '.join(unsupported)}")
        omega = 2.0 * math.pi * frequency_offset_hz
        local_beta2 = el.beta2_s2_per_m - el.beta3_s3_per_m * omega  # D''(ω) at the channel
        a = attenuation_per_m(el)
        gamma.append(el.gamma_per_w_per_km * 1e-3)
        alpha.append(a)
        beta2.append(local_beta2)
        length.append(el.length_km * 1e3)
        power.append(p)
        dispersion.append(b)
        p *= math.exp(-a * el.length_km * 1e3)
        b += local_beta2 * el.length_km * 1e3
    if not gamma:
        raise ValueError("the link has no fiber")
    return _Profile(
        gamma=tuple(gamma),
        alpha=tuple(alpha),
        beta2=tuple(beta2),
        length=tuple(length),
        power=tuple(power),
        dispersion=tuple(dispersion),
        amplifier_gains=tuple(gains),
        amplifier_power_after=tuple(after),
        amplifier_ase_psd=tuple(ase),
    )


def _raised_cosine(f: npt.NDArray[np.float64], rs: float, beta: float) -> npt.NDArray[np.float64]:
    """Raised-cosine spectrum, peak 1, :math:`\\int = R_s` (the RRC matched filter's |H|²)."""
    x = np.abs(f)
    lo, hi = (1.0 - beta) * rs / 2.0, (1.0 + beta) * rs / 2.0
    out = np.where(x <= lo, 1.0, 0.0)
    if beta > 0:
        edge = (x > lo) & (x <= hi)
        out = np.where(edge, 0.5 * (1.0 + np.cos(math.pi / (beta * rs) * (x - lo))), out)
    return np.asarray(out, dtype=np.float64)


class _Channels(HpcomModel):
    """The transmitted channels: centre frequencies, powers, symbol rate, roll-off."""

    frequencies_hz: tuple[float, ...]
    powers_w: tuple[float, ...]
    symbol_rate_hz: float
    roll_off: float
    n_polarizations: int

    @property
    def half_width_hz(self) -> float:
        return self.symbol_rate_hz * (1.0 + self.roll_off) / 2.0

    def psd(self, f: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Total launch PSD (both polarisations), W/Hz."""
        out = np.zeros(np.shape(f))
        for fc, p in zip(self.frequencies_hz, self.powers_w, strict=True):
            out += (
                p / self.symbol_rate_hz * _raised_cosine(f - fc, self.symbol_rate_hz, self.roll_off)
            )
        return out


def _channels(transmitter: Transmitter) -> _Channels:
    rs = getattr(transmitter, "symbol_rate_hz", None)
    beta = getattr(transmitter, "roll_off", None)
    if rs is None or beta is None:
        raise TypeError(
            f"the NLI models need root-raised-cosine channels; {type(transmitter).__name__} "
            "has none"
        )
    wdm = getattr(transmitter, "wdm", None)
    freqs = (0.0,) if wdm is None else tuple(float(f) for f in wdm.channel_frequencies_hz())
    if transmitter.n_cores != 1:
        raise ValueError("the NLI models are single-core")
    p = transmitter.launch_power_per_channel_w
    return _Channels(
        frequencies_hz=freqs,
        powers_w=(p,) * len(freqs),
        symbol_rate_hz=float(rs),
        roll_off=float(beta),
        n_polarizations=transmitter.n_polarizations,
    )


def modulation_phi(transmitter: Transmitter) -> float:
    r""":math:`\Phi = 2 - \mu_4/\mu_2^2` of the transmitted symbols (per polarisation).

    1 for QPSK, 17/25 for 16-QAM, 13/21 for 64-QAM, 0 for Gaussian symbols
    — the strength of the EGN correction [Poggiolini2015]_.
    """
    cfg = transmitter.resolved_constellation
    pmf = transmitter.dm.empirical_pmf if transmitter.dm is not None else cfg.pmf
    m = np.abs(np.asarray(cfg.points)) ** 2
    mu2 = float(np.dot(pmf, m))
    mu4 = float(np.dot(pmf, m**2))
    return 2.0 - mu4 / mu2**2


# ---------------------------------------------------------------------------
# GN reference formula
# ---------------------------------------------------------------------------


def _span_terms(prof: _Profile, w: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    r"""Per-fiber link-function terms :math:`c_s(w)`, shape ``(n_fibers, len(w))``.

    :math:`c_s = \gamma_sp_s\,e^{-i\Omega B_s}\,(1 - e^{-(\alpha_s + i\Omega\beta_{2,s})L_s})/(\alpha_s + i\Omega\beta_{2,s})`
    with :math:`\Omega = 4\pi^2 w`; the link function is :math:`\sum_s c_s`.
    """
    omega = 4.0 * math.pi**2 * np.asarray(w)[None, :]
    g, a, b2, length, p, disp = (
        np.asarray(x)[:, None]
        for x in (prof.gamma, prof.alpha, prof.beta2, prof.length, prof.power, prof.dispersion)
    )
    rate = a + 1j * omega * b2
    small = np.abs(rate * length) < 1e-8
    safe = np.where(small, 1.0, rate)
    integral = np.where(small, length, -np.expm1(-safe * length) / safe)
    out: npt.NDArray[np.complex128] = g * p * np.exp(-1j * omega * disp) * integral
    return out


def _incoherent_kernel(
    prof: _Profile, w: npt.NDArray[np.float64], cell: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Σ_s |c_s|², with the in-span oscillation averaged over quadrature cells of width ``cell``."""
    omega = 4.0 * math.pi**2 * np.asarray(w)[None, :]
    g, a, b2, length, p = (
        np.asarray(x)[:, None]
        for x in (prof.gamma, prof.alpha, prof.beta2, prof.length, prof.power)
    )
    theta = omega * b2 * length
    decay = np.exp(-a * length)
    spread = 4.0 * math.pi**2 * np.abs(b2) * length * np.asarray(cell)[None, :]
    numerator = 1.0 + decay**2 - 2.0 * decay * np.cos(theta) * np.sinc(spread / (2.0 * math.pi))
    denominator = a**2 + (omega * b2) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        lossless = np.where(
            denominator == 0, length**2, numerator / np.where(denominator == 0, 1, denominator)
        )
    out: npt.NDArray[np.float64] = np.sum((g * p) ** 2 * lossless, axis=0)
    return out


def _quadrature(
    prof: _Profile, w_max: float, coherent: bool, max_points: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Nodes |w|, trapezoid weights and the kernel |Σ c_s|² on them.

    Three regions: logarithmic near 0 (the log singularity of S and the
    peak of the kernel), uniform in w where the coherent oscillations of
    the span sum must be resolved (12 points per period), logarithmic
    again beyond ``max_points`` with the incoherent, cell-averaged kernel.
    """
    theta = (
        4.0
        * math.pi**2
        * max(
            abs(d + b * length)
            for d, b, length in zip(prof.dispersion, prof.beta2, prof.length, strict=True)
        )
    )
    period = math.inf if theta == 0 else 2.0 * math.pi / theta
    ds = 2e-3
    dw = period / 12.0
    w_min = (1e-9 * period) if math.isfinite(period) else w_max * 1e-30
    w_1 = min(w_max, dw / ds) if math.isfinite(period) else w_max
    nodes, weights, kernel = [], [], []

    def log_region(lo: float, hi: float, incoherent: bool) -> None:
        if hi <= lo:
            return
        n = max(2, math.ceil(math.log(hi / lo) / ds) + 1)
        s = np.linspace(math.log(lo), math.log(hi), n)
        w = np.exp(s)
        wt = np.full(n, (s[1] - s[0])) * w
        wt[0] /= 2.0
        wt[-1] /= 2.0
        nodes.append(w)
        weights.append(wt)
        if incoherent:
            kernel.append(_incoherent_kernel(prof, w, w * (s[1] - s[0])))
        else:
            kernel.append(_coherent(w))

    def _coherent(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.empty(w.size)
        step = max(1, 2_000_000 // len(prof.gamma))
        for i in range(0, w.size, step):
            terms = _span_terms(prof, w[i : i + step])
            out[i : i + step] = (
                np.abs(terms.sum(axis=0)) ** 2 if coherent else np.sum(np.abs(terms) ** 2, axis=0)
            )
        return out

    log_region(w_min, w_1, incoherent=False)
    w_c = w_1
    if w_1 < w_max:
        n = min(max_points, math.ceil((w_max - w_1) / dw) + 1)
        w_c = min(w_max, w_1 + (n - 1) * dw)
        w = np.linspace(w_1, w_c, n)
        wt = np.full(n, w[1] - w[0] if n > 1 else 0.0)
        wt[0] /= 2.0
        wt[-1] /= 2.0
        nodes.append(w)
        weights.append(wt)
        kernel.append(_coherent(w))
    log_region(w_c, w_max, incoherent=True)
    return np.concatenate(nodes), np.concatenate(weights), np.concatenate(kernel)


def _s_table(
    channels: _Channels, f: float, w_abs: npt.NDArray[np.float64], n_t: int = 1000
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r""":math:`S_f(\pm w) = \int du/|u|\,G(f+u)G(f+w/u)G(f+u+w/u)` for ``w_abs`` > 0.

    With :math:`u = \sigma e^t`, :math:`du/|u| = dt`; trapezoid in ``t``
    over the support of the integrand.
    """
    lo = min(channels.frequencies_hz) - channels.half_width_hz
    hi = max(channels.frequencies_hz) + channels.half_width_hz
    width = max(hi - f, f - lo)
    n_grid = 200_001
    step = (hi - lo) / (n_grid - 1)
    table = np.append(channels.psd(np.linspace(lo, hi, n_grid)), 0.0)

    def psd(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # linear interpolation on the uniform grid (0 outside the band)
        pos = (x - lo) / step
        i = np.floor(pos)
        frac = pos - i
        inside = (i >= 0) & (i < n_grid - 1)
        k = np.where(inside, i, n_grid).astype(np.int64)
        out: npt.NDArray[np.float64] = np.where(
            inside, table[k] * (1.0 - frac) + table[np.minimum(k + 1, n_grid)] * frac, 0.0
        )
        return out

    out = []
    for sign_w in (1.0, -1.0):
        s = np.zeros(w_abs.size)
        for i in range(0, w_abs.size, 50):
            w = sign_w * w_abs[i : i + 50, None]
            t_lo = np.log(np.abs(w) / width)
            t_hi = math.log(width)
            tau = np.linspace(0.0, 1.0, n_t)[None, :]
            t = t_lo + tau * (t_hi - t_lo)
            dt = (t_hi - t_lo) / (n_t - 1)
            for sigma in (1.0, -1.0):
                u = sigma * np.exp(t)
                v = w / u
                vals = psd(f + u) * psd(f + v) * psd(f + u + v)
                vals[:, 0] /= 2.0
                vals[:, -1] /= 2.0
                s[i : i + 50] += np.where(t_hi > t_lo[:, 0], vals.sum(axis=1) * dt[:, 0], 0.0)
        out.append(s)
    return out[0], out[1]


def gn_psd(
    link: Link,
    transmitter: Transmitter,
    frequencies_hz: npt.ArrayLike,
    *,
    coherent: bool = True,
    max_points: int = 400_000,
) -> npt.NDArray[np.float64]:
    r"""NLI power spectral density of the GN model, W/Hz, referred to the link input.

    .. math::

       G_\mathrm{NLI}(f) = c\iint G(f_1)G(f_2)G(f_1 + f_2 - f)
          \Big|\sum_s c_s\big((f_1 - f)(f_2 - f)\big)\Big|^2 df_1\,df_2,

    with the launch PSD :math:`G` of all channels (both polarisations), the
    link function of the spans :math:`c_s` (``docs/physics.rst``) and
    :math:`c = 16/27` for Manakov dual polarisation, 2 for one polarisation.
    Since the link function depends on :math:`w = (f_1 - f)(f_2 - f)`
    only, the double integral is evaluated as
    :math:`\int|\sum_s c_s(w)|^2S_f(w)\,dw` with
    :math:`S_f(w) = \int G(f + u)G(f + w/u)G(f + u + w/u)\,du/|u|`.
    ``coherent=False`` sums the spans' NLI powers instead (the incoherent GN
    model). Dispersion is :math:`\beta_2` at each channel (with
    :math:`\beta_3`, the local value at the channel under test).
    """
    prof = _profile(link)
    channels = _channels(transmitter)
    c = 16.0 / 27.0 if channels.n_polarizations == 2 else 2.0
    lo = min(channels.frequencies_hz) - channels.half_width_hz
    hi = max(channels.frequencies_hz) + channels.half_width_hz
    w_max = (hi - lo) ** 2
    nodes, weights, kernel = _quadrature(prof, w_max, coherent, max_points)
    w_tab = np.geomspace(nodes.min(), w_max, 800)  # S is smooth in log w (1e-5 dB at 600)
    out = []
    for f in np.atleast_1d(np.asarray(frequencies_hz, dtype=np.float64)):
        s_plus, s_minus = _s_table(channels, float(f), w_tab)
        x = np.log(nodes)
        s = np.interp(x, np.log(w_tab), s_plus) + np.interp(x, np.log(w_tab), s_minus)
        out.append(c * float(np.sum(weights * kernel * s)))
    return np.asarray(out, dtype=np.float64)


def _egn_correction_psd(link: Link, transmitter: Transmitter, channel: int) -> float:
    r"""Closed-form EGN correction :math:`G_\mathrm{corr}` of [Poggiolini2015]_, Eq. 2 (W/Hz).

    .. math::

       G_\mathrm{corr} = \frac{40}{81}\frac{\gamma^2P_mN_s\bar L_\mathrm{eff}^2}
          {R_m\pi|\beta_2|\bar L_s}\Big[\sum_{n\ne m}\Phi_n\frac{P_n^2}{R_n|f_n - f_m|}
          + \Phi_m\frac{2P_m^2}{R_m^2}\Big]

    (dual polarisation, identical spans: averages over the link's fibers).
    """
    prof = _profile(link)
    channels = _channels(transmitter)
    if channels.n_polarizations != 2:
        raise ValueError("the closed-form EGN correction is derived for dual polarisation")
    phi = modulation_phi(transmitter)
    n_s = len(prof.length)
    length = float(np.mean(prof.length))
    alpha = np.asarray(prof.alpha)
    l_eff = float(
        np.mean(
            np.where(
                alpha > 0,
                -np.expm1(-alpha * np.asarray(prof.length)) / np.where(alpha > 0, alpha, 1),
                prof.length,
            )
        )
    )
    beta2 = abs(float(np.mean(prof.beta2)))
    gamma = float(np.mean(prof.gamma))
    rs = channels.symbol_rate_hz
    p = channels.powers_w
    f = channels.frequencies_hz
    m = channel
    xci = sum(phi * p[n] ** 2 / (rs * abs(f[n] - f[m])) for n in range(len(f)) if n != m)
    sci = phi * 2.0 * p[m] ** 2 / rs**2
    return (
        40.0
        / 81.0
        * gamma**2
        * p[m]
        * n_s
        * l_eff**2
        / (rs * math.pi * beta2 * length)
        * (xci + sci)
    )


def nli_coefficient(
    link: Link,
    transmitter: Transmitter,
    *,
    model: NLIModel = "gn",
    channel: int | None = None,
    coherent: bool = True,
    nodes_per_region: int = 6,
) -> float:
    r"""NLI coefficient :math:`\eta` (1/W²) of a channel: :math:`P_\mathrm{NLI} = \eta P^3`.

    :math:`P_\mathrm{NLI} = \int G_\mathrm{NLI}(f)\,|H(f - f_m)|^2df`, the NLI
    in the channel's matched filter (raised-cosine :math:`|H|^2`), with
    every channel at the transmitter's launch power :math:`P`. ``channel``
    defaults to the centre one. The integral is composite Gauss–Legendre
    with ``nodes_per_region`` nodes on each region of the raised cosine
    (the two roll-offs and the flat top): :math:`G_\mathrm{NLI}` falls
    steeply towards the channel edges. ``model="egn"`` subtracts the
    closed-form correction (flat over the channel).
    """
    channels = _channels(transmitter)
    m = len(channels.frequencies_hz) // 2 if channel is None else channel
    if not 0 <= m < len(channels.frequencies_hz):
        raise ValueError(f"channel {m} is outside [0, {len(channels.frequencies_hz)})")
    fc = channels.frequencies_hz[m]
    rs, beta = channels.symbol_rate_hz, channels.roll_off
    lo, hi = (1.0 - beta) * rs / 2.0, (1.0 + beta) * rs / 2.0
    regions = [(-lo, lo)] if beta == 0 else [(-hi, -lo), (-lo, lo), (lo, hi)]
    x, w = np.polynomial.legendre.leggauss(nodes_per_region)
    offsets = np.concatenate([(a + b) / 2.0 + (b - a) / 2.0 * x for a, b in regions])
    weights = np.concatenate([(b - a) / 2.0 * w for a, b in regions])
    values = gn_psd(link, transmitter, fc + offsets, coherent=coherent)
    p_nli = float(np.sum(weights * values * _raised_cosine(offsets, rs, beta)))
    if model == "egn":
        p_nli -= _egn_correction_psd(link, transmitter, m) * channels.symbol_rate_hz
    elif model != "gn":
        raise ValueError(f"model must be 'gn' or 'egn', got {model!r}")
    return p_nli / channels.powers_w[m] ** 3


class SNRPrediction(HpcomModel):
    """Output of :func:`predict_snr` (powers per channel, referred to the link input)."""

    model: NLIModel
    launch_power_w: float
    eta_per_w2: float
    ase_power_w: float
    nli_power_w: float
    snr_db: float
    snr_ase_db: float
    snr_nli_db: float
    optimal_launch_power_dbm: float
    optimal_snr_db: float


def predict_snr(
    link: Link,
    transmitter: Transmitter,
    *,
    model: NLIModel = "gn",
    channel: int | None = None,
    coherent: bool = True,
    eta_per_w2: float | None = None,
) -> SNRPrediction:
    r"""SNR of a channel, :math:`P/(P_\mathrm{ASE} + \eta P^3)`, and its optimum.

    :math:`P_\mathrm{ASE} = n_\mathrm{pol}R_s\sum_k S_k/p_k`: each EDFA's ASE
    PSD per polarisation :math:`S_k` in the matched filter's noise
    bandwidth :math:`R_s`, referred to the input by the signal power
    :math:`p_k` after it — the convention of :class:`~hpcom.EDFA` and of the
    receiver's SNR. ``eta_per_w2`` overrides the model's coefficient (e.g.
    with :func:`first_order_nli`'s).
    """
    prof = _profile(link)
    channels = _channels(transmitter)
    eta = (
        nli_coefficient(link, transmitter, model=model, channel=channel, coherent=coherent)
        if eta_per_w2 is None
        else eta_per_w2
    )
    p = channels.powers_w[0]
    ase = (
        channels.n_polarizations
        * channels.symbol_rate_hz
        * sum(
            s / after
            for s, after in zip(prof.amplifier_ase_psd, prof.amplifier_power_after, strict=True)
        )
    )
    nli = eta * p**3
    p_opt = (ase / (2.0 * eta)) ** (1.0 / 3.0) if ase > 0 else math.inf

    def db(x: float) -> float:
        return 10.0 * math.log10(x) if x > 0 else math.inf

    return SNRPrediction(
        model=model,
        launch_power_w=p,
        eta_per_w2=eta,
        ase_power_w=ase,
        nli_power_w=nli,
        snr_db=db(p / (ase + nli)),
        snr_ase_db=db(p / ase) if ase > 0 else math.inf,
        snr_nli_db=db(p / nli),
        optimal_launch_power_dbm=w_to_dbm(p_opt) if math.isfinite(p_opt) else math.inf,
        optimal_snr_db=db(p_opt / (1.5 * ase)) if math.isfinite(p_opt) else math.inf,
    )


# ---------------------------------------------------------------------------
# First-order perturbation, evaluated numerically
# ---------------------------------------------------------------------------


class FirstOrderNLI(HpcomModel):
    """Output of :func:`first_order_nli`.

    Attributes
    ----------
    eta_per_w2
        NLI coefficient (mean over blocks).
    eta_stderr_per_w2
        Its standard error over the blocks.
    snr_nli_db
        :math:`P/(\\eta P^3)` at the transmitter's launch power.
    linear_floor_db
        SNR of the linear (NLI-free) received signal: the receiver's own
        floor, removed from the estimate.
    """

    eta_per_w2: float
    eta_stderr_per_w2: float = Field(ge=0)
    snr_nli_db: float
    linear_floor_db: float


def _first_order_fields(
    field: ComplexArray, link: Link, dt_s: float, nodes_per_fiber: int
) -> tuple[ComplexArray, ComplexArray]:
    r"""Linear output and first-order NLI field :math:`\delta A` at the link end.

    Per fiber, :math:`\delta A(L) = \int_0^L e^{\hat L(L-z)}\,
    \mathcal F[i\gamma_\mathrm{eff}P(z)A(z)]\,dz` with :math:`A(z)` the
    linear field; Gauss–Legendre in :math:`\zeta = (1 - e^{-\alpha z})/\alpha`,
    in which the integrand has no loss decay. :math:`\delta A` from earlier
    fibers propagates linearly through the rest of the link.
    """
    a = np.fft.fft(np.asarray(field, dtype=np.complex128), axis=-1)
    delta = np.zeros_like(a)
    x, wts = np.polynomial.legendre.leggauss(nodes_per_fiber)
    n = a.shape[-1]
    for el in link.elements:
        if isinstance(el, EDFA):
            g = math.sqrt(el.gain_linear)
            a, delta = a * g, delta * g
            continue
        assert isinstance(el, Fiber)
        exponent = linear_operator_exponent(n, dt_s, el)
        length = el.length_km * 1e3
        alpha = attenuation_per_m(el)
        gamma = effective_gamma_per_m(el)
        dual = el.polarization == "dual"
        zeta_max = -math.expm1(-alpha * length) / alpha if alpha > 0 else length
        zeta = (x + 1.0) * zeta_max / 2.0
        z = -np.log1p(-alpha * zeta) / alpha if alpha > 0 else zeta
        weights = wts * zeta_max / 2.0 * np.exp(alpha * z)  # dz = e^{αz} dζ
        contribution = np.zeros_like(a)
        for zj, wj in zip(z, weights, strict=True):
            aj = np.fft.ifft(a * np.exp(exponent * zj), axis=-1)
            p = np.abs(aj) ** 2
            if dual:
                p = p.sum(axis=-2, keepdims=True)
            nonlinear = np.fft.fft(1j * gamma * p * aj, axis=-1)
            contribution += wj * nonlinear * np.exp(exponent * (length - zj))
        transfer = np.exp(exponent * length)
        a = a * transfer
        delta = delta * transfer + contribution
    return np.fft.ifft(a, axis=-1), np.fft.ifft(delta, axis=-1)


def first_order_nli(
    link: Link,
    transmitter: Transmitter,
    *,
    channel: int | None = None,
    n_blocks: int = 4,
    seed: int = 0,
    gaussian: bool = False,
    nodes_per_fiber: int = 64,
) -> FirstOrderNLI:
    r"""NLI coefficient from the first-order perturbation of the actual signal.

    For each of ``n_blocks`` independent blocks of data: the linear field
    and the first-order NLI field :math:`\delta A` (:func:`_first_order_fields`)
    at the link end; the transmitter's matched receiver (CD compensation,
    matched filter, complex gain — WDM: the channel ``channel``, default the
    centre one) scores :math:`A_\mathrm{lin} + \delta A` exactly as
    :class:`~hpcom.Simulation` does, and
    :math:`\eta = (1/\mathrm{SNR} - 1/\mathrm{SNR}_\mathrm{lin})/(sP)^2`, with the
    receiver's floor :math:`\mathrm{SNR}_\mathrm{lin}` (no NLI) removed. The
    receiver sees :math:`A_\mathrm{lin} + s\,\delta A` with :math:`s` chosen so
    that :math:`\|s\,\delta A\| = 10^{-4}\|A_\mathrm{lin}\|`: the first-order
    field turns the mean nonlinear rotation :math:`e^{i\phi}` into
    :math:`1 + i\phi`, and at the launch power its amplitude
    :math:`|1 + i\phi|` would be mistaken for signal gain (≈1 dB for three
    channels over five spans at 0 dBm). The result does not depend on the
    launch power. ``gaussian`` replaces the data by complex Gaussian symbols (the
    GN model's assumption). The fibers' own step settings are not used; the
    z-integral takes ``nodes_per_fiber`` Gauss–Legendre nodes.
    """
    from hpcom.simulation import Simulation
    from hpcom.wdm import WDMTransmitter

    _profile(link)  # validates the elements
    channels = _channels(transmitter)
    overrides: dict[str, object] = {"cd_compensation_s2": link.total_dispersion_s2}
    if isinstance(transmitter, WDMTransmitter):
        m = len(channels.frequencies_hz) // 2 if channel is None else channel
        overrides["channels"] = (m,)
    receiver = transmitter.matched_receiver(**overrides)
    sim = Simulation(transmitter=transmitter, link=link, receiver=receiver, seed=seed)
    p = transmitter.launch_power_per_channel_w
    etas, floors = [], []
    for b in range(n_blocks):
        rng = np.random.default_rng([seed, b])
        sent = transmitter.generate(rng=rng)
        if gaussian:
            shape = np.shape(sent.data_symbols())
            symbols = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)
            sent = transmitter.generate(symbols=symbols)
        linear, delta = _first_order_fields(sent.field, link, sent.dt_s, nodes_per_fiber)
        # Score δ scaled to 1e-4 of the signal: first order turns the mean SPM/XPM rotation
        # e^{iφ} into 1 + iφ, which at the launch power would inflate the signal by |1 + iφ|².
        scale = 1e-4 * float(np.linalg.norm(linear) / np.linalg.norm(delta))
        with_nli = sim._receive(sent, linear + scale * delta, 0, 0.0)
        floor = sim._receive(sent, linear, 0, 0.0)
        inv = 10.0 ** (-with_nli.metrics.snr_db / 10.0) - 10.0 ** (-floor.metrics.snr_db / 10.0)
        etas.append(inv / (scale * p) ** 2)
        floors.append(floor.metrics.snr_db)
    eta = float(np.mean(etas))
    stderr = float(np.std(etas, ddof=1) / math.sqrt(n_blocks)) if n_blocks > 1 else 0.0
    return FirstOrderNLI(
        eta_per_w2=eta,
        eta_stderr_per_w2=stderr,
        snr_nli_db=-10.0 * math.log10(eta * p**2),
        linear_floor_db=float(np.min(floors)),
    )
