"""Abstract base class and shared helpers for fiber propagation backends."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt

from hpcom._typing import ComplexArray, Direction, FiberLike
from hpcom.units import SPEED_OF_LIGHT_M_PER_S


class BackendUnavailableError(RuntimeError):
    """A registered backend cannot be used because its dependencies are missing."""


class CPUFallbackWarning(RuntimeWarning):
    """A span ran on the NumPy (CPU) backend because the requested one lacks its scheme."""


class Propagator(ABC):
    """Backend-agnostic propagation of an optical field through one fiber span.

    **Contract** (checked for every registered backend by the conformance
    suite in ``tests/backends/``):

    * ``fiber.polarization == "single"``: ``field`` has shape
      ``(..., n_samples)`` and is propagated with the scalar NLSE.
      ``"dual"``: shape ``(..., 2, n_samples)`` (X, Y) and the Manakov
      equation. Leading axes are independent batch dimensions.
    * The output has the input's shape and dtype ``complex128``; the input is
      not modified.
    * Equation and sign conventions are those of
      :doc:`/physics` — in NumPy's FFT convention the linear operator is
      ``exp[(−i β₁ω + i β₂ω²/2 − i β₃ω³/6 + … − α(ω)/2) z]``
      (:func:`linear_operator_exponent`) and the Kerr term is
      ``+i γ |A|² A`` (``+i (8/9) γ (|A_x|²+|A_y|²) A`` for Manakov), with
      ``|A|²`` replaced by its convolution with the Raman response
      (:func:`kerr_power_response`) and the shock factor
      (:func:`shock_factor`) applied when the fiber has them.
    * ``direction="backward"`` applies the exact inverse step sequence (same
      steps, negated length), as used by digital back-propagation.
    """

    #: Stable identifier; conventionally equal to the registry name.
    name: str = ""

    #: ``Scheme.method`` values this backend implements (with every Kerr-step
    #: option and step placement). Others run on NumPy with a
    #: :class:`CPUFallbackWarning`.
    supported_methods: frozenset[str] = frozenset({"ssfm"})

    @abstractmethod
    def propagate(
        self,
        field: ComplexArray,
        fiber: FiberLike,
        dt_s: float,
        direction: Direction = "forward",
    ) -> ComplexArray:
        """Propagate ``field`` (sample period ``dt_s``) through one span of ``fiber``."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Helpers shared by the built-in backends
# ---------------------------------------------------------------------------


def signed_length_m(fiber: FiberLike, direction: Direction) -> float:
    """Span length in metres, negated for backward propagation."""
    if direction == "forward":
        return fiber.length_km * 1e3
    if direction == "backward":
        return -fiber.length_km * 1e3
    raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")


def step_boundaries_m(fiber: FiberLike) -> npt.NDArray[np.float64]:
    r"""Step boundaries :math:`0 = z_0 < z_1 < … < z_K = L` of a span, in metres.

    ``"uniform"``: :math:`z_n = nL/K`. ``"logarithmic"`` with factor
    :math:`k` (``fiber.step_log_factor``) and power attenuation :math:`\alpha`:

    .. math::

        z_n = -\frac{1}{k\alpha}\ln\!\left(1 - n\,\frac{1 - e^{-k\alpha L}}{K}\right),

    so every step carries the same share of :math:`\int e^{-k\alpha z}\,dz`
    (``k = 1``: equal nonlinear phase per step). Without loss it is uniform.
    Derivation and references: ``docs/numerics.rst``.
    """
    k_steps = int(fiber.n_steps)
    if k_steps < 1:
        raise ValueError(f"fiber.n_steps must be >= 1, got {k_steps}")
    length_m = fiber.length_km * 1e3
    n = np.arange(k_steps + 1, dtype=np.float64)
    if fiber.step_distribution == "uniform":
        return n * (length_m / k_steps)
    if fiber.step_distribution != "logarithmic":
        raise ValueError(f"unknown step_distribution {fiber.step_distribution!r}")
    rate = fiber.step_log_factor * fiber.alpha_db_per_km * np.log(10.0) / 10.0 / 1e3  # kα, 1/m
    if rate * length_m == 0.0:
        return n * (length_m / k_steps)
    delta = -np.expm1(-rate * length_m) / k_steps
    z: npt.NDArray[np.float64] = -np.log1p(-n * delta) / rate
    z[-1] = length_m  # exact end point despite rounding
    return z


def step_sizes_m(fiber: FiberLike, direction: Direction) -> npt.NDArray[np.float64]:
    """Signed step lengths in the order they are taken.

    Forward: :math:`h_1, …, h_K` from the span input. Backward: the same
    steps from the span output, negated (:math:`-h_K, …, -h_1`) — the exact
    reverse of the forward sequence.
    """
    sign = 1.0 if signed_length_m(fiber, direction) > 0 else -1.0
    if int(fiber.n_steps) < 1:
        raise ValueError(f"fiber.n_steps must be >= 1, got {fiber.n_steps}")
    if fiber.step_distribution == "uniform":
        h = np.full(int(fiber.n_steps), sign * fiber.length_km * 1e3 / int(fiber.n_steps))
        return h
    sizes = np.diff(step_boundaries_m(fiber))
    return sign * (sizes if sign > 0 else sizes[::-1])


def merged_linear_steps(sizes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Lengths of the ``K+1`` linear sub-steps of the symmetric scheme.

    ``[h₁/2, (h₁+h₂)/2, …, (h_{K−1}+h_K)/2, h_K/2]`` — adjacent half steps merged.
    """
    merged = np.empty(sizes.size + 1)
    merged[0], merged[-1] = sizes[0] / 2.0, sizes[-1] / 2.0
    merged[1:-1] = (sizes[:-1] + sizes[1:]) / 2.0
    return merged


def check_field_shape(field: np.ndarray, fiber: FiberLike) -> None:
    """Validate ``field`` against ``fiber.polarization`` and cores (see :class:`Propagator`).

    Single core: ``(..., n)`` / ``(..., 2, n)``; ``M`` cores: ``(..., M, n)`` /
    ``(..., M, 2, n)``.
    """
    if fiber.polarization not in ("single", "dual"):
        raise ValueError(f"unknown fiber polarization {fiber.polarization!r}")
    dual = fiber.polarization == "dual"
    cores = int(fiber.n_cores)
    tail = (2,) if dual else ()
    if cores > 1:
        tail = (cores, *tail)
    if field.ndim < len(tail) + 1 or tuple(field.shape[-len(tail) - 1 : -1]) != tail:
        layout = ("M, " if cores > 1 else "") + ("2, " if dual else "")
        what = f"{cores}-core " if cores > 1 else ""
        raise ValueError(
            f"a {what}{fiber.polarization}-polarisation fiber expects a field of shape "
            f"(..., {layout}n_samples){f' with M = {cores}' if cores > 1 else ''}; "
            f"got {field.shape}"
        )


def core_axis(fiber: FiberLike) -> int:
    """Axis of the core index in a multicore field (−2 single, −3 dual polarisation)."""
    return -3 if fiber.polarization == "dual" else -2


def multicore_linear_system(
    n_samples: int, dt_s: float, fiber: Any
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    r"""Eigen-decomposition of the multicore linear operator, per frequency.

    The linear part of the coupled equations is
    :math:`\partial_z\tilde A = [\,iK(\omega) - \tfrac{\alpha}{2}\,]\tilde A`
    with the Hermitian matrix
    :math:`K(\omega) = \operatorname{diag}_m\big(k(\omega) + \delta\beta_m -
    \delta\beta_{1,m}\omega\big) + \kappa`, :math:`k(\omega) = -\beta_1\omega +
    \beta_2\omega^2/2 - \beta_3\omega^3/6`. With :math:`K = V\Lambda V^\dagger`,

    .. math:: e^{h[iK - \alpha/2]} = V\,\operatorname{diag}\big(e^{h(i\lambda - \alpha/2)}\big)\,V^\dagger

    exactly, for any :math:`h`. Returns ``(V, exponent)``: ``V`` of shape
    ``(F, M, M)`` (``F = 1`` when there is no skew, since :math:`K - k(\omega)I`
    is then frequency independent) and ``exponent`` :math:`i\lambda - \alpha/2`
    of shape ``(n, M)``.
    """
    common = linear_operator_exponent(n_samples, dt_s, fiber)  # i k(ω) − α(ω)/2
    alpha_half = -common.real  # α(ω)/2, common to every core
    k = common.imag  # k(ω)
    kappa = np.asarray(fiber.coupling_per_m, dtype=np.float64)
    m = kappa.shape[0]
    beta0 = fiber.core_beta0_offsets_per_m
    beta1 = fiber.core_beta1_offsets_s_per_m
    offsets = np.zeros(m) if beta0 is None else np.asarray(beta0, dtype=np.float64)
    if beta1 is None or not np.any(beta1):
        mu, vectors = np.linalg.eigh(kappa + np.diag(offsets))
        lam = k[:, None] + mu[None, :]
        v = vectors[None].astype(np.complex128)
    else:
        omega = 2.0 * np.pi * np.fft.fftfreq(n_samples, d=dt_s)
        diag = k[:, None] + offsets[None, :] - omega[:, None] * np.asarray(beta1)[None, :]
        matrices = np.broadcast_to(kappa, (n_samples, m, m)).copy()
        matrices[:, np.arange(m), np.arange(m)] = diag
        lam, vectors = np.linalg.eigh(matrices)
        v = vectors.astype(np.complex128)
    exponent: npt.NDArray[np.complex128] = (1j * lam - alpha_half[:, None]).astype(np.complex128)
    return v, exponent


_CBRT2, _CBRT4 = 2.0 ** (1.0 / 3.0), 4.0 ** (1.0 / 3.0)
_YOSHIDA_W1 = 1.0 / (2.0 - _CBRT2)
_SUZUKI_P = 1.0 / (4.0 - _CBRT4)

#: Sub-step weights of the symmetric compositions of the Strang step, per method:
#: Yoshida's triple jump and Suzuki's fractal (5-stage) fourth-order compositions.
COMPOSITION_WEIGHTS: dict[str, tuple[float, ...]] = {
    "ssfm": (1.0,),
    "ssfm4-yoshida": (_YOSHIDA_W1, 1.0 - 2.0 * _YOSHIDA_W1, _YOSHIDA_W1),
    "ssfm4-suzuki": (_SUZUKI_P, _SUZUKI_P, 1.0 - 4.0 * _SUZUKI_P, _SUZUKI_P, _SUZUKI_P),
}

#: Every ``Scheme.method`` hpcom defines.
ALL_METHODS: frozenset[str] = frozenset(COMPOSITION_WEIGHTS) | {"rk4ip", "ssfm-local-error"}

#: Step-size factors of the local-error controller (Sinkin et al. 2003).
LOCAL_ERROR_GROWTH = 2.0 ** (1.0 / 3.0)


def attenuation_per_m(fiber: FiberLike) -> float:
    """Power attenuation α in 1/m (``P(z) = P(0)·e^{−αz}``)."""
    return float(fiber.alpha_db_per_km * np.log(10.0) / 10.0 / 1e3)


def uses_fast_path(fiber: FiberLike) -> bool:
    """Uniform steps with the plain symmetric split step: one precomputed ``D(h)``."""
    s = fiber.scheme
    return (
        fiber.n_cores == 1
        and fiber.raman is None
        and fiber.effective_shock_time_s == 0.0
        and float(getattr(fiber, "pmd_ps_per_sqrt_km", 0.0)) == 0.0
        and fiber.step_distribution == "uniform"
        and s.method == "ssfm"
        and s.nonlinear_position == 0.5
        and s.kerr_length == "step"
        and s.kerr_filter_bandwidth_hz is None
    )


def split_step_substeps(
    fiber: FiberLike, direction: Direction
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Geometry of :func:`split_step_sequence`: linear lengths, signed sub-steps, Kerr position.

    Returns ``(m, h, r)``: the ``J+1`` linear lengths, the ``J`` signed
    sub-step lengths :math:`w_i h` and the Kerr position along the traversal
    (:math:`r` forward, :math:`1-r` backward).
    """
    scheme = fiber.scheme
    if scheme.method not in COMPOSITION_WEIGHTS:
        raise ValueError(f"{scheme.method!r} is not a split-step method")
    sizes = step_sizes_m(fiber, direction)
    weights = np.asarray(COMPOSITION_WEIGHTS[scheme.method], dtype=np.float64)
    sub = (sizes[:, None] * weights[None, :]).ravel()
    r = float(scheme.nonlinear_position)
    r_trav = r if direction == "forward" else 1.0 - r
    pre, post = r_trav * sub, (1.0 - r_trav) * sub
    linear = np.empty(sub.size + 1)
    linear[0], linear[-1] = pre[0], post[-1]
    linear[1:-1] = post[:-1] + pre[1:]
    return linear, sub, r_trav


def split_step_sequence(
    fiber: FiberLike, direction: Direction
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Signed linear and Kerr sub-step lengths of a span, in the order they are applied.

    Returns ``(m, c)`` with ``m`` of length ``J+1`` and ``c`` of length ``J``:

    .. math:: A_\text{out} = D(m_J)\,N(c_J)\cdots D(m_1)\,N(c_1)\,D(m_0)\,A_\text{in}.

    Each step :math:`h` (split into sub-steps :math:`w_i h` for compositions)
    becomes linear :math:`r h`, Kerr, linear :math:`(1-r)h`, with :math:`r`
    the Kerr position in the fiber's coordinate — :math:`1-r` along the
    traversal when going backward, so that backward is the exact inverse.
    ``c`` is the step length, or the effective length
    :math:`(e^{\alpha r|h|} - e^{-\alpha(1-r)|h|})/\alpha` with the sign of
    :math:`h`. See ``docs/numerics.rst``.
    """
    linear, sub, _ = split_step_substeps(fiber, direction)
    r = float(fiber.scheme.nonlinear_position)
    alpha = attenuation_per_m(fiber)
    if fiber.scheme.kerr_length == "step" or alpha == 0.0:
        return linear, sub
    mag = np.abs(sub)
    kerr = np.sign(sub) * (np.exp(alpha * r * mag) - np.exp(-alpha * (1.0 - r) * mag)) / alpha
    return linear, kerr


def kerr_filter_response(
    n_samples: int, dt_s: float, fiber: FiberLike
) -> npt.NDArray[np.float64] | None:
    """Real, even transfer function of the Kerr-step intensity filter (``None``: no filter)."""
    scheme = fiber.scheme
    fc = scheme.kerr_filter_bandwidth_hz
    if fc is None:
        return None
    f = np.fft.fftfreq(n_samples, d=dt_s)
    if scheme.kerr_filter_shape == "gaussian":
        response: npt.NDArray[np.float64] = np.exp(-np.log(2.0) / 2.0 * (f / fc) ** 2)
        return response
    if scheme.kerr_filter_shape == "rectangular":
        window: npt.NDArray[np.float64] = (np.abs(f) <= fc).astype(np.float64)
        return window
    raise ValueError(f"unknown kerr_filter_shape {scheme.kerr_filter_shape!r}")


def linear_operator_exponent(
    n_samples: int, dt_s: float, fiber: FiberLike
) -> npt.NDArray[np.complex128]:
    r"""Per-metre exponent of the linear step, :math:`\hat L(\omega) = iD(\omega) - \alpha(\omega)/2`.

    .. math::

        D(\omega) = \sum_{n\ge1} \frac{\beta_n}{n!}(-\omega)^n
                   = -\beta_1\omega + \frac{\beta_2}{2}\omega^2 - \frac{\beta_3}{6}\omega^3 + …

    ``ω`` is the angular-frequency grid in NumPy FFT order. The signs follow
    from Agrawal's NLSE written in NumPy's FFT convention (see
    :doc:`/physics`): grid frequency :math:`+\omega` is the optical
    frequency :math:`\omega_0 - \omega`, so :math:`D(\omega) = \beta(\omega_0 -
    \omega) - \beta_0` and β₁ > 0 delays the field by β₁z. The loss is
    ``alpha_db_per_km``, or ``fiber.loss_spectrum`` at the wavelength
    :math:`2\pi c/(\omega_0 - \omega)` of each frequency.
    """
    omega = 2.0 * np.pi * np.fft.fftfreq(n_samples, d=dt_s)
    beta3 = fiber.beta3_s3_per_m
    exponent: npt.NDArray[np.complex128] = (
        -1j * fiber.beta1_s_per_m * omega
        + 1j * (fiber.beta2_s2_per_m / 2.0) * omega**2
        - 1j * (beta3 / 6.0) * omega**3
    ).astype(np.complex128)
    for n, beta in enumerate(fiber.higher_order_betas, start=4):
        if beta:
            exponent += 1j * (beta / math.factorial(n)) * (-omega) ** n
    return exponent - attenuation_spectrum_per_m(n_samples, dt_s, fiber) / 2.0


def attenuation_spectrum_per_m(
    n_samples: int, dt_s: float, fiber: FiberLike
) -> npt.NDArray[np.float64] | float:
    """Power attenuation α(ω) in 1/m on the FFT grid (a scalar without a loss spectrum)."""
    spectrum = fiber.loss_spectrum
    if spectrum is None:
        return attenuation_per_m(fiber)
    c = SPEED_OF_LIGHT_M_PER_S
    nu = c / fiber.wavelength_m - np.fft.fftfreq(n_samples, d=dt_s)  # optical frequency
    wavelength = np.full(n_samples, np.inf)
    wavelength[nu > 0] = c / nu[nu > 0]
    return np.asarray(spectrum.at(wavelength) * np.log(10.0) / 10.0 / 1e3, dtype=np.float64)


def kerr_power_response(
    n_samples: int, dt_s: float, fiber: FiberLike
) -> npt.NDArray[np.complex128] | None:
    r"""Frequency response applied to the power in the Kerr term (``None``: none).

    .. math:: \tilde R(\omega) = \big[(1-f_R) + f_R H_R(\omega)\big]\,H_\mathrm{filter}(\omega)

    — the Raman response (``fiber.raman``) and the Kerr-step intensity
    filter of the scheme (:func:`kerr_filter_response`). The Kerr term uses
    :math:`\operatorname{Re}\mathcal F^{-1}[\tilde R\,\mathcal F|A|^2]`; the real
    part drops the unpaired Nyquist bin of an even-length grid, where a
    real response would have to be real.
    """
    raman = fiber.raman
    response = kerr_filter_response(n_samples, dt_s, fiber)
    if raman is None:
        return None if response is None else response.astype(np.complex128)
    omega = 2.0 * np.pi * np.fft.fftfreq(n_samples, d=dt_s)
    f_r = raman.f_r
    total: npt.NDArray[np.complex128] = ((1.0 - f_r) + f_r * raman.transfer(omega)).astype(
        np.complex128
    )
    return total if response is None else total * response


def shock_factor(n_samples: int, dt_s: float, fiber: FiberLike) -> npt.NDArray[np.float64] | None:
    r"""Self-steepening factor :math:`1 - \tau_\mathrm{shock}\,\omega` (``None``: none).

    The operator :math:`1 + i\tau\,\partial_t` in NumPy's convention
    (:math:`\partial_t \to i\omega`); with :math:`\tau = 1/\omega_0` it is
    the optical frequency over the carrier's, :math:`(\omega_0-\omega)/\omega_0`.
    """
    tau = fiber.effective_shock_time_s
    if tau == 0.0:
        return None
    omega = 2.0 * np.pi * np.fft.fftfreq(n_samples, d=dt_s)
    return np.asarray(1.0 - tau * omega, dtype=np.float64)


# ---------------------------------------------------------------------------
# Polarisation-mode dispersion (coarse-step Manakov-PMD model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PMDRealization:
    r"""One random birefringence realisation of a span (see :func:`pmd_realization`).

    ``axes[k]`` is the unit Stokes vector :math:`\hat b_k` of section ``k``'s
    birefringence, ``output_rotation`` the frequency-independent Jones matrix
    :math:`Q_N` applied at the span output, ``dgd_per_m`` the differential
    group delay per metre :math:`\Delta\beta_1 = \tau_s/L_s`.
    """

    axes: npt.NDArray[np.float64]
    output_rotation: npt.NDArray[np.complex128]
    dgd_per_m: float
    section_length_m: float


@lru_cache(maxsize=64)
def _pmd_frames(
    seed: int, n_sections: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Section axes and output rotation of a Haar-random cascade (cached per seed)."""
    rng = np.random.default_rng(seed)
    sigma3 = np.diag([1.0, -1.0]).astype(np.complex128)
    q = np.eye(2, dtype=np.complex128)
    axes = np.empty((n_sections, 3))
    for k in range(n_sections):
        v = rng.standard_normal(4)
        v /= np.linalg.norm(v)
        a, b = complex(v[0], v[3]), complex(v[2], v[1])
        q = np.array([[a, -np.conj(b)], [b, np.conj(a)]]) @ q
        m = q.conj().T @ sigma3 @ q  # = b·σ
        axes[k] = (m[1, 0].real, m[1, 0].imag, m[0, 0].real)
    axes.flags.writeable = False
    q.flags.writeable = False
    return axes, q


def pmd_realization(fiber: FiberLike) -> PMDRealization | None:
    r"""The span's birefringence realisation, or ``None`` without PMD.

    Coarse-step model (Marcuse, Menyuk & Wai 1997): ``fiber.pmd_sections``
    sections of length :math:`L_s = L/N`, each a Haar-random rotation
    :math:`R_k` followed by linear birefringence with DGD
    :math:`\tau_s = \langle\tau\rangle\sqrt{3\pi/8N}`, so that the span's DGD is
    Maxwellian with mean :math:`\langle\tau\rangle = D_\mathrm{PMD}\sqrt L`
    (:math:`\langle\tau^2\rangle = N\tau_s^2`). Writing
    :math:`Q_k = R_k\cdots R_1`, the span's Jones matrix is
    :math:`Q_N\prod_k e^{-i\omega\tau_s(\hat b_k\cdot\vec\sigma)/2}` with
    :math:`\hat b_k\cdot\vec\sigma = Q_k^\dagger\sigma_3Q_k`, and the Manakov
    Kerr term commutes with every :math:`Q_k`. The realisation is a function
    of ``fiber.pmd_seed`` and ``pmd_sections`` only.
    """
    coefficient = float(getattr(fiber, "pmd_ps_per_sqrt_km", 0.0))
    if coefficient == 0.0:
        return None
    n = int(fiber.pmd_sections)  # type: ignore[attr-defined]
    axes, rotation = _pmd_frames(int(fiber.pmd_seed), n)  # type: ignore[attr-defined]
    mean_dgd = coefficient * 1e-12 * math.sqrt(fiber.length_km)
    length_m = fiber.length_km * 1e3
    tau_s = mean_dgd * math.sqrt(3.0 * math.pi / (8.0 * n))
    return PMDRealization(axes, rotation, tau_s / (length_m / n), length_m / n)


def pmd_pieces(
    realization: PMDRealization, start_m: float, lengths_m: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Split consecutive signed linear moves into per-section pieces.

    Starting at ``start_m`` (0 forward, ``L`` backward), move ``lengths_m[j]``
    after ``lengths_m[j-1]``. Returns ``(sections, lengths)`` of shape
    ``(J, P)``: the sections crossed by move ``j`` in the order crossed and
    the signed length in each, padded with zero-length pieces.
    """
    ls = realization.section_length_m
    n = realization.axes.shape[0]
    tol = 1e-9 * ls
    rows: list[list[tuple[int, float]]] = []
    z = float(start_m)
    for m in np.asarray(lengths_m, dtype=np.float64):
        row: list[tuple[int, float]] = []
        remaining = float(m)
        while abs(remaining) > tol:
            if remaining > 0:
                k = min(max(math.floor((z + tol) / ls), 0), n - 1)
                step = remaining if k == n - 1 else min(remaining, (k + 1) * ls - z)
            else:
                k = min(max(math.ceil((z - tol) / ls) - 1, 0), n - 1)
                step = remaining if k == 0 else max(remaining, k * ls - z)
            row.append((k, step))
            z += step
            remaining -= step
        rows.append(row)
    width = max(1, max(len(r) for r in rows))
    sections = np.zeros((len(rows), width), dtype=np.intp)
    pieces = np.zeros((len(rows), width))
    for j, row in enumerate(rows):
        for p, (k, step) in enumerate(row):
            sections[j, p], pieces[j, p] = k, step
    return sections, pieces


def pmd_rotate(
    spectrum: npt.NDArray[np.complex128],
    axis: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    r""":math:`e^{-i\theta(\hat b\cdot\vec\sigma)}` on a ``(..., 2, n)`` spectrum, per frequency."""
    c, s = np.cos(theta), np.sin(theta)
    b1, b2, b3 = float(axis[0]), float(axis[1]), float(axis[2])
    x, y = spectrum[..., 0, :], spectrum[..., 1, :]
    out = np.empty_like(spectrum)
    out[..., 0, :] = (c - 1j * s * b3) * x - 1j * s * (b1 - 1j * b2) * y
    out[..., 1, :] = -1j * s * (b1 + 1j * b2) * x + (c + 1j * s * b3) * y
    return out


def apply_jones(
    field: ComplexArray, matrix: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """A frequency-independent 2×2 Jones matrix on the polarisation axis (−2) of a field."""
    out = np.asarray(np.einsum("pq,...qn->...pn", matrix, field), dtype=np.complex128)
    return out


def effective_gamma_per_m(fiber: FiberLike) -> float:
    """Kerr coefficient in 1/(W·m) as used by the nonlinear step.

    Includes the Manakov 8/9 factor (polarisation averaging over the
    Poincaré sphere; Marcuse, Menyuk & Wai, JLT 15(9) 1997) for
    dual-polarisation fibers.
    """
    gamma = fiber.gamma_per_w_per_km * 1e-3
    return gamma * 8.0 / 9.0 if fiber.polarization == "dual" else gamma
