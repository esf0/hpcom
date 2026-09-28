r"""Performance metrics of received symbols.

All functions compare received (equalised) symbols with the *transmitted*
reference — the metrics are data-aided, as in the fiber-optics literature.
Definitions (see :doc:`/physics` for derivations and references):

* **BER** — fraction of differing information bits.
* **SER** — fraction of wrongly decided symbols.
* **EVM** (RMS) — :math:`\sqrt{\sum|y-x|^2 / \sum|x|^2}`.
* **SNR** (effective) — :math:`\sum|x|^2 / \sum|y-x|^2 = 1/\text{EVM}^2`, in dB.
* **Q-factor** — :math:`20\log_{10}\!\big(\sqrt{2}\,\mathrm{erfc}^{-1}(2\,\text{BER})\big)` dB.
* **MI** and **GMI** — achievable rates (bits per symbol) for a
  circularly-symmetric Gaussian auxiliary channel with variance
  :math:`\sigma^2 = \langle|y-x|^2\rangle`, estimated by Monte Carlo over the
  received samples (Alvarado et al., *J. Lightwave Technol.* 36(2), 2018):

  .. math::
     \text{MI} = H(X) - \frac{1}{N}\sum_n \log_2
       \frac{\sum_{x'} p(x')\,q(y_n|x')}{p(x_n)\,q(y_n|x_n)},
     \qquad
     \text{GMI} = H(X) - \frac{1}{N}\sum_n \sum_{i=1}^{m} \log_2
       \frac{\sum_{x'} p(x')\,q(y_n|x')}
            {\sum_{x' \in \mathcal{X}^i_{b_{n,i}}} p(x')\,q(y_n|x')},

  with :math:`q(y|x) \propto e^{-|y-x|^2/\sigma^2}`, ``p`` the symbol PMF and
  :math:`\mathcal{X}^i_b` the points whose label bit ``i`` equals ``b``. Both
  estimators are anchored at the source entropy :math:`H(X)` (rather than
  the sample mean of :math:`-\log_2 p(x_n)`, which has the same expectation
  but more variance), so noiseless links give exactly ``MI = GMI = H(X)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import Field
from scipy.special import erfcinv, logsumexp

from hpcom._model import HpcomModel
from hpcom._typing import ComplexArray

__all__ = [
    "Metrics",
    "StreamMetrics",
    "bit_error_rate",
    "evm_rms",
    "gmi_gaussian",
    "mi_gaussian",
    "q_factor_db",
    "snr_db",
]

_LN2 = float(np.log(2.0))
_CHUNK = 1 << 20  # samples × points evaluated at once in MI/GMI


class Metrics(HpcomModel):
    """Aggregate quality metrics (over all scored streams).

    Rates (``mi``, ``gmi``) are per symbol per polarisation, averaged over
    streams. ``q_factor_db`` is ``None`` when BER is 0 or ≥ 0.5.
    """

    n_bits: int = Field(ge=0)
    n_bit_errors: int = Field(ge=0)
    ber: float
    n_symbols: int = Field(ge=0)
    n_symbol_errors: int = Field(ge=0)
    ser: float
    evm_rms: float
    snr_db: float
    q_factor_db: float | None
    mi_bits: float
    gmi_bits: float


class StreamMetrics(Metrics):
    """Metrics of one stream; ``polarization`` / ``channel`` are indices (``None`` if absent)."""

    polarization: int | None = None
    channel: int | None = None


def bit_error_rate(
    sent_bits: npt.ArrayLike, received_bits: npt.ArrayLike
) -> tuple[float, int, int]:
    """``(ber, n_errors, n_bits)``; arrays must have the same shape."""
    a, b = np.asarray(sent_bits), np.asarray(received_bits)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: sent {a.shape} vs received {b.shape}")
    n = int(a.size)
    errors = int(np.count_nonzero(a != b))
    return (errors / n if n else 0.0), errors, n


def evm_rms(received: ComplexArray, reference: ComplexArray) -> float:
    """RMS error-vector magnitude relative to the reference RMS amplitude."""
    y, x = np.asarray(received), np.asarray(reference)
    if y.shape != x.shape:
        raise ValueError(f"shape mismatch: received {y.shape} vs reference {x.shape}")
    ref_power = float(np.sum(np.abs(x) ** 2))
    err_power = float(np.sum(np.abs(y - x) ** 2))
    if ref_power == 0:
        return float("inf") if err_power > 0 else 0.0
    return float(np.sqrt(err_power / ref_power))


def snr_db(received: ComplexArray, reference: ComplexArray) -> float:
    """Effective (data-aided) SNR ``Σ|x|² / Σ|y−x|²`` in dB (``inf`` when error-free)."""
    evm = evm_rms(received, reference)
    return float("inf") if evm == 0 else float(-20.0 * np.log10(evm))


def q_factor_db(ber: float) -> float | None:
    """``20·log₁₀(√2·erfcinv(2·BER))``; ``None`` for BER ≤ 0 or ≥ 0.5."""
    if not 0.0 < ber < 0.5:
        return None
    return float(20.0 * np.log10(np.sqrt(2.0) * erfcinv(2.0 * ber)))


def _log_q(
    y: npt.NDArray[np.complex128], points: npt.NDArray[np.complex128], sigma2: float
) -> npt.NDArray[np.float64]:
    """``ln q(y|x) + const`` for all (sample, point) pairs: ``−|y−x|²/σ²``."""
    d = (y.real[:, None] - points.real) ** 2 + (y.imag[:, None] - points.imag) ** 2
    out: npt.NDArray[np.float64] = -d / sigma2
    return out


def _noise_variance(y: npt.NDArray[np.complex128], x: npt.NDArray[np.complex128]) -> float:
    return float(np.mean(np.abs(y - x) ** 2))


def mi_gaussian(
    received: ComplexArray,
    sent_indices: npt.ArrayLike,
    points: ComplexArray,
    pmf: npt.ArrayLike | None = None,
) -> float:
    """Mutual information (bits/symbol) with a Gaussian auxiliary channel (module docstring)."""
    y = np.asarray(received, dtype=np.complex128).reshape(-1)
    idx = np.asarray(sent_indices, dtype=np.intp).reshape(-1)
    pts = np.asarray(points, dtype=np.complex128)
    p = np.full(pts.size, 1.0 / pts.size) if pmf is None else np.asarray(pmf, dtype=np.float64)
    sigma2 = _noise_variance(y, pts[idx])
    if sigma2 == 0.0:
        nz = p[p > 0]
        return float(-np.sum(nz * np.log2(nz)))
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    log_p = np.log(np.where(p > 0, p, 1.0)) + np.where(p > 0, 0.0, -np.inf)
    loss = 0.0
    step = max(1, _CHUNK // pts.size)
    for s in range(0, y.size, step):
        metric = _log_q(y[s : s + step], pts, sigma2) + log_p  # ln p(x) q(y|x), (n, M)
        sent = metric[np.arange(metric.shape[0]), idx[s : s + step]]
        loss += float(np.sum(logsumexp(metric, axis=1) - sent))
    return entropy - loss / y.size / _LN2


def gmi_gaussian(
    received: ComplexArray,
    sent_indices: npt.ArrayLike,
    points: ComplexArray,
    bit_labels: npt.ArrayLike,
    pmf: npt.ArrayLike | None = None,
) -> float:
    """Generalised mutual information (bits/symbol) for bit-wise decoding (module docstring)."""
    y = np.asarray(received, dtype=np.complex128).reshape(-1)
    idx = np.asarray(sent_indices, dtype=np.intp).reshape(-1)
    pts = np.asarray(points, dtype=np.complex128)
    labels = np.asarray(bit_labels, dtype=bool)
    p = np.full(pts.size, 1.0 / pts.size) if pmf is None else np.asarray(pmf, dtype=np.float64)
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    sigma2 = _noise_variance(y, pts[idx])
    if sigma2 == 0.0:
        return entropy
    log_p = np.log(np.where(p > 0, p, 1.0)) + np.where(p > 0, 0.0, -np.inf)
    sent_bits = labels[idx]  # (n, m)
    loss = 0.0
    step = max(1, _CHUNK // pts.size)
    for s in range(0, y.size, step):
        metric = _log_q(y[s : s + step], pts, sigma2) + log_p  # (n, M)
        den_all = logsumexp(metric, axis=1)  # (n,)
        for i in range(labels.shape[1]):
            match = labels[:, i][None, :] == sent_bits[s : s + step, i][:, None]  # (n, M)
            num = logsumexp(np.where(match, metric, -np.inf), axis=1)
            loss += float(np.sum(den_all - num))
    return entropy - loss / y.size / _LN2
