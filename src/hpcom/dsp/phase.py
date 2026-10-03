r"""Blind carrier-phase recovery: blind phase search (Pfau, Hoffmann & Noé, JLT 2009).

The received symbol-rate samples carry a slowly varying carrier phase
:math:`\theta_k` (laser phase noise, residual frequency offset, nonlinear
phase): :math:`r_k = a_k e^{i\theta_k} + n_k`. Blind phase search tries
:math:`B` test phases over one symmetry period of the constellation,

.. math:: \varphi_b = \Big(\frac bB - \frac12\Big)\frac{2\pi}{S},\qquad b = 0…B-1,

(:math:`S` = 4 for square QAM), measures for each the squared distance of
every rotated sample to its nearest point,
:math:`d_{b,k} = |r_ke^{-i\varphi_b} - \hat a(r_ke^{-i\varphi_b})|^2`, sums it
over a window of :math:`2N+1` symbols, picks the test phase with the least
sum, and unwraps the picks (period :math:`2\pi/S`) into a continuous
estimate :math:`\hat\theta_k`. The samples are rotated back,
:math:`r_ke^{-i\hat\theta_k}`. The estimate is ambiguous by multiples of
:math:`2\pi/S`; the receiver's complex-gain stage that follows removes the
constant part (data-aided or with pilots).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pydantic import Field

from hpcom._model import HpcomModel
from hpcom.modulation.mapping import nearest_symbol_indices

__all__ = ["BlindPhaseSearch", "blind_phase_search"]


class BlindPhaseSearch(HpcomModel):
    """Blind phase search carrier-phase recovery (see the module docstring).

    Attributes
    ----------
    test_phases
        :math:`B`, test phases per symmetry period.
    half_window
        :math:`N`: the distances are summed over :math:`2N+1` symbols.
    symmetry
        :math:`S`, the rotational symmetry of the constellation (4: square
        QAM; 2: BPSK).
    temperature
        Differentiable version only (:func:`hpcom.autodiff.demodulate`): the
        gradient follows a softmax over the test phases with this
        temperature, relative to the mean windowed distance; the forward
        values are those of the hard search.
    """

    test_phases: int = Field(default=32, ge=2)
    half_window: int = Field(default=16, ge=0)
    symmetry: int = Field(default=4, ge=1)
    temperature: float = Field(default=0.05, gt=0)

    def test_phase_grid(self) -> npt.NDArray[np.float64]:
        """The test phases :math:`\\varphi_b`, rad."""
        period = 2.0 * math.pi / self.symmetry
        return (np.arange(self.test_phases) / self.test_phases - 0.5) * period


def _window_sum(d: npt.NDArray[np.float64], half: int) -> npt.NDArray[np.float64]:
    """Circular sum over ``2·half + 1`` samples along the last axis."""
    out: npt.NDArray[np.float64] = d.copy()
    for j in range(1, half + 1):
        out += np.roll(d, j, axis=-1) + np.roll(d, -j, axis=-1)
    return out


def blind_phase_search(
    samples: npt.ArrayLike,
    points: npt.ArrayLike,
    config: BlindPhaseSearch,
    energy: float | None = None,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    r"""Recover the carrier phase of symbol-rate ``samples`` ``(*streams, n)``.

    ``points`` is the constellation and ``energy`` its mean symbol energy
    (default: the mean of :math:`|\text{points}|^2`); each stream is first
    scaled to that energy. Returns the derotated samples (original
    amplitude) and the unwrapped phase estimate :math:`\hat\theta_k`,
    both ``(*streams, n)``.
    """
    r = np.asarray(samples, dtype=np.complex128)
    pts = np.asarray(points, dtype=np.complex128)
    es = float(np.mean(np.abs(pts) ** 2)) if energy is None else energy
    scale = np.sqrt(np.mean(np.abs(r) ** 2, axis=-1, keepdims=True) / es)
    unit = r / scale
    phis = config.test_phase_grid()
    rotated = unit[..., None, :] * np.exp(-1j * phis)[:, None]  # (*streams, B, n)
    decided = pts[nearest_symbol_indices(rotated, pts)]
    dist = _window_sum(np.abs(rotated - decided) ** 2, config.half_window)
    picked = phis[np.argmin(dist, axis=-2)]  # (*streams, n)
    s = config.symmetry
    phase = np.unwrap(s * picked, axis=-1) / s
    return r * np.exp(-1j * phase), phase
