r"""Adaptive butterfly equalisers: constant modulus (CMA) and least mean squares (LMS).

The FIR of ``equalize="mimo"`` — fractionally spaced, :math:`T` taps per
input mode, :math:`y_p[k] = \sum_q\sum_j w_{pq,j}\,x_q[Sk + j]` (:math:`S`
samples per symbol, :math:`j = -(T-1)/2 … (T-1)/2`, circular) — but
adapted symbol by symbol, as in a real receiver, instead of fitted by least
squares to known data. With :math:`u_q = (x_q[Sk + j])_j`:

* **CMA** (Godard 1980), blind: :math:`e_p = y_p\,(R^2 - |y_p|^2)` with
  :math:`R^2 = E|a|^4/E|a|^2`, and
  :math:`w_{pq} \leftarrow w_{pq} + \mu\,e_p\,\bar u_q`. Converges to the
  inverse channel up to a phase and an order of the modes; for multi-ring
  constellations (16-QAM and up) its error never reaches zero (the
  constant-modulus misadjustment).
* **LMS**: CMA first (pre-convergence), then each mode's constant phase is
  removed from the taps — blind, from the fourth power of the output,
  :math:`\tfrac14\arg(\langle y^4\rangle / E[a^4])` (up to a multiple of
  π/2) — then :math:`e_p = \hat a_p - y_p` with the decision
  :math:`\hat a_p` (decision-directed); or, with
  ``lms_reference="data"``, the transmitted symbol (data-aided: phase from
  the least-squares fit to the data, decisions where no symbol is known,
  e.g. pilot slots).

  With carrier-phase recovery (``cpr``), the phase is not constant: the
  carrier phase :math:`\hat\theta_{p,k}` is estimated on the CMA output
  (blind phase search; data-aided, its constant ambiguity is fixed by the
  data) and kept in the LMS loop — decision on
  :math:`y_pe^{-i\hat\theta_{p,k}}`, error
  :math:`e_p = \hat a_pe^{i\hat\theta_{p,k}} - y_p` — so that the taps do
  not chase the laser phase. The output then still carries the carrier
  phase, which the receiver's own carrier-phase recovery removes.

When the transmitted symbols are given, the order of the outputs is
identified after CMA (output → the mode it correlates with most), as the
training header of a real receiver does; otherwise CMA's order is kept.

Taps start as a centre spike on the diagonal; each group's input is
scaled to the constellation's mean energy. Each stage sweeps the block
``passes`` times (the block is periodic); the output is that of the last
sweep.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field

from hpcom._model import HpcomModel
from hpcom.dsp.phase import BlindPhaseSearch, blind_phase_search

__all__ = ["AdaptiveEqualizer", "adaptive_equalize", "constant_modulus_radius"]

AdaptiveKind = Literal["cma", "lms"]


class AdaptiveEqualizer(HpcomModel):
    """Settings of ``equalize="cma"`` and ``equalize="lms"`` (see the module docstring).

    The number of taps is the receiver's ``mimo_taps``.

    Attributes
    ----------
    cma_step, lms_step
        Step sizes :math:`\\mu` (for input at the constellation's energy).
    passes
        Sweeps over the block per stage (CMA; then LMS).
    lms_reference
        ``"decisions"`` (decision-directed, blind) or ``"data"``
        (data-aided: needs the transmitted symbols).
    """

    cma_step: float = Field(default=1e-3, gt=0)
    lms_step: float = Field(default=1e-3, gt=0)
    passes: int = Field(default=3, ge=1)
    lms_reference: Literal["decisions", "data"] = "decisions"


def constant_modulus_radius(points: npt.ArrayLike, pmf: npt.ArrayLike | None = None) -> float:
    """:math:`R^2 = E|a|^4/E|a|^2`: CMA's target, met by the constellation itself."""
    pts = np.asarray(points, dtype=np.complex128)
    p = np.full(pts.size, 1.0 / pts.size) if pmf is None else np.asarray(pmf, dtype=np.float64)
    return float(np.dot(p, np.abs(pts) ** 4) / np.dot(p, np.abs(pts) ** 2))


def fourth_moment(points: npt.ArrayLike, pmf: npt.ArrayLike | None = None) -> complex:
    """:math:`E[a^4]` (negative real for square QAM; 0 if the constellation has no 4-fold phase)."""
    pts = np.asarray(points, dtype=np.complex128)
    p = np.full(pts.size, 1.0 / pts.size) if pmf is None else np.asarray(pmf, dtype=np.float64)
    m4 = complex(np.dot(p, pts**4))
    scale = float(np.dot(p, np.abs(pts) ** 4))
    return m4 if abs(m4) > 1e-9 * scale else 0j


def taps_index(n_samples: int, samples_per_symbol: int, n_taps: int) -> npt.NDArray[np.intp]:
    """Sample index of tap ``j`` for symbol ``k``: ``(n_slots, n_taps)``, circular."""
    half = (n_taps - 1) // 2
    n_slots = n_samples // samples_per_symbol
    index = samples_per_symbol * np.arange(n_slots)[:, None] + np.arange(-half, half + 1)
    return index % n_samples


def adaptive_equalize(
    samples: npt.NDArray[np.complex128],
    samples_per_symbol: int,
    n_taps: int,
    mode_axes: tuple[int, ...],
    kind: AdaptiveKind,
    config: AdaptiveEqualizer,
    points: npt.ArrayLike,
    pmf: npt.ArrayLike | None = None,
    train_slots: npt.NDArray[np.intp] | None = None,
    targets: npt.NDArray[np.complex128] | None = None,
    cpr: BlindPhaseSearch | None = None,
) -> npt.NDArray[np.complex128]:
    r"""CMA or LMS butterfly equaliser (module docstring) of ``samples`` ``(*lead, S·n)``.

    Axes ``mode_axes`` of the leading ones are equalised jointly, every
    other leading axis is an independent group (as in
    :func:`hpcom.receiver.mimo_equalize`). ``targets``
    ``(*lead, len(train_slots))`` (optional) are the transmitted symbols at
    ``train_slots`` (default: all slots): they identify the order of the
    outputs and train data-aided LMS. ``cpr`` puts the carrier phase in the
    LMS loop. Returns ``(*lead, n)`` at the constellation's scale.
    """
    x = np.asarray(samples, dtype=np.complex128)
    pts = np.asarray(points, dtype=np.complex128)
    lead = x.shape[:-1]
    n_lead = len(lead)
    others = [a for a in range(n_lead) if a not in mode_axes]
    order = (*others, *mode_axes, n_lead)
    group_shape = tuple(lead[a] for a in others)
    mode_shape = tuple(lead[a] for a in mode_axes)
    d = math.prod(mode_shape)
    n_samples = x.shape[-1]
    flat = np.transpose(x, order).reshape(-1, d, n_samples)
    weights = np.full(pts.size, 1.0 / pts.size) if pmf is None else np.asarray(pmf, np.float64)
    energy = float(np.dot(weights, np.abs(pts) ** 2))
    flat = flat * np.sqrt(energy / np.mean(np.abs(flat) ** 2, axis=(1, 2), keepdims=True))
    index = taps_index(n_samples, samples_per_symbol, n_taps)
    n_slots = index.shape[0]
    regress = flat[:, :, index]  # (G, D, n, T)
    data_aided = kind == "lms" and config.lms_reference == "data"
    known = np.zeros(n_slots, dtype=bool)
    full_t = np.zeros((flat.shape[0], d, n_slots), dtype=np.complex128)
    if targets is not None:
        train = np.arange(n_slots) if train_slots is None else np.asarray(train_slots)
        tt = np.transpose(np.asarray(targets, dtype=np.complex128), order)
        full_t[:, :, train] = tt.reshape(-1, d, tt.shape[-1])
        known[train] = True
    elif data_aided:
        raise ValueError('lms_reference="data" needs the transmitted symbols')
    taps = np.zeros((flat.shape[0], d, d, n_taps), dtype=np.complex128)
    taps[:, np.arange(d), np.arange(d), (n_taps - 1) // 2] = 1.0
    out = np.empty((flat.shape[0], d, n_slots), dtype=np.complex128)
    r2 = constant_modulus_radius(pts, pmf)
    carrier = np.ones_like(out)  # e^{iθ̂} kept in the LMS loop

    def sweep(stage: Literal["cma", "lms"]) -> None:
        for _ in range(config.passes):
            for k in range(n_slots):
                u = regress[:, :, k, :]  # (G, D, T)
                y = np.einsum("gpqt,gqt->gp", taps, u)
                if stage == "cma":
                    e = config.cma_step * y * (r2 - np.abs(y) ** 2)
                else:
                    c = carrier[:, :, k]
                    z = y * np.conj(c)
                    want = pts[np.argmin(np.abs(z[..., None] - pts) ** 2, axis=-1)]
                    if data_aided and known[k]:
                        want = full_t[:, :, k]
                    e = config.lms_step * (want * c - y)
                taps[...] += e[:, :, None, None] * np.conj(u)[:, None, :, :]
                out[:, :, k] = y

    sweep("cma")
    if targets is not None:  # identify the modes: output feeding each transmitted mode
        corr = np.abs(np.einsum("gpn,gqn->gpq", out[:, :, known], np.conj(full_t[:, :, known])))
        source = np.argmax(corr, axis=1)  # (G, D)
        taps = np.take_along_axis(taps, source[:, :, None, None], axis=1)
        out = np.take_along_axis(out, source[:, :, None], axis=1)
    if kind == "lms":
        if cpr is not None:
            theta = blind_phase_search(out, pts, cpr, energy)[1]
            if data_aided:
                fit = np.sum(
                    out[:, :, known]
                    * np.exp(-1j * theta[:, :, known])
                    * np.conj(full_t[:, :, known]),
                    axis=-1,
                )
                theta = theta + np.angle(fit)[:, :, None]
            carrier = np.exp(1j * theta)
        else:
            if data_aided:
                fit = np.sum(out[:, :, known] * np.conj(full_t[:, :, known]), axis=-1)
                rotation = np.angle(fit)
            else:
                m4 = fourth_moment(pts, pmf)
                rotation = (
                    np.angle(np.mean(out**4, axis=-1) / m4) / 4.0 if m4 else np.zeros(out.shape[:2])
                )
            taps *= np.exp(-1j * rotation)[:, :, None, None]
        sweep("lms")
    y = out.reshape(*group_shape, *mode_shape, n_slots)
    return np.transpose(y, np.argsort(order))
