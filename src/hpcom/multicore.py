r"""Weakly coupled multicore fibers (coupled NLSE / Manakov per core).

Each core :math:`m = 1…M` carries its own envelope (or Manakov pair), coupled
linearly to the others (coupled-mode theory, e.g. Chekhovskoy et al. 2016):

.. math::

    \partial_z A_m = \hat L_m A_m + i\gamma_\mathrm{eff} P_m A_m
                     + i\sum_{n\ne m} \kappa_{mn} A_n,
    \qquad
    \hat L_m = \hat L + i\,\delta\beta_m - \delta\beta_{1,m}\,\partial_t,

with :math:`\hat L` the single-core operator of :class:`~hpcom.fiber.Fiber`
(common loss, β₁…β₃), :math:`\kappa` the real symmetric coupling matrix
(``coupling_per_m``), :math:`\delta\beta_m` a per-core propagation-constant
mismatch and :math:`\delta\beta_{1,m}` a per-core group-delay offset (skew).
The Kerr term is per core only — :math:`P_m = |A_m|^2` or
:math:`|A_{m,x}|^2 + |A_{m,y}|^2` with the Manakov 8/9 — since the cores do
not overlap (inter-core XPM is orders of magnitude weaker and neglected).

Fields carry the core axis before the polarisation axis: ``(..., M, n)``
single polarisation, ``(..., M, 2, n)`` dual. Every propagation scheme and
backend supports multicore fibers; the linear step, including coupling, is
exact (``docs/physics.rst``, *Multicore fibers*).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from hpcom._model import FloatArrayField
from hpcom.fiber import Fiber

__all__ = [
    "MulticoreFiber",
    "hexagonal_coupling",
    "linear_array_coupling",
    "ring_coupling",
]


class MulticoreFiber(Fiber):
    """A span of weakly coupled multicore fiber (see the module docstring).

    All :class:`~hpcom.fiber.Fiber` parameters apply to every core.

    Attributes
    ----------
    coupling_per_m
        ``(M, M)`` real symmetric coupling matrix :math:`\\kappa_{mn}` in
        1/m, zero diagonal (see :func:`ring_coupling`,
        :func:`hexagonal_coupling`, :func:`linear_array_coupling`).
    core_beta0_offsets_per_m
        ``(M,)`` propagation-constant mismatch :math:`\\delta\\beta_m` in 1/m
        (``None``: identical cores).
    core_beta1_offsets_s_per_m
        ``(M,)`` group-delay offset :math:`\\delta\\beta_{1,m}` in s/m — the
        inter-core skew (``None``: none).
    """

    coupling_per_m: FloatArrayField
    core_beta0_offsets_per_m: FloatArrayField | None = None
    core_beta1_offsets_s_per_m: FloatArrayField | None = None

    @model_validator(mode="after")
    def _validate_cores(self) -> MulticoreFiber:
        c = self.coupling_per_m
        if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 2:
            raise ValueError(f"coupling_per_m must be an (M, M) matrix with M >= 2, got {c.shape}")
        if not np.all(np.isfinite(c)):
            raise ValueError("coupling_per_m must be finite")
        if not np.allclose(c, c.T, rtol=0, atol=1e-15 * max(1.0, float(np.max(np.abs(c))))):
            raise ValueError("coupling_per_m must be symmetric (lossless reciprocal coupling)")
        if np.any(np.diag(c) != 0):
            raise ValueError(
                "coupling_per_m must have a zero diagonal; use core_beta0_offsets_per_m"
            )
        for name in ("core_beta0_offsets_per_m", "core_beta1_offsets_s_per_m"):
            v = getattr(self, name)
            if v is not None and (v.shape != (c.shape[0],) or not np.all(np.isfinite(v))):
                raise ValueError(f"{name} must be {c.shape[0]} finite values, got shape {v.shape}")
        return self

    @property
    def n_cores(self) -> int:
        """Number of cores ``M``."""
        return int(self.coupling_per_m.shape[0])

    @classmethod
    def from_fiber(
        cls,
        fiber: Fiber,
        coupling_per_m: npt.ArrayLike,
        *,
        core_beta0_offsets_per_m: npt.ArrayLike | None = None,
        core_beta1_offsets_s_per_m: npt.ArrayLike | None = None,
    ) -> MulticoreFiber:
        """Cores with ``fiber``'s parameters (loss, dispersion, γ, steps, scheme, backend)."""
        return cls.model_validate(
            {
                **{k: getattr(fiber, k) for k in Fiber.model_fields},
                "coupling_per_m": coupling_per_m,
                "core_beta0_offsets_per_m": core_beta0_offsets_per_m,
                "core_beta1_offsets_s_per_m": core_beta1_offsets_s_per_m,
            }
        )

    def recentred(self, frequency_offset_hz: float) -> MulticoreFiber:
        r"""As :meth:`Fiber.recentred`; each core's skew adds :math:`-\delta\beta_{1,m}\omega_k`.

        That per-core constant cannot be taken out as a common phase, so it
        goes into ``core_beta0_offsets_per_m``.
        """
        fiber = super().recentred(frequency_offset_hz)
        skew = self.core_beta1_offsets_s_per_m
        if skew is None or frequency_offset_hz == 0.0:
            return fiber  # type: ignore[return-value]
        base = self.core_beta0_offsets_per_m
        offsets = (np.zeros(self.n_cores) if base is None else base) - skew * (
            2.0 * np.pi * frequency_offset_hz
        )
        return fiber.replace(core_beta0_offsets_per_m=offsets)  # type: ignore[return-value]

    def core(self, index: int) -> Fiber:
        """The single-core :class:`~hpcom.fiber.Fiber` of core ``index`` with no coupling."""
        if not 0 <= index < self.n_cores:
            raise IndexError(f"core {index} of {self.n_cores}")
        fields = {k: getattr(self, k) for k in Fiber.model_fields}
        if self.core_beta1_offsets_s_per_m is not None:
            fields["beta1_s_per_m"] = self.beta1_s_per_m + float(
                self.core_beta1_offsets_s_per_m[index]
            )
        return Fiber(**fields)


def ring_coupling(n_cores: int, coupling_per_m: float) -> npt.NDArray[np.float64]:
    """Nearest-neighbour coupling of ``n_cores`` cores on a ring (``n_cores ≥ 3``)."""
    if n_cores < 3:
        raise ValueError(f"a ring needs at least 3 cores, got {n_cores}")
    c = np.zeros((n_cores, n_cores))
    i = np.arange(n_cores)
    c[i, (i + 1) % n_cores] = c[(i + 1) % n_cores, i] = coupling_per_m
    return c


def linear_array_coupling(n_cores: int, coupling_per_m: float) -> npt.NDArray[np.float64]:
    """Nearest-neighbour coupling of ``n_cores ≥ 2`` cores in a line (two cores: a coupler)."""
    if n_cores < 2:
        raise ValueError(f"need at least 2 cores, got {n_cores}")
    c = np.zeros((n_cores, n_cores))
    i = np.arange(n_cores - 1)
    c[i, i + 1] = c[i + 1, i] = coupling_per_m
    return c


def hexagonal_coupling(coupling_per_m: float) -> npt.NDArray[np.float64]:
    """7-core hexagonal fiber, nearest neighbours: core 0 at the centre, cores 1–6 on the ring.

    The centre couples to all six ring cores; each ring core to the centre and
    its two ring neighbours (equal pitch).
    """
    c = np.zeros((7, 7))
    c[0, 1:] = c[1:, 0] = coupling_per_m
    c[1:, 1:] = ring_coupling(6, coupling_per_m)
    return c
