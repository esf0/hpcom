"""OFDM (orthogonal frequency-division multiplexing) configuration.

An :class:`OFDMConfig` describes one OFDM "symbol" — a length-``n_subcarriers``
IFFT block — and how to slice it into guard, pilot, and data subcarriers.
The transmitter (added in T12.2) treats the parent ``Transmitter.n_symbols``
parameter as the number of OFDM symbols, each carrying ``n_data`` QAM
symbols on data subcarriers and ``len(pilot_subcarrier_indices)`` known
symbols on pilot subcarriers, surrounded by guard bands.

Subcarrier indices are in NumPy ``fftfreq`` natural order ``0, 1, …, N-1``
(equivalently, baseband from DC up to Nyquist, then wrapping into negative
frequencies). The two guard bands sit at the edges of that frequency range
— ``n_guard`` subcarriers nearest Nyquist (indices ``N/2 - n_guard``
through ``N/2 + n_guard - 1`` in fftshift order, equivalently ``N/2 ±
n_guard`` straddling the wrap-around in natural order).

DC (index 0) is always treated as a data position (it can be marked as a
pilot via ``pilot_subcarrier_indices`` if the application wants a DC pilot
or DC null).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


def _is_power_of_two(n: int) -> bool:
    return n >= 2 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class OFDMConfig:
    """One OFDM symbol's structure: subcarriers, guard, cyclic prefix, pilots.

    Parameters
    ----------
    n_subcarriers
        IFFT size; must be a power of 2 and ≥ 4.
    cp_len
        Cyclic-prefix length in samples. ``0 ≤ cp_len < n_subcarriers``.
    n_guard
        Number of guard subcarriers on each edge of the band. ``0 ≤ n_guard
        < n_subcarriers/2``; total guard = ``2 * n_guard`` (the inner block
        of high-frequency subcarriers near Nyquist).
    pilot_subcarrier_indices
        Tuple of subcarrier indices (in natural ``fftfreq`` order) that
        carry known pilot symbols. Must be disjoint from the guard set and
        within ``[0, n_subcarriers)``.
    """

    n_subcarriers: int
    cp_len: int = 0
    n_guard: int = 0
    pilot_subcarrier_indices: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not _is_power_of_two(self.n_subcarriers):
            raise ValueError(
                f"n_subcarriers must be a power of 2 ≥ 4, got {self.n_subcarriers}"
            )
        if self.n_subcarriers < 4:
            raise ValueError(
                f"n_subcarriers must be ≥ 4, got {self.n_subcarriers}"
            )
        if not 0 <= self.cp_len < self.n_subcarriers:
            raise ValueError(
                f"cp_len must satisfy 0 ≤ cp_len < n_subcarriers "
                f"({self.n_subcarriers}), got {self.cp_len}"
            )
        if not 0 <= self.n_guard < self.n_subcarriers // 2:
            raise ValueError(
                f"n_guard must satisfy 0 ≤ n_guard < n_subcarriers/2 "
                f"({self.n_subcarriers // 2}), got {self.n_guard}"
            )

        pilots = tuple(int(p) for p in self.pilot_subcarrier_indices)
        if len(set(pilots)) != len(pilots):
            raise ValueError(
                f"pilot_subcarrier_indices must be unique, got {pilots}"
            )
        for p in pilots:
            if not 0 <= p < self.n_subcarriers:
                raise ValueError(
                    f"pilot subcarrier {p} is out of range "
                    f"[0, {self.n_subcarriers})"
                )
        guard_set = set(int(i) for i in self.guard_subcarrier_indices())
        bad = guard_set & set(pilots)
        if bad:
            raise ValueError(
                f"pilot subcarriers {sorted(bad)} collide with the guard band"
            )
        # Canonicalise the pilot tuple (sorted, ints).
        object.__setattr__(self, "pilot_subcarrier_indices", tuple(sorted(pilots)))

    # ---- subcarrier sets -------------------------------------------------

    def guard_subcarrier_indices(self) -> npt.NDArray[np.intp]:
        """Indices (natural ``fftfreq`` order) that fall in the guard bands.

        Guard subcarriers sit symmetrically around Nyquist (``N/2``):
        ``N/2 - n_guard, …, N/2 + n_guard - 1``. Empty when ``n_guard == 0``.
        """
        if self.n_guard == 0:
            return np.empty(0, dtype=np.intp)
        half = self.n_subcarriers // 2
        return np.arange(half - self.n_guard, half + self.n_guard, dtype=np.intp)

    def data_subcarrier_indices(self) -> npt.NDArray[np.intp]:
        """Indices that carry QAM data symbols."""
        all_idx = np.arange(self.n_subcarriers, dtype=np.intp)
        non_data = set(self.guard_subcarrier_indices().tolist()) | set(
            self.pilot_subcarrier_indices
        )
        return np.array(
            [i for i in all_idx if int(i) not in non_data], dtype=np.intp
        )

    @property
    def n_data(self) -> int:
        """Number of data subcarriers per OFDM symbol."""
        return self.n_subcarriers - 2 * self.n_guard - len(self.pilot_subcarrier_indices)

    @property
    def n_pilots(self) -> int:
        """Number of pilot subcarriers per OFDM symbol."""
        return len(self.pilot_subcarrier_indices)

    @property
    def samples_per_ofdm_symbol(self) -> int:
        """Time-domain samples per OFDM symbol (IFFT + cyclic prefix)."""
        return self.n_subcarriers + self.cp_len


__all__ = ["OFDMConfig"]
