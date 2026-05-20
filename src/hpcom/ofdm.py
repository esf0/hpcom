"""OFDM (orthogonal frequency-division multiplexing) configuration and
modulation primitives.

An :class:`OFDMConfig` describes one OFDM "symbol" — a length-``n_subcarriers``
IFFT block — and how to slice it into guard, pilot, and data subcarriers.
The :class:`hpcom.transmitter.OFDMTransmitter` treats the parent
``Transmitter.n_symbols`` parameter as the number of OFDM frames, each
carrying ``n_data`` QAM symbols on data subcarriers and ``n_pilots`` known
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

Power normalisation convention (see :func:`_ofdm_modulate_frames`): the
time-domain field is scaled by ``N / sqrt(n_data)`` so that — with a
unit-power constellation and no pilots — the average sample power is 1.
This matches the single-carrier ``Transmitter`` convention and lets the
same launch-power machinery downstream operate without OFDM-aware logic.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hpcom._typing import ComplexArray, _is_power_of_two
from hpcom.modulation import nearest_symbol_indices
from hpcom.receiver import ReceivedSymbols, Receiver
from hpcom.transmitter import (
    Polarization,
    TransmittedSignal,
    Transmitter,
    _as_1d,
    _seed_seq_from,
)


class OFDMConfig(BaseModel):
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
    pilot_symbol
        Complex value placed on every pilot subcarrier. Defaults to
        ``1.0 + 0j``. The same value is used on every pilot subcarrier
        across every frame (BPSK-style reference); per-subcarrier sequences
        (Zadoff-Chu, etc.) are out of scope for v1.
    """

    model_config = ConfigDict(frozen=True)

    n_subcarriers: int
    cp_len: int = Field(default=0, ge=0)
    n_guard: int = Field(default=0, ge=0)
    pilot_subcarrier_indices: tuple[int, ...] = ()
    pilot_symbol: complex = 1.0 + 0.0j

    @model_validator(mode="after")
    def _validate(self) -> OFDMConfig:
        if not _is_power_of_two(self.n_subcarriers) or self.n_subcarriers < 4:
            raise ValueError(
                f"n_subcarriers must be a power of 2 ≥ 4, got {self.n_subcarriers}"
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
        # Canonicalise the pilot tuple (sorted, ints). With frozen=True the
        # public setattr is blocked, but model_validator can assign via
        # __dict__ before pydantic finalises the instance.
        self.__dict__["pilot_subcarrier_indices"] = tuple(sorted(pilots))
        return self

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


# ---------------------------------------------------------------------------
# OFDM signal + transmitter (importing Transmitter ABC from hpcom.transmitter)
# ---------------------------------------------------------------------------


def _ofdm_modulate_frames(
    data_symbols: ComplexArray,
    cfg: OFDMConfig,
) -> ComplexArray:
    """IFFT-based OFDM modulation with cyclic prefix.

    Parameters
    ----------
    data_symbols
        Shape ``(n_frames, n_data)``. Per-frame QAM symbols at the data
        subcarrier positions, in the same order as
        ``cfg.data_subcarrier_indices()``.
    cfg
        OFDM configuration: subcarrier count, CP length, guard size,
        pilot positions and value.

    Returns
    -------
    A 1-D complex field of length ``n_frames * (n_subcarriers + cp_len)``,
    obtained by:
    1. Building each frame's frequency-domain vector (data symbols at data
       subcarrier indices, ``cfg.pilot_symbol`` at pilot indices, zero
       elsewhere — including guard subcarriers).
    2. IFFT (NumPy convention: ``x_n = (1/N) Σ_k X_k exp(2πi kn/N)``).
    3. Scaling by ``N / sqrt(n_data)`` so the data-driven part of the
       field has unit average sample power (pilots add a small bias).
    4. Prepending the last ``cp_len`` samples of each IFFT block as the
       cyclic prefix (no-op when ``cp_len == 0``).
    5. Concatenating frames in time order.

    The scale factor is derived from Parseval's theorem: for unit-power
    data symbols X with n_data non-zero entries out of N,
    ``E[|x_n|²] = n_data / N²``, so scaling by ``N / sqrt(n_data)`` gives
    ``E[|x_n|²] = 1`` over the data ensemble.
    """
    data_symbols = np.asarray(data_symbols, dtype=np.complex128)
    if data_symbols.ndim != 2 or data_symbols.shape[1] != cfg.n_data:
        raise ValueError(
            f"data_symbols must have shape (n_frames, {cfg.n_data}), "
            f"got {data_symbols.shape}"
        )
    n_frames = data_symbols.shape[0]
    n = cfg.n_subcarriers

    # Build the frequency-domain matrix in one allocation.
    freq = np.zeros((n_frames, n), dtype=np.complex128)
    freq[:, cfg.data_subcarrier_indices()] = data_symbols
    if cfg.n_pilots > 0:
        freq[:, list(cfg.pilot_subcarrier_indices)] = cfg.pilot_symbol
    # Guard subcarriers remain zero by construction.

    # IFFT along the subcarrier axis, then scale to unit per-sample power.
    time = np.fft.ifft(freq, axis=1)
    time *= n / np.sqrt(cfg.n_data)

    # Prepend cyclic prefix (last cp_len samples of each frame).
    if cfg.cp_len > 0:
        cp = time[:, -cfg.cp_len :]
        time_with_cp = np.concatenate([cp, time], axis=1)
    else:
        time_with_cp = time

    return time_with_cp.reshape(-1)


class OFDMSignal(TransmittedSignal):
    """OFDM transmitted signal.

    Shape conventions (``n_symbols`` here is the number of OFDM frames):

    * Single-pol: ``field=(n_symbols * (N + cp_len),)``,
      ``symbols=(n_symbols * n_data,)``,
      ``bits=(n_bits_per_stream,)``.
    * Dual-pol:   ``field=(2, n_symbols * (N + cp_len))``,
      ``symbols=(2, n_symbols * n_data)``,
      ``bits=(2, n_bits_per_stream)``.
    """

    ofdm: OFDMConfig


class OFDMTransmitter(Transmitter):
    """Orthogonal-frequency-division-multiplexing transmitter.

    Generates ``n_symbols`` OFDM frames; each frame carries ``ofdm.n_data``
    QAM symbols on data subcarriers, ``ofdm.n_pilots`` known pilot symbols
    on pilot subcarriers, and ``2 * ofdm.n_guard`` zero-valued guard
    subcarriers, then IFFTs and prepends a cyclic prefix.

    API conventions:

    * ``symbol_rate_hz`` is the **total OFDM signal bandwidth** (= sample
      rate of the generated field). ``dt_s = 1 / symbol_rate_hz``.
    * ``n_symbols`` is the number of OFDM frames. The field length is
      ``n_symbols * (n_subcarriers + cp_len)`` samples per polarisation.
    * The base-class ``pilots`` field (time-domain periodic pilots) is
      rejected — frequency-domain pilots are configured via
      :class:`OFDMConfig` instead.

    Power normalisation follows the unit-mean-power convention: with a
    unit-power constellation and no pilots, the field's average sample
    power equals 1.

    Probabilistic shaping via ``dm`` is supported; the block-alignment
    requirement is ``n_symbols * ofdm.n_data % dm.block_length == 0``.
    """

    ofdm: OFDMConfig

    @model_validator(mode="after")
    def _validate_ofdm(self) -> OFDMTransmitter:
        if self.pilots is not None:
            raise ValueError(
                "OFDMTransmitter does not accept base-class `pilots` "
                "(time-domain periodic pilots). Configure frequency-domain "
                "pilots via `ofdm.pilot_subcarrier_indices` instead."
            )
        if self.ofdm.n_data == 0:
            raise ValueError(
                "ofdm.n_data == 0 — no data subcarriers available. "
                "Reduce n_guard or pilot count to leave room for data."
            )
        return self

    @property
    def dt_s(self) -> float:
        """OFDM sample period = 1 / total bandwidth."""
        return 1.0 / self.symbol_rate_hz

    def _data_symbols_per_stream(self) -> int:
        """Override: one OFDM frame carries ``ofdm.n_data`` QAM symbols."""
        return self.n_symbols * self.ofdm.n_data

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> OFDMSignal:
        """Generate an OFDM transmitted signal.

        Shapes for deterministic inputs:

        - single-pol: ``bits=(n_bits_per_stream,)``,
          ``symbols=(n_symbols * ofdm.n_data,)``.
        - dual-pol:   ``bits=(2, n_bits_per_stream)``,
          ``symbols=(2, n_symbols * ofdm.n_data)``.
        """
        if bits is not None and symbols is not None:
            raise ValueError("pass at most one of `bits` or `symbols`, not both")

        if self.polarization == "single":
            single_rng = rng if rng is not None else np.random.default_rng(self.seed)
            ch_bits, ch_syms, ch_field = self._generate_one_ofdm_stream(
                single_rng,
                bits=_as_1d(bits, "bits") if bits is not None else None,
                symbols=_as_1d(symbols, "symbols") if symbols is not None else None,
            )
            return self._make_ofdm_signal(
                field=ch_field, symbols=ch_syms, bits=ch_bits, polarization="single",
            )

        # Dual-pol
        bps = self._bits_per_stream()
        total_data_syms = self.n_symbols * self.ofdm.n_data
        if bits is not None:
            b2 = np.asarray(bits, dtype=np.uint8)
            if b2.shape != (2, bps):
                raise ValueError(
                    f"dual-pol `bits` must have shape (2, {bps}), got {b2.shape}"
                )
        if symbols is not None:
            s2 = np.asarray(symbols, dtype=np.complex128)
            if s2.shape != (2, total_data_syms):
                raise ValueError(
                    f"dual-pol `symbols` must have shape (2, {total_data_syms}), "
                    f"got {s2.shape}"
                )

        seed_seq = _seed_seq_from(rng, self.seed)
        pol_seqs = seed_seq.spawn(2)
        per_pol_bits: list[npt.NDArray[np.uint8]] = []
        per_pol_syms: list[ComplexArray] = []
        per_pol_fields: list[ComplexArray] = []
        for pol_idx, sub_seq in enumerate(pol_seqs):
            pol_rng = np.random.default_rng(sub_seq)
            p_bits, p_syms, p_field = self._generate_one_ofdm_stream(
                pol_rng,
                bits=bits[pol_idx] if bits is not None else None,
                symbols=symbols[pol_idx] if symbols is not None else None,
            )
            per_pol_bits.append(p_bits)
            per_pol_syms.append(p_syms)
            per_pol_fields.append(p_field)

        return self._make_ofdm_signal(
            field=np.stack(per_pol_fields, axis=0),
            symbols=np.stack(per_pol_syms, axis=0),
            bits=np.stack(per_pol_bits, axis=0),
            polarization="dual",
        )

    def _generate_one_ofdm_stream(
        self,
        rng: np.random.Generator,
        *,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> tuple[npt.NDArray[np.uint8], ComplexArray, ComplexArray]:
        """Generate (bits, data-symbols, field) for one polarisation stream."""
        from hpcom.modulation import nearest_symbol_indices

        cfg = self._resolved_constellation()
        dm = self.dm
        n_bits = self._bits_per_stream()
        total_data_syms = self.n_symbols * self.ofdm.n_data

        if symbols is not None:
            ch_symbols = np.asarray(symbols, dtype=np.complex128)
            if ch_symbols.shape != (total_data_syms,):
                raise ValueError(
                    f"`symbols` must have shape ({total_data_syms},), "
                    f"got {ch_symbols.shape}"
                )
            if dm is not None:
                indices = nearest_symbol_indices(ch_symbols, cfg.points).astype(np.intp)
                ch_bits = dm.decode(indices)
            else:
                ch_bits = cfg.demap(ch_symbols)
        elif bits is not None:
            ch_bits = np.asarray(bits, dtype=np.uint8)
            if ch_bits.shape != (n_bits,):
                raise ValueError(
                    f"`bits` must have shape ({n_bits},), got {ch_bits.shape}"
                )
            if dm is not None:
                indices = dm.encode(ch_bits)
                ch_symbols = cfg.points[indices].astype(np.complex128)
            else:
                ch_symbols = np.asarray(cfg.map(ch_bits), dtype=np.complex128)
        else:
            ch_bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
            if dm is not None:
                indices = dm.encode(ch_bits)
                ch_symbols = cfg.points[indices].astype(np.complex128)
            else:
                ch_symbols = np.asarray(cfg.map(ch_bits), dtype=np.complex128)

        data_matrix = ch_symbols.reshape(self.n_symbols, self.ofdm.n_data)
        ch_field = _ofdm_modulate_frames(data_matrix, self.ofdm)
        return ch_bits, ch_symbols, ch_field

    def _make_ofdm_signal(
        self,
        *,
        field: ComplexArray,
        symbols: ComplexArray,
        bits: npt.NDArray[np.uint8],
        polarization: Polarization,
    ) -> OFDMSignal:
        return OFDMSignal(
            field=field,
            dt_s=self.dt_s,
            symbols=symbols,
            bits=bits,
            n_symbols=self.n_symbols,
            polarization=polarization,
            pilots=None,  # OFDM rejects base pilots; freq-domain pilots in OFDMConfig
            ofdm=self.ofdm,
        )


# ---------------------------------------------------------------------------
# OFDM demodulation (receiver side)
# ---------------------------------------------------------------------------


def _ofdm_demodulate_frames(
    field: ComplexArray,
    cfg: OFDMConfig,
) -> tuple[ComplexArray, ComplexArray | None]:
    """FFT-based OFDM demodulation: undo CP, FFT, extract data and pilots.

    Parameters
    ----------
    field
        1-D complex baseband field of length ``n_frames * (N + cp_len)``.
    cfg
        OFDM configuration matching the transmitter.

    Returns
    -------
    data_symbols
        Shape ``(n_frames, n_data)`` — recovered QAM symbols at data
        subcarrier positions.
    pilot_symbols
        Shape ``(n_frames, n_pilots)`` when pilots are configured, else
        ``None``.

    The inverse of :func:`_ofdm_modulate_frames`: strips the cyclic prefix,
    FFTs each frame, undoes the Tx power scaling (multiply by
    ``sqrt(n_data) / N``), and slices out data and pilot subcarrier values.
    """
    field = np.asarray(field, dtype=np.complex128)
    n = cfg.n_subcarriers
    sps = cfg.samples_per_ofdm_symbol  # N + cp_len
    if field.ndim != 1 or field.size % sps != 0:
        raise ValueError(
            f"field length {field.size} is not a multiple of "
            f"samples_per_ofdm_symbol ({sps})"
        )
    n_frames = field.size // sps

    # Reshape into frames and strip CP.
    frames = field.reshape(n_frames, sps)
    useful = frames[:, cfg.cp_len:]  # (n_frames, N)

    # FFT along the subcarrier axis, then undo Tx power scaling.
    spectrum = np.fft.fft(useful, axis=1)
    spectrum *= np.sqrt(cfg.n_data) / n

    # Extract data subcarrier values.
    data_idx = cfg.data_subcarrier_indices()
    data_symbols: ComplexArray = spectrum[:, data_idx]

    # Extract pilot subcarrier values (if any).
    pilot_symbols: ComplexArray | None = None
    if cfg.n_pilots > 0:
        pilot_symbols = spectrum[:, list(cfg.pilot_subcarrier_indices)]

    return data_symbols, pilot_symbols


def _pilot_equalize(
    data_2d: ComplexArray,
    pilot_2d: ComplexArray,
    cfg: OFDMConfig,
) -> ComplexArray:
    """One-tap channel equalisation using pilot subcarriers.

    For each frame, the channel is estimated at pilot positions as
    ``H_pilot = received_pilot / known_pilot``, then linearly interpolated
    to all data subcarrier positions, and data symbols are divided by the
    interpolated channel estimate.

    Parameters
    ----------
    data_2d
        Shape ``(n_frames, n_data)`` — received (noisy) data symbols.
    pilot_2d
        Shape ``(n_frames, n_pilots)`` — received pilot symbols.
    cfg
        OFDM configuration (supplies pilot/data indices and known pilot
        value).

    Returns
    -------
    Equalized data symbols, same shape as *data_2d*.
    """
    pilot_indices = np.array(cfg.pilot_subcarrier_indices, dtype=np.float64)
    data_indices = cfg.data_subcarrier_indices().astype(np.float64)

    # Channel estimate at pilot positions.
    h_pilot = pilot_2d / cfg.pilot_symbol  # (n_frames, n_pilots)

    n_frames = data_2d.shape[0]
    equalized = np.empty_like(data_2d)

    for f in range(n_frames):
        # Interpolate channel magnitude and phase separately would be
        # fragile; interpolate real and imaginary parts independently.
        h_re = np.interp(data_indices, pilot_indices, h_pilot[f].real)
        h_im = np.interp(data_indices, pilot_indices, h_pilot[f].imag)
        h_data = h_re + 1j * h_im
        equalized[f] = data_2d[f] / h_data

    return equalized


class OFDMReceivedSymbols(ReceivedSymbols):
    """Output of :class:`OFDMReceiver`.

    Shape conventions (``n_symbols`` is the number of OFDM frames):

    * Single-pol: ``symbols=(n_frames * n_data,)``,
      ``bits=(n_bits_per_stream,)``.
    * Dual-pol:   ``symbols=(2, n_frames * n_data)``,
      ``bits=(2, n_bits_per_stream)``.

    OFDM has no time-domain pilots, so :meth:`data_bits` and
    :meth:`data_symbols` just return :attr:`bits` / :attr:`symbols` directly.
    """

    ofdm: OFDMConfig


class OFDMReceiver(Receiver):
    """Orthogonal-frequency-division-multiplexing receiver.

    Demodulates ``n_frames`` OFDM frames: strips the cyclic prefix, FFTs,
    extracts data subcarriers, optionally applies one-tap pilot-based
    channel equalisation, then hard-decodes to nearest constellation point.

    API conventions:

    * ``ofdm`` must match the transmitter's ``OFDMConfig`` (subcarrier
      count, CP length, guard/pilot layout).
    * The base-class ``pilots`` field (time-domain periodic pilots) is
      rejected — frequency-domain pilots are configured via
      :class:`OFDMConfig` instead.
    * ``samples_per_symbol`` is not used by OFDM; it is always set to the
      default (inherited from :class:`Receiver`) and ignored.

    Probabilistic shaping via ``dm`` is supported; the block-alignment
    requirement is ``n_frames * ofdm.n_data % dm.block_length == 0``.
    """

    ofdm: OFDMConfig

    @model_validator(mode="after")
    def _validate_ofdm_rx(self) -> OFDMReceiver:
        if self.pilots is not None:
            raise ValueError(
                "OFDMReceiver does not accept base-class `pilots` "
                "(time-domain periodic pilots). Configure frequency-domain "
                "pilots via `ofdm.pilot_subcarrier_indices` instead."
            )
        return self

    def receive(
        self, field: ComplexArray, dt_s: float | None = None
    ) -> OFDMReceivedSymbols:
        """Demodulate a received OFDM field.

        For ``polarization="single"`` the input is 1-D; for ``"dual"`` it
        is ``(2, n_samples)`` and each polarisation is demodulated
        independently.
        """
        if self.polarization == "single":
            if field.ndim != 1:
                raise ValueError(
                    f"single-pol receiver expects 1-D field, got shape {field.shape}"
                )
            return self._receive_one_ofdm_stream(field)

        # Dual-pol
        if field.ndim != 2 or field.shape[0] != 2:
            raise ValueError(
                f"dual-pol receiver expects 2-D field with leading axis 2, "
                f"got shape {field.shape}"
            )
        per_pol = [self._receive_one_ofdm_stream(field[i]) for i in (0, 1)]
        return OFDMReceivedSymbols(
            symbols=np.stack([p.symbols for p in per_pol], axis=0),
            bits=np.stack([p.bits for p in per_pol], axis=0),
            decisions=np.stack([p.decisions for p in per_pol], axis=0),
            modulation_order=self.effective_modulation_order,
            polarization="dual",
            pilots=None,
            pilot_symbols=None,
            ofdm=self.ofdm,
        )

    def _receive_one_ofdm_stream(
        self, field: ComplexArray
    ) -> OFDMReceivedSymbols:
        """Single 1-D OFDM demodulation pipeline."""
        data_2d, pilot_2d = _ofdm_demodulate_frames(field, self.ofdm)

        # One-tap pilot equalisation (when pilots are configured).
        if pilot_2d is not None and self.ofdm.n_pilots > 0:
            data_2d = _pilot_equalize(data_2d, pilot_2d, self.ofdm)

        symbols_rx = data_2d.reshape(-1)

        cfg = self._resolved_constellation()
        decisions = nearest_symbol_indices(symbols_rx, cfg.points)

        if self.dm is not None:
            if decisions.size % self.dm.block_length != 0:
                raise ValueError(
                    f"received {decisions.size} symbols, which is not a "
                    f"multiple of dm.block_length ({self.dm.block_length}); "
                    "check that the field length matches the transmitter"
                )
            bits = self.dm.decode(decisions.astype(np.intp))
        else:
            bits = cfg.demap(symbols_rx)

        return OFDMReceivedSymbols(
            symbols=symbols_rx,
            bits=bits,
            decisions=decisions,
            modulation_order=self.effective_modulation_order,
            polarization="single",
            pilots=None,
            pilot_symbols=None,
            ofdm=self.ofdm,
        )


__all__ = [
    "OFDMConfig",
    "OFDMReceivedSymbols",
    "OFDMReceiver",
    "OFDMSignal",
    "OFDMTransmitter",
]
