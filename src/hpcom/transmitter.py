"""Optical transmitters: abstract base + concrete subclasses per signal shape.

This module owns the abstract :class:`Transmitter` and :class:`TransmittedSignal`
bases, plus the concrete :class:`SingleCarrierTransmitter` /
:class:`SingleCarrierSignal` pair (RRC-shaped QAM). The WDM and OFDM variants
live in :mod:`hpcom.wdm` and :mod:`hpcom.ofdm` respectively — each module owns
the config, the transmitter, and the signal class for its signal shape.

Polarisation, pilots, and probabilistic shaping (``dm``) are universal fields
on the abstract base because they compose orthogonally with every signal shape.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hpcom._pulse import root_raised_cosine
from hpcom._typing import ComplexArray
from hpcom.modulation import (
    ConstantCompositionDM,
    ConstellationConfig,
    bits_per_symbol,
)
from hpcom.pilots import PilotConfig

Polarization = Literal["single", "dual"]


# ---------------------------------------------------------------------------
# TransmittedSignal: ABC + SingleCarrierSignal
# ---------------------------------------------------------------------------


class TransmittedSignal(BaseModel):
    """Abstract base for transmitter output.

    Every transmitter returns a subclass of this carrying:

    * ``field`` — the complex baseband output; shape depends on polarisation.
    * ``dt_s`` — sample period in seconds.
    * ``symbols`` — the QAM symbols that were modulated.
    * ``bits`` — the bit stream (per stream / polarisation / WDM channel).
    * ``n_symbols`` — the *generation count* used by the transmitter. Its
      physical meaning varies by signal shape (number of QAM symbols for
      single-carrier; number of OFDM frames for OFDM). Each concrete signal
      type's docstring states what ``n_symbols`` means for it.
    * ``polarization`` and ``pilots`` — universal, shape-agnostic flags.

    Signal-shape-specific metadata (RRC parameters for SC/WDM, OFDM
    parameters for OFDM, per-channel views for WDM, per-frame views for
    OFDM) lives on subclasses.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    field: ComplexArray
    dt_s: float
    symbols: ComplexArray
    bits: npt.NDArray[np.uint8]
    n_symbols: int
    polarization: Polarization = "single"
    pilots: PilotConfig | None = None


class SingleCarrierSignal(TransmittedSignal):
    """Single-carrier RRC-shaped QAM signal output.

    Shape conventions:

    * Single-pol: ``field=(n_samples,)``, ``bits=(n_symbols*k,)``,
      ``symbols=(n_symbols,)``.
    * Dual-pol:   ``field=(2, n_samples)``, ``bits=(2, n_symbols*k)``,
      ``symbols=(2, n_symbols)``.

    When ``pilots`` is set, ``symbols`` at pilot positions equal the configured
    pilot value, and the corresponding ``bits`` are zero-filled. Use
    :meth:`data_bits` / :meth:`data_symbols` for the pilot-stripped views.
    """

    samples_per_symbol: int
    modulation_order: int
    roll_off: float
    pulse_span_symbols: int

    @property
    def pilot_indices(self) -> npt.NDArray[np.intp]:
        """Symbol-grid positions of pilots, or empty if no pilots configured."""
        if self.pilots is None:
            return np.empty(0, dtype=np.intp)
        return self.pilots.pilot_indices(self.n_symbols)

    @property
    def data_indices(self) -> npt.NDArray[np.intp]:
        """Symbol-grid positions of data symbols; all positions if no pilots."""
        if self.pilots is None:
            return np.arange(self.n_symbols, dtype=np.intp)
        return self.pilots.data_indices(self.n_symbols)

    def data_symbols(self) -> ComplexArray:
        """Symbols at data positions only. Preserves leading dims (pol, WDM)."""
        if self.pilots is None:
            return self.symbols
        return self.symbols[..., self.data_indices]

    def data_bits(self) -> npt.NDArray[np.uint8]:
        """Bits at data positions only.

        Preserves the leading dims of :attr:`bits` (pol, WDM channel) and
        shrinks the trailing bit axis by the pilot fraction.
        """
        if self.pilots is None:
            return self.bits
        k = bits_per_symbol(self.modulation_order)
        idx = self.data_indices
        reshaped = self.bits.reshape(*self.bits.shape[:-1], -1, k)
        selected = reshaped[..., idx, :]
        return selected.reshape(*self.bits.shape[:-1], -1)


# ---------------------------------------------------------------------------
# Transmitter: ABC + SingleCarrierTransmitter
# ---------------------------------------------------------------------------


class Transmitter(BaseModel):
    """Abstract optical transmitter.

    Concrete subclasses (:class:`SingleCarrierTransmitter`,
    :class:`~hpcom.wdm.WDMTransmitter`, :class:`~hpcom.ofdm.OFDMTransmitter`)
    implement :meth:`generate`. The base owns the universal fields and the
    shared validation: constellation/modulation_order exclusivity, DM
    block-alignment, polarisation, RNG seed.

    Constellation specification: either ``modulation_order`` (Gray-coded
    square QAM) or ``constellation`` (a :class:`ConstellationConfig`, for
    geometric or probabilistic shaping). Exactly one must be set.

    Probabilistic shaping is enabled by setting ``dm`` (a distribution
    matcher whose ``target_pmf`` matches ``constellation.probabilities``).
    With ``dm`` active, the bit-stream length per polarisation/channel is
    ``(n_data_symbols / dm.block_length) * dm.bits_per_block`` rather than
    the usual ``n_data_symbols * log₂ M``, where ``n_data_symbols`` is what
    :meth:`_data_symbols_per_stream` returns (subclass-specific). Pilots
    and ``dm`` are mutually exclusive.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    symbol_rate_hz: float = Field(gt=0)
    n_symbols: int = Field(ge=1)
    modulation_order: int | None = None
    constellation: ConstellationConfig | None = None
    dm: ConstantCompositionDM | None = None
    seed: int | None = None
    polarization: Polarization = "single"
    pilots: PilotConfig | None = None

    @model_validator(mode="after")
    def _validate_base(self) -> Transmitter:
        if (self.modulation_order is None) == (self.constellation is None):
            raise ValueError(
                "exactly one of `modulation_order` or `constellation` must be set"
            )
        if self.modulation_order is not None:
            resolved = ConstellationConfig.from_square_qam(self.modulation_order)
        else:
            assert self.constellation is not None  # for mypy
            resolved = self.constellation
        # Cached resolved constellation; bypass frozen for internal caching.
        self.__dict__["_constellation"] = resolved

        if self.dm is not None:
            if self.pilots is not None:
                raise ValueError(
                    "`pilots` and `dm` are mutually exclusive — pilots displace data "
                    "symbols and would break the DM block composition"
                )
            if self.constellation is None or self.constellation.probabilities is None:
                raise ValueError(
                    "`dm` requires `constellation` with non-None probabilities"
                )
            if self.dm.target_pmf.shape != self.constellation.probabilities.shape or not (
                np.allclose(self.dm.target_pmf, self.constellation.probabilities, atol=1e-12)
            ):
                raise ValueError(
                    "dm.target_pmf must match constellation.probabilities elementwise"
                )
            n_data_syms = self._data_symbols_per_stream()
            if n_data_syms % self.dm.block_length != 0:
                raise ValueError(
                    f"data-symbols-per-stream ({n_data_syms}) must be a multiple of "
                    f"dm.block_length ({self.dm.block_length})"
                )
        return self

    @abstractmethod
    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> TransmittedSignal:
        """Generate a transmitted signal. Concrete subclasses implement this."""

    @property
    @abstractmethod
    def dt_s(self) -> float:
        """Sample period of the generated field in seconds."""

    # ---- shared accessors ------------------------------------------------

    @property
    def effective_modulation_order(self) -> int:
        """Constellation size — always defined, even when ``modulation_order`` is None."""
        return self._resolved_constellation().modulation_order

    def _resolved_constellation(self) -> ConstellationConfig:
        """The internal :class:`ConstellationConfig` (built from
        ``modulation_order`` or passed in directly)."""
        return self.__dict__["_constellation"]  # type: ignore[no-any-return]

    def _data_symbols_per_stream(self) -> int:
        """Number of QAM data symbols emitted per polarisation/channel stream.

        For single-carrier this equals :attr:`n_symbols`. Subclasses with a
        different symbol/frame structure (e.g. OFDM, where one ``n_symbols``
        increment = one OFDM frame = ``ofdm.n_data`` QAM symbols) override
        this so the DM block-alignment check and bit-stream sizing land on
        the actual QAM symbol count.
        """
        return self.n_symbols

    def _bits_per_stream(self) -> int:
        """Length of the bit-stream input expected per polarisation/channel.

        Without DM: ``n_data_symbols * k``. With DM:
        ``(n_data_symbols / block_length) * bits_per_block``.
        """
        n_data_syms = self._data_symbols_per_stream()
        if self.dm is not None:
            return (n_data_syms // self.dm.block_length) * self.dm.bits_per_block
        return n_data_syms * self._resolved_constellation().bits_per_symbol


# ---------------------------------------------------------------------------
# SingleCarrierTransmitter
# ---------------------------------------------------------------------------


class SingleCarrierTransmitter(Transmitter):
    """Single-carrier QAM optical transmitter (optionally dual-polarisation).

    Produces one RRC-pulse-shaped baseband stream per polarisation. For WDM
    (multiple carriers), use :class:`~hpcom.wdm.WDMTransmitter`. For OFDM,
    use :class:`~hpcom.ofdm.OFDMTransmitter`.

    Setting ``polarization="dual"`` gives bits/symbols/field a leading axis
    of size 2; the two polarisations are seeded independently.
    """

    samples_per_symbol: int = Field(default=16, ge=2)
    roll_off: float = Field(default=0.1, ge=0.0, le=1.0)
    pulse_span_symbols: int = Field(default=64, ge=0)

    @property
    def dt_s(self) -> float:
        """Sample period of the generated field in seconds."""
        return 1.0 / (self.symbol_rate_hz * self.samples_per_symbol)

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> SingleCarrierSignal:
        """Generate a single-carrier transmitted signal.

        ``rng`` overrides the configured ``seed`` (useful when a parent
        Simulation threads reproducibility across stages).

        For deterministic inputs (testing, replay) pass either ``bits`` or
        ``symbols``. Shapes:

        - single-pol: ``bits=(n_symbols*k,)``, ``symbols=(n_symbols,)``.
        - dual-pol:  ``bits=(2, n_symbols*k)``, ``symbols=(2, n_symbols)``.

        ``bits`` and ``symbols`` are mutually exclusive.
        """
        if bits is not None and symbols is not None:
            raise ValueError("pass at most one of `bits` or `symbols`, not both")

        if self.polarization == "single":
            single_rng = rng if rng is not None else np.random.default_rng(self.seed)
            ch_bits, ch_syms, field = self._generate_one_channel(
                single_rng,
                bits=_as_1d(bits, "bits") if bits is not None else None,
                symbols=_as_1d(symbols, "symbols") if symbols is not None else None,
            )
            return self._make_signal(
                field=field, symbols=ch_syms, bits=ch_bits, polarization="single"
            )

        # Dual-pol single-carrier.
        bps = self._bits_per_stream()
        if bits is not None:
            b2 = np.asarray(bits, dtype=np.uint8)
            if b2.shape != (2, bps):
                raise ValueError(
                    f"dual-pol `bits` must have shape (2, {bps}), got {b2.shape}"
                )
        if symbols is not None:
            s2 = np.asarray(symbols, dtype=np.complex128)
            if s2.shape != (2, self.n_symbols):
                raise ValueError(
                    f"dual-pol `symbols` must have shape (2, {self.n_symbols}), "
                    f"got {s2.shape}"
                )

        seed_seq = _seed_seq_from(rng, self.seed)
        pol_seqs = seed_seq.spawn(2)

        per_pol_bits: list[npt.NDArray[np.uint8]] = []
        per_pol_syms: list[ComplexArray] = []
        per_pol_fields: list[ComplexArray] = []
        for pol_idx, sub_seq in enumerate(pol_seqs):
            pol_rng = np.random.default_rng(sub_seq)
            p_bits, p_syms, p_field = self._generate_one_channel(
                pol_rng,
                bits=bits[pol_idx] if bits is not None else None,
                symbols=symbols[pol_idx] if symbols is not None else None,
            )
            per_pol_bits.append(p_bits)
            per_pol_syms.append(p_syms)
            per_pol_fields.append(p_field)

        return self._make_signal(
            field=np.stack(per_pol_fields, axis=0),
            symbols=np.stack(per_pol_syms, axis=0),
            bits=np.stack(per_pol_bits, axis=0),
            polarization="dual",
        )

    def _make_signal(
        self,
        *,
        field: ComplexArray,
        symbols: ComplexArray,
        bits: npt.NDArray[np.uint8],
        polarization: Polarization,
    ) -> SingleCarrierSignal:
        """Bundle a ``SingleCarrierSignal`` with this Tx's metadata."""
        return SingleCarrierSignal(
            field=field,
            dt_s=self.dt_s,
            symbols=symbols,
            bits=bits,
            samples_per_symbol=self.samples_per_symbol,
            modulation_order=self.effective_modulation_order,
            roll_off=self.roll_off,
            pulse_span_symbols=self.pulse_span_symbols,
            n_symbols=self.n_symbols,
            polarization=polarization,
            pilots=self.pilots,
        )

    def _generate_one_channel(
        self,
        rng: np.random.Generator,
        *,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> tuple[npt.NDArray[np.uint8], ComplexArray, ComplexArray]:
        """Generate ``(bits, symbols, baseband-field)`` for one 1-D stream.

        WDM subclasses orchestrate multiple calls to this method (one per
        channel × polarisation) to build their composite output.
        """
        from hpcom.modulation import nearest_symbol_indices  # local to avoid cycle

        cfg = self._resolved_constellation()
        dm = self.dm
        n_bits = self._bits_per_stream()

        if symbols is not None:
            ch_symbols = np.asarray(symbols, dtype=np.complex128)
            if ch_symbols.shape != (self.n_symbols,):
                raise ValueError(
                    f"`symbols` must have shape ({self.n_symbols},), "
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

        if self.pilots is not None:
            k = cfg.bits_per_symbol
            pilot_idx = self.pilots.pilot_indices(self.n_symbols)
            ch_symbols = ch_symbols.copy()
            ch_symbols[pilot_idx] = self.pilots.symbol
            ch_bits = ch_bits.copy()
            for p in pilot_idx:
                ch_bits[int(p) * k : (int(p) + 1) * k] = 0

        n_samples = self.n_symbols * self.samples_per_symbol
        upsampled = np.zeros(n_samples, dtype=np.complex128)
        upsampled[:: self.samples_per_symbol] = ch_symbols

        n_taps = self.pulse_span_symbols * self.samples_per_symbol + 1
        rrc = root_raised_cosine(n_taps, self.samples_per_symbol, self.roll_off)
        full = np.convolve(upsampled, rrc, mode="full")
        delay = (n_taps - 1) // 2
        field = full[delay : delay + n_samples].astype(np.complex128)
        return ch_bits, ch_symbols, field


# ---------------------------------------------------------------------------
# Internal helpers (used by SC and WDM/OFDM subclasses living in other modules)
# ---------------------------------------------------------------------------


def _as_1d(arr: np.ndarray, name: str) -> np.ndarray:
    """Validate that a deterministic-input array is 1-D (single-channel path)."""
    if arr.ndim != 1:
        raise ValueError(
            f"`{name}` must be 1-D for a single-channel Transmitter, got {arr.ndim}-D"
        )
    return arr


def _seed_seq_from(
    rng: np.random.Generator | None, seed: int | None
) -> np.random.SeedSequence:
    """Build a reproducible SeedSequence from either a generator or a seed."""
    if rng is not None:
        return np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
    return np.random.SeedSequence(seed)
