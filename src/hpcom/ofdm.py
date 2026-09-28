r"""OFDM: frame configuration, transmitter, receiver.

An :class:`OFDMConfig` describes one OFDM frame: ``n_subcarriers`` (``N``)
subcarriers spaced by ``Δf``, split into data, pilot and guard subcarriers,
plus a cyclic prefix. Subcarrier indices are in NumPy ``fftfreq`` order
(``0 … N/2−1`` non-negative, ``N/2 … N−1`` negative frequencies); subcarrier
``k`` sits at baseband frequency ``k·Δf`` (``(k−N)·Δf`` for ``k ≥ N/2``).
The ``2·n_guard`` guard subcarriers straddle the band edge ``±N·Δf/2``.

**Oversampling.** With ``oversampling = L`` the IFFT has ``L·N`` points: the
``N`` subcarriers occupy the central ``N·Δf`` of an ``L·N·Δf`` sample rate
and the rest of the spectrum is empty. ``L = 1`` fills the whole simulated
band — fine for linear channels, but in nonlinear (SSFM) propagation the
spectral broadening then aliases; use ``L ≥ 2`` there.

**Normalisation.** Each frame's time samples are

.. math:: x_n = \frac{L N}{\sqrt{E_\text{frame}}}\,\sqrt{P_\text{pol}}\;
          \text{IFFT}_{LN}\{X\}_n,
          \qquad E_\text{frame} = n_\text{data} E_s + n_\text{pilot}|p|^2,

so the expected field power is the launch power per polarisation
:math:`P_\text{pol}`; the cyclic prefix repeats the last ``cp_len·L``
samples. The receiver undoes the scaling up to :math:`\sqrt{P_\text{pol}}`
and the channel, which the equaliser removes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator, model_validator

from hpcom._model import HpcomModel, Polymorphic
from hpcom._typing import ComplexArray, _is_power_of_two
from hpcom.receiver import ReceivedSymbols, Receiver
from hpcom.transmitter import TransmittedSignal, Transmitter

__all__ = ["OFDMConfig", "OFDMReceivedSymbols", "OFDMReceiver", "OFDMSignal", "OFDMTransmitter"]


class OFDMConfig(HpcomModel):
    """Structure of one OFDM frame.

    Attributes
    ----------
    n_subcarriers
        Number of subcarriers ``N`` (a power of two, ≥ 4).
    cp_len
        Cyclic-prefix length in non-oversampled samples, ``0 ≤ cp_len < N``.
    n_guard
        Unused subcarriers on each side of the band edge, ``0 ≤ n_guard < N/2``.
    pilot_subcarrier_indices
        Subcarriers carrying ``pilot_symbol`` in every frame (``fftfreq``
        order, stored sorted, outside the guard band).
    pilot_symbol
        Pilot value on the unit-power constellation scale.
    oversampling
        IFFT oversampling factor ``L`` (see module docstring).
    """

    n_subcarriers: int
    cp_len: int = Field(default=0, ge=0)
    n_guard: int = Field(default=0, ge=0)
    pilot_subcarrier_indices: tuple[int, ...] = ()
    pilot_symbol: complex = 1.0 + 0.0j
    oversampling: int = Field(default=1, ge=1)

    @field_validator("pilot_subcarrier_indices", mode="after")
    @classmethod
    def _sort_pilots(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(v)) != len(v):
            raise ValueError(f"pilot_subcarrier_indices must be unique, got {v}")
        return tuple(sorted(v))

    @model_validator(mode="after")
    def _validate(self) -> OFDMConfig:
        n = self.n_subcarriers
        if not _is_power_of_two(n) or n < 4:
            raise ValueError(f"n_subcarriers must be a power of 2 ≥ 4, got {n}")
        if self.cp_len >= n:
            raise ValueError(f"cp_len must be < n_subcarriers ({n}), got {self.cp_len}")
        if self.n_guard >= n // 2:
            raise ValueError(f"n_guard must be < n_subcarriers/2 ({n // 2}), got {self.n_guard}")
        out_of_range = [p for p in self.pilot_subcarrier_indices if not 0 <= p < n]
        if out_of_range:
            raise ValueError(f"pilot subcarriers {out_of_range} are outside [0, {n})")
        clash = set(self.pilot_subcarrier_indices) & set(self.guard_subcarrier_indices().tolist())
        if clash:
            raise ValueError(f"pilot subcarriers {sorted(clash)} fall in the guard band")
        # n_data == 0 (a pilot-only frame) is a valid structure; a transmitter
        # rejects it because it has no data to send.
        return self

    # ---- subcarrier sets ---------------------------------------------------

    def guard_subcarrier_indices(self) -> npt.NDArray[np.intp]:
        """Guard subcarriers ``N/2 − n_guard … N/2 + n_guard − 1`` (around the band edge)."""
        half = self.n_subcarriers // 2
        return np.arange(half - self.n_guard, half + self.n_guard, dtype=np.intp)

    def data_subcarrier_indices(self) -> npt.NDArray[np.intp]:
        """Subcarriers carrying data, in ``fftfreq`` order."""
        used = np.ones(self.n_subcarriers, dtype=bool)
        used[self.guard_subcarrier_indices()] = False
        used[list(self.pilot_subcarrier_indices)] = False
        return np.flatnonzero(used).astype(np.intp)

    def subcarrier_frequency_index(self, k: npt.ArrayLike) -> npt.NDArray[np.intp]:
        """Signed frequency index of subcarriers ``k`` (``k``, or ``k − N`` for ``k ≥ N/2``)."""
        k = np.asarray(k, dtype=np.intp)
        return np.where(k < self.n_subcarriers // 2, k, k - self.n_subcarriers)

    def fft_bins(self, k: npt.ArrayLike) -> npt.NDArray[np.intp]:
        """Position of subcarriers ``k`` in the ``L·N``-point (oversampled) FFT."""
        return np.asarray(self.subcarrier_frequency_index(k) % self.fft_size, dtype=np.intp)

    @property
    def n_data(self) -> int:
        return self.n_subcarriers - 2 * self.n_guard - len(self.pilot_subcarrier_indices)

    @property
    def n_pilots(self) -> int:
        return len(self.pilot_subcarrier_indices)

    @property
    def fft_size(self) -> int:
        """IFFT/FFT length ``L·N``."""
        return self.n_subcarriers * self.oversampling

    @property
    def samples_per_frame(self) -> int:
        """Time samples per frame, ``L·(N + cp_len)``."""
        return self.oversampling * (self.n_subcarriers + self.cp_len)

    def frame_energy(self, data_symbol_energy: float) -> float:
        """``E_frame = n_data·E_s + n_pilots·|p|²`` (see module docstring)."""
        return self.n_data * data_symbol_energy + self.n_pilots * abs(self.pilot_symbol) ** 2


class OFDMSignal(TransmittedSignal):
    """Output of :class:`OFDMTransmitter`.

    ``field``: ``(n_frames·samples_per_frame,)`` or ``(2, …)``;
    ``symbols``: data symbols frame after frame, ``(…, n_frames·n_data)``;
    ``bits``: ``(…, n_bits_per_stream)``.
    """

    transmitter: Polymorphic[OFDMTransmitter]

    @property
    def data_indices(self) -> npt.NDArray[np.intp]:
        return np.arange(self.symbols.shape[-1], dtype=np.intp)


class OFDMTransmitter(Transmitter):
    """CP-OFDM transmitter (optionally dual-polarisation).

    Attributes
    ----------
    subcarrier_spacing_hz
        ``Δf``. The subcarrier grid spans ``N·Δf``; the sample rate is ``L·N·Δf``.
    n_frames
        Number of OFDM frames.
    ofdm
        Frame structure, :class:`OFDMConfig`.
    """

    subcarrier_spacing_hz: float = Field(gt=0)
    n_frames: int = Field(ge=1)
    ofdm: OFDMConfig

    @property
    def dt_s(self) -> float:
        return 1.0 / (self.ofdm.fft_size * self.subcarrier_spacing_hz)

    @property
    def bandwidth_hz(self) -> float:
        """Subcarrier grid width ``N·Δf``."""
        return self.ofdm.n_subcarriers * self.subcarrier_spacing_hz

    @property
    def n_data_symbols_per_stream(self) -> int:
        return self.n_frames * self.ofdm.n_data

    @property
    def n_samples(self) -> int:
        return self.n_frames * self.ofdm.samples_per_frame

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.ArrayLike | None = None,
        symbols: npt.ArrayLike | None = None,
    ) -> OFDMSignal:
        cfg = self.ofdm
        out_bits, data = self._draw_data(rng, bits, symbols)
        lead = data.shape[:-1]
        grid = np.zeros((*lead, self.n_frames, cfg.fft_size), dtype=np.complex128)
        grid[..., cfg.fft_bins(cfg.data_subcarrier_indices())] = data.reshape(
            *lead, self.n_frames, cfg.n_data
        )
        if cfg.n_pilots:
            grid[..., cfg.fft_bins(cfg.pilot_subcarrier_indices)] = cfg.pilot_symbol
        scale = cfg.fft_size * np.sqrt(
            self.launch_power_per_polarization_w / cfg.frame_energy(self.data_symbol_energy)
        )
        frames = np.fft.ifft(grid, axis=-1) * scale
        cp = cfg.cp_len * cfg.oversampling
        if cp:
            frames = np.concatenate([frames[..., -cp:], frames], axis=-1)
        return OFDMSignal(
            transmitter=self, field=frames.reshape(*lead, -1), symbols=data, bits=out_bits
        )


class OFDMReceivedSymbols(ReceivedSymbols):
    """Output of :class:`OFDMReceiver`; ``symbols`` are frame after frame (``…, n_frames·n_data``)."""


class OFDMReceiver(Receiver):
    """CP-OFDM receiver.

    After (optional) CD compensation, every frame's cyclic prefix is dropped,
    the ``L·N``-point FFT taken and the subcarriers extracted. ``equalize``:

    * ``"auto"`` (default) — ``"pilot"`` when the frame has pilot
      subcarriers, otherwise as for the other receivers (``"data_aided"``
      with a reference, else ``"unit_power"``).
    * ``"pilot"`` — one-tap per-subcarrier equaliser: the channel is measured
      on the pilot subcarriers of each frame and linearly interpolated (real
      and imaginary parts, in frequency order; held constant beyond the
      outermost pilots) to the data subcarriers. Needs pilot subcarriers.
    * ``"none"``, ``"unit_power"``, ``"data_aided"`` — one complex gain per
      polarisation, as for the other receivers.
    """

    ofdm: OFDMConfig

    @model_validator(mode="after")
    def _validate_ofdm_rx(self) -> OFDMReceiver:
        if self.equalize == "pilot" and self.ofdm.n_pilots == 0:
            raise ValueError('equalize="pilot" requires ofdm.pilot_subcarrier_indices')
        return self

    def compatibility_issues(self, transmitter: Transmitter) -> list[str]:
        if not isinstance(transmitter, OFDMTransmitter):
            return [f"OFDMReceiver cannot receive a {type(transmitter).__name__}"]
        issues = self._base_issues(transmitter)
        if transmitter.ofdm != self.ofdm:
            issues.append(f"ofdm: transmitter {transmitter.ofdm!r}, receiver {self.ofdm!r}")
        return issues

    def reference(
        self, sent: TransmittedSignal
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.complex128]]:
        return np.asarray(sent.bits), np.asarray(sent.symbols)

    def receive(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        reference: TransmittedSignal | None = None,
    ) -> OFDMReceivedSymbols:
        cfg = self.ofdm
        x = self._compensate_dispersion(self._check_field(field), dt_s)
        spf = cfg.samples_per_frame
        if x.shape[-1] % spf:
            raise ValueError(f"field length {x.shape[-1]} is not a multiple of {spf} samples/frame")
        lead = x.shape[:-1]
        frames = x.reshape(*lead, -1, spf)[..., cfg.cp_len * cfg.oversampling :]
        spectrum = np.fft.fft(frames, axis=-1) * (
            np.sqrt(cfg.frame_energy(self.data_symbol_energy)) / cfg.fft_size
        )
        data = spectrum[..., cfg.fft_bins(cfg.data_subcarrier_indices())]  # (*lead, F, n_data)
        pilots = spectrum[..., cfg.fft_bins(cfg.pilot_subcarrier_indices)] if cfg.n_pilots else None
        reference_data = self.reference(reference)[1] if reference is not None else None

        mode = self._resolve_equalizer(cfg.n_pilots > 0, reference_data is not None)
        if mode == "pilot":
            assert pilots is not None
            h = self._interpolate_channel(pilots / cfg.pilot_symbol)
            mean_gain: npt.NDArray[np.complex128] = np.asarray(
                np.mean(h, axis=(-2, -1)), dtype=np.complex128
            )
            out = self._detect(
                (data / h).reshape(*lead, -1),
                gain=np.ones(lead, dtype=np.complex128),
                equalizer="pilot",
            )
            out["gain"] = mean_gain
            out["pilot_symbols"] = (pilots / mean_gain[..., None, None]).reshape(*lead, -1)
        else:
            out = self._detect(
                data.reshape(*lead, -1),
                raw_pilots=None if pilots is None else pilots.reshape(*lead, -1),
                reference_data=reference_data,
                equalizer=mode,
            )
        return OFDMReceivedSymbols(receiver=self, **out)

    def _interpolate_channel(
        self, h_pilot: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        """Pilot channel estimates ``(…, F, n_pilots)`` → data subcarriers ``(…, F, n_data)``."""
        cfg = self.ofdm
        f_pil = cfg.subcarrier_frequency_index(cfg.pilot_subcarrier_indices)
        f_dat = cfg.subcarrier_frequency_index(cfg.data_subcarrier_indices())
        order = np.argsort(f_pil)
        f_pil = f_pil[order]
        flat = h_pilot[..., order].reshape(-1, order.size)
        out = np.empty((flat.shape[0], f_dat.size), dtype=np.complex128)
        for i, row in enumerate(flat):
            out[i] = np.interp(f_dat, f_pil, row.real) + 1j * np.interp(f_dat, f_pil, row.imag)
        return out.reshape(*h_pilot.shape[:-1], f_dat.size)


OFDMSignal.model_rebuild()
OFDMReceivedSymbols.model_rebuild()
