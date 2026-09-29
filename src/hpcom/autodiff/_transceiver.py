r"""Differentiable transmitters, receivers, demappers and information rates.

The same chain as :class:`~hpcom.SingleCarrierTransmitter` /
:class:`~hpcom.WDMTransmitter` → receiver → :mod:`hpcom.metrics`, as JAX
functions: constellation (geometric and probabilistic shaping) → pulse
shaping, WDM and launch power → … → dispersion compensation, channel
selection, front end, compensation stage (DBP or any JAX function), matched
filter, equaliser → Gaussian-channel MI, GMI and bit LLRs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from hpcom._pulse import _circular_responses, fir_taps
from hpcom.autodiff._fiber import Array
from hpcom.autodiff._link import LinkParameters, compensate
from hpcom.dsp.compensation import DigitalBackPropagation
from hpcom.modulation.shaping import ConstellationConfig
from hpcom.receiver import PulseShapedReceiver
from hpcom.transmitter import PulseShapedTransmitter
from hpcom.wdm import WDMReceiver, WDMTransmitter

__all__ = [
    "CompensateFn",
    "TransmitterParameters",
    "bit_llrs",
    "constellation",
    "demodulate",
    "information_rates",
    "modulate",
    "sample_symbols",
    "snr_db",
]

_LN2 = math.log(2.0)

#: A custom compensation stage: ``(samples, dt_s, frequency_offset_hz) → samples``.
CompensateFn = Callable[[Array, float, float], Array]


class TransmitterParameters(NamedTuple):
    """Trainable transmitter quantities as a JAX pytree.

    ``points`` are normalised to unit mean power (under the symbol PMF)
    wherever they are used, so they can be trained freely (geometric
    shaping). ``logits`` (probabilistic shaping) give the PMF
    ``softmax(logits)``; ``None`` keeps the constellation's own PMF, fixed.
    ``launch_power_dbm`` is per channel, summed over polarisations.
    """

    points: Array
    launch_power_dbm: Array
    logits: Array | None = None

    @classmethod
    def from_transmitter(
        cls, transmitter: PulseShapedTransmitter, *, trainable_pmf: bool = False
    ) -> TransmitterParameters:
        """The transmitter's constellation and launch power (``logits`` if ``trainable_pmf``)."""
        cfg = transmitter.resolved_constellation
        dtype = jnp.result_type(float)
        with np.errstate(divide="ignore"):
            logits = np.log(cfg.pmf)
        return cls(
            points=jnp.asarray(cfg.points, jnp.result_type(dtype, jnp.complex64)),
            launch_power_dbm=jnp.asarray(
                10.0 * math.log10(transmitter.launch_power_per_channel_w / 1e-3), dtype
            ),
            logits=jnp.asarray(logits, dtype) if trainable_pmf else None,
        )

    def to_constellation(self, like: ConstellationConfig) -> ConstellationConfig:
        """A :class:`~hpcom.ConstellationConfig` with these points (and PMF), ``like``'s labels."""
        points = np.asarray(self.points, dtype=np.complex128)
        pmf = None if self.logits is None else np.asarray(jax.nn.softmax(self.logits), np.float64)
        return ConstellationConfig.from_points(
            points,
            bit_labels=like.bit_labels,
            probabilities=pmf if pmf is not None else like.probabilities,
        )


def constellation(
    transmitter: PulseShapedTransmitter, params: TransmitterParameters | None = None
) -> tuple[Array, Array]:
    """``(points, pmf)``: the constellation normalised to :math:`E|s|^2 = 1`, and its PMF."""
    p = TransmitterParameters.from_transmitter(transmitter) if params is None else params
    cfg = transmitter.resolved_constellation
    pmf = jnp.asarray(cfg.pmf) if p.logits is None else jax.nn.softmax(p.logits)
    points = jnp.asarray(p.points)
    energy = jnp.sum(pmf * jnp.abs(points) ** 2)
    return points / jnp.sqrt(energy), pmf


def sample_symbols(
    key: Array, points: Array, logits: Array, shape: tuple[int, ...], temperature: float = 1.0
) -> tuple[Array, Array]:
    r"""Draw ``shape`` symbols from ``softmax(logits)``, differentiable in the PMF.

    Straight-through Gumbel-softmax (as in end-to-end probabilistic shaping,
    Stark, Ait Aoudia & Hoydis 2019): the forward pass sends exact
    constellation points ``points[indices]``; the backward pass uses the
    gradient of the relaxed one-hot :math:`\mathrm{softmax}((\log p + g)/T)`.
    Returns ``(symbols, indices)``.
    """
    gumbel = jax.random.gumbel(key, (*shape, logits.shape[-1]), dtype=jnp.real(points).dtype)
    soft = jax.nn.softmax((jax.nn.log_softmax(logits) + gumbel) / temperature, axis=-1)
    indices = jnp.argmax(soft, axis=-1)
    hard = jax.nn.one_hot(indices, logits.shape[-1], dtype=soft.dtype)
    onehot = hard + soft - jax.lax.stop_gradient(soft)
    return onehot.astype(points.dtype) @ points, indices


# ---------------------------------------------------------------------------
# Transmitter
# ---------------------------------------------------------------------------


def _slot_stream(transmitter: PulseShapedTransmitter, data: Array) -> Array:
    """Insert the pilots: ``(..., n_data)`` → ``(..., n_symbols)``."""
    if transmitter.pilots is None:
        return data
    slots = jnp.full(
        (*data.shape[:-1], transmitter.n_symbols), transmitter.pilots.symbol, data.dtype
    )
    return slots.at[..., transmitter.data_indices].set(data)


def _stream_energy(transmitter: PulseShapedTransmitter) -> float:
    """Expected ``|s|²`` per slot for unit-energy data symbols, pilots included."""
    n_pil = transmitter.pilot_indices.size
    pilot = abs(transmitter.pilots.symbol) ** 2 if transmitter.pilots is not None else 0.0
    return (transmitter.n_data_symbols_per_stream + n_pil * pilot) / transmitter.n_symbols


def _linear_same(x: Array, taps: np.ndarray) -> Array:
    delay = (taps.size - 1) // 2
    n = x.shape[-1]
    flat = x.reshape(-1, n)
    k = jnp.asarray(taps, jnp.real(x).dtype)
    out = jax.vmap(lambda row: jnp.convolve(row, k, mode="full")[delay : delay + n])(flat)
    return out.reshape(x.shape)


def _pulse_shape(transmitter: PulseShapedTransmitter, slots: Array) -> Array:
    sps = transmitter.samples_per_symbol
    n = slots.shape[-1] * sps
    up = jnp.zeros((*slots.shape[:-1], n), slots.dtype).at[..., ::sps].set(slots)
    if transmitter.convolution == "circular":
        h_tx, _ = _circular_responses(n, sps, transmitter.roll_off, transmitter.pulse_span_symbols)
        return jnp.fft.ifft(jnp.fft.fft(up, axis=-1) * jnp.asarray(h_tx), axis=-1)
    assert transmitter.pulse_span_symbols is not None
    taps = np.sqrt(sps) * fir_taps(sps, transmitter.roll_off, transmitter.pulse_span_symbols)
    return _linear_same(up, taps)


def modulate(
    symbols: Array,
    transmitter: PulseShapedTransmitter,
    *,
    launch_power_dbm: Array | None = None,
) -> Array:
    """:meth:`generate(symbols=…) <hpcom.SingleCarrierTransmitter.generate>` as a JAX function.

    ``symbols`` are the data symbols on the unit-power scale, shape
    ``(*transmitter.stream_shape, n_data_symbols_per_stream)`` — e.g.
    ``points[indices]`` from :func:`constellation`, or :func:`sample_symbols`.
    Pilots are inserted, every stream is RRC-shaped, scaled to the launch
    power (``launch_power_dbm`` per channel, default the transmitter's) and,
    for WDM, moved to its channel frequency and summed. Returns the field,
    ``(n_samples,)`` or ``(2, n_samples)``.
    """
    if transmitter.dm is not None:
        raise NotImplementedError("distribution matchers are not differentiable")
    s = jnp.asarray(symbols)
    if s.shape != (*transmitter.stream_shape, transmitter.n_data_symbols_per_stream):
        raise ValueError(
            f"symbols must have shape {(*transmitter.stream_shape, transmitter.n_data_symbols_per_stream)}, "
            f"got {s.shape}"
        )
    rdtype = jnp.real(s).dtype
    if launch_power_dbm is None:
        per_channel_w = jnp.asarray(transmitter.launch_power_per_channel_w, rdtype)
    else:
        per_channel_w = 1e-3 * 10.0 ** (jnp.asarray(launch_power_dbm, rdtype) / 10.0)
    per_pol_w = per_channel_w / transmitter.n_polarizations
    scale = jnp.sqrt(per_pol_w / _stream_energy(transmitter))
    shaped = _pulse_shape(transmitter, _slot_stream(transmitter, s)) * scale
    if isinstance(transmitter, WDMTransmitter):
        t = np.arange(transmitter.n_samples) * transmitter.dt_s
        carriers = np.exp(2j * np.pi * transmitter.wdm.channel_frequencies_hz()[:, None] * t)
        shaped = jnp.sum(shaped * jnp.asarray(carriers, shaped.dtype), axis=-2)
    return shaped


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------


def _lowpass(x: Array, dt_s: float, bandwidth_hz: float) -> Array:
    keep = np.abs(np.fft.fftfreq(x.shape[-1], d=dt_s)) <= bandwidth_hz / 2.0
    return jnp.fft.ifft(jnp.fft.fft(x, axis=-1) * jnp.asarray(keep, jnp.real(x).dtype), axis=-1)


def _resample(x: Array, n_samples: int) -> Array:
    """:func:`hpcom.dsp.frontend.resample` (band-limited, periodic)."""
    n_in = x.shape[-1]
    if n_samples == n_in:
        return x
    m = min(n_in, n_samples)
    k = np.fft.fftfreq(m, d=1.0 / m).round().astype(np.intp)
    k = k[np.abs(2 * k) < m]
    spectrum = jnp.fft.fft(x, axis=-1)
    out = jnp.zeros((*x.shape[:-1], n_samples), spectrum.dtype)
    out = out.at[..., k % n_samples].set(spectrum[..., k % n_in])
    return jnp.fft.ifft(out, axis=-1) * (n_samples / n_in)


def _matched_filter(receiver: PulseShapedReceiver, x: Array, sps: int) -> Array:
    n = x.shape[-1]
    if n % sps:
        raise ValueError(f"field length {n} is not a multiple of samples_per_symbol={sps}")
    if receiver.convolution == "circular":
        _, h_rx = _circular_responses(n, sps, receiver.roll_off, receiver.pulse_span_symbols)
        y = jnp.fft.ifft(jnp.fft.fft(x, axis=-1) * jnp.asarray(h_rx), axis=-1)
    else:
        assert receiver.pulse_span_symbols is not None
        y = _linear_same(
            x, fir_taps(sps, receiver.roll_off, receiver.pulse_span_symbols) / np.sqrt(sps)
        )
    return y[..., ::sps]


def _cd_compensate(receiver: PulseShapedReceiver, x: Array, dt_s: float) -> Array:
    if receiver.cd_compensation_s2 == 0.0 and receiver.cd_compensation_s3 == 0.0:
        return x
    omega = 2.0 * np.pi * np.fft.fftfreq(x.shape[-1], d=dt_s)
    phase = (
        -receiver.cd_compensation_s2 / 2.0 * omega**2 + receiver.cd_compensation_s3 / 6.0 * omega**3
    )
    return jnp.fft.ifft(jnp.fft.fft(x, axis=-1) * jnp.asarray(np.exp(1j * phase), x.dtype), axis=-1)


def _offsets(receiver: PulseShapedReceiver) -> np.ndarray | None:
    if isinstance(receiver, WDMReceiver):
        return receiver.wdm.channel_frequencies_hz()[list(receiver.selected_channels)]
    return None


def _baseband(receiver: PulseShapedReceiver, x: Array, dt_s: float) -> Array:
    offsets = _offsets(receiver)
    if offsets is None:
        return x
    t = np.arange(x.shape[-1]) * dt_s
    down = np.exp(-2j * np.pi * offsets[:, None] * t)
    return x[..., None, :] * jnp.asarray(down, x.dtype)


def _run_compensation(
    receiver: PulseShapedReceiver,
    samples: Array,
    dt_s: float,
    offset_hz: float,
    compensation: CompensateFn | None,
    compensation_params: LinkParameters | None,
    nonlinearity_scale: Array | None,
) -> Array:
    if compensation is not None:
        return compensation(samples, dt_s, offset_hz)
    stage = receiver.compensation
    if not isinstance(stage, DigitalBackPropagation):
        raise NotImplementedError(
            f"{type(stage).__name__} has no differentiable model; pass compensation=<a JAX function>"
        )
    return compensate(
        samples,
        stage,
        dt_s,
        params=compensation_params,
        nonlinearity_scale=nonlinearity_scale,
        frequency_offset_hz=offset_hz,
    )


def demodulate(
    field: Array,
    receiver: PulseShapedReceiver,
    dt_s: float,
    *,
    reference: Array | None = None,
    compensation: CompensateFn | None = None,
    compensation_params: LinkParameters | None = None,
    nonlinearity_scale: Array | None = None,
) -> Array:
    """:meth:`receive(...).symbols <hpcom.SingleCarrierReceiver.receive>` as a JAX function.

    Dispersion compensation, channel selection (WDM), the compensation
    stage's front end (low-pass, resampling) and stage, matched filter,
    symbol-rate sampling and the per-stream complex-gain equaliser, exactly
    as the receiver does. Returns the equalised data symbols,
    ``(*streams, n_data)``.

    ``reference`` — the transmitted data symbols ``(*streams, n_data)`` —
    feeds the data-aided equaliser (``equalize="auto"`` without pilots).
    With ``receiver.compensation`` set, the stage is ``compensation`` (any
    JAX function ``(samples, dt_s, frequency_offset_hz) → samples``, e.g. a
    neural network) or, by default, the receiver's
    :class:`~hpcom.DigitalBackPropagation` with trainable
    ``compensation_params`` (its link model) and ``nonlinearity_scale``.
    """
    if receiver.dm is not None:
        raise NotImplementedError("distribution matchers are not differentiable")
    x = _cd_compensate(receiver, jnp.asarray(field), dt_s)
    stage = receiver.compensation
    sps = receiver.samples_per_symbol

    def run(samples: Array, dt: float, offset_hz: float) -> Array:
        return _run_compensation(
            receiver, samples, dt, offset_hz, compensation, compensation_params, nonlinearity_scale
        )

    if stage is not None and stage.bandwidth_hz is None and stage.samples_per_symbol in (None, sps):
        x = run(x, dt_s, 0.0)  # identity front end: the whole field, once
        stage = None
    streams = _baseband(receiver, x, dt_s)
    if stage is not None:
        n = streams.shape[-1]
        sps_out = stage.samples_per_symbol or sps
        if stage.bandwidth_hz is not None:
            streams = _lowpass(streams, dt_s, stage.bandwidth_hz)
        if sps_out != sps:
            streams = _resample(streams, n // sps * sps_out)
        dt_c = dt_s * sps / sps_out
        offsets = _offsets(receiver)
        if offsets is None:
            streams = run(streams, dt_c, 0.0)
        else:
            streams = jnp.stack(
                [run(streams[..., c, :], dt_c, float(f)) for c, f in enumerate(offsets)], axis=-2
            )
        sps = sps_out
    slots = _matched_filter(receiver, streams, sps)
    return _equalize(receiver, slots, reference)


def _equalize(receiver: PulseShapedReceiver, slots: Array, reference: Array | None) -> Array:
    n_slots = slots.shape[-1]
    pilots = receiver.pilots
    if pilots is None:
        data, raw_pilots = slots, None
    else:
        data = slots[..., pilots.data_indices(n_slots)]
        raw_pilots = slots[..., pilots.pilot_indices(n_slots)]
    mode = receiver.equalize
    if mode == "auto":
        mode = (
            "pilot"
            if raw_pilots is not None
            else ("data_aided" if reference is not None else "unit_power")
        )
    if mode == "none":
        return data
    if mode == "unit_power":
        gain = jnp.sqrt(jnp.mean(jnp.abs(data) ** 2, axis=-1) / receiver.data_symbol_energy)
    elif mode == "pilot":
        assert raw_pilots is not None and pilots is not None
        gain = jnp.sum(np.conj(pilots.symbol) * raw_pilots, axis=-1) / (
            raw_pilots.shape[-1] * abs(pilots.symbol) ** 2
        )
    else:  # data_aided
        if reference is None:
            raise ValueError('equalize="data_aided" needs reference= (the sent data symbols)')
        ref = jnp.asarray(reference)
        if ref.shape != data.shape:
            raise ValueError(f"reference has shape {ref.shape}, received data {data.shape}")
        gain = jnp.sum(jnp.conj(ref) * data, axis=-1) / jnp.sum(jnp.abs(ref) ** 2, axis=-1)
    return data / gain[..., None]


# ---------------------------------------------------------------------------
# Demapper and information rates
# ---------------------------------------------------------------------------


def _metric(y: Array, points: Array, pmf: Array, sigma2: Array) -> Array:
    """``ln p(x) q(y|x)`` for every (sample, point): ``(n, M)``."""
    d = jnp.abs(y.reshape(-1)[:, None] - points[None, :]) ** 2
    return -d / sigma2 + jnp.log(pmf)[None, :]


def information_rates(
    received: Array,
    sent_indices: Array,
    points: Array,
    bit_labels: np.ndarray,
    pmf: Array | None = None,
) -> tuple[Array, Array]:
    """``(MI, GMI)`` in bits/symbol — :func:`hpcom.metrics.information_rates` in JAX.

    Gaussian auxiliary channel with :math:`\\sigma^2 = \\langle|y - x|^2\\rangle`,
    anchored at the source entropy; differentiable in the received samples,
    the points and the PMF (maximise either to train a transceiver).
    """
    y = jnp.asarray(received).reshape(-1)
    idx = jnp.asarray(sent_indices).reshape(-1)
    pts = jnp.asarray(points)
    p = jnp.full(pts.shape[0], 1.0 / pts.shape[0], jnp.real(pts).dtype) if pmf is None else pmf
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log2(jnp.where(p > 0, p, 1.0)), 0.0))
    sigma2 = jnp.mean(jnp.abs(y - pts[idx]) ** 2)
    metric = _metric(y, pts, p, sigma2)
    log_den = logsumexp(metric, axis=1)
    sent = jnp.take_along_axis(metric, idx[:, None], axis=1)[:, 0]
    mi = entropy - jnp.mean(log_den - sent) / _LN2
    labels = jnp.asarray(bit_labels, jnp.real(metric).dtype)  # (M, m)
    sent_bits = labels[idx]  # (n, m)
    match = sent_bits[:, :, None] == labels.T[None, :, :]  # (n, m, M)
    log_num = logsumexp(metric[:, None, :], axis=2, b=match.astype(metric.dtype))
    gmi = entropy - jnp.mean(jnp.sum(log_den[:, None] - log_num, axis=1)) / _LN2
    return mi, gmi


def bit_llrs(
    received: Array, points: Array, bit_labels: np.ndarray, sigma2: Array, pmf: Array | None = None
) -> Array:
    r"""Max-free (exact) bit LLRs :math:`\ln\frac{\sum_{b_i=0} p\,q}{\sum_{b_i=1} p\,q}`, ``(n, m)``."""
    pts = jnp.asarray(points)
    p = jnp.full(pts.shape[0], 1.0 / pts.shape[0], jnp.real(pts).dtype) if pmf is None else pmf
    metric = _metric(jnp.asarray(received), pts, p, sigma2)
    ones = jnp.asarray(bit_labels, metric.dtype).T  # (m, M)
    l0 = logsumexp(metric[:, None, :], axis=2, b=1.0 - ones[None])
    l1 = logsumexp(metric[:, None, :], axis=2, b=ones[None])
    return l0 - l1


def snr_db(received: Array, reference: Array) -> Array:
    """Effective SNR ``Σ|x|² / Σ|y − x|²`` in dB (:func:`hpcom.metrics.snr_db`)."""
    x = jnp.asarray(reference)
    return 10.0 * jnp.log10(
        jnp.sum(jnp.abs(x) ** 2) / jnp.sum(jnp.abs(jnp.asarray(received) - x) ** 2)
    )
