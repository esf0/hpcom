"""Blocks of an end-to-end system: encoders, pre-distorters, receiver front ends, decoders.

Every block is a validated configuration (:class:`~hpcom.HpcomModel`) with
``init`` (its trainable parameters, a JAX pytree; ``{}`` for a fixed block)
and the function it applies. :class:`~hpcom.learn.EndToEnd` chains them.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from hpcom._model import HpcomModel
from hpcom.autodiff import information_rates, learned_dbp, learned_dbp_init
from hpcom.autodiff._fiber import Array
from hpcom.dsp.compensation import DigitalBackPropagation
from hpcom.link import Link
from hpcom.receiver import PulseShapedReceiver
from hpcom.transmitter import PulseShapedTransmitter

__all__ = [
    "CDC",
    "DBP",
    "QAM",
    "CNNDecoder",
    "CNNPredistorter",
    "Constellation",
    "Decoder",
    "EffectiveSNR",
    "Encoder",
    "FrontEnd",
    "GaussianDemapper",
    "LearnedDBP",
    "NNDecoder",
    "NNEncoder",
    "Predistorter",
]

Params = Any  # a JAX pytree


class Context(NamedTuple):
    """What the blocks need to know about the system (built by EndToEnd)."""

    transmitter: PulseShapedTransmitter
    link: Link
    channels: tuple[int, ...] | None  # scored WDM channels; None: single carrier
    qam: Array  # (M,) the transmitter's constellation, unit power
    bit_labels: np.ndarray
    n_data: int  # data symbols per stream
    rdtype: Any
    cdtype: Any


# ---------------------------------------------------------------------------
# Small networks (circular in time: the blocks of symbols are periodic)
# ---------------------------------------------------------------------------


def _mlp_init(key: Array, sizes: list[int], dtype: Any, last_scale: float = 1.0) -> list[Any]:
    keys = jax.random.split(key, len(sizes) - 1)
    layers = [
        (jax.random.normal(k, (a, b), dtype) * math.sqrt(2.0 / (a + b)), jnp.zeros(b, dtype))
        for k, a, b in zip(keys, sizes[:-1], sizes[1:], strict=True)
    ]
    w, b = layers[-1]
    layers[-1] = (w * last_scale, b)
    return layers


def _mlp(layers: list[Any], x: Array) -> Array:
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = jax.nn.relu(x)
    return x


def _conv_init(
    key: Array, channels_in: int, width: int, kernel: int, n_layers: int, dtype: Any
) -> list[Any]:
    keys = jax.random.split(key, n_layers)
    layers, c = [], channels_in
    for k in keys:
        w = jax.random.normal(k, (width, c, kernel), dtype) * math.sqrt(2.0 / (c * kernel))
        layers.append((w, jnp.zeros(width, dtype)))
        c = width
    return layers


def _conv(layers: list[Any], x: Array, dilations: tuple[int, ...]) -> Array:
    """Dilated circular 1-D convolutions with ReLU: (channels, n) → (width, n)."""
    x = x[None]
    for (w, b), d in zip(layers, dilations, strict=True):
        pad = d * (w.shape[-1] - 1) // 2
        x = jnp.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="wrap")
        x = jax.lax.conv_general_dilated(x, w, (1,), "VALID", rhs_dilation=(d,))
        x = jax.nn.relu(x + b[None, :, None])
    return x[0]


class Block(HpcomModel, ABC):
    """A block of the chain; ``learning_rate`` is that of its parameters (Adam)."""

    learning_rate: float = Field(default=1e-3, gt=0)

    def init(self, key: Array, ctx: Context) -> Params:
        """Initial trainable parameters (``{}``: nothing to train)."""
        del key, ctx
        return {}


# ---------------------------------------------------------------------------
# Encoders: symbol index → point
# ---------------------------------------------------------------------------


class Encoder(Block, ABC):
    """Maps the ``M`` symbol indices to complex points (normalised to unit power)."""

    @abstractmethod
    def points(self, params: Params, ctx: Context) -> Array: ...


def _unit(points: Array) -> Array:
    return points / jnp.sqrt(jnp.mean(jnp.abs(points) ** 2))


class QAM(Encoder):
    """The transmitter's own constellation, fixed."""

    def points(self, params: Params, ctx: Context) -> Array:
        return ctx.qam


class Constellation(Encoder):
    """Free points (geometric shaping), starting at the transmitter's constellation."""

    learning_rate: float = Field(default=1e-2, gt=0)

    def init(self, key: Array, ctx: Context) -> Params:
        return {"points": ctx.qam}

    def points(self, params: Params, ctx: Context) -> Array:
        return _unit(params["points"])


class NNEncoder(Encoder):
    """The transmitter's constellation plus an MLP correction of every point (one-hot in)."""

    hidden: int = Field(default=64, ge=1)

    def init(self, key: Array, ctx: Context) -> Params:
        m = ctx.qam.shape[0]
        return {"mlp": _mlp_init(key, [m, self.hidden, 2], ctx.rdtype, last_scale=0.01)}

    def points(self, params: Params, ctx: Context) -> Array:
        m = ctx.qam.shape[0]
        out = _mlp(params["mlp"], jnp.eye(m, dtype=ctx.rdtype))
        return _unit(ctx.qam + (out[:, 0] + 1j * out[:, 1]).astype(ctx.cdtype))


# ---------------------------------------------------------------------------
# Pre-distorters: a transmitter with memory
# ---------------------------------------------------------------------------


class Predistorter(Block, ABC):
    """Transforms one channel's symbol sequence, ``(polarisations, n)`` → same."""

    @abstractmethod
    def apply(self, params: Params, symbols: Array, ctx: Context) -> Array: ...


class CNNPredistorter(Predistorter):
    """Residual dilated CNN over (Re, Im) of every polarisation; starts at the identity.

    Receptive field ``1 + (kernel − 1)·Σ dilations`` symbols (169 by default).
    """

    width: int = Field(default=32, ge=1)
    kernel: int = Field(default=9, ge=1)
    dilations: tuple[int, ...] = (1, 4, 16)

    def init(self, key: Array, ctx: Context) -> Params:
        k_conv, k_out = jax.random.split(key)
        pols = 2 if ctx.transmitter.polarization == "dual" else 1
        conv = _conv_init(
            k_conv, 2 * pols, self.width, self.kernel, len(self.dilations), ctx.rdtype
        )
        out = jax.random.normal(k_out, (self.width, 2 * pols), ctx.rdtype)
        return {
            "conv": conv,
            "out": (out * math.sqrt(2.0 / self.width) * 0.01, jnp.zeros(2 * pols, ctx.rdtype)),
        }

    def apply(self, params: Params, symbols: Array, ctx: Context) -> Array:
        pols = symbols.shape[0]
        x = jnp.concatenate([symbols.real, symbols.imag]).astype(ctx.rdtype)
        h = _conv(params["conv"], x, self.dilations)
        w, b = params["out"]
        y = (h.T @ w + b).T  # (2·pols, n)
        return symbols + (y[:pols] + 1j * y[pols:]).astype(symbols.dtype)


# ---------------------------------------------------------------------------
# Receiver front ends
# ---------------------------------------------------------------------------


class FrontEnd(Block, ABC):
    """The receiver up to the matched filter: CD compensation or (learned) DBP."""

    @abstractmethod
    def receiver(self, ctx: Context) -> PulseShapedReceiver: ...

    def compensation(self, params: Params, structure: Any) -> Any:
        """The JAX compensation stage for :func:`hpcom.autodiff.demodulate` (None: the receiver's own)."""
        del params, structure
        return None

    def structure(self, ctx: Context) -> Any:
        return None


def _channel_overrides(ctx: Context) -> dict[str, Any]:
    return {} if ctx.channels is None else {"channels": ctx.channels}


class CDC(FrontEnd):
    """Chromatic-dispersion compensation of the whole field (frequency domain)."""

    def receiver(self, ctx: Context) -> PulseShapedReceiver:
        rx = ctx.transmitter.matched_receiver(
            cd_compensation_s2=ctx.link.total_dispersion_s2, **_channel_overrides(ctx)
        )
        assert isinstance(rx, PulseShapedReceiver)
        return rx


class DBP(FrontEnd):
    """Digital back-propagation with the link's own parameters (fixed).

    ``wideband``: the whole received field at the simulation rate (every
    channel, as a receiver that digitises the whole band; with
    ``steps_per_span=None`` the forward steps — ideal full-field DBP).
    Otherwise each scored channel alone, low-passed to
    ``bandwidth_factor·R_s`` at ``samples_per_symbol``.
    """

    steps_per_span: int | None = Field(default=4, ge=1)
    wideband: bool = False
    samples_per_symbol: int = Field(default=2, ge=1)
    bandwidth_factor: float = Field(default=1.1, gt=0)

    def stage(self, ctx: Context) -> DigitalBackPropagation:
        if self.wideband:
            return DigitalBackPropagation(
                link=ctx.link, steps_per_span=self.steps_per_span, samples_per_symbol=None
            )
        return DigitalBackPropagation(
            link=ctx.link,
            steps_per_span=self.steps_per_span,
            samples_per_symbol=self.samples_per_symbol,
            bandwidth_hz=self.bandwidth_factor * ctx.transmitter.symbol_rate_hz,
        )

    def receiver(self, ctx: Context) -> PulseShapedReceiver:
        rx = ctx.transmitter.matched_receiver(
            compensation=self.stage(ctx), **_channel_overrides(ctx)
        )
        assert isinstance(rx, PulseShapedReceiver)
        return rx


class LearnedDBP(DBP):
    """DBP with every step trainable (:func:`hpcom.autodiff.learned_dbp`), starting as :class:`DBP`.

    Also trains one global Kerr scale ξ. Single-channel learned DBP scores
    one channel (its structure is built for that channel's offset).
    """

    knots: int = Field(default=33, ge=2)
    intensity_taps: int = Field(default=8, ge=0)

    def _sampling(self, ctx: Context) -> tuple[int, float, float]:
        tx = ctx.transmitter
        if self.wideband:
            return tx.n_symbols * tx.samples_per_symbol, tx.dt_s, 0.0
        if ctx.channels is not None and len(ctx.channels) != 1:
            raise ValueError("single-channel learned DBP scores exactly one WDM channel")
        offset = 0.0
        if ctx.channels is not None:
            offset = float(tx.wdm.channel_frequencies_hz()[ctx.channels[0]])  # type: ignore[attr-defined]
        dt = 1.0 / (self.samples_per_symbol * tx.symbol_rate_hz)
        return tx.n_symbols * self.samples_per_symbol, dt, offset

    def structure(self, ctx: Context) -> Any:
        n, dt, offset = self._sampling(ctx)
        structure, _ = learned_dbp_init(
            ctx.link,
            n,
            dt,
            steps_per_span=self.steps_per_span,
            frequency_offset_hz=offset,
            knots=self.knots,
            intensity_taps=self.intensity_taps,
        )
        return structure

    def init(self, key: Array, ctx: Context) -> Params:
        n, dt, offset = self._sampling(ctx)
        _, params = learned_dbp_init(
            ctx.link,
            n,
            dt,
            steps_per_span=self.steps_per_span,
            frequency_offset_hz=offset,
            knots=self.knots,
            intensity_taps=self.intensity_taps,
        )
        params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, ctx.rdtype), params)
        return {"dbp": params, "xi": jnp.asarray(1.0, ctx.rdtype)}

    def compensation(self, params: Params, structure: Any) -> Any:
        p = params["dbp"]._replace(nonlinear=params["dbp"].nonlinear * params["xi"])

        def stage(samples: Array, dt_s: float, offset_hz: float) -> Array:
            del dt_s, offset_hz  # fixed by the structure
            return learned_dbp(samples, structure, p)

        return stage


# ---------------------------------------------------------------------------
# Decoders: received samples → rate
# ---------------------------------------------------------------------------


class Decoder(Block, ABC):
    """Scores the received samples ``(streams, n)`` against the sent indices, in bit/symbol."""

    def calibrate(self, params: Params, z: Array, idx: Array, points: Array) -> Params:
        """Adjust the initial parameters on one received block (default: nothing)."""
        del z, idx, points
        return params

    @abstractmethod
    def rate(self, params: Params, z: Array, idx: Array, points: Array, ctx: Context) -> Array: ...


class GaussianDemapper(Decoder):
    """The mismatched Gaussian receiver: MI or GMI per stream (σ² estimated), averaged."""

    metric: Literal["mi", "gmi"] = "mi"

    def rate(self, params: Params, z: Array, idx: Array, points: Array, ctx: Context) -> Array:
        which = 0 if self.metric == "mi" else 1
        rates = [
            information_rates(z[s], idx[s], points, ctx.bit_labels)[which]
            for s in range(z.shape[0])
        ]
        return sum(rates) / len(rates)


class EffectiveSNR(Decoder):
    """Scores the effective SNR in dB, :math:`\\langle|x|^2\\rangle/\\langle|z - x|^2\\rangle` per stream, averaged.

    Not a rate: the figure of merit of DSP studies (e.g. learned DBP), and
    the objective that minimises the mean-square error.
    """

    def rate(self, params: Params, z: Array, idx: Array, points: Array, ctx: Context) -> Array:
        x = points[idx]
        signal = jnp.mean(jnp.abs(x) ** 2, axis=-1)
        error = jnp.mean(jnp.abs(z - x) ** 2, axis=-1)
        return jnp.mean(10.0 * jnp.log10(signal / error))


class _NeuralDecoder(Decoder, ABC):
    """logits = −e^s·|z − x_m|² + correction(z): starts at the Gaussian metric, s = 1/σ² measured.

    The rate is the cross-entropy bound H(X) − CE/ln 2 (a lower bound on the MI).
    """

    def calibrate(self, params: Params, z: Array, idx: Array, points: Array) -> Params:
        sigma2 = jnp.mean(jnp.abs(z - points[idx]) ** 2)
        return {**params, "log_scale": -jnp.log(sigma2).astype(params["log_scale"].dtype)}

    @abstractmethod
    def correction(self, params: Params, z: Array) -> Array: ...

    def rate(self, params: Params, z: Array, idx: Array, points: Array, ctx: Context) -> Array:
        m = points.shape[0]

        def cross_entropy(zs: Array, ids: Array) -> Array:
            metric = -jnp.exp(params["log_scale"]) * jnp.abs(zs[:, None] - points[None, :]) ** 2
            logits = metric + self.correction(params, zs)
            picked = jnp.take_along_axis(logits, ids[:, None], axis=-1)[:, 0]
            return jnp.mean(jax.nn.logsumexp(logits, axis=-1) - picked)

        ce = jnp.mean(jax.vmap(cross_entropy)(z, idx))
        return math.log2(m) - ce / math.log(2.0)


class NNDecoder(_NeuralDecoder):
    """An MLP over a ±``window``-symbol window (Re, Im) and :math:`|z_k|^2`."""

    window: int = Field(default=3, ge=0)
    hidden: tuple[int, ...] = (128, 128)

    def init(self, key: Array, ctx: Context) -> Params:
        m = ctx.qam.shape[0]
        sizes = [4 * self.window + 3, *self.hidden, m]
        return {
            "mlp": _mlp_init(key, sizes, ctx.rdtype, last_scale=0.01),
            "log_scale": jnp.asarray(0.0, ctx.rdtype),
        }

    def correction(self, params: Params, z: Array) -> Array:
        cols = [jnp.roll(z, -j) for j in range(-self.window, self.window + 1)]
        feats = jnp.stack(
            [c.real for c in cols] + [c.imag for c in cols] + [jnp.abs(z) ** 2], axis=-1
        )
        return _mlp(params["mlp"], feats.astype(params["log_scale"].dtype))


class CNNDecoder(_NeuralDecoder):
    """Dilated circular CNN over (Re z, Im z, :math:`|z|^2`) — long memory (169 symbols by default)."""

    width: int = Field(default=32, ge=1)
    kernel: int = Field(default=9, ge=1)
    dilations: tuple[int, ...] = (1, 4, 16)
    head: int = Field(default=128, ge=1)

    def init(self, key: Array, ctx: Context) -> Params:
        k_conv, k_head = jax.random.split(key)
        m = ctx.qam.shape[0]
        return {
            "conv": _conv_init(k_conv, 3, self.width, self.kernel, len(self.dilations), ctx.rdtype),
            "head": _mlp_init(k_head, [self.width, self.head, m], ctx.rdtype, last_scale=0.01),
            "log_scale": jnp.asarray(0.0, ctx.rdtype),
        }

    def correction(self, params: Params, z: Array) -> Array:
        x = jnp.stack([z.real, z.imag, jnp.abs(z) ** 2]).astype(params["log_scale"].dtype)
        return _mlp(params["head"], _conv(params["conv"], x, self.dilations).T)
