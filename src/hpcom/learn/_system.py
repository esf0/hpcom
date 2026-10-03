"""An end-to-end system: encoder → (pre-distorter) → transmitter → link → front end → decoder."""

from __future__ import annotations

import math
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import model_validator

from hpcom._model import HpcomModel, Polymorphic
from hpcom.autodiff import demodulate, modulate, propagate_link
from hpcom.autodiff._fiber import Array, GradientMode
from hpcom.learn._blocks import (
    CDC,
    QAM,
    Context,
    Decoder,
    Encoder,
    FrontEnd,
    GaussianDemapper,
    Params,
    Predistorter,
)
from hpcom.link import Link
from hpcom.transmitter import PulseShapedTransmitter, Transmitter

__all__ = ["EndToEnd", "Precision"]

#: Floating-point precision of training and evaluation.
Precision = Literal["single", "double"]


class EndToEnd(HpcomModel):
    r"""A transmitter, a link and a receiver with trainable blocks, scored by a rate.

    Per call of :meth:`rate`: random symbol indices for every stream
    (polarisation, WDM channel; all channels use the same encoder and
    pre-distorter), the encoder's points, the pre-distorter per channel
    (renormalised to unit power), pulse shaping and WDM
    (:func:`hpcom.autodiff.modulate`), the link with its ASE
    (:func:`hpcom.autodiff.propagate_link`), the front end and the
    transmitter's matched receiver (:func:`hpcom.autodiff.demodulate`:
    CD compensation or DBP, matched filter, data-aided complex gain), and the
    decoder's rate of the scored streams in bit/symbol per polarisation.
    Everything is differentiable: :class:`~hpcom.learn.Trainer` maximises it.

    Attributes
    ----------
    transmitter
        A single-carrier or WDM transmitter: symbol rate, pulse, constellation
        size (its constellation is the encoder's starting point), launch power.
    link
        Fibers and EDFAs.
    channels
        WDM channels scored (default: the centre one); ``None`` for a single
        carrier.
    encoder, predistorter, front_end, decoder
        The blocks (:mod:`hpcom.learn`); fixed or trainable.
    """

    transmitter: Polymorphic[Transmitter]
    link: Link
    channels: tuple[int, ...] | None = None
    encoder: Polymorphic[Encoder] = QAM()
    predistorter: Polymorphic[Predistorter] | None = None
    front_end: Polymorphic[FrontEnd] = CDC()
    decoder: Polymorphic[Decoder] = GaussianDemapper()

    @model_validator(mode="after")
    def _validate(self) -> EndToEnd:
        tx = self.transmitter
        if not isinstance(tx, PulseShapedTransmitter):
            raise ValueError("EndToEnd needs a pulse-shaped (single-carrier or WDM) transmitter")
        if tx.n_cores != 1:
            raise ValueError("EndToEnd is single-core")
        if tx.dm is not None or tx.resolved_constellation.probabilities is not None:
            raise ValueError("EndToEnd sends uniform symbols; use a constellation without a PMF")
        n_ch = tx.n_channels
        if n_ch > 1:
            if self.channels is not None and any(not 0 <= c < n_ch for c in self.channels):
                raise ValueError(f"channels {self.channels} are outside [0, {n_ch})")
        elif self.channels is not None:
            raise ValueError("channels are for WDM transmitters")
        return self

    @property
    def scored_channels(self) -> tuple[int, ...] | None:
        """The WDM channels scored (default: the centre one); ``None`` for a single carrier."""
        n_ch = self.transmitter.n_channels
        if n_ch == 1:
            return None
        return self.channels if self.channels is not None else (n_ch // 2,)

    # ---- runtime ---------------------------------------------------------

    def context(self, precision: Precision = "single", link: Link | None = None) -> Context:
        rdtype = jnp.float64 if precision == "double" else jnp.float32
        cdtype = jnp.complex128 if precision == "double" else jnp.complex64
        tx = self.transmitter
        assert isinstance(tx, PulseShapedTransmitter)
        cfg = tx.resolved_constellation
        points = np.asarray(cfg.points)
        points = points / np.sqrt(np.mean(np.abs(points) ** 2))
        n_data = int(np.shape(tx.generate().data_symbols())[-1])
        return Context(
            transmitter=tx,
            link=self.link if link is None else link,
            channels=self.scored_channels,
            qam=jnp.asarray(points, cdtype),
            bit_labels=np.asarray(cfg.bit_labels),
            n_data=n_data,
            rdtype=rdtype,
            cdtype=cdtype,
        )

    def init(self, key: Array, precision: Precision = "single") -> Params:
        """Initial parameters of every block, with the neural decoders calibrated on one block."""
        ctx = self.context(precision)
        k_enc, k_pre, k_front, k_dec, k_cal = jax.random.split(key, 5)
        params = {
            "encoder": self.encoder.init(k_enc, ctx),
            "predistorter": {} if self.predistorter is None else self.predistorter.init(k_pre, ctx),
            "front_end": self.front_end.init(k_front, ctx),
            "decoder": self.decoder.init(k_dec, ctx),
        }
        received = jax.jit(self._received_fn(ctx, "store", ctx.link))
        points, idx, z = received(params, k_cal)
        params["decoder"] = self.decoder.calibrate(params["decoder"], z, idx, points)
        return params

    def learning_rates(self) -> dict[str, float]:
        return {
            "encoder": self.encoder.learning_rate,
            "predistorter": 0.0 if self.predistorter is None else self.predistorter.learning_rate,
            "front_end": self.front_end.learning_rate,
            "decoder": self.decoder.learning_rate,
        }

    def rate_fn(
        self,
        precision: Precision = "single",
        gradient: GradientMode = "store",
        link: Link | None = None,
    ) -> Any:
        """``rate(params, key) → bit/symbol``: a pure function (jit it, differentiate it).

        ``link`` (default: :attr:`link`) is the channel; a finer simulation of
        the same link evaluates a trained system, while the receiver keeps
        :attr:`link` as its model.
        """
        ctx = self.context(precision)
        received = self._received_fn(ctx, gradient, self.link if link is None else link)

        def rate(params: Params, key: Array) -> Array:
            points, idx, z = received(params, key)
            return self.decoder.rate(params["decoder"], z, idx, points, ctx)

        return rate

    def _received_fn(self, ctx: Context, gradient: GradientMode, link: Link) -> Any:
        tx = ctx.transmitter
        stream_shape = tuple(tx.stream_shape)
        m = ctx.qam.shape[0]
        wdm = ctx.channels is not None
        dual = tx.polarization == "dual"
        structure = self.front_end.structure(ctx)
        rx = self.front_end.receiver(ctx)

        def received(params: Params, key: Array) -> tuple[Array, Array, Array]:
            k_data, k_noise = jax.random.split(key)
            points = self.encoder.points(params["encoder"], ctx)
            idx = jax.random.randint(k_data, (*stream_shape, ctx.n_data), 0, m)
            symbols = points[idx]
            if self.predistorter is not None:
                symbols = self._predistort(params["predistorter"], symbols, ctx, wdm, dual)
            field = modulate(symbols, tx)
            field = propagate_link(field, link, tx.dt_s, key=k_noise, gradient=gradient)
            sent = jnp.take(idx, jnp.asarray(ctx.channels), axis=-2) if wdm else idx
            z = demodulate(
                field,
                rx,
                tx.dt_s,
                reference=points[sent],
                compensation=self.front_end.compensation(params["front_end"], structure),
            )
            return points, sent.reshape(-1, ctx.n_data), z.reshape(-1, ctx.n_data)

        return received

    def _predistort(
        self, params: Params, symbols: Array, ctx: Context, wdm: bool, dual: bool
    ) -> Array:
        assert self.predistorter is not None
        # one channel's symbols as (polarisations, n)
        s = symbols if dual else symbols[None]  # (pol, [ch,] n)
        if wdm:
            s = jnp.moveaxis(s, -2, 0)  # (ch, pol, n)
            s = jax.vmap(lambda c: self.predistorter.apply(params, c, ctx))(s)
            s = s / jnp.sqrt(jnp.mean(jnp.abs(s) ** 2, axis=(1, 2), keepdims=True))
            s = jnp.moveaxis(s, 0, -2)
        else:
            s = self.predistorter.apply(params, s, ctx)
            s = s / jnp.sqrt(jnp.mean(jnp.abs(s) ** 2))
        return s if dual else s[0]

    def bits_per_symbol(self) -> float:
        tx = self.transmitter
        assert isinstance(tx, PulseShapedTransmitter)
        return math.log2(tx.resolved_constellation.modulation_order)
