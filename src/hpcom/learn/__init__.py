r"""End-to-end learning through the physical channel: blocks, systems, training, evaluation.

Compose a system from blocks, train it by gradient ascent on its rate
through the split-step simulation of the link, and evaluate it on fixed
(paired) blocks — on the training simulation or a finer one::

    from hpcom import EDFA, Fiber, Link, WDMConfig, WDMTransmitter, learn

    fiber = Fiber.ssmf(80.0, n_steps=50, polarization="dual")
    link = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
    tx = WDMTransmitter(modulation_order=64, symbol_rate_hz=32e9, n_symbols=2048,
                        polarization="dual", samples_per_symbol=8,
                        wdm=WDMConfig(n_channels=5, spacing_hz=50e9))
    system = learn.EndToEnd(transmitter=tx, link=link,
                            encoder=learn.Constellation(),
                            front_end=learn.LearnedDBP(steps_per_span=2, wideband=True))
    trained = learn.Trainer(steps=2000).fit(system)
    print(learn.evaluate(system, trained.params).mean)

Blocks (each fixed or trainable, with its own learning rate):

* encoders — :class:`QAM`, :class:`Constellation` (free points),
  :class:`NNEncoder` (QAM + MLP correction);
* pre-distorters — :class:`CNNPredistorter` (a transmitter with memory);
* receiver front ends — :class:`CDC`, :class:`DBP` (single-channel or
  wideband, fixed), :class:`LearnedDBP` (every step trainable);
* decoders — :class:`GaussianDemapper` (MI or GMI), :class:`NNDecoder`
  (±window MLP), :class:`CNNDecoder` (long memory).

:class:`Trainer` runs Adam with cosine decay and a Polyak average, through
the reversible adjoint by default. Needs JAX (``pip install "hpcom[jax]"``).
"""

from __future__ import annotations

from hpcom.learn._blocks import (
    CDC,
    DBP,
    QAM,
    CNNDecoder,
    CNNPredistorter,
    Constellation,
    Decoder,
    Encoder,
    FrontEnd,
    GaussianDemapper,
    LearnedDBP,
    NNDecoder,
    NNEncoder,
    Predistorter,
)
from hpcom.learn._system import EndToEnd, Precision
from hpcom.learn._train import Evaluation, Trained, Trainer, evaluate

__all__ = [
    "CDC",
    "DBP",
    "QAM",
    "CNNDecoder",
    "CNNPredistorter",
    "Constellation",
    "Decoder",
    "Encoder",
    "EndToEnd",
    "Evaluation",
    "FrontEnd",
    "GaussianDemapper",
    "LearnedDBP",
    "NNDecoder",
    "NNEncoder",
    "Precision",
    "Predistorter",
    "Trained",
    "Trainer",
    "evaluate",
]
