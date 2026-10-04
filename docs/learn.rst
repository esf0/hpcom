End-to-end learning and benchmarks
==================================

:mod:`hpcom.learn` trains transmitters and receivers by gradient ascent on
an information rate, with the gradient taken through the split-step
simulation of the whole link (:doc:`autodiff`). :mod:`hpcom.bench` fixes
systems, data and scoring so that different methods can be compared.

A system from blocks
--------------------

An :class:`~hpcom.learn.EndToEnd` system is a transmitter, a link and four
blocks. Each block is fixed or has trainable parameters with its own
learning rate.

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Position
     - Fixed
     - Trainable
   * - encoder (index → point)
     - :class:`~hpcom.learn.QAM` (the transmitter's constellation)
     - :class:`~hpcom.learn.Constellation` (free points, geometric shaping);
       :class:`~hpcom.learn.NNEncoder` (QAM + MLP correction)
   * - pre-distorter (optional, per channel)
     - —
     - :class:`~hpcom.learn.CNNPredistorter` (residual dilated CNN over the
       symbol sequence of both polarisations: a transmitter with memory)
   * - front end (receiver before the matched filter)
     - :class:`~hpcom.learn.CDC`; :class:`~hpcom.learn.DBP` (one channel at
       2 samples/symbol, or ``wideband``: the whole field)
     - :class:`~hpcom.learn.LearnedDBP` (every step's filter, Kerr
       coefficient and power filter; starts as :class:`~hpcom.learn.DBP`)
   * - decoder (samples → rate)
     - :class:`~hpcom.learn.GaussianDemapper` (MI or GMI, the mismatched
       Gaussian receiver)
     - :class:`~hpcom.learn.NNDecoder` (±window MLP),
       :class:`~hpcom.learn.CNNDecoder` (long memory); both start at the
       Gaussian metric with the measured noise variance

One evaluation of the rate draws random symbol indices for every stream
(polarisation and WDM channel; every channel uses the same encoder), maps
them, pre-distorts, shapes the pulses, propagates the field with ASE,
applies the front end, the matched filter and the data-aided complex gain,
and scores the selected channels: bit/symbol per polarisation, averaged
over the scored streams. CD compensation is always before the decoder.

.. code-block:: python

   from hpcom import EDFA, Fiber, Link, WDMConfig, WDMTransmitter, learn

   fiber = Fiber.ssmf(80.0, n_steps=50, polarization="dual")
   link = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
   tx = WDMTransmitter(modulation_order=64, symbol_rate_hz=32e9, n_symbols=2048,
                       polarization="dual", samples_per_symbol=8,
                       launch_power_per_channel_dbm=0.0,
                       wdm=WDMConfig(n_channels=5, spacing_hz=50e9))
   system = learn.EndToEnd(transmitter=tx, link=link,
                           encoder=learn.Constellation(),
                           front_end=learn.LearnedDBP(steps_per_span=2, wideband=True))
   trained = learn.Trainer(steps=2000).fit(system)
   print(learn.evaluate(system, trained.params).mean)

Training and evaluation
-----------------------

:class:`~hpcom.learn.Trainer` runs Adam for ``steps`` steps, a fresh block
of data and noise each. By default:

* the gradient through the link uses the **reversible adjoint**, so memory
  does not grow with the number of split steps;
* the learning rates decay to zero (cosine);
* the returned parameters are the **Polyak average** (factor 0.99) of the
  iterates. Without it, parameters whose gradients are mostly noise drift:
  a learned DBP ended below its own starting point in our first run.

:func:`~hpcom.learn.evaluate` scores fixed blocks drawn from a seed, the
same for every system: differences between systems are paired, with far
less variance than independent runs. ``link`` can be a finer simulation of
the same link, so that a method cannot profit from the training
simulation's own numerical errors.

Benchmark scenarios
-------------------

A :class:`~hpcom.bench.Scenario` fixes the transmitter, the link, the
power sweep, the scored channels, the evaluation blocks and the scoring
simulation (four times the training steps). Its sampling rate must be at
least twice the occupied bandwidth :math:`B`: the Kerr products of the
signal reach :math:`\pm 1.5\,B`, and at a lower rate they alias back into
the signal band (5 × 32 GBd at 8 samples/symbol loses 0.08 bit/4D at the
optimum of ``jovanovic2023``). A
:class:`~hpcom.bench.Method` is a set of blocks.
:func:`~hpcom.bench.run` trains a method at every power and scores it;
:func:`~hpcom.bench.leaderboard` ranks the results.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Scenario
     - System
   * - ``sc-dp-10x80``
     - one 32 GBd dual-polarisation channel, RRC 0.1, 1024 points (DBP
       reaches 9.1 bit/symbol: 256 points would cap it at 8), 10 × 80 km SSMF,
       EDFA NF 5 dB, −3 to +9 dBm
   * - ``wdm5-dp-10x80``
     - 5 × 32 GBd dual-polarisation on a 50 GHz grid (centre channel scored),
       16 samples/symbol, 256 points, same link, −5 to +5 dBm per channel
   * - ``jovanovic2023``
     - the system of Jovanovic et al. (JLT 2023): 5 × 32 GBd
       dual-polarisation, RRC 0.01, 16 samples/symbol, 64 points,
       10 × 100 km SSMF with
       D = 16.464 ps/(nm·km) and γ = 1.3 /W/km

Standard methods (:data:`~hpcom.bench.METHODS`): ``qam+cdc``, ``gs+cdc``,
``qam+dbp1``, ``qam+dbp4``, ``gs+ldbp1``, ``qam+wdbp2``, ``gs+wldbp2`` and
``qam+ffdbp`` (ideal full-field DBP with the forward steps). The DBP
baselines are tuned, so that a learned method is not compared with a weak
DBP: every Kerr step uses its effective length (the default of
:class:`~hpcom.learn.DBP`), and with one step per span it sits 0.15 of the
span from the input (:data:`~hpcom.bench.DBP1_SCHEME`; +0.9 to
+1.6 bit/symbol over a whole-step Kerr length at mid-span in
``sc-dp-10x80``). From a shell, e.g. on a compute node:

.. code-block:: bash

   python -m hpcom.bench --scenario wdm5-dp-10x80 --method qam+cdc gs+cdc gs+wldbp2 --out results.json

Re-running a method into the same ``--out`` file adds its new powers
(``--powers``) to the stored ones. The leaderboard marks a best power at
the edge of the sweep "(edge)": the optimum may lie outside it.

Leaderboard
-----------

Results of 4 October 2026 (``benchmarks/leaderboard/results``): 2000
training steps per power on an RTX 3070. The rate is the best over the
power sweep on the scoring simulation, in bit/symbol per polarisation,
± the standard error over the 16 paired blocks. Every best power is
inside its sweep.

.. list-table:: ``sc-dp-10x80`` (one channel, 1024 points)
   :header-rows: 1
   :widths: 30 25 20 25

   * - Method
     - Rate
     - At
     - Training per power
   * - ``qam+dbp4``
     - 9.070 ± 0.005
     - +7 dBm
     - —
   * - ``gs+ldbp1``
     - 8.344 ± 0.008
     - +5 dBm
     - 111 s
   * - ``qam+dbp1``
     - 7.551 ± 0.006
     - +3 dBm
     - —
   * - ``gs+cdc``
     - 6.872 ± 0.009
     - −1 dBm
     - 98 s
   * - ``qam+cdc``
     - 6.566 ± 0.007
     - −1 dBm
     - —

.. list-table:: ``wdm5-dp-10x80`` (5 channels, centre scored, 256 points)
   :header-rows: 1
   :widths: 30 25 20 25

   * - Method
     - Rate
     - At
     - Training per power
   * - ``qam+ffdbp`` (ideal)
     - 7.355 ± 0.005
     - +3 dBm
     - —
   * - ``gs+wldbp2``
     - 6.971 ± 0.013
     - +1 dBm
     - 146 s
   * - ``gs+ldbp1``
     - 6.881 ± 0.018
     - +1 dBm
     - 140 s
   * - ``qam+dbp4``
     - 6.724 ± 0.019
     - +1 dBm
     - —
   * - ``gs+cdc``
     - 6.585 ± 0.010
     - −1 dBm
     - 123 s
   * - ``qam+cdc``
     - 6.369 ± 0.010
     - −1 dBm
     - —
   * - ``qam+wdbp2``
     - 5.426 ± 0.008
     - −3 dBm
     - —

.. list-table:: ``jovanovic2023`` (5 channels, centre scored, 64 points)
   :header-rows: 1
   :widths: 30 25 20 25

   * - Method
     - Rate
     - At
     - Training per power
   * - ``gs+cdc``
     - 5.472 ± 0.007
     - +0.5 dBm
     - 125 s
   * - ``qam+cdc``
     - 5.357 ± 0.007
     - +0.5 dBm
     - —

What the numbers say:

* **One channel.** Fixed DBP with 4 steps per span is the best method,
  2.5 bit/symbol above CD compensation. Learned DBP at one step per span
  with learned points (8.34) lies between tuned one-step DBP (7.55) and
  4-step DBP (9.07).
* **Five channels.** Single-channel DBP gains only 0.36 bit/symbol: the
  neighbours' interference dominates. Learned wideband DBP at 2 steps per
  span with learned points (6.97) is the best practical method. It closes
  half the gap between learned points with CD compensation (6.59) and ideal
  full-field DBP (7.36). Fixed wideband DBP with the same 2 steps per span
  is 0.94 bit/symbol *below* CD compensation: over a 250 GHz band it needs
  many more steps (with three channels, 16 steps per span were needed to
  pass single-channel DBP with 4, ``research/2026-10-01-learned-dbp``).
* **Learned points** add 0.31 (one channel) and 0.22 (five channels)
  bit/symbol to QAM with CD compensation. In ``jovanovic2023`` they add
  0.115 bit/symbol per polarisation (0.23 bit/4D); the paper reports
  0.21 bit/4D. At the optimum the QAM rate equals the paper's Fig. 5
  (10.71 bit/4D) and the learned one is 0.02 bit/4D above it
  (``gallery/jovanovic2023``).

[``tests/numerical/test_learn.py``, ``tests/numerical/test_bench.py``:
the Gaussian-demapper rate of QAM over a linear link equals the MI of QAM
on the AWGN channel at the predicted SNR (0.02 bit/symbol); learned DBP
starts at DBP's rate; neural decoders start at the Gaussian demapper's;
training a constellation on an AWGN-like link beats QAM; every layout
trains through the reversible adjoint; scoring is paired and
reproducible.]
