Concepts
========

A simulation is assembled from a few objects that each correspond to a piece
of hardware or a DSP stage::

   Transmitter ──field──▶ Link ──field──▶ Receiver ──▶ ReceivedSymbols
       │                (Fiber, EDFA,                     │
       └──TransmittedSignal  AdditiveNoise, …)            ▼
                                        Simulation ──▶ Result (metrics, provenance)

All of them are **immutable, validated pydantic models** (:class:`hpcom._model.HpcomModel`):

* unknown keyword arguments raise (a typo never silently falls back to a default);
* array fields are read-only;
* ``a == b`` compares by value, arrays included;
* ``obj.replace(field=value)`` returns a re-validated copy;
* ``model_dump_json()`` / ``model_validate_json()`` round-trip everything,
  including which transmitter/receiver/link-element subclass was used.


Transmitters
------------

.. list-table::
   :widths: 45 55

   * - :class:`~hpcom.SingleCarrierTransmitter`
     - RRC-shaped QAM, one channel
   * - :class:`~hpcom.WDMTransmitter`
     - ``n_channels`` RRC-shaped channels on a grid
   * - :class:`~hpcom.OFDMTransmitter`
     - CP-OFDM with pilot and guard subcarriers
   * - :class:`hpcom.experimental.ExperimentalTransmission`
     - template for custom waveforms

Shared fields (:class:`~hpcom.Transmitter`): the constellation
(``modulation_order`` for Gray-coded square QAM, or a
:class:`~hpcom.ConstellationConfig`), an optional distribution matcher
``dm``, ``polarization`` (``"single"``/``"dual"``), the launch power
(``launch_power_per_channel_dbm`` or ``launch_power_total_dbm``) and ``seed``.

:meth:`~hpcom.Transmitter.generate` returns a
:class:`~hpcom.TransmittedSignal`: the field in √W, the transmitted symbols
and the information bits. Pass ``bits=`` or ``symbols=`` for deterministic
input.

Array layout
^^^^^^^^^^^^

Time or symbol index is always the last axis. Leading axes are
``(polarisation, channel)``, each present only when applicable:

==============================  =================  =======================
Signal                          ``field``          ``symbols``/``bits``
==============================  =================  =======================
single-carrier, single-pol      ``(n,)``           ``(n_sym,)``
single-carrier, dual-pol        ``(2, n)``         ``(2, n_sym)``
WDM, single-pol                 ``(n,)``           ``(n_ch, n_sym)``
WDM, dual-pol                   ``(2, n)``         ``(2, n_ch, n_sym)``
==============================  =================  =======================


Link
----

A :class:`~hpcom.Link` is an ordered tuple of components, each with
``apply(field, dt_s, *, rng)``: :class:`~hpcom.Fiber`,
:class:`~hpcom.EDFA`, :class:`~hpcom.AdditiveNoise`, or your own.
``Link.from_spans(n, fiber, amplifier)`` builds the usual amplified link;
``link.total_dispersion_s2`` is what the receiver must compensate.
``link.backpropagate(field, dt_s, steps_per_span=…)`` runs it backwards
(digital back-propagation).

A fiber's ``polarization`` selects the equation (NLSE or Manakov) and its
``backend`` the numerical engine (see :doc:`backends`).


Receivers
---------

:class:`~hpcom.SingleCarrierReceiver`, :class:`~hpcom.WDMReceiver` (any
subset of ``channels``) and :class:`~hpcom.OFDMReceiver` mirror the
transmitters. All compensate chromatic dispersion
(``cd_compensation_s2``/``_s3``) and equalise one complex gain per stream
(``equalize="auto" | "none" | "unit_power" | "pilot" | "data_aided"``; see
:doc:`physics`). The default ``"auto"`` uses the receiver's pilots if it has
any, otherwise the transmitted symbols when they are supplied (as in every
``Simulation`` — the removal of the mean nonlinear phase rotation that
pre-v1 hpcom always applied), otherwise blind amplitude normalisation. ``receive(field, dt_s, reference=sent)`` returns
:class:`~hpcom.ReceivedSymbols`: equalised symbols, decisions, decoded bits
and the removed gain per stream.


Simulation and Result
---------------------

:class:`~hpcom.Simulation` checks that transmitter, link and receiver fit
together (listing every mismatch), runs them with a single seed and scores
the result. :class:`~hpcom.Result` holds the transmitted signal, the received
field, the receiver output, aggregate :class:`~hpcom.Metrics`, one
:class:`~hpcom.StreamMetrics` per (polarisation, channel), and
:class:`~hpcom.SimulationMetadata` with the full configuration as JSON and
the seed entropy, enough to replay the run exactly.
