hpcom
=====

Simulation of coherent optical communication links: QAM, WDM and OFDM
transceivers, scalar (NLSE) and dual-polarisation (Manakov) fiber
propagation with split-step, fourth-order and adaptive solvers on NumPy, JAX
or TensorFlow (GPU),
EDFA noise, digital back-propagation, and data-aided performance metrics
(BER, SNR, Q, MI, GMI).

Every configuration object is an immutable, validated model; every numerical
convention is written down in :doc:`physics` and pinned by a test against
theory.

Install
-------

.. code-block:: bash

   pip install hpcom              # NumPy backend
   pip install "hpcom[jax-cuda]"  # + JAX backend on NVIDIA GPUs (CPU: "hpcom[jax]")
   pip install "hpcom[tf-gpu]"    # + TensorFlow backend with CUDA (CPU: "hpcom[tf]")

Quick start
-----------

.. code-block:: python

   from hpcom import EDFA, Fiber, Link, Simulation, SingleCarrierTransmitter

   fiber = Fiber.ssmf(length_km=80.0, polarization="dual")
   link = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))

   tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=4096,
                                 polarization="dual", launch_power_per_channel_dbm=0.0)
   rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)

   result = Simulation(transmitter=tx, link=link, receiver=rx, seed=0).run()
   print(result)   # Result(ber=..., snr=... dB, evm=..., gmi=... b/sym, ...)

Runnable scenarios live in ``examples/``: back-to-back BER vs theory, a
launch-power sweep over an amplified link, Manakov propagation, dataset
generation, a custom waveform, WDM, OFDM, probabilistic shaping and digital
back-propagation.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   concepts
   physics
   numerics
   impairments
   autodiff
   learn
   backends
   migration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

* :ref:`genindex`
* :ref:`modindex`
