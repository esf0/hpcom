Migrating from pre-v1 hpcom
===========================

The pre-v1 modules (``hpcom.channel``, ``hpcom.signal``, …) still import —
they are loaded from :mod:`hpcom._legacy` on first use and emit a
:class:`DeprecationWarning`. They need ``pip install "hpcom[legacy]"`` and
will be removed in v2.

Concepts
--------

========================================  ===============================================
pre-v1                                    v1
========================================  ===============================================
``wdm`` dict (``create_wdm_parameters``)  :class:`~hpcom.WDMTransmitter` +
                                          :class:`~hpcom.WDMConfig` (or
                                          :class:`~hpcom.SingleCarrierTransmitter`)
``wdm['p_ave_dbm']``                      ``launch_power_per_channel_dbm`` — same meaning
``wdm['n_polarisations'] = 2``            ``polarization="dual"``
``wdm['upsampling']``                     ``samples_per_symbol``
``channel`` dict                          :class:`~hpcom.Fiber` (per span) +
                                          :class:`~hpcom.EDFA`, combined with
                                          ``Link.from_spans``
``channel['dz']``                         ``Fiber.n_steps`` (steps per span)
``full_line_model_wdm(channel, wdm)``     ``Simulation(transmitter, link, receiver).run()``
``channels_type='all'`` / ``'middle'``    ``WDMReceiver(channels=None)`` / ``channels=(N//2,)``
``nonlinear_shift``                       ``equalize="data_aided"``
``dbp=[steps]``                           ``compensation=DigitalBackPropagation(link=…,
                                          steps_per_span=…, samples_per_symbol=None)``
``dbp_model_wdm``                         ``DigitalBackPropagation(…, samples_per_symbol=…)``
``noise_figure_db=-200`` (noiseless)      ``EDFA(…, noise_figure_db=None)``
``result['ber_x']``, ``['q_x']``, …       ``result.stream(polarization=0, channel=…)``
``create_ofdm_parameters`` / …            :class:`~hpcom.OFDMTransmitter`,
                                          :class:`~hpcom.OFDMReceiver`
========================================  ===============================================

Numerical differences
---------------------

* **Same bits, same field.** The default square-QAM labelling
  (``labeling="q_descending"``) is the pre-v1 one, and launch-power scaling
  is identical, so the same bits give the same symbols and the same WDM
  field, channel for channel. The pre-v1 RRC was a time-sampled,
  window-truncated impulse response (≈3e-4 relative field difference); v1
  uses the exact RRC by default. The alternative labelling
  ``ConstellationConfig.from_square_qam(M, labeling="q_ascending")`` maps
  the same bits to the complex-conjugate symbols (same BER/GMI statistics).
* **ASE noise**: v1 follows the standard per-polarisation PSD
  :math:`n_\mathrm{sp}h\nu(G-1)` (see :doc:`physics`). Pre-v1 injected more
  noise — equivalent to a noise figure 3 dB higher (Manakov) or 6 dB higher
  (scalar). To reproduce a pre-v1 result, raise ``noise_figure_db``
  accordingly. The pre-v1 idiom ``noise_figure_db=-200`` for a noiseless
  amplifier is ``EDFA(gain_db=…, noise_figure_db=None)``.
* **Digital back-propagation**: pre-v1 ``full_line_model(dbp=[…])``
  back-propagated at the simulation rate; ``samples_per_symbol=None``
  reproduces it (3e-6 RMS). Pre-v1 ``dbp_model_wdm`` re-generated a
  waveform from given symbols at the reduced rate and back-propagated that;
  v1 instead band-limits and resamples the *received* field, as a
  receiver's ADC does, and runs the matched filter at that rate.
* **Random bits** come from NumPy's ``SeedSequence``/``Generator`` instead of
  Python's ``random``; the same seed gives different bits. Pass ``bits=`` for
  bit-exact comparisons.
* **Metrics** are data-aided (EVM, SNR against the transmitted symbols), and
  MI/GMI use a Gaussian auxiliary channel instead of the hard-decision
  contingency table.
