Physics conventions
===================

This page states every equation, sign convention and normalisation hpcom
uses. Each statement is pinned by a test (named in brackets) — if the code
changes, the test fails before the documentation becomes wrong.

.. contents::
   :local:
   :depth: 1


Field and units
---------------

A signal is the complex envelope :math:`A(t)` of one polarisation, in
:math:`\sqrt{\mathrm{W}}`, so :math:`|A(t)|^2` is the instantaneous optical
power. It is sampled at :math:`t_n = n\,\Delta t` with ``dt_s`` =
:math:`\Delta t`; time is always the **last** array axis. A dual-polarisation
field has shape ``(2, n)`` = :math:`(A_x, A_y)`, and a WDM signal is one
composite field (channels are summed, not stacked). Leading axes beyond these
are independent batch dimensions.

Public parameters are SI unless the name says otherwise (``length_km``,
``alpha_db_per_km``, ``gamma_per_w_per_km``, ``*_dbm``). :mod:`hpcom.units`
converts from the units of the literature.

All signals are **periodic** over the simulated window
:math:`T = N\Delta t`: the split-step solver works with FFTs, and the default
pulse shaping is circular so that the signal agrees with that assumption (see
`Pulse shaping`_).


Propagation equations
---------------------

hpcom integrates the nonlinear Schrödinger equation in the frame moving with
the group velocity (Agrawal, *Nonlinear Fiber Optics*, ch. 2):

.. math::

   \frac{\partial A}{\partial z} = -\frac{\alpha}{2}A
     - \frac{i\beta_2}{2}\frac{\partial^2 A}{\partial t^2}
     + \frac{\beta_3}{6}\frac{\partial^3 A}{\partial t^3}
     + i\gamma|A|^2A ,

for ``Fiber(polarization="single")``, and the Manakov equation for
``Fiber(polarization="dual")``, :math:`\mathbf A = (A_x, A_y)^T`:

.. math::

   \frac{\partial \mathbf A}{\partial z} = -\frac{\alpha}{2}\mathbf A
     - \frac{i\beta_2}{2}\frac{\partial^2 \mathbf A}{\partial t^2}
     + \frac{\beta_3}{6}\frac{\partial^3 \mathbf A}{\partial t^3}
     + i\,\frac{8}{9}\gamma\left(|A_x|^2 + |A_y|^2\right)\mathbf A .

The factor 8/9 averages the Kerr effect over the rapidly varying
polarisation state (Marcuse, Menyuk & Wai, *J. Lightwave Technol.* 15(9),
1997). [``test_manakov_with_empty_y_equals_scalar_with_eight_ninths_gamma``]

Fiber parameters
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - ``alpha_db_per_km``
     - Power loss; :math:`\alpha\,[\mathrm{m^{-1}}] = \alpha_\mathrm{dB}\ln(10)/10/1000`,
       :math:`P(L) = P(0)\,e^{-\alpha L}`.
   * - ``beta2_s2_per_m``
     - :math:`\beta_2 = -\lambda^2 D/(2\pi c)`; SSMF with :math:`D = 17` ps/(nm·km) gives
       :math:`\beta_2 = -2.17\times10^{-26}` s²/m.
   * - ``beta3_s3_per_m``
     - :math:`\beta_3 = (\lambda^2/2\pi c)^2 (S + 2D/\lambda)`; SSMF
       (:math:`S = 0.057` ps/(nm²·km)) gives :math:`1.28\times10^{-40}` s³/m.
   * - ``gamma_per_w_per_km``
     - Kerr coefficient :math:`\gamma`.
   * - ``Fiber.effective_length_km``
     - :math:`L_\mathrm{eff} = (1-e^{-\alpha L})/\alpha`.

[``tests/unit/test_units.py``, ``tests/unit/test_fiber.py``]

Fourier convention and sign of frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spectra use NumPy's FFT, :math:`\tilde A_k = \sum_n A_n e^{-2\pi i kn/N}`, with
angular frequencies :math:`\omega_k = 2\pi\,` ``fftfreq(N, dt)``. In this
convention :math:`\partial/\partial t \leftrightarrow +i\omega`, and the linear
part of the equations above is multiplication by
:math:`e^{\hat L(\omega) z}` with

.. math::

   \hat L(\omega) = \frac{i\beta_2}{2}\omega^2 - \frac{i\beta_3}{6}\omega^3 - \frac{\alpha}{2}.

Because the equation is Agrawal's (field :math:`\propto A\,e^{i(\beta_0 z-\omega_0 t)}`)
while the FFT is NumPy's, a **positive baseband frequency** :math:`+f` in hpcom
is the optical frequency :math:`\nu_0 - f` (longer wavelength). Only
odd-order effects notice this; for :math:`\beta_3 > 0` a pulse develops its
oscillatory tail on the trailing edge and its centroid is delayed by
:math:`\beta_3 L/(4T_0^2)` for a Gaussian :math:`e^{-t^2/2T_0^2}`.
[``test_third_order_dispersion_delays_pulse_centroid``,
``test_gaussian_dispersion_matches_closed_form``]

Split-step solver
^^^^^^^^^^^^^^^^^

Every nonlinear backend implements the symmetric (Strang) split-step Fourier
method with the loss inside the linear operator. With :math:`h = L/n` and
``n = fiber.n_steps``,

.. math::

   A(L) = D_{h/2}\,\big[N_h\,D_h\big]^{n-1} N_h\,D_{h/2}\,A(0),\qquad
   D_h = \mathcal F^{-1} e^{h\hat L}\mathcal F,\quad
   N_h A = A\,e^{i\gamma_\mathrm{eff} h |A|^2},

with :math:`\gamma_\mathrm{eff} = \gamma` (scalar) or :math:`\tfrac89\gamma`
and :math:`|A|^2 \to |A_x|^2+|A_y|^2` (Manakov). Properties:

* second-order accurate — halving the step divides the global error by 4
  [``test_split_step_error_is_second_order``];
* without loss every step is unitary, so energy is conserved to rounding
  [``test_energy_conserved_without_loss``];
* :math:`N_h` does not change :math:`|A|`, so the same sequence with
  :math:`h \to -h` is the exact inverse: ``direction="backward"`` undoes a
  forward propagation to rounding error, which is what digital
  back-propagation (``Link.backpropagate``) uses
  [``test_backward_propagation_is_exact_inverse``].

The NumPy and TensorFlow backends execute the identical step sequence and
agree to ~1e-13; both reproduce the original ``ssfm-gpu`` kernels
[``test_matches_ssfm_gpu_golden_output``]. Choose ``n_steps`` so that the
nonlinear phase per step, :math:`\gamma P_\mathrm{peak} h`, is small (≲ 0.01
rad) and confirm convergence by doubling it.


Launch power
------------

The launch power belongs to the transmitter and is defined **per WDM channel,
summed over both polarisations** — the convention of the pre-v1 ``p_ave_dbm``:

.. math::

   P_\mathrm{ch} = 10^{\,\text{launch\_power\_per\_channel\_dbm}/10}\ \mathrm{mW},\qquad
   P_\mathrm{pol} = P_\mathrm{ch}/n_\mathrm{pol},\qquad
   P_\mathrm{total} = N_\mathrm{ch}\,P_\mathrm{ch}.

``launch_power_total_dbm`` may be given instead (it is split equally between
channels); at most one of the two may be set, and 0 dBm per channel is used
when neither is. Read the values back with ``launch_power_per_channel_w``,
``launch_power_per_polarization_w`` and ``launch_power_total_w``.

The field is scaled so that its **expected** mean power equals the launch
power (pilots included); a finite realisation fluctuates by
:math:`O(1/\sqrt{n})`. Symbols (``TransmittedSignal.symbols``) stay on the
unit-energy constellation scale. [``test_field_power_is_the_launch_power``,
``test_received_power_is_the_launch_power_per_channel_summed_over_polarisations``]


Amplifier noise
---------------

``EDFA`` multiplies the field by :math:`\sqrt G` and adds white circular
Gaussian ASE, independent in every sample and polarisation, with one-sided
power spectral density **per polarisation** (Essiambre et al., *J. Lightwave
Technol.* 28(4), 2010; identical to OptiCommPy's ``edfa``)

.. math::

   S_\mathrm{ASE} = n_\mathrm{sp}\,h\nu\,(G-1),\qquad
   n_\mathrm{sp} = \frac{G F - 1}{2(G-1)}
   \quad\Longleftrightarrow\quad
   F = \frac{1 + 2 n_\mathrm{sp}(G-1)}{G},

i.e. :math:`S_\mathrm{ASE} = (GF-1)h\nu/2 \approx F h\nu G/2`, with :math:`F`
the linear noise figure and :math:`\nu = c/\lambda`. Each complex sample
receives variance :math:`S_\mathrm{ASE}/\Delta t`. A single-polarisation
field only receives the co-polarised ASE (the orthogonal part cannot reach a
single-polarisation coherent receiver). ``noise_figure_db = 0`` is below the
quantum limit :math:`F \ge 2 - 1/G` but still gives :math:`n_\mathrm{sp}=1/2`.
[``tests/unit/test_amplifier.py``]

For :math:`N` spans with gain equal to span loss, the ASE-limited SNR after
the matched filter is

.. math:: \mathrm{SNR} = \frac{P_\mathrm{pol}}{N\,S_\mathrm{ASE}\,R_s}.

[``test_single_carrier_snr_matches_closed_form`` — 1 to 20 spans, ±0.1 dB]

.. note::
   Earlier versions used a larger level per polarisation: pre-v1 hpcom
   :math:`h\nu(G-1)F` in the Manakov model and :math:`2h\nu(G-1)F` in the
   scalar model; hpcom 0.2.0.dev :math:`h\nu(G-1)F` in both. At high gain
   these equal the present model with a noise figure larger by 3 dB, 6 dB
   and 3 dB respectively.

Noise loading
^^^^^^^^^^^^^

``AdditiveNoise`` adds the same white noise without gain. Its constructors
translate figures of merit into the per-polarisation PSD :math:`N_0`:

* ``for_snr``: :math:`\mathrm{SNR} = E_s/N_0 = P_\mathrm{pol}/(N_0 R_s)`, the
  SNR at the matched-filter output;
* ``for_osnr``: :math:`\mathrm{OSNR} = P_\mathrm{ch}/(2N_0B_\mathrm{ref})`
  with :math:`B_\mathrm{ref} = 12.5` GHz (0.1 nm) and noise counted in both
  polarisations; for a dual-polarisation channel
  :math:`\mathrm{SNR} = \mathrm{OSNR}\,B_\mathrm{ref}/R_s`.

[``tests/numerical/test_awgn_ber.py``,
``test_osnr_noise_loading_gives_snr_equal_osnr_times_bref_over_rs``]


Pulse shaping
-------------

Single-carrier and WDM transmitters shape symbols with a root-raised-cosine
pulse of roll-off :math:`\beta`. With :math:`f` in units of the symbol rate,
the raised-cosine spectrum is

.. math::

   \mathrm{RC}(f) = \begin{cases}
     1 & |f| < \tfrac{1-\beta}{2}\\[2pt]
     \tfrac12\left[1+\cos\!\left(\tfrac{\pi}{\beta}\left(|f|-\tfrac{1-\beta}{2}\right)\right)\right]
       & \tfrac{1-\beta}{2}\le |f| \le \tfrac{1+\beta}{2}\\[2pt]
     0 & \text{otherwise.}
   \end{cases}

Three implementations, chosen by ``pulse_span_symbols`` and ``convolution``:

.. list-table::
   :widths: 35 65

   * - ``pulse_span_symbols=None`` (default)
     - Exact RRC applied in the frequency domain,
       :math:`H_\mathrm{tx} = \mathrm{sps}\sqrt{\mathrm{RC}}`,
       :math:`H_\mathrm{rx} = \sqrt{\mathrm{RC}}`. Strictly band-limited; ISI-free to
       rounding error.
   * - ``pulse_span_symbols=S``, ``convolution="circular"``
     - FIR of :math:`S\cdot\mathrm{sps}+1` taps (unit-energy :math:`h`), applied
       circularly: Tx :math:`\sqrt{\mathrm{sps}}\,h`, Rx :math:`h/\sqrt{\mathrm{sps}}`.
       A realistic filter without block edges; truncation ISI shrinks with :math:`S`.
   * - ``pulse_span_symbols=S``, ``convolution="linear"``
     - Same FIR, linear convolution of a finite burst. The :math:`\approx S` symbols at
       each end carry transients and are excluded from scoring by default.

In every mode the transmitter output has mean power :math:`E|a|^2` before
launch-power scaling, the noiseless transmit–receive cascade returns each
symbol with unit gain, and white noise of PSD :math:`N_0` leaves the matched
filter with variance :math:`N_0 R_s` per symbol — hence
:math:`\mathrm{SNR} = P_\mathrm{pol}/(N_0 R_s)`.
[``tests/unit/test_pulse_shaping.py``; end to end in
``tests/numerical/test_awgn_ber.py``: BER within 5σ of exact theory and SNR
within ±0.05 dB for all three modes]

Truncated FIRs are not free: with :math:`\beta = 0.1`, a 64-symbol span has
an ISI floor of 2.7e-3 (−51 dB) and −53 dB out-of-band leakage; 256 symbols
reach 2e-4 and −76 dB.


WDM
---

Channel :math:`i = 0,\dots,N-1` is centred at the baseband frequency

.. math:: f_i = \left(i - \tfrac{N-1}{2}\right)\Delta f ,

so the grid is symmetric about the carrier. Each channel/polarisation stream
is shaped independently, multiplied by :math:`e^{2\pi i f_i t_n}` and summed.
Two conditions are enforced:

* the sampled band must contain the grid,
  :math:`R_s\,\mathrm{sps} \ge 2\left(|f_0| + R_s(1+\beta)/2\right)`;
* for circular (periodic) signals every carrier must complete an integer
  number of cycles in the window, :math:`\Delta f\,n_\mathrm{symbols}/R_s \in \mathbb Z`
  (:math:`/2` for even :math:`N`); the error message states the required
  multiple of ``n_symbols``.

The receiver removes dispersion from the **composite** field, then shifts
each selected channel to baseband and matched-filters it. (Compensating after
the shift would leave each channel a walk-off of :math:`\beta_2\omega_i L`.)
With the exact RRC and :math:`\Delta f \ge R_s(1+\beta)`, linear crosstalk
between channels is zero to rounding error.
[``tests/unit/test_wdm*.py``]


OFDM
----

A frame has :math:`N` subcarriers spaced by :math:`\Delta f` (``fftfreq``
order: subcarrier :math:`k` at :math:`k\Delta f`, or :math:`(k-N)\Delta f` for
:math:`k \ge N/2`), of which :math:`n_\mathrm{data}` carry data,
:math:`n_\mathrm{pilot}` a known pilot :math:`p`, and :math:`2n_\mathrm{guard}`
around the band edge are empty. With oversampling :math:`L` the IFFT has
:math:`LN` points and the sample rate is :math:`LN\Delta f`:

.. math::

   x_n = \frac{LN}{\sqrt{E_\mathrm{frame}}}\sqrt{P_\mathrm{pol}}\;
         \mathrm{IFFT}_{LN}\{X\}_n,\qquad
   E_\mathrm{frame} = n_\mathrm{data}E_s + n_\mathrm{pilot}|p|^2 ,

followed by a cyclic prefix of ``cp_len``·:math:`L` samples. :math:`L = 1`
fills the whole simulated band; use :math:`L \ge 2` for nonlinear
propagation, otherwise the spectral broadening aliases.

With noise of PSD :math:`N_0` over the band, the SNR on each data subcarrier
is

.. math::

   \mathrm{SNR}_\mathrm{sc} = \frac{P_\mathrm{pol}}{N_0 N\Delta f}\cdot
      \frac{N E_s}{E_\mathrm{frame}} .

The ``"pilot"`` equaliser measures the channel on the pilot subcarriers of
each frame and interpolates real and imaginary parts linearly **in frequency
order** (constant beyond the outermost pilots).
[``tests/unit/test_ofdm*.py``, ``test_ofdm`` in ``test_awgn_ber.py``]


Receivers and equalisation
--------------------------

1. **Dispersion compensation** — multiply the spectrum of the whole field by
   :math:`\exp\!\left(-\tfrac{i}{2}B_2\omega^2 + \tfrac{i}{6}B_3\omega^3\right)`
   with :math:`B_2` = ``cd_compensation_s2`` (use ``link.total_dispersion_s2``
   = :math:`\sum\beta_2 L`) and :math:`B_3` = ``cd_compensation_s3``
   (``link.total_beta3_s3``). This is the exact inverse of the linear fiber
   operator. [``test_cd_compensation_undoes_beta2_and_beta3_exactly``]
2. **Demodulation** to symbol-rate samples :math:`r_k` (matched filter, or FFT).
3. **Equalisation** by one complex gain :math:`g` per stream,
   :math:`y_k = r_k/g`:

   ============== =================================================================
   ``none``       :math:`g = 1` (raw samples, in √W)
   ``unit_power`` :math:`g = \sqrt{\langle|r|^2\rangle/E_s}` — blind amplitude only;
                  biased at low SNR (it normalises signal *plus* noise)
   ``pilot``      :math:`g = \sum \bar p_k r_k / \sum|p_k|^2` over the pilots
   ``data_aided`` :math:`g = \sum \bar a_k r_k / \sum|a_k|^2` over the *transmitted*
                  data symbols — the least-squares removal of the mean (e.g.
                  nonlinear) phase rotation, pre-v1 ``nonlinear_shift``
   ============== =================================================================

   ``ReceivedSymbols.gain`` reports :math:`g`; its phase is the mean phase
   rotation of the link.
4. **Decision**: nearest constellation point (minimum Euclidean distance),
   then its bit label, or distribution-matcher decoding. A DM block whose
   decisions are not a valid codeword cannot be inverted; its bits are set to
   zero and ``dm_block_valid`` is ``False`` (pre-FEC, the block is lost).


Metrics
-------

All metrics compare received symbols :math:`y` with the transmitted ones
:math:`x` (data-aided):

* **BER** — differing information bits / bits; **SER** — wrong decisions / symbols.
* **EVM** — :math:`\sqrt{\sum|y-x|^2/\sum|x|^2}`; **SNR** —
  :math:`\sum|x|^2/\sum|y-x|^2 = \mathrm{EVM}^{-2}`.
* **Q-factor** — :math:`20\log_{10}\!\big(\sqrt2\,\mathrm{erfc}^{-1}(2\,\mathrm{BER})\big)` dB.
* **MI**, **GMI** — rates in bits/symbol for a Gaussian auxiliary channel
  :math:`q(y|x)\propto e^{-|y-x|^2/\sigma^2}`, :math:`\sigma^2 = \langle|y-x|^2\rangle`
  (Alvarado et al., *J. Lightwave Technol.* 36(2), 2018):

  .. math::

     \mathrm{MI} = H(X) - \frac1N\sum_n\log_2\frac{\sum_{x'}p(x')q(y_n|x')}{p(x_n)q(y_n|x_n)},\qquad
     \mathrm{GMI} = H(X) - \frac1N\sum_n\sum_{i=1}^{m}\log_2\frac{\sum_{x'}p(x')q(y_n|x')}
       {\sum_{x'\in\mathcal X^i_{b_{n,i}}}p(x')q(y_n|x')} .

  For Gray-coded QPSK on AWGN both equal twice the BPSK capacity
  [``test_qpsk_generalised_mutual_information_matches_theory``].

Aggregate metrics sum counts over all streams (polarisations × channels); MI
and GMI are averaged per stream. ``Simulation`` excludes
``discard_symbols`` data symbols at each end (default: the receiver's filter
transient — zero for circular pulse shaping; rounded up to whole DM blocks).


Randomness and reproducibility
------------------------------

``Simulation(seed=s)`` builds ``SeedSequence(s)`` and spawns two children: one
for the transmitter data, one for the link. The transmitter spawns one child
per stream in (polarisation, channel) order, so stream 0 of any transmitter
with the same seed carries the same data. Link noise is drawn sequentially,
element by element, from the link generator. A transmitter with its own
``seed`` keeps it for the data. With ``seed=None`` the entropy drawn from the
OS is stored in ``result.metadata.seed_entropy``; together with
``result.metadata.config`` (the full simulation as JSON) a run can be
replayed exactly:

.. code-block:: python

   replay = Simulation.model_validate(result.metadata.config).replace(
       seed=result.metadata.seed_entropy
   ).run()


Relation to pre-v1 hpcom
------------------------

For the same bits, the pre-v1 WDM transmitter's field equals the complex
conjugate of hpcom's (legacy constellation points are the conjugates, times
their RMS) with legacy channel :math:`i` carrying hpcom channel
:math:`N-1-i`. Power conventions agree exactly. The only numerical difference
is the pulse: legacy samples the RRC impulse response over the window
(≈3e-4 relative difference in the field); with that filter substituted the
fields agree to 1e-11. [``tests/numerical/test_legacy_parity.py``]
