Differentiable propagation (JAX)
================================

:mod:`hpcom.autodiff` runs hpcom's whole chain — transmitter, fiber link,
receiver, demapper and information rates — as pure JAX functions, so that
:func:`jax.grad`, :func:`jax.jit` and :func:`jax.vmap` apply to any part of
it. Use it to fit fiber parameters to measurements, to learn
back-propagation or equaliser parameters (or plug a neural network into the
receiver), or to train transmitters and receivers end to end through a
physical channel model.

.. contents::
   :local:
   :depth: 1


What is differentiable
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Differentiable in
   * - :func:`~hpcom.autodiff.propagate`
     - the input field and every :class:`~hpcom.autodiff.FiberParameters`
       leaf: loss, :math:`\beta_1…\beta_N`, :math:`\gamma`, Raman
       :math:`f_R, \tau_1, \tau_2, \tau_b`, shock time, loss-spectrum values,
       multicore coupling matrix, core mismatch and skew, PMD coefficient
       (the realisation's axes stay fixed), and for fiber lasers the gain,
       gain bandwidth, saturation energy and nonlinear gain
   * - :func:`~hpcom.autodiff.amplify`, :func:`~hpcom.autodiff.propagate_link`
     - the field, fiber parameters, EDFA gain and noise figure (the ASE is
       reparameterised: :math:`n = \sigma(G, F)\,\xi` with :math:`\xi` drawn
       from a JAX PRNG key)
   * - :func:`~hpcom.autodiff.backpropagate`, :func:`~hpcom.autodiff.compensate`
     - the field, the receiver's link model (every fiber and amplifier
       parameter, span by span) and the nonlinearity scale :math:`\xi`
   * - :func:`~hpcom.autodiff.constellation`, :func:`~hpcom.autodiff.modulate`
     - the constellation points (geometric shaping; normalised to unit
       power wherever used), the PMF logits (probabilistic shaping), the
       launch power, the data symbols
   * - :func:`~hpcom.autodiff.sample_symbols`
     - the PMF, through straight-through Gumbel-softmax sampling (exact
       constellation points forward, relaxed one-hot gradient backward)
   * - :func:`~hpcom.autodiff.demodulate`
     - the field, the DBP stage's link model and :math:`\xi`, or the
       parameters of any JAX function used as the compensation stage (e.g.
       a neural network); dispersion compensation, front end (low-pass,
       resampling), matched filter and equalisers — including the MIMO
       butterfly FIR, solved by the normal equations — are exact and
       differentiable
   * - :func:`~hpcom.autodiff.information_rates`, :func:`~hpcom.autodiff.bit_llrs`,
       :func:`~hpcom.autodiff.snr_db`
     - the received samples, points and PMF — MI and GMI (the estimators of
       :mod:`hpcom.metrics`) as training objectives, exact bit LLRs

The pydantic configuration (:class:`~hpcom.Fiber`, :class:`~hpcom.Link`,
:class:`~hpcom.DigitalBackPropagation`) fixes the *structure* — lengths,
step count and placement, scheme, polarisation, cores, which terms exist —
and is a static argument under :func:`jax.jit`. The continuous values are
pytrees; ``FiberParameters.from_fiber`` / ``to_fiber`` convert both ways,
so a fitted parameter set goes back into an ordinary simulation.

Not differentiable: hard decisions, BER/SER, distribution matchers (CCDM)
— use the GMI, the LLRs or Gumbel sampling instead — and the adaptive
local-error scheme. Every fixed-step scheme is supported: the symmetric split step with any
Kerr position, effective Kerr length or intensity filter, the Yoshida and
Suzuki compositions, and RK4IP. The adaptive local-error method is not —
its step sequence depends on the field — and is rejected.

**Gradients of what.** The step grid is fixed by the configuration and does
not move with the parameters, so a gradient is the exact derivative of the
*discretised* solution. It converges to the derivative of the continuous
solution as the steps are refined, at the scheme's order.

**Complex arrays.** For a real loss :math:`\mathcal L` of a complex array
:math:`z = x + iy`, JAX returns :math:`\partial\mathcal L/\partial x -
i\,\partial\mathcal L/\partial y`; steepest descent is
:math:`z \leftarrow z - \eta\,\overline{\nabla}`.


Units and precision
-------------------

:class:`~hpcom.autodiff.FiberParameters` is in ps/km units — β_n in psⁿ/km
(SSMF: :math:`\beta_2 = -21.7` ps²/km, :math:`\beta_3 = 0.13` ps³/km), α in
dB/km, γ in 1/(W·km), Raman and shock times in fs — and the propagation
runs internally with time in ps and length in km. Every quantity is then
O(1): in SI units :math:`\beta_3 \approx 10^{-40}` s³/m is below float32's
normal range.

Precision follows the field: complex128 needs ``jax_enable_x64`` (or a
``jax.enable_x64`` context) and then matches the NumPy reference to rounding
(≤ 1e-10 relative; the band-edge dispersion phase, up to 1e5 rad, is
computed in different units). complex64 is faster on GPUs; it resolves a
phase :math:`\varphi` to about :math:`10^{-7}\varphi`, so its accuracy is
set by the largest dispersion phase per step at the band edge: 2e-4 for
SSMF at 100 GHz, 1e-2 for a 25 THz generalised-NLSE run (use complex128
there).


Gradient memory: store, checkpoint, reversible
----------------------------------------------

Reverse-mode AD needs every intermediate field of the forward pass. For a
span of :math:`J` sub-steps that is :math:`O(J)` fields — 1.2 GiB for 800
steps of :math:`2^{14}` samples. ``gradient=`` chooses how the backward pass
gets them:

``"store"``
   JAX's default: keep them all.

``"checkpoint"``
   Keep the field at every :math:`\sqrt J`-th step and recompute each block
   during the backward pass [Chen2016]_ (memory :math:`O(\sqrt J)`, about
   twice the forward work).

``"reversible"``
   Keep nothing. The split step is exactly invertible — the Kerr step is a
   pure phase, :math:`N(c)^{-1} = N(-c)`, including the Raman response and
   the intensity filter, since :math:`|A|` is unchanged; the linear step
   :math:`e^{h\hat D}` is invertible — so the backward pass *reconstructs*
   each state from the next one while it propagates the adjoint:

   .. math::

      a_j = F_j^{-1}(a_{j+1}),\qquad
      \bar a_j = \Big(\frac{\partial F_j}{\partial a_j}\Big)^{\!\dagger}\bar a_{j+1},\qquad
      \bar\theta \mathrel{+}= \Big(\frac{\partial F_j}{\partial\theta}\Big)^{\!\dagger}\bar a_{j+1},

   with :math:`F_j = D(m_{j+1})\,N(c_j)` the :math:`j`-th step, as in
   reversible networks [Gomez2017]_. Memory does not depend on :math:`J`;
   the work is about three forward passes; the result equals the stored
   gradient to rounding (reconstruction runs the span backwards, which is
   exact to ~1e-12 — ``docs/numerics.rst``). Nonlinear gain keeps the Kerr
   step exactly invertible too. RK4IP and saturated gain are not
   invertible and support ``"store"`` and ``"checkpoint"``.

Measured compile-time temporary memory for the gradient of an 80 km SSMF span,
:math:`2^{14}` samples (field: 0.25 MiB):

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 25

   * - steps
     - store
     - checkpoint
     - reversible
   * - 50
     - 77 MiB
     - 15 MiB
     - 3.0 MiB
   * - 200
     - 302 MiB
     - 27 MiB
     - 3.0 MiB
   * - 800
     - 1202 MiB
     - 52 MiB
     - 3.0 MiB


Speed
-----

Value and gradient through 10 × 80 km of dual-polarisation SSMF with
noiseless EDFAs, :math:`2^{16}` samples, 50 steps per span (500 sub-steps),
``jax.jit``-compiled (laptop RTX 3070 vs. its CPU):

.. list-table::
   :header-rows: 1
   :widths: 30 17 17 17 17

   * -
     - forward c128
     - value+grad c128
     - forward c64
     - value+grad c64
   * - GPU, ``store``
     - 0.17 s
     - 0.40 s
     - 0.022 s
     - 0.079 s
   * - GPU, ``reversible``
     - 0.17 s
     - 0.67 s
     - 0.022 s
     - 0.091 s
   * - CPU, ``store``
     - 1.27 s
     - 3.72 s
     - 0.68 s
     - 1.88 s


Multicore coupling
------------------

The coupled linear step :math:`e^{imK}` (:math:`K` real symmetric: coupling,
mismatch and, per frequency, skew) is computed from the eigen-decomposition
:math:`K = V\Lambda V^\top`, as in the NumPy backend — a generic matrix
exponential loses up to 1e-8 on the large phases skew produces. Its
derivative is the Daleckii–Krein formula [Higham2008]_ (Thm. 3.11),

.. math::

   d\,e^{imK} = V\,(F\circ V^\top dK\,V)\,V^\top,\qquad
   F_{ij} = \frac{e^{im\lambda_i} - e^{im\lambda_j}}{\lambda_i-\lambda_j}
          = im\,e^{im(\lambda_i+\lambda_j)/2}\,\mathrm{sinc}\frac{m(\lambda_i-\lambda_j)}{2},

implemented as a custom JVP in the sinc form, which stays finite for the
degenerate eigenvalues of symmetric layouts (ring, hexagon), where
differentiating the eigenvectors would divide by zero.


Validation
----------

[``tests/numerical/test_autodiff.py``, ``tests/unit/test_autodiff_api.py``]

.. list-table::
   :widths: 30 70

   * - Values
     - Equal the NumPy reference (≤ 1e-10) for the split step (single,
       Manakov, Kerr position + effective length + logarithmic steps,
       filtered), Yoshida, Suzuki with β₄, β₅, loss spectrum and Raman,
       RK4IP with Raman and self-steepening, multicore ring with mismatch and
       7-core Manakov with skew — forward and backward, in every gradient
       mode; noiseless links equal ``Link.propagate``; ``compensate`` equals
       ``DigitalBackPropagation.compensate`` at a WDM channel offset with
       logarithmic steps and effective Kerr length.
   * - Gradients vs finite differences
     - Every parameter leaf of all those cases (Richardson-extrapolated
       central differences on the plateau of step sizes) to 1e-6.
   * - Closed forms
     - :math:`\partial/\partial\gamma` of lossy SPM (exact with the effective
       Kerr length) to 1e-12; :math:`\partial/\partial\beta_2` of the peak of
       a dispersed Gaussian, :math:`-T_0^2\beta_2z^2/(T_0^4+\beta_2^2z^2)^{3/2}`,
       to 1e-9; the input-field gradient of :math:`\mathrm{Re}\langle c, Ux\rangle`
       is :math:`\overline{U^\dagger c}`.
   * - Modes
     - store, checkpoint and reversible gradients agree to 1e-10; the
       reversible mode's memory is independent of the step count.
   * - Noise
     - :func:`~hpcom.autodiff.amplify` has the EDFA's ASE variance; the
       gradient of the noise power in the noise figure is exact.
   * - Transceivers
     - :func:`~hpcom.autodiff.modulate` equals ``generate(symbols=…)`` and
       :func:`~hpcom.autodiff.demodulate` equals ``receive(...).symbols``
       (1e-12) for single carrier, dual polarisation with pilots and FIR
       pulses, linear convolution, 3-channel dual-polarisation WDM (channel
       subsets, band-limited front end) and probabilistic shaping, with CD
       compensation, DBP at 2 samples/symbol and full-field DBP;
       :func:`~hpcom.autodiff.information_rates` equals
       :func:`hpcom.metrics.information_rates` (1e-12) and the GMI from the
       LLRs; gradients through transmitter → link → DBP receiver → GMI equal
       finite differences; Gumbel samples have the softmax frequencies.
   * - Applications
     - L-BFGS on exact gradients recovers (α, β₂, β₃, γ) of a 20 km span
       from 3–30 % off to 1e-6; gradient descent on the DBP nonlinearity
       scale lands on the grid-search optimum; learned geometric and
       probabilistic shaping beat QAM on new data.


Example: learned DBP
--------------------

``examples/10_learned_dbp.py`` trains the per-span γ and β₂ of a
one-step-per-span DBP (2 samples/symbol, 16-QAM, 32 GBd, 10 × 80 km,
+5 dBm) on one simulated data set by L-BFGS with ``gradient="reversible"``,
converts the result back with ``FiberParameters.to_fiber`` and scores it with
the standard receiver chain on new data and noise:

.. list-table::
   :header-rows: 1
   :widths: 60 20

   * - receiver
     - SNR
   * - linear CD compensation
     - 11.0 dB
   * - DBP, 1 step/span, link parameters
     - 13.4 dB
   * - DBP, 1 step/span, learned parameters
     - 17.7 dB
   * - DBP, 16 steps/span, link parameters
     - 31.2 dB

The learned γ is ≈ 1.6× the fiber's in every span: one lumped Kerr step at
the span midpoint sees the power :math:`e^{-\alpha L/2}P_\mathrm{in}`, while
the span accumulates :math:`L_\mathrm{eff}P_\mathrm{in}`, a ratio of 0.7 for
80 km (see also the effective Kerr length in :doc:`numerics`).


Example: end-to-end learning
----------------------------

``examples/11_end_to_end_learning.py`` trains 64 constellation points and
the launch power jointly, by Adam on the GMI after the whole chain —
pulse shaping, 20 × 80 km of SSMF with EDFA noise, CD compensation, matched
filter, data-aided phase recovery, bit-wise demapper — with a fresh batch of
data and noise per step (150 steps, 45 s on a laptop CPU). Scored by
:class:`~hpcom.Simulation` on new data (mean of 8 draws): 64-QAM at its best
launch power (−3 dBm) 5.484 b/sym, the learned constellation 5.534 b/sym.
On the AWGN channel at :math:`E_s/N_0` = 12 dB the same machinery gives
+0.05 b/sym for 16 learned points over 16-QAM (geometric shaping) and
+0.16 b/sym for a learned 64-QAM PMF over uniform (probabilistic shaping by
Gumbel-softmax; capacity is 0.11 b/sym above the shaped rate)
[``test_learned_*_beats_*``].


Relation to other work
----------------------

Differentiable split-step models are established in optical communications:
learned digital back-propagation [HagerPfister2018]_, [HagerPfister2021]_,
end-to-end autoencoders through the NLSE [Li2018]_, joint geometric and
probabilistic shaping by Gumbel-softmax [Stark2019]_, and JAX
signal-processing libraries [Fan2020]_. What :mod:`hpcom.autodiff` adds is the same validated
simulator — every fixed-step scheme, the generalised NLSE, Manakov and
multicore fibers, amplifier noise, DBP at channel offsets — differentiable
in all its physical parameters, with gradients checked against finite
differences and closed forms, float32-safe parameter units, and an exact
constant-memory adjoint for the split step.


References
----------

.. [Chen2016] T. Chen, B. Xu, C. Zhang, C. Guestrin, "Training deep nets
   with sublinear memory cost," arXiv:1604.06174, 2016.

.. [Fan2020] Q. Fan, G. Zhou, T. Gui, C. Lu, A. P. T. Lau, "Advancing
   theoretical understanding and practical performance of signal processing
   for nonlinear optical communications through machine learning," *Nat.
   Commun.* 11, 3694, 2020.

.. [Gomez2017] A. N. Gomez, M. Ren, R. Urtasun, R. B. Grosse, "The
   reversible residual network: backpropagation without storing
   activations," *NeurIPS* 2017.

.. [HagerPfister2018] C. Häger, H. D. Pfister, "Nonlinear interference
   mitigation via deep neural networks," *OFC* 2018, W3A.4.

.. [HagerPfister2021] C. Häger, H. D. Pfister, "Physics-based deep learning
   for fiber-optic communication systems," *IEEE J. Sel. Areas Commun.*
   39(1), 280–294, 2021.

.. [Higham2008] N. J. Higham, *Functions of Matrices: Theory and
   Computation*, SIAM, 2008.

.. [Stark2019] M. Stark, F. Ait Aoudia, J. Hoydis, "Joint learning of
   geometric and probabilistic constellation shaping," *IEEE Globecom
   Workshops* 2019.

.. [Li2018] S. Li, C. Häger, N. Garcia, H. Wymeersch, "Achievable
   information rates for nonlinear fiber communication via end-to-end
   autoencoder learning," *ECOC* 2018.
