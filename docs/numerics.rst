Numerical methods
=================

This page states exactly what the fiber solvers compute: the operators, the
split-step composition for arbitrary step boundaries, its order of accuracy,
the step-placement rules and backward propagation (digital
back-propagation). The equations themselves and their sign conventions are in
:doc:`physics`. Every claim cites the test that checks it; the references are
at the end.


Operators
---------

Each span solves :math:`\partial_z A = (\hat L + \hat N)A` for the scalar
NLSE or the Manakov equation (:doc:`physics`; [Agrawal2013]_, ch. 2). In NumPy's FFT convention
(:math:`\partial_t \leftrightarrow i\omega`) the **linear operator** is
diagonal in frequency,

.. math::
   :label: linear-operator

   \hat L(\omega) = -i\beta_1\omega + \frac{i\beta_2}{2}\omega^2
                    - \frac{i\beta_3}{6}\omega^3 - \frac{\alpha}{2},

and its sub-problem :math:`\partial_z A = \hat L A` is solved exactly over a
length :math:`h`:

.. math::
   :label: linear-flow

   D(h)A = \mathcal F^{-1}\!\left[e^{h\hat L(\omega)}\,\mathcal F A\right].

The **nonlinear operator** is :math:`\hat N A = i\gamma_\mathrm{eff}P A` with
:math:`P = |A|^2` (scalar, :math:`\gamma_\mathrm{eff} = \gamma`) or
:math:`P = |A_x|^2 + |A_y|^2` (Manakov, :math:`\gamma_\mathrm{eff} =
\tfrac89\gamma`). Its sub-problem conserves :math:`P` pointwise, so it too is
solved exactly:

.. math::
   :label: nonlinear-flow

   N(h)A = A\,e^{i\gamma_\mathrm{eff} h\,P}.

Both flows are exact; all numerical error comes from composing them.


Symmetric split-step scheme
---------------------------

A span of length :math:`L` is divided at boundaries
:math:`0 = z_0 < z_1 < \dots < z_K = L` into steps
:math:`h_k = z_k - z_{k-1}` (``Fiber.n_steps`` :math:`= K`, placement below).
One step is the symmetric (Strang) composition [Strang1968]_

.. math::
   :label: strang-step

   S(h) = D\!\left(\tfrac h2\right)\,N(h)\,D\!\left(\tfrac h2\right),

i.e. the nonlinear phase is taken with the power at the **midpoint** of the
step, after the first half of the loss and dispersion. The span is

.. math::
   :label: span

   A(L) = S(h_K)\cdots S(h_2)\,S(h_1)\,A(0)
        = D\!\left(\tfrac{h_K}{2}\right) N(h_K)\,
          D\!\left(\tfrac{h_{K-1}+h_K}{2}\right) \cdots
          D\!\left(\tfrac{h_1+h_2}{2}\right) N(h_1)\,
          D\!\left(\tfrac{h_1}{2}\right) A(0),

where adjacent half steps merge because :math:`D(a)D(b) = D(a+b)`. A span
costs :math:`K` nonlinear steps, :math:`K+1` linear steps and
:math:`2K+2` FFTs. For uniform steps every inner linear step is the same
:math:`D(h)` and is computed once; other placements evaluate
:math:`e^{\hat L m}` for each merged length :math:`m`.

Implementation (``hpcom.backends.numpy.ssfm``; the TensorFlow backend runs
the identical sequence, compiled)::

    m = [h1/2, (h1+h2)/2, …, (h_{K-1}+h_K)/2, h_K/2]        # merged_linear_steps
    X = FFT(A) · exp(L̂ m[0])
    for k = 1..K:
        A = IFFT(X);  A ← A · exp(i γ_eff h_k P(A))          # N(h_k)
        X = FFT(A) · exp(L̂ m[k])
    return IFFT(X)

Loss sits in :math:`\hat L`, so it is applied exactly; the Kerr phase of a
step uses the midpoint power. [``test_agrees_with_numpy_reference``: NumPy
and TensorFlow agree to 1e-12 for every placement;
``test_scalar_ssfm_equals_opticommpy_to_rounding``: OptiCommPy's ``ssfm``,
the same scheme, agrees to 1e-14.]


Order of accuracy
-----------------

**Local error.** For operators :math:`\hat A,\hat B`, the
Baker–Campbell–Hausdorff formula gives

.. math::
   :label: bch

   e^{\frac h2\hat A}\,e^{h\hat B}\,e^{\frac h2\hat A}
   = \exp\!\Big(h(\hat A+\hat B)
     - \frac{h^3}{24}[\hat A,[\hat A,\hat B]]
     + \frac{h^3}{12}[\hat B,[\hat B,\hat A]] + O(h^5)\Big),

so one symmetric step errs by :math:`O(h^3)` — against :math:`O(h^2)` for
the asymmetric step :math:`D(h)N(h)`. The commutators involve dispersion
*and* nonlinearity (:math:`[\hat L,\hat N] \ne 0` only when both act),
which is why the error grows with :math:`\beta_2\omega^2` (signal bandwidth)
and :math:`\gamma P` (power). For the cubic NLS, second-order convergence of
Strang splitting is proven by [Lubich2008]_.

**Global error.** Summing local errors,
:math:`e_\mathrm{global} = O\big(\sum_k h_k^3\big)`. For every placement
below the steps scale as :math:`h_k = L\,[\varphi(k/K) - \varphi((k-1)/K)]`
with a fixed smooth :math:`\varphi`, so :math:`\sum_k h_k^3 = O(K^{-2})`:
**second order in** :math:`1/K` **for every placement**; the placement only
changes the constant. [``test_split_step_error_is_second_order``: error
ratio 3.8–4.6 per step doubling, uniform and logarithmic, both backends.]

**Loss and the midpoint rule.** With dispersion switched off, the exact
Kerr phase over a step is :math:`\gamma P(z_{k-1}) L_\mathrm{eff}(h)`,
:math:`L_\mathrm{eff}(h) = (1-e^{-\alpha h})/\alpha`; the midpoint rule gives
:math:`\gamma P(z_{k-1}) e^{-\alpha h/2} h`. Their ratio,

.. math::

   \frac{\alpha h/2}{\sinh(\alpha h/2)} = 1 - \frac{(\alpha h)^2}{24} + O(h^4),

is consistent with the :math:`O(h^3)` local error, but for one step per
80 km span (:math:`\alpha h \approx 3.7`) the phase is 40 % low — relevant for
coarse DBP (see *Planned schemes*).

**Asymptotic vs. practical regime.** The :math:`O(K^{-2})` law holds once
the steps are small against the dispersion and nonlinear lengths. At coarse
steps — the regime of digital back-propagation — the local errors of
successive steps are only weakly correlated and accumulate more like a
random walk [Musetti2018]_. hpcom's tests therefore check the order where it
is asymptotic and report coarse-step behaviour as measured numbers.


Step placement
--------------

``Fiber.step_distribution`` selects the boundaries :math:`z_n`
(``Fiber.step_boundaries_km``; :func:`hpcom.backends.base.step_boundaries_m`).

Uniform (default)
^^^^^^^^^^^^^^^^^

.. math::

   z_n = \frac{nL}{K}, \qquad h_k = \frac LK.

Logarithmic, factor :math:`k`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With power :math:`P(z) \propto e^{-\alpha z}` (:math:`\alpha` the *power*
attenuation, 1/m), choose the boundaries so that every step carries the same
share of :math:`\int P^k\,dz`:

.. math::
   :label: log-rule

   \int_{z_{n-1}}^{z_n} e^{-k\alpha z}\,dz = \frac{1-e^{-k\alpha L}}{k\alpha K}
   \quad\Longrightarrow\quad
   z_n = -\frac{1}{k\alpha}\ln\!\left(1 - n\,\frac{1-e^{-k\alpha L}}{K}\right),
   \quad n = 0,\dots,K.

The step size then grows along the span as :math:`h(z) \propto e^{k\alpha z}`
— short steps where the power is high — with
:math:`h_K/h_1 \to e^{k\alpha L}` for large :math:`K` (≈ 40 for
:math:`k=1` and 80 km at 0.2 dB/km). :math:`k \to 0` and :math:`\alpha = 0`
give uniform steps. [``test_logarithmic_steps_carry_equal_shares_of_the_weighted_length``,
``test_logarithmic_steps_without_loss_are_uniform``]

The factor ``step_log_factor`` :math:`= k` covers the deterministic rules of
the literature:

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - :math:`k`
     - Rule
     - Origin
   * - 1
     - Equal effective length, hence equal nonlinear phase
       :math:`\gamma P_k L_\mathrm{eff}(h_k)`, per step.
     - Logarithmic step size of [Bosco2000]_; the nonlinear-phase criterion of
       [Sinkin2003]_ with exponentially decaying power reduces to it
       ([Musetti2018]_, eq. 9). Used for DBP by [Asif2011]_.
   * - 1/3
     - Constant local error of the **symmetric** scheme:
       :math:`h_{k+1} = h_k (P_k/P_{k+1})^{1/3} \simeq h_k e^{\alpha h_k/3}`.
     - [Zhang2008]_ (see [Musetti2018]_, eq. 10 with :math:`q = 3`).
   * - 1/2
     - The same rule for the asymmetric scheme (:math:`q = 2`).
     - [Zhang2005]_, [Zhang2008]_.

The iterative rules of those papers start from a chosen first step and let
:math:`K` follow; hpcom fixes :math:`K` and :math:`L` and uses the closed form
:eq:`log-rule`, which has the same step-size profile :math:`h \propto
e^{k\alpha z}`. [Asif2011]_ write the rule as
:math:`h_n = -\frac{1}{A\Gamma}\ln\frac{1-n\sigma}{1-(n-1)\sigma}`,
:math:`\sigma = (1-e^{-2\Gamma L})/K` with a slope coefficient :math:`A`;
its steps sum to :math:`L` only for :math:`A = 2`, where it is
:eq:`log-rule` with :math:`k\alpha = 2\Gamma`.

Measured error constants
^^^^^^^^^^^^^^^^^^^^^^^^

Relative field error against a converged reference; 30 GHz band-limited
signal, 10 mW, 80 km, 0.2 dB/km, :math:`\beta_2 = -21.6` ps²/km
[``test_placement_changes_the_error_constant_not_the_order``]:

=========== ========= ========= =========== ===========
steps K     uniform   k = 1     k = 1/3     k = 1/2
=========== ========= ========= =========== ===========
80          5.3e-4    4.0e-3    2.3e-4      4.4e-4
320         3.1e-5    1.8e-4    1.4e-5      1.2e-5
1280        1.9e-6    4.6e-6    8.5e-7      7.4e-7
=========== ========= ========= =========== ===========

All converge as :math:`K^{-2}`; the constant-local-error placements
(:math:`k = 1/3, 1/2`) are 2.3–2.6× more accurate than uniform steps. The
equal-nonlinear-phase placement (:math:`k = 1`) is *less* accurate here: its
tail steps are ≈ 40× longer and dominate the error, and it reaches the
asymptotic regime only beyond ≈ 1000 steps per span. This agrees with
[Musetti2018]_: the nonlinear-phase criterion ignores the signal bandwidth,
while the local error grows with it. Long steps also over-estimate
four-wave mixing (numerical phase matching; [Francia1999]_, [Bosco2000]_):
at 100 GHz spacing and 400 steps the idler error is 13 % (:math:`k = 1`),
10 % (uniform) and 3.6 % (:math:`k = 1/3`); all vanish as :math:`K^{-2}`
[``test_spurious_fwm_of_coarse_steps_vanishes_as_h_squared``].


Backward propagation and DBP
----------------------------

Each factor of :eq:`strang-step` is invertible —
:math:`D(h)^{-1} = D(-h)` and, because :math:`N` conserves :math:`P`,
:math:`N(h)^{-1} = N(-h)` — so :math:`S(h)^{-1} = S(-h)` and

.. math::

   A(0) = S(-h_1)\,S(-h_2)\cdots S(-h_K)\,A(L):

the **same steps, in reverse order, with negated lengths**
(``direction="backward"``;
:func:`hpcom.backends.base.step_sizes_m` returns
:math:`-h_K,\dots,-h_1`). This undoes a forward propagation to rounding for
every placement [``test_backward_propagation_is_exact_inverse``]; a different
placement or step count is an accurate but not an exact inverse
[``test_dbp_inverts_a_link_simulated_with_logarithmic_steps``].

:class:`~hpcom.dsp.DigitalBackPropagation` runs this per span with its own
``steps_per_span``, ``step_distribution`` and ``step_log_factor`` and
:math:`\gamma \to \xi\gamma` (``nonlinearity_scale``) [IpKahn2008]_. The
boundaries are those of the span, so a logarithmic placement puts the short
steps at the *end* of each backward span, where the back-propagated power is
highest [Asif2011]_. The receiver front end and the per-channel frequency
offset are described in :doc:`physics`.

Measured (5 × 80 km, 64-QAM 32 GBd dual-polarisation at 7 dBm, noiseless,
single-channel DBP at 4 samples/symbol; SNR in dB)
[``test_constant_local_error_placement_beats_uniform_steps``,
``test_equal_nonlinear_phase_placement_over_concentrates_the_steps``]:

=============== ===== ===== ===== ===== =====
steps per span  1     2     4     8     16
=============== ===== ===== ===== ===== =====
uniform         18.3  24.1  37.6  49.3  62.3
k = 1           18.3  24.7  32.2  41.5  52.6
k = 1/3         18.3  27.5  43.3  55.5  69.4
=============== ===== ===== ===== ===== =====

(Plain CD compensation gives 15.9 dB.) With noise and inter-channel effects
the optimum can differ; ``Simulation.evaluate`` compares placements on the
same received field.


Backends and GPU policy
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 20 25

   * - Scheme
     - ``numpy``
     - ``tensorflow``
     - ``numpy-linear``
   * - Symmetric SSFM, uniform steps
     - ✓
     - ✓ (GPU)
     - exact, no steps
   * - Symmetric SSFM, logarithmic steps
     - ✓
     - ✓ (GPU)
     - exact, no steps

Policy for new schemes: a scheme that can run on a GPU gets its GPU
implementation (TensorFlow now, JAX planned) and must pass the conformance
suite in ``tests/backends/`` on every backend that offers it. A scheme that
cannot run on the GPU must say so at run time — a warning that the CPU
implementation is used — never fall back silently.


Planned schemes
---------------

Not implemented yet; listed with the literature they follow.

* **Effective-length Kerr step** — replace :math:`h` in :math:`N(h)` by
  :math:`2\sinh(\alpha h/2)/\alpha`, which makes the midpoint rule exact for
  pure SPM with loss (still second order); matters for coarse DBP.
* **Nonlinear-step position** :math:`r` within a step (asymmetric
  :math:`r = 0`, symmetric :math:`r = 1/2`, optimised in between) and
  **filtered DBP** (low-pass filtered power in the Kerr step) [DuLowery2010]_.
* **Bandwidth-aware first step** :math:`h_1 = \Phi_\mathrm{FWM}/\big(|\beta_2|
  (2\pi B)^2\big)` with nonlinear-phase or constant-local-error updates
  [Musetti2018]_.
* **Adaptive local-error control** (step doubling, third-order accurate)
  [Sinkin2003]_.
* **Higher-order methods**: fourth-order symmetric compositions of the
  Strang step [Yoshida1990]_, [Muslu2005]_; Runge–Kutta in the interaction
  picture (RK4IP) [Hult2007]_. A comparison of split-step variants for
  optical links is [Shao2014]_.


References
----------

.. [Agrawal2013] G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed. Academic
   Press, 2013.

.. [Asif2011] R. Asif, C.-Y. Lin, B. Schmauss, "Digital backward
   propagation: a technique to compensate fiber dispersion and non-linear
   impairments," in C. Cuadrado-Laborde (ed.), *Applications of Digital
   Signal Processing*, InTech, 2011, ch. 2 (Eq. 16: logarithmic step size
   for DBP). See also R. Asif, C.-Y. Lin, B. Schmauss, *J. Mod. Opt.*
   59(2), 95–101, 2012.

.. [Bosco2000] G. Bosco, A. Carena, V. Curri, R. Gaudino, P. Poggiolini,
   S. Benedetto, "Suppression of spurious tones induced by the split-step
   method in fiber systems simulation," *IEEE Photon. Technol. Lett.* 12(5),
   489–491, 2000.

.. [DuLowery2010] L. B. Du, A. J. Lowery, "Improved single channel
   backpropagation for intra-channel fiber nonlinearity compensation in
   long-haul optical communication systems," *Opt. Express* 18(16),
   17075–17088, 2010.

.. [Francia1999] C. Francia, "Constant step-size analysis in numerical
   simulation for correct four-wave-mixing power evaluation in optical fiber
   transmission systems," *IEEE Photon. Technol. Lett.* 11(1), 69–71, 1999.

.. [Hult2007] J. Hult, "A fourth-order Runge–Kutta in the interaction
   picture method for simulating supercontinuum generation in optical
   fibers," *J. Lightw. Technol.* 25(12), 3770–3775, 2007.

.. [IpKahn2008] E. Ip, J. M. Kahn, "Compensation of dispersion and nonlinear
   impairments using digital backpropagation," *J. Lightw. Technol.* 26(20),
   3416–3425, 2008.

.. [Lubich2008] C. Lubich, "On splitting methods for Schrödinger–Poisson and
   cubic nonlinear Schrödinger equations," *Math. Comp.* 77(264), 2141–2153,
   2008.

.. [Musetti2018] S. Musetti, P. Serena, A. Bononi, "On the accuracy of
   split-step Fourier simulations for wideband nonlinear optical
   communications," *J. Lightw. Technol.* 36(23), 5669–5677, 2018.

.. [Muslu2005] G. M. Muslu, H. A. Erbay, "Higher-order split-step Fourier
   schemes for the generalized nonlinear Schrödinger equation," *Math.
   Comput. Simul.* 67(6), 581–595, 2005.

.. [Shao2014] J. Shao, X. Liang, S. Kumar, "Comparison of split-step Fourier
   schemes for simulating fiber optic communication systems," *IEEE Photon.
   J.* 6(4), 2014.

.. [Sinkin2003] O. V. Sinkin, R. Holzlöhner, J. Zweck, C. R. Menyuk,
   "Optimization of the split-step Fourier method in modeling optical-fiber
   communications systems," *J. Lightw. Technol.* 21(1), 61–68, 2003.

.. [Strang1968] G. Strang, "On the construction and comparison of difference
   schemes," *SIAM J. Numer. Anal.* 5(3), 506–517, 1968.

.. [Yoshida1990] H. Yoshida, "Construction of higher order symplectic
   integrators," *Phys. Lett. A* 150(5–7), 262–268, 1990.

.. [Zhang2005] Q. Zhang, M. I. Hayee, "An SSF scheme to achieve comparable
   global simulation accuracy in WDM systems," *IEEE Photon. Technol. Lett.*
   17(9), 1869–1871, 2005.

.. [Zhang2008] Q. Zhang, M. I. Hayee, "Symmetrized split-step Fourier scheme
   to control global simulation accuracy in fiber-optic communication
   systems," *J. Lightw. Technol.* 26(2), 302–316, 2008.
