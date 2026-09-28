Backends
========

hpcom has two seams for alternative numerical implementations: **propagators**
(fiber propagation) and **DSP kernel sets** (the hot loops of the
transmitter/receiver). Both are registries of named implementations, both are
held to the same conformance test suites, and switching between them changes
speed, never results.


Propagators
-----------

========================  ====================================================
``"numpy"`` (default)     Symmetric split-step Fourier, scalar NLSE and
                          Manakov, in NumPy. The reference implementation.
``"jax"``                 Every scheme on JAX/XLA (runs on the GPU with
                          ``pip install "hpcom[jax-cuda]"``; CPU with
                          ``"hpcom[jax]"``). Computes in complex128 inside a
                          scoped ``jax.enable_x64`` (no global side effect).
``"tensorflow"``          Every scheme on TensorFlow (runs on the GPU when
                          available). ``pip install "hpcom[tf]"``.
``"numpy-linear"``        Exact linear solution (loss and dispersion in one
                          FFT); ignores γ and ``n_steps``.
========================  ====================================================

**GPU backends.** JAX is the recommended GPU backend. TensorFlow stays a
fully supported alternative for systems where it is the better fit (an
existing TensorFlow stack, a platform without a JAX GPU build). Both
implement every scheme and pass the same conformance suite; their results
agree with NumPy to rounding.

*Measured* (``benchmarks/backends.py``; NVIDIA RTX 3070 Laptop GPU, JAX
0.11.2 / CUDA 12 against NumPy on the same PC's CPU; dual polarisation,
80 km, 100 steps; seconds per span). Every JAX result agrees with NumPy to
≤ 3e-13:

============ ======================= ======================= ======================= =======================
samples      Strang (uniform)        Suzuki (4th order)      RK4IP                   local error 1e-5
============ ======================= ======================= ======================= =======================
16 384       0.075 → 0.019 (4×)      0.38 → 0.077 (5×)       0.31 → 0.051 (6×)       2.31 → 0.149 (16×)
65 536       0.74 → 0.043 (17×)      3.85 → 0.18 (21×)       3.43 → 0.13 (26×)       14.7 → 0.36 (41×)
262 144      4.09 → 0.153 (27×)      19.3 → 0.73 (27×)       15.9 → 0.49 (32×)       50.4 → 1.46 (35×)
============ ======================= ======================= ======================= =======================

*JAX memory.* By default JAX reserves 75 % of the GPU memory when it starts;
set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` to allocate on demand when the
GPU is shared (``benchmarks/gpu_validation.sh`` does).

Select one per fiber (``Fiber(..., backend="jax")``,
``fiber.with_backend("jax")``). Backends are imported lazily:
``import hpcom`` never imports JAX or TensorFlow, and asking for an uninstalled
backend raises :class:`~hpcom.backends.BackendUnavailableError` with the
install command. :func:`~hpcom.backends.available_propagators` lists the
usable ones.

Adding a propagator
^^^^^^^^^^^^^^^^^^^

1. Subclass :class:`~hpcom.backends.Propagator` and implement
   ``propagate(field, fiber, dt_s, direction="forward")``. The contract
   (shapes, dtype, equation and sign conventions, exact backward inverse) is
   in the class docstring, :doc:`physics` and :doc:`numerics`; the helpers
   :func:`~hpcom.backends.base.linear_operator_exponent`,
   :func:`~hpcom.backends.base.effective_gamma_per_m`,
   :func:`~hpcom.backends.base.step_sizes_m` and
   :func:`~hpcom.backends.base.merged_linear_steps` give the operators and
   the step sequence (every ``step_distribution`` must be supported).
2. Register it::

      from hpcom.backends import register_propagator
      register_propagator("jax", JaxPropagator)          # class: instantiated lazily

3. Run the conformance suite. ``tests/conftest.py`` parametrises
   ``tests/backends/test_propagator_conformance.py`` over every available
   backend, so once the backend is registered at import time (e.g. from
   ``hpcom/backends/__init__.py``) it is held to the same closed-form
   dispersion, β₁/β₃, exact SPM/Manakov phase, solitons (N = 1, 2, vector),
   FWM and modulation-instability closed forms, energy conservation, exact
   inverse and second-order convergence for every step placement, and
   ``ssfm-gpu``-parity tests as the built-ins. A backend that runs on the
   CPU only must warn when it is used where a GPU scheme was expected
   (:doc:`numerics`, *Backends and GPU policy*).


DSP kernel sets
---------------

Transmitters and receivers call their hot loops through
:func:`hpcom.backends.kernels.get_kernels`:

==========================  =================================================
``nearest_indices``         nearest constellation point for every sample
                            (memory-bounded)
``ccdm_encode``             bits → symbol indices of a constant-composition
                            distribution matcher (multiset unranking)
``ccdm_decode``             inverse of ``ccdm_encode``
==========================  =================================================

The pure-NumPy set lives in ``hpcom/backends/python/kernels.py`` and is
registered as ``"python"``. A compiled set (e.g. Rust via PyO3) is any module
or object with the same three functions and bit-exact outputs::

   from hpcom.backends.kernels import register_kernels, set_default_kernels
   register_kernels("rust", hpcom_rust)
   set_default_kernels("rust")

``tests/backends/test_kernel_conformance.py`` runs every registered set
against brute-force references (exhaustive lexicographic enumeration for the
matcher, direct distance matrices for the decisions).
