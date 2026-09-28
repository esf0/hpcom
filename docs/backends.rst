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
``"tensorflow"``          The same step sequence on TensorFlow (runs on the
                          GPU when available). ``pip install "hpcom[tf]"``.
``"numpy-linear"``        Exact linear solution (loss and dispersion in one
                          FFT); ignores γ and ``n_steps``.
========================  ====================================================

Select one per fiber (``Fiber(..., backend="tensorflow")``,
``fiber.with_backend("tensorflow")``). Backends are imported lazily:
``import hpcom`` never imports TensorFlow, and asking for an uninstalled
backend raises :class:`~hpcom.backends.BackendUnavailableError` with the
install command. :func:`~hpcom.backends.available_propagators` lists the
usable ones.

Adding a propagator
^^^^^^^^^^^^^^^^^^^

1. Subclass :class:`~hpcom.backends.Propagator` and implement
   ``propagate(field, fiber, dt_s, direction="forward")``. The contract
   (shapes, dtype, equation and sign conventions, exact backward inverse) is
   in the class docstring and in :doc:`physics`; the helpers
   :func:`~hpcom.backends.base.linear_operator_exponent` and
   :func:`~hpcom.backends.base.effective_gamma_per_m` give the operators.
2. Register it::

      from hpcom.backends import register_propagator
      register_propagator("jax", JaxPropagator)          # class: instantiated lazily

3. Run the conformance suite. ``tests/conftest.py`` parametrises
   ``tests/backends/test_propagator_conformance.py`` over every available
   backend, so once the backend is registered at import time (e.g. from
   ``hpcom/backends/__init__.py``) it is held to the same closed-form
   dispersion, β₃ sign, exact SPM/Manakov phase, soliton, energy
   conservation, exact inverse, second-order convergence and
   ``ssfm-gpu``-parity tests as the built-ins.


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
