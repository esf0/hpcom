Propagator backends
===================

A ``Propagator`` is the seam between the user-facing ``Fiber.propagate()``
call and the actual numerics. Backends register themselves under a string
name; ``Fiber.backend`` selects which one is invoked.

Built-in backends
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Name
     - Implementation
     - When to use
   * - ``numpy-linear``
     - Pure NumPy, dispersion + loss, no nonlinearity (γ ignored).
     - Sanity checks, CPU benchmarks, CI without TensorFlow.
   * - ``tensorflow``
     - Vendored ``ssfm-gpu`` SSFM and Manakov kernels.
     - Production scenarios, GPU runs.

Switching backends
------------------

Either at construction::

   fiber = Fiber.ssmf(length_km=80.0, backend="numpy-linear")

…or with ``with_backend``::

   fast = fiber.with_backend("tensorflow")

The output is bit-identical across backends in the linear regime; see
``tests/numerical/test_cross_backend_equivalence.py`` for the pinned tolerance.

Adding a new backend
--------------------

1. Subclass ``hpcom.backends.base.Propagator`` and implement::

      def propagate(self, field, fiber, dt_s, direction="forward") -> field

2. Register::

      from hpcom.backends import register_propagator
      register_propagator("my-backend", MyPropagator())

3. Set ``Fiber(..., backend="my-backend")``.

The ``fiber`` argument is typed as ``hpcom._typing.FiberLike`` — a Protocol
covering the parameters every backend needs (``length_km``,
``alpha_db_per_km``, ``beta2_s2_per_m``, ``gamma_per_w_per_km``, ``n_steps``,
``polarization``). Any duck-typed object will work, which is convenient for
testing without instantiating a full ``Fiber``.
