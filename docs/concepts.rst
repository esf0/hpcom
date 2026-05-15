Concepts
========

A simulation in hpcom is composed from a small set of domain objects that
each map to a real piece of optical-comm hardware or signal-processing stage.
All of them are frozen dataclasses with validated config.

Pipeline
--------

::

   Transmitter  →  Link  →  Receiver
        │           │          │
        │           ├── Fiber  │
        │           ├── Amplifier
        │           └── ... (any LinkComponent)
        │
        ↓
     Result (BER, EVM, decoded symbols, metadata)

You compose them yourself, or hand them to ``Simulation`` which runs the
whole pipeline once and returns a ``Result``.

Transmitter
-----------

Generates a random bit stream, maps it to a square-QAM constellation
(Gray-coded, unit average power), upsamples to the sample grid and
pulse-shapes with a root-raised-cosine filter.

Key parameters: ``modulation_order`` (4/16/64/256/1024), ``symbol_rate_hz``,
``n_symbols``, ``samples_per_symbol``, ``roll_off``, ``seed``.

Fiber
-----

Frozen physics: ``length_km``, ``alpha_db_per_km``, ``beta2_s2_per_m``,
``gamma_per_w_per_km``, ``n_steps``, ``polarization`` (``"single"`` for NLSE,
``"dual"`` for Manakov), and ``backend`` (the propagator name).

The shortcut ``Fiber.ssmf(length_km)`` returns standard single-mode fibre
parameters (D = 17 ps/nm/km, α = 0.2 dB/km, γ = 1.27 1/W/km).

Amplifier (EDFA)
----------------

``EDFA(gain_db, noise_figure_db, wavelength_m=1550e-9)`` scales the field by
``√G`` and adds zero-mean complex Gaussian ASE noise of PSD
``N₀ = h · f_c · (G − 1) · NF`` per polarization. Accepts 1-D or 2-D
(dual-pol) fields.

Link
----

An ordered tuple of ``LinkComponent`` instances. Built-ins are ``Fiber`` and
``Amplifier``; any object with ``.apply(field, dt_s, *, rng) -> field``
satisfies the protocol.

``Link.from_spans(n, fiber, amp)`` builds a uniform N-span EDFA-compensated
link. ``link.total_dispersion_s2`` returns ``Σ β₂·L`` over all spans —
useful for receiver-side CD compensation.

Receiver
--------

Matched RRC filter, downsample at the symbol rate, optional unit-power
equalization, hard decision to nearest constellation point. The optional
``cd_compensation_s2`` parameter undoes accumulated chromatic dispersion
before the matched filter.

Simulation
----------

The runner. Takes a ``Transmitter``, a ``Link``, a ``Receiver``, and an
optional ``seed`` and ``launch_power_dbm``. Returns a ``Result`` with BER,
EVM, decoded symbols, and reproducibility metadata. The single ``seed`` is
split via ``numpy.random.SeedSequence`` into independent sub-streams for the
Tx and the link's amplifiers.

Result
------

Carries the received field, decoded symbols and bits, ``Metrics`` (BER, EVM,
counts), and ``SimulationMetadata`` (seed, runtime, config snapshot). Use
``result.to_dict()`` for a serializable summary; ``np.savez`` the arrays
yourself if you need to persist the signal.
