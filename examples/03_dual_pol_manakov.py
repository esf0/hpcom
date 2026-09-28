"""03 — Dual-polarization (Manakov) propagation through a fiber.

Demonstrates the dual-pol propagation path. We bypass the single-pol
Transmitter / Receiver here and drive the backend directly so the example
stays focused on the Manakov equation. WDM and full dual-pol Tx/Rx live in a
later release.

Requires the TensorFlow extra.

Run::

    python examples/03_dual_pol_manakov.py
"""

from __future__ import annotations

import numpy as np

from hpcom import Fiber


def main() -> None:
    fiber = Fiber.ssmf(length_km=80.0, n_steps=50, backend="tensorflow")
    # Same physics with dual-polarization mode → Manakov dispatch under the hood.
    fiber_dual = Fiber(
        length_km=fiber.length_km,
        alpha_db_per_km=fiber.alpha_db_per_km,
        beta2_s2_per_m=fiber.beta2_s2_per_m,
        gamma_per_w_per_km=fiber.gamma_per_w_per_km,
        n_steps=fiber.n_steps,
        backend="tensorflow",
        polarization="dual",
    )

    n_samples = 1024
    dt_s = 1e-11
    rng = np.random.default_rng(0)
    field_x = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) * 1e-3
    field_y = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) * 1e-3
    field = np.stack([field_x, field_y], axis=0).astype(np.complex128)

    out = fiber_dual.propagate(field, dt_s=dt_s)

    p_in = float(np.mean(np.abs(field) ** 2))
    p_out = float(np.mean(np.abs(out) ** 2))
    print(f"Manakov over {fiber_dual.length_km:.0f} km SSMF, dual-pol")
    print(f"  input mean power:  {p_in:.3e}")
    print(f"  output mean power: {p_out:.3e}  (loss {10 * np.log10(p_in / p_out):.2f} dB)")
    print(f"  output shape:      {out.shape}")


if __name__ == "__main__":
    main()
