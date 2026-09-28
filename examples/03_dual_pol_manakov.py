"""03 — The Manakov equation directly: fiber propagation without a transceiver.

Fibers can be used on their own. A dual-polarisation field of shape
``(2, n)`` (or a batch ``(..., 2, n)``) propagates with the Manakov equation
— the Kerr term acts on |A_x|² + |A_y|² with the 8/9 averaging factor — and
backward propagation undoes it exactly.

Run::

    python examples/03_dual_pol_manakov.py
"""

from __future__ import annotations

import numpy as np

from hpcom import Fiber


def main() -> None:
    fiber = Fiber.ssmf(length_km=80.0, n_steps=100, polarization="dual")
    rng = np.random.default_rng(0)
    batch = (rng.standard_normal((4, 2, 4096)) + 1j * rng.standard_normal((4, 2, 4096))) * 0.03
    dt_s = 1 / 256e9

    out = fiber.propagate(batch, dt_s)
    back = fiber.propagate(out, dt_s, direction="backward")

    p_in, p_out = np.mean(np.abs(batch) ** 2), np.mean(np.abs(out) ** 2)
    print(f"Manakov over {fiber.length_km:.0f} km SSMF, batch of {batch.shape[0]} dual-pol fields")
    print(f"  output shape:            {out.shape}")
    print(
        f"  power loss:              {10 * np.log10(p_in / p_out):.3f} dB (fiber: {fiber.loss_db:.1f} dB)"
    )
    print(f"  backward ∘ forward error: {np.max(np.abs(back - batch)) / np.max(np.abs(batch)):.1e}")


if __name__ == "__main__":
    main()
