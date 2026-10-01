"""12 — A soliton mode-locked fiber laser, started from noise.

A unidirectional ring of anomalous-dispersion fiber: an erbium-like gain
fiber whose gain saturates on the pulse energy and has a 30-nm (3.75 THz)
bandwidth, standard fiber, a fast saturable absorber and a 50 % output
coupler (:mod:`hpcom.laser`). Each round trip the absorber favours the
strongest noise spike; within a few dozen round trips one pulse takes all
the gain and settles into a soliton. Two textbook signatures check the
physics:

* the **soliton area theorem** — energy × width fixed by the average
  dispersion and nonlinearity, :math:`E\\,T_0 = 2|\\bar\\beta_2|/\\bar\\gamma`;
* **Kelly sidebands** — the periodic perturbation (gain, loss, coupler)
  sheds dispersive waves that are phase-matched to the soliton at
  :math:`\\Delta\\omega_m T_0 = \\sqrt{8m z_0/L - 1}`,
  :math:`z_0 = \\pi T_0^2/2|\\bar\\beta_2|` (Kelly 1992).

Run::

    python examples/12_mode_locked_fiber_laser.py
"""

from __future__ import annotations

import math

import numpy as np

from hpcom import Fiber, GainFiber, OutputCoupler, RingCavity, SaturableAbsorber

N, DT = 2048, 20e-15  # 41 ps window, 50 THz bandwidth


def fwhm(power: np.ndarray, spacing: float) -> float:
    """Full width at half maximum of a single peak (linear interpolation), FFT order."""
    p = np.roll(power, power.size // 2 - int(np.argmax(power)))
    above = np.flatnonzero(p >= p.max() / 2)
    lo, hi = above[0], above[-1]
    left = lo - 1 + (p.max() / 2 - p[lo - 1]) / (p[lo] - p[lo - 1])
    right = hi + (p[hi] - p.max() / 2) / (p[hi] - p[hi + 1])
    return float((right - left) * spacing)


def main() -> None:
    gain = GainFiber(
        length_km=2e-3,
        alpha_db_per_km=0.0,
        beta2_s2_per_m=-2.0e-26,
        gamma_per_w_per_km=3.0,
        gain_db_per_m=5.0,
        gain_bandwidth_hz=3.75e12,
        saturation_energy_j=40e-12,
        n_steps=20,
    )
    smf = Fiber(
        length_km=8e-3,
        alpha_db_per_km=0.0,
        beta2_s2_per_m=-2.17e-26,
        gamma_per_w_per_km=1.3,
        n_steps=80,
    )
    cavity = RingCavity(
        elements=(
            gain,
            smf,
            SaturableAbsorber(modulation_depth=0.3, saturation_power_w=10.0),
            OutputCoupler(output_ratio=0.5),
        )
    )
    noise = np.random.default_rng(1).standard_normal((N, 2)) @ np.array([1, 1j]) * 1e-2
    result = cavity.run(noise, DT, n_round_trips=600, tol=1e-6)

    energy = result.output_energy_j
    print(f"converged: {result.converged} after {result.round_trips} round trips")
    print("output energy (pJ) every 25 round trips:")
    print("  " + " ".join(f"{e * 1e12:.1f}" for e in energy[:200:25]))

    power = np.abs(result.output) ** 2
    peaks = np.flatnonzero(
        (power > np.roll(power, 1)) & (power >= np.roll(power, -1)) & (power > 0.1 * power.max())
    )
    width = fwhm(power, DT)
    t0 = width / (2 * math.acosh(math.sqrt(2)))  # sech²: FWHM = 1.763 T₀
    spectrum = np.abs(np.fft.fft(result.output)) ** 2
    f = np.fft.fftfreq(N, DT)
    bandwidth = fwhm(spectrum, 1 / (N * DT))
    print(
        f"pulses in the window: {peaks.size}; FWHM {width * 1e15:.0f} fs, "
        f"peak {power.max():.0f} W, time-bandwidth product {width * bandwidth:.3f} "
        "(sech²: 0.315)"
    )

    # Soliton area theorem with the length-averaged β₂ and γ
    lengths = np.array([gain.length_m, smf.length_m])
    beta2 = np.dot(lengths, [gain.beta2_s2_per_m, smf.beta2_s2_per_m]) / lengths.sum()
    gamma = np.dot(lengths, [gain.gamma_per_w_per_km, smf.gamma_per_w_per_km]) / lengths.sum()
    gamma *= 1e-3  # 1/(W·m)
    # Path-averaged energy in the fibers: it rises in the gain fiber and drops at the
    # lumped losses. The averaged (guiding-centre) soliton is an approximation when the
    # energy swings this much within a round trip of ~1.7 soliton periods: ~20 % agreement.
    x, ends = result.intracavity, []
    for fiber in (gain, smf):
        y = fiber.propagate(x, DT)
        ends.append((np.sum(np.abs(x) ** 2) + np.sum(np.abs(y) ** 2)) / 2 * DT)
        x = y
    e_avg = np.dot(lengths, ends) / lengths.sum()
    swing = np.sum(np.abs(gain.propagate(result.intracavity, DT)) ** 2)
    swing /= np.sum(np.abs(result.intracavity) ** 2)
    print(
        f"soliton area theorem: E·T0 = {e_avg * t0 * 1e24:.1f} pJ·ps, "
        f"2|β2|/γ = {2 * abs(beta2) / gamma * 1e24:.1f} pJ·ps "
        f"(path-averaged; the energy swings {swing:.1f}× per round trip)"
    )

    # Kelly sidebands: the strongest spectral peaks away from the soliton
    db = 10 * np.log10(spectrum / spectrum.max() + 1e-30)
    z0 = math.pi * t0**2 / (2 * abs(beta2))
    length = lengths.sum()
    local = np.flatnonzero((db > np.roll(db, 1)) & (db > np.roll(db, -1)) & (db > -45) & (f > 0))
    print("Kelly sidebands (THz):")
    for m in range(1, 5):
        predicted = math.sqrt(8 * m * z0 / length - 1) / t0 / (2 * math.pi)
        nearest = local[np.argmin(np.abs(f[local] - predicted))]
        print(
            f"  m = {m}: predicted {predicted / 1e12:.2f}, "
            f"found {f[nearest] / 1e12:.2f} at {db[nearest]:.0f} dB"
        )


if __name__ == "__main__":
    main()
