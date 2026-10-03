# Kelly 1992: sidebands of a soliton fiber laser

S. M. J. Kelly, "Characteristic sideband instability of periodically
amplified average soliton," *Electron. Lett.* 28(8), 806–807, 1992.

**Claim.** A soliton in a laser cavity sheds dispersive waves at the
frequency offsets where they are phase-matched to it once per round trip:
Δω_m T₀ = √(8 m z₀ / L − 1), with z₀ = πT₀²/2|β̄₂| and L the cavity length.

**hpcom** (`examples/12_mode_locked_fiber_laser.py`, `hpcom.laser`): a ring
of an erbium-like gain fiber (saturable gain, 3.75 THz bandwidth), SMF, a
fast saturable absorber and a 50 % coupler, started from noise. One 0.49 ps
soliton forms within about 75 round trips.

| sideband | predicted (THz) | hpcom (THz) | level |
|---|---|---|---|
| m = 1 | 1.08 | 1.15 | −22 dB |
| m = 2 | 1.63 | 1.59 | −23 dB |
| m = 3 | 2.04 | 2.00 | −31 dB |
| m = 4 | 2.37 | 2.37 | −37 dB |

The prediction uses the length-averaged β₂ and the measured pulse width.
The guiding-centre soliton is approximate here (the energy changes 2× per
round trip of 1.7 soliton periods), so the agreement is to a few percent.

Run: `python examples/12_mode_locked_fiber_laser.py` (5 s on a CPU).
