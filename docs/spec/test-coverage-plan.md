# Test coverage expansion plan

Drafted 2026-05-17 in response to "more meaningful tests for all crucial parts."

This is a planning doc only — no test code yet. Sections **A–F** are the proposed
new test files; section **G** is the pilots feature (new code, not just tests);
section **H** is what we explicitly defer.

The grading column uses ⚫ for "exists and is solid", 🟡 for "partial / shallow
test exists", ⚪ for "nothing tests this today."

---

## Current test inventory

`tests/unit/`: amplifier, back_to_back, backend_registry, fiber, modulation, wdm.
`tests/numerical/`: awgn_ber, cross_backend_equivalence, energy_conservation,
linear_dispersion, round_trip, soliton, tf_parity.
`tests/integration/`: ssmf_edfa.

82 tests total, all green.

---

## Coverage matrix — by subsystem

### Modulation / Gray coding
| Property                                              | Today | Gap                              |
|-------------------------------------------------------|-------|----------------------------------|
| `bits_per_symbol` correctness                          | ⚫     |                                  |
| Unit average constellation power                       | ⚫     |                                  |
| Bit labels unique per symbol                            | ⚫     |                                  |
| Gray along each axis (row & column)                    | ⚫     |                                  |
| **True Hamming-1 adjacency in the 2-D plane**          | 🟡    | per-axis test misses diagonal-neighbour bugs |
| **Round-trip bits → symbols → bits**                   | ⚫     |                                  |
| **Known fixed mappings (QPSK 00 → corner, etc.)**      | ⚪     | catches endianness/MSB-LSB regressions |
| **Symmetry of constellation** (point-set inversion)    | ⚪     | sanity for any future order      |

### RRC pulse (`src/hpcom/_pulse.py`)
| Property                                              | Today | Gap                              |
|-------------------------------------------------------|-------|----------------------------------|
| Unit ℓ²-norm energy                                    | ⚪    | direct                           |
| Even symmetry around center tap                        | ⚪    | direct                           |
| Zero crossings at symbol multiples (Nyquist ISI=0)     | ⚪    | the whole point of RRC           |
| Behaviour at roll_off ∈ {0, 0.5, 1.0}                  | ⚪    | edge cases                       |
| Singularity handling at t = ±1/(4β)                    | 🟡    | reviewer flagged tolerance       |

### Transmitter (single-channel)
| Property                                              | Today | Gap                              |
|-------------------------------------------------------|-------|----------------------------------|
| Output shape, dt_s, determinism via seed              | ⚫     |                                  |
| **Inject deterministic bits/symbols → field carries them** | ⚪    | needs `bits=` / `symbols=` param on `generate()` |
| **Field is band-limited (spectral support ≤ Rs(1+β))** | ⚪    | FFT check                        |
| Average power matches constellation E[|s|²]·g(0)      | ⚪    | unit-power constellation → near-unit avg power |

### Transmitter (WDM)
| Property                                              | Today | Gap                              |
|-------------------------------------------------------|-------|----------------------------------|
| Per-channel bits/symbols shape                         | ⚫     |                                  |
| 3-channel B2B per-channel BER = 0                     | ⚫     |                                  |
| Nyquist guard                                          | ⚫     |                                  |
| Seeded determinism                                     | ⚫     |                                  |
| **Channel i's energy localised at f_i in the FFT**     | ⚪    | spectral verification of grid placement |
| **Frequency shift exactness** (single-tone test)       | ⚪    | feed a CW carrier, see it land at f_i |
| **Per-channel deterministic symbols → round-trip**     | ⚪    | needs deterministic-bits API     |
| **Cross-channel leakage at receiver**                  | ⚪    | quantify suppression of channel j≠i at RX output |
| **Different per-channel modulation orders**            | ⚪    | currently only uniform — out of scope unless trivial |

### Receiver
| Property                                              | Today | Gap                              |
|-------------------------------------------------------|-------|----------------------------------|
| B2B BER = 0 across orders                              | ⚫     |                                  |
| CD compensation undoes linear dispersion (integration) | ⚫    | tested via cross_backend          |
| **Downshift identity** (RX wdm_channel demux of unshifted Tx = original) | ⚪ | direct unit test |
| **Pilot extraction** (see §G)                         | ⚪    |                                  |

### Fiber / propagation backends
| Property                                              | Today | Gap                              |
|-------------------------------------------------------|-------|----------------------------------|
| NumPy linear matches closed-form dispersion           | ⚫     |                                  |
| NumPy linear loss matches Beer-Lambert                | ⚫     |                                  |
| Forward → backward round-trip                          | ⚫     |                                  |
| Energy conservation (lossless)                         | ⚫     |                                  |
| TF parity vs golden npz                                | ⚫     |                                  |
| Cross-backend equivalence in linear regime             | ⚫     |                                  |
| Fundamental soliton preservation                       | ⚫     |                                  |
| **TF wrapper vs direct `_vendor.propagate_schrodinger` call** | ⚪ | "live" parity (the golden npz could itself be wrong) |
| **TF wrapper vs direct `_vendor.propagate_manakov` call** (dual-pol) | ⚪ | ditto for Manakov path |
| Dispersion β₂ ↔ D conversion (`fiber._dispersion_to_beta2_s2_per_m`) | ⚪ | unit conversion correctness |

### EDFA / amplifier
| Property                                              | Today | Gap                              |
|-------------------------------------------------------|-------|----------------------------------|
| Variance matches `N₀·fs`                               | ⚫     |                                  |
| Gain restores attenuated power                         | ⚫     |                                  |
| Zero-gain identity                                     | ⚫     |                                  |
| Dual-pol independence                                   | ⚫     |                                  |
| Seeded determinism                                     | ⚫     |                                  |
| **Cross-quadrature independence** (E[Re·Im] = 0)       | ⚪    | proper circularly-symmetric Gaussian |
| **Gaussianity** (e.g. excess kurtosis ≈ 0)             | ⚪    | catches if RNG path ever swaps to non-Gaussian |

### Simulation / end-to-end
| Property                                              | Today | Gap                              |
|-------------------------------------------------------|-------|----------------------------------|
| SSMF + EDFA chain runs, BER bounded                    | ⚫     |                                  |
| Rejects mismatched modulation order                    | ⚫     |                                  |
| Rejects WDM Tx (raises NotImplementedError)            | ⚪    | one-liner test of the guard      |
| Launch-power scaling sets target mean power            | ⚪    | direct check of `launch_power_dbm` |
| Reproducibility under `seed`                            | ⚪    | two runs same seed → identical Result |

---

## Proposed new test files (priority order)

| # | File                                          | Tests        | Why this matters                                     |
|---|-----------------------------------------------|-------------|------------------------------------------------------|
| A | `tests/unit/test_modulation_extra.py`         | ~6           | True 2-D Hamming-1 adjacency; known-fixed mappings; round-trip with edge patterns (all-zero / all-one / walking-one). Closes the highest-leverage gap — a Gray-coding regression silently doubles BER. |
| B | `tests/unit/test_pulse_shaping.py`            | ~5           | RRC unit energy, even symmetry, Nyquist ISI=0 at symbol multiples, roll_off edge cases. No coverage at all today. |
| C | `tests/unit/test_transmitter_extra.py`        | ~4           | Deterministic-symbol injection round-trip, spectral band-limiting via FFT, average-power vs constellation. **Depends on G1**. |
| D | `tests/unit/test_wdm_spectrum.py`             | ~5           | FFT shows energy at f_i; single-tone-per-channel landing test; cross-channel suppression measured in dB; deterministic-symbol round-trip per channel. **Depends on G1**. |
| E | `tests/numerical/test_tf_live_parity.py`      | ~3           | Call `_vendor.propagate_schrodinger` / `propagate_manakov` directly and assert byte-equal output vs `Fiber.propagate(backend="tensorflow")`. Independent of golden npz so a stale npz can't hide a wrapper bug. |
| F | `tests/unit/test_amplifier_statistics.py`     | ~3           | Cross-quadrature ⟨Re·Im⟩ ≈ 0; kurtosis ≈ 0; per-sample independence (autocorrelation ≈ δ). |

**Approximate effort**: ~30 new tests, ~350 LOC.

---

## G. Pilot symbols — new feature

Pilots are a real DSP need: known reference symbols interleaved with data so the
receiver can estimate phase noise, carrier-frequency offset, or per-channel
gain. The user explicitly asked for "multichannels with pilots — we need to
check generation as well."

### G1. Minimal `PilotConfig` design

```python
@dataclass(frozen=True)
class PilotConfig:
    period: int              # every `period`-th symbol slot is a pilot
    symbol: complex = 1+0j   # what to transmit at each pilot slot
```

Properties:
- `period >= 2`, otherwise nothing is data.
- For a transmitter with `n_symbols=N`, pilot indices are `0, period, 2·period, …` (offset 0 for v1; offset configurability is a follow-up).
- Pilot `symbol` is on the constellation but does NOT need to be a constellation point — receivers use it as a reference, not for bit-extraction. We'll keep it unit-magnitude by default.

### G2. Transmitter integration

- Add `pilots: PilotConfig | None = None` to `Transmitter`.
- Per-channel application: when both `wdm` and `pilots` are set, the same pilot
  config is applied to each channel independently. Bits at pilot slots are zero-filled
  in `TransmittedSignal.bits` (and not used for BER scoring at the receiver).
- New helper on `TransmittedSignal`: `pilot_indices()` returns the symbol-index
  positions of pilots.

### G3. Receiver integration

- Add `pilots: PilotConfig | None = None` to `Receiver`.
- New return field `pilot_symbols: ComplexArray | None` on `ReceivedSymbols` —
  the received complex amplitudes at the pilot positions, after matched filter
  and downsampling. No estimation logic in v1; the receiver just hands them
  back. Estimators (CPE, CFO) are separate downstream work.

### G4. Test coverage for pilots

`tests/unit/test_pilots.py` (~6 tests):
1. `PilotConfig` validation (`period >= 2`).
2. B2B noiseless: `received.pilot_symbols == cfg.symbol` (within fp tolerance) at every pilot index.
3. Pilot slots in `sent.symbols` equal `cfg.symbol`; data slots are random.
4. With `pilots` and a known seed, data-slot bits are deterministic; bit count at non-pilot positions matches expected.
5. WDM + pilots: per-channel B2B, each channel's `pilot_symbols` equal the configured pilot value (within tolerance), proving pilots survive the frequency-shift + demux roundtrip.
6. BER scoring rule: a `metrics.bit_error_rate` aware of pilots should skip pilot positions. **This requires extending metrics or the Receiver to mask pilot bits — flag for design decision.** Cleanest: pilots are excluded from `bits` entirely (only data bits are returned), and `pilot_symbols` is separate.

### G5. Open question for you

For test #6 above, two paths:
- **Option α** — `Transmitter.bits` and `Receiver.bits` only contain *data* bits (pilots are stripped at the receiver). Cleanest; matches the "BER on data" intuition. Means `bits.shape[0]` shrinks by `1/period`.
- **Option β** — `bits` keeps the full bit stream (zero-filled at pilot slots). BER computation excludes pilot positions explicitly. More flexible but pushes bookkeeping onto every caller.

Recommend **α**.

---

## H. Explicitly deferred

- Pilot-based estimators (CPE, CFO, per-channel gain) — separate task, v1.1.
- Polarization-multiplexed pilots (interleaved per polarization) — needs dual-pol API surface first.
- Non-uniform pilot patterns (e.g. compressed-sensing schedules) — research code, not core.
- Comparison against external `ssfm_gpu` package — package isn't installed and is what the vendor copy supersedes. The `test_tf_live_parity.py` (item E) covers the same concern by calling the vendored functions directly.
- Per-channel different modulation orders in WDM — current design assumes uniform; revisit if a real use case appears.

---

## Decision points for you

1. **Approve A–F as the new test files** (or trim/expand)?
2. **Pilots design (§G1–G3) acceptable**? In particular Option α for test #6 (§G5)?
3. **Ordering preference**: ship pilots first (A through F minus C/D, which need pilots? actually they don't — they need deterministic bits, which is a smaller helper) then tests, or interleave?
