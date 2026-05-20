# Session resume — hpcom v1 next steps

**Last updated: 2026-05-18.** 338 tests passing; ruff + mypy clean.

## What's done

- M0–M11 complete (infrastructure through Phase 4 shaping)
- T12.1 OFDMConfig dataclass done
- Code review of T11.x/T12.1 done; all 10 review items fixed

## Remaining implementation

### Phase A — OFDM (T12.2–T12.4)

**Architecture decisions locked for OFDM implementation:**

- **D1** — `symbol_rate_hz` = total OFDM bandwidth; `dt_s = 1 / symbol_rate_hz` (override property in OFDMTransmitter/Receiver).
- **D2** — `samples_per_symbol` is single-carrier-only vocabulary ("samples per QAM symbol") — it does not apply to OFDM, where there is no 1-QAM-symbol-to-M-samples relation. OFDM oversampling, if added later, will be a separate parameter on `OFDMTransmitter` (e.g. zero-padded IFFT size or an explicit `oversampling_factor`), not the base-class `samples_per_symbol`. Today `OFDMTransmitter.dt_s = 1 / symbol_rate_hz` (no oversampling).
- **D3** — `OFDMTransmitter.generate()` and `OFDMReceiver.receive()` are complete overrides; RRC pipeline not used.
- **D4** — Power normalisation: Tx multiplies IFFT output by `N / sqrt(n_data)`; Rx divides FFT by `N / sqrt(n_data)`. Unit average power in field; noiseless B2B recovers `Y_data = data_syms` exactly.
- **D5** — `dm` field reused from base Transmitter; block-alignment: `n_symbols * n_data % dm.block_length == 0`.

**A1: `OFDMTransmitter`**
- Files: `transmitter.py` (relax sps check, add class), `ofdm.py` (add `_ofdm_modulate_frames`), `tests/unit/test_ofdm_transmitter.py`
- Key checks: CP duplicates tail; guard subcarriers zero in FFT; pilot values correct; avg power ≈ 1.0; shape = `(n_symbols * (N + cp_len),)`.

**A2: `OFDMReceiver`**
- Files: `receiver.py` (add class), `ofdm.py` (add `_ofdm_demodulate_frames`), `tests/unit/test_ofdm_receiver.py`
- Key checks: noiseless B2B BER = 0 for QPSK/16-QAM/64-QAM; one-tap pilot eq; dual-pol; DM path.

**A3: OFDM AWGN BER test (T12.4)**
- File: `tests/numerical/test_ofdm_awgn.py`
- Physics: per-subcarrier SNR = `(N / n_data) * Es/N0`; compare to QAM-AWGN formula; tolerance 0.5 dB at BER=1e-3.
- Mark `@pytest.mark.slow`.

**🛑 USER CHECKPOINT after A3:** OFDM scope (dual-pol, WDM), M13 go/no-go.

### Phase B — OptiCommPy gap (M13, T13.1)

- Survey → `docs/spec/opticomm_gap.md`
- **🛑 USER CHECKPOINT:** user picks candidates before any implementation.

### Phase C — Component code reviews (after all implementation)

| # | Target | Key physics risks |
|---|--------|-------------------|
| CR1 | `modulation/` | MB normalization, CC-DM bijection, Gray labeling |
| CR2 | `transmitter.py` + `receiver.py` | RRC delay, WDM crosstalk, pilot accounting |
| CR3 | `ofdm.py` | CP length, FFT normalization, N/n_data factor |
| CR4 | `amplifier.py` + `fiber.py` + `link.py` | ASE formula, NF=0 floor, SSFM step size |
| CR5 | `simulation.py` + `result.py` + `metrics.py` | BER counting, EVM reference |
| CR6 | `backends/` | β₂ sign, SSFM symmetry, Manakov dispatch |
| CR7 | `__init__.py`, `_typing.py`, `pilots.py`, `wdm.py`, examples/ | Re-exports, DeprecationWarning |
| CR8 | Test suite | Physics invariant coverage, slow-mark audit |

Each CR session: read → five-axis check → physics check → fix Critical/Important → run tests.

**🛑 FINAL USER CHECKPOINT (Checkpoint 3):** all CRs done → pre-release check → tag + merge to master.

## Useful artefacts

- `scripts/` — legacy probes (untracked, keep for re-running)
- `docs/spec/hpcom-v1-tasks.md` — full task list (M0–M13)
- Agent memory: `MEMORY.md` → `project_hpcom_v1_status.md` for prior context
