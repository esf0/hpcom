"""Generate ``opticommpy_v0_10.npz``: OptiCommPy reference outputs for hpcom's cross-checks.

OptiCommPy (GPL-3.0, https://github.com/edsonportosilva/OptiCommPy) is an
independent implementation of the same channel models. Only its *outputs*
are stored (with the inputs that produced them); hpcom does not depend on or
include its code. OptiCommPy needs numba, so run this in a separate
environment::

    uv venv ocp && VIRTUAL_ENV=$PWD/ocp uv pip install OptiCommPy==0.10.0
    ocp/bin/python tests/numerical/_data/make_opticommpy_golden.py

Cases (all 2 × 80 km spans, D = 16 ps/(nm·km), α = 0.2 dB/km, γ = 1.3 /W/km,
F_c = 193.1 THz, ideal loss-compensating amplifiers):

* ``ssfm_h{hz}`` — scalar NLSE, ``optic.models.channels.ssfm`` (symmetric
  split step, fixed step ``hz`` km);
* ``manakov_h{hz}`` — ``manakovSSF`` with fixed steps (``nlprMethod=False``;
  its nonlinear step is an iterated trapezoidal rule, a different scheme);
* ``edfa_var_*`` — complex noise variance per sample of
  ``optic.models.devices.edfa`` on a zero input, for a grid of gains and noise
  figures (``n_edfa`` samples each).
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
from optic.models.channels import manakovSSF, ssfm
from optic.models.devices import edfa
from optic.utils import parameters

OUT = Path(__file__).with_name("opticommpy_v0_10.npz")
N, FS, FC = 1024, 200e9, 193.1e12
SCALAR_POWER_W, MANAKOV_POWER_W = 10e-3, 10e-3
SCALAR_STEPS_KM = (8.0, 0.5)
MANAKOV_STEPS_KM = (2.0, 0.5, 0.125)
EDFA_GRID = ((16.0, 5.0), (20.0, 4.5), (10.0, 6.0), (25.0, 3.5))
N_EDFA = 1 << 20


def _field(shape: tuple[int, ...], power_w: float, seed: int) -> np.ndarray:
    """Random field band-limited to |f| ≤ 0.15·Fs with the given mean power (per row)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    spectrum = np.fft.fft(x, axis=0)
    spectrum[np.abs(np.fft.fftfreq(shape[0])) > 0.15] = 0
    x = np.fft.ifft(spectrum, axis=0)
    return x * np.sqrt(power_w / np.mean(np.abs(x) ** 2, axis=0))


def _channel(hz: float) -> parameters:
    p = parameters()
    p.Ltotal, p.Lspan, p.hz = 160.0, 80.0, hz
    p.alpha, p.D, p.gamma, p.Fc, p.Fs = 0.2, 16.0, 1.3, FC, FS
    p.amp, p.prgsBar = "ideal", False
    return p


def main() -> None:
    data: dict[str, np.ndarray] = {
        "n": np.array(N),
        "fs_hz": np.array(FS),
        "fc_hz": np.array(FC),
        "opticommpy_version": np.array(importlib.metadata.version("OptiCommPy")),
    }
    scalar_in = _field((N,), SCALAR_POWER_W, seed=1)
    data["ssfm_input"] = scalar_in
    for hz in SCALAR_STEPS_KM:
        data[f"ssfm_h{hz}"] = ssfm(scalar_in.copy(), _channel(hz))

    manakov_in = _field((N, 2), MANAKOV_POWER_W / 2, seed=2)  # columns x, y
    data["manakov_input"] = manakov_in
    for hz in MANAKOV_STEPS_KM:
        p = _channel(hz)
        p.nlprMethod = False
        p.saveSpanN = [2]
        data[f"manakov_h{hz}"] = manakovSSF(manakov_in.copy(), p)

    gains, nfs, variances = [], [], []
    for i, (gain_db, nf_db) in enumerate(EDFA_GRID):
        amp = parameters()
        amp.G, amp.NF, amp.Fc, amp.Fs, amp.seed = gain_db, nf_db, FC, FS, 100 + i
        noise = edfa(np.zeros(N_EDFA, dtype=np.complex128), amp)
        gains.append(gain_db)
        nfs.append(nf_db)
        variances.append(np.mean(np.abs(noise) ** 2))
    data["edfa_gain_db"] = np.array(gains)
    data["edfa_nf_db"] = np.array(nfs)
    data["edfa_var"] = np.array(variances)
    data["n_edfa"] = np.array(N_EDFA)
    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
