"""Time every nonlinear backend on the same spans and check they agree with NumPy.

Run on the machine to be measured (e.g. a GPU node)::

    uv run python benchmarks/backends.py [--sizes 16384 65536 262144] [--repeats 3]

Prints the devices each backend sees, then one row per (field size, scheme,
backend): the median wall time of repeated runs after a warm-up (which
includes JIT compilation) and the largest deviation from the NumPy result.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from hpcom import Fiber, Scheme
from hpcom.backends import BackendUnavailableError, get_propagator

BACKENDS = ("numpy", "jax", "tensorflow")
SCHEMES = {
    "ssfm (uniform)": Scheme(),
    "ssfm4-suzuki": Scheme(method="ssfm4-suzuki"),
    "rk4ip": Scheme(method="rk4ip"),
    "local-error 1e-5": Scheme(method="ssfm-local-error", local_error=1e-5),
}


def _devices() -> None:
    try:
        import jax

        print("jax devices:", jax.devices())
    except ImportError:
        print("jax: not installed")
    try:
        import tensorflow as tf

        print("tensorflow GPUs:", tf.config.list_physical_devices("GPU"))
    except ImportError:
        print("tensorflow: not installed")


def _field(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
    spectrum = np.fft.fft(x, axis=-1)
    spectrum[..., np.abs(np.fft.fftfreq(n)) > 0.2] = 0
    x = np.fft.ifft(spectrum, axis=-1)
    return x * np.sqrt(5e-3 / np.mean(np.abs(x) ** 2))


def main(sizes: tuple[int, ...] = (1 << 14, 1 << 16, 1 << 18), repeats: int = 3) -> None:
    _devices()
    available = []
    for name in BACKENDS:
        try:
            get_propagator(name)
            available.append(name)
        except (BackendUnavailableError, KeyError):
            pass
    print(f"backends: {available}\n")
    print(
        f"{'samples':>8}  {'scheme':<18} {'backend':<11} {'time [s]':>9}  {'max |Δ| / max|A|':>17}"
    )
    for n in sizes:
        field, dt = _field(n), 1 / 256e9
        for label, scheme in SCHEMES.items():
            fiber = Fiber.ssmf(80.0, n_steps=100, polarization="dual").replace(scheme=scheme)
            reference = None
            for name in available:
                f = fiber.replace(backend=name)
                out = f.propagate(field, dt)  # warm-up / JIT
                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    out = f.propagate(field, dt)
                    times.append(time.perf_counter() - t0)
                if reference is None:
                    reference = out
                dev = np.max(np.abs(out - reference)) / np.max(np.abs(reference))
                print(
                    f"{n:>8}  {label:<18} {name:<11} {statistics.median(times):>9.3f}  {dev:>17.1e}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1 << 14, 1 << 16, 1 << 18])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    main(tuple(args.sizes), args.repeats)
