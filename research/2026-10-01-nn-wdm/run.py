"""End-to-end NN encoder and decoder through a WDM fiber link (centre channel scored).

Follow-up of ``research/2026-09-30-nn-autoencoder`` with inter-channel
nonlinearity: 3 channels, 50 GHz apart, all transmitting with the *same*
(learned) constellation; the receiver decodes the centre channel. Training
is by Adam through the physical channel (``hpcom.autodiff``: split-step NLSE
of the whole WDM field + ASE), a fresh block of data and noise per step;
evaluation is paired on fresh blocks, and repeated on a 4× finer simulation.

Receiver chain of the scored channel (every method):

1. CD compensation of the whole field (or DBP, below), frequency domain;
2. shift the centre channel to 0 Hz, RRC matched filter, symbol-rate
   sampling;
3. data-aided complex gain (one LS complex number per block: mean phase and
   amplitude);
4. demapper: Gaussian metric, or NN decoder (Gaussian metric with a learned
   scale + MLP over a ±K-symbol window).

Methods:

* ``qam+gauss``, ``qam+nn``, ``gs+gauss``, ``ae+mem`` — as in the
  single-channel study (learned parts: points / NN encoder of all channels,
  NN decoder of the centre channel);
* ``qam+dbp``, ``gs+dbp``, ``ae+mem+dbp`` — single-channel DBP of the centre
  channel instead of CD compensation (low-pass to (1+β)R_s, 2 samples/symbol,
  4 steps/span): what a real receiver can do; the neighbours' XPM/FWM stays;
* ``qam+ffdbp`` — full-field DBP of all three channels at the simulation
  rate with the forward steps: an ideal, non-causal reference.

Run::

    python run.py --out results           # full sweep (GPU)
    python run.py --quick --out smoke     # seconds
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import hpcom
from hpcom import (
    EDFA,
    ConstellationConfig,
    DigitalBackPropagation,
    Fiber,
    Link,
    WDMConfig,
    WDMTransmitter,
)
from hpcom import autodiff as ad

M = 64  # constellation size; set by --order
CUT = 1  # centre channel of 3


def mlp_init(key, sizes, dtype):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (n_in, n_out) in zip(keys, itertools.pairwise(sizes), strict=True):
        w = jax.random.normal(k, (n_in, n_out), dtype) * math.sqrt(2.0 / (n_in + n_out))
        params.append((w, jnp.zeros(n_out, dtype)))
    return params


def mlp(params, x):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": 0}


def adam_step(params, grads, state, lr):
    """Gradient *ascent* on a rate; complex leaves take the conjugate gradient."""
    t = state["t"] + 1
    g = jax.tree_util.tree_map(jnp.conj, grads)
    m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, state["m"], g)
    v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * jnp.abs(b) ** 2, state["v"], g)
    new = jax.tree_util.tree_map(
        lambda p, a, b: p + lr * (a / (1 - 0.9**t)) / (jnp.sqrt(b / (1 - 0.999**t)) + 1e-8),
        params,
        m,
        v,
    )
    return new, {"m": m, "v": v, "t": t}


def system(args, steps_per_span=None):
    fiber = Fiber.ssmf(80.0, n_steps=steps_per_span or args.steps_per_span).with_steps(
        distribution="logarithmic", log_factor=1 / 3
    )
    link = Link.from_spans(args.spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
    tx = WDMTransmitter(
        constellation=ConstellationConfig.from_square_qam(M),
        symbol_rate_hz=32e9,
        n_symbols=args.symbols,
        samples_per_symbol=5,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )
    return tx, link


def receiver(name, tx, link):
    if name.endswith("ffdbp"):
        stage = DigitalBackPropagation(link=link, steps_per_span=None, samples_per_symbol=None)
        return tx.matched_receiver(channels=(CUT,), compensation=stage)
    if name.endswith("dbp"):
        stage = DigitalBackPropagation(
            link=link, steps_per_span=4, samples_per_symbol=2, bandwidth_hz=1.1 * 32e9
        )
        return tx.matched_receiver(channels=(CUT,), compensation=stage)
    return tx.matched_receiver(channels=(CUT,), cd_compensation_s2=link.total_dispersion_s2)


def window(z, half):
    """Decoder features: Re/Im of z[k−K … k+K] (circular) and |z[k]|²."""
    cols = [jnp.roll(z, -j) for j in range(-half, half + 1)]
    return jnp.stack([c.real for c in cols] + [c.imag for c in cols] + [jnp.abs(z) ** 2], axis=-1)


def make_method(name, args, tx, link, key):
    rdtype = jnp.float32 if args.dtype == "c64" else jnp.float64
    cdtype = jnp.complex64 if args.dtype == "c64" else jnp.complex128
    qam = jnp.asarray(tx.resolved_constellation.points, cdtype)
    half = args.window if "mem" in name or name == "qam+nn" else 0
    k_enc, k_dec = jax.random.split(key)
    params: dict = {}
    if name.startswith("ae"):
        enc = mlp_init(k_enc, [M, 64, 2], rdtype)
        params["enc"] = [*enc[:-1], (enc[-1][0] * 0.01, enc[-1][1])]
    if name.startswith("gs"):
        params["points"] = qam
    if name == "qam+nn" or name.startswith("ae"):
        dec = mlp_init(k_dec, [4 * half + 3, 128, 128, M], rdtype)
        params["dec"] = [*dec[:-1], (dec[-1][0] * 0.01, dec[-1][1])]
        params["log_scale"] = jnp.asarray(0.0, rdtype)
    power = jnp.asarray(args.power, rdtype)
    rx = receiver(name, tx, link)

    def points_of(p):
        if "enc" in p:
            out = mlp(p["enc"], jnp.eye(M, dtype=rdtype))
            pts = qam + (out[:, 0] + 1j * out[:, 1]).astype(cdtype)
        else:
            pts = p.get("points", qam)
        return pts / jnp.sqrt(jnp.mean(jnp.abs(pts) ** 2))

    def received(p, key, channel=link):
        k_data, k_noise = jax.random.split(key)
        pts = points_of(p)
        idx = jax.random.randint(k_data, (3, tx.n_symbols), 0, M)  # every channel
        x = ad.modulate(pts[idx], tx, launch_power_dbm=power)
        y = ad.propagate_link(x, channel, tx.dt_s, key=k_noise, gradient=args.gradient)
        ref = pts[idx[CUT : CUT + 1]]
        z = ad.demodulate(y, rx, tx.dt_s, reference=ref)[0]
        return pts, idx[CUT], z

    if "log_scale" in params:  # start exactly at the Gaussian demapper
        pts0, idx0, z0 = jax.jit(received)(params, jax.random.key(7))
        params["log_scale"] = -jnp.log(jnp.mean(jnp.abs(z0 - pts0[idx0]) ** 2)).astype(rdtype)

    def rate(p, key, channel=link):
        pts, idx, z = received(p, key, channel)
        if "dec" not in p:
            return ad.information_rates(z, idx, pts, tx.resolved_constellation.bit_labels)[0]
        metric = -jnp.exp(p["log_scale"]) * jnp.abs(z[:, None] - pts[None, :]) ** 2
        logits = metric + mlp(p["dec"], window(z, half))
        picked = jnp.take_along_axis(logits, idx[:, None], axis=-1)[:, 0]
        ce = jnp.mean(jax.nn.logsumexp(logits, axis=-1) - picked)
        return math.log2(M) - ce / math.log(2.0)

    lr = 1e-2 if name.startswith("gs") else args.lr
    return params, rate, points_of, lr


def train_and_evaluate(name, args, tx, link, seed):
    k_init, k_train = jax.random.split(jax.random.key(seed))
    k_eval = jax.random.key(int(1e6 + 10 * args.power))  # shared by every method: paired
    params, rate, points_of, lr = make_method(name, args, tx, link, k_init)
    history = []
    t0 = time.perf_counter()
    if params:
        value_and_grad = jax.jit(jax.value_and_grad(rate))
        state = adam_init(params)
        for i, k in enumerate(jax.random.split(k_train, args.iters)):
            value, grads = value_and_grad(params, k)
            params, state = adam_step(params, grads, state, lr)
            if i % 25 == 0 or i == args.iters - 1:
                history.append((i, float(value)))
    evaluate = jax.jit(rate)
    keys = jax.random.split(k_eval, args.eval_blocks)
    rates = [float(evaluate(params, k)) for k in keys]
    out = {
        "method": name,
        "power_dbm": args.power,
        "rate_mean": float(np.mean(rates)),
        "rate_std": float(np.std(rates) / math.sqrt(len(rates))),
        "rates": rates,
        "history": history,
        "points": np.asarray(points_of(params)).astype(np.complex128).view(np.float64).tolist(),
        "train_s": time.perf_counter() - t0,
    }
    if args.eval_steps_per_span:
        fine_link = system(args, steps_per_span=args.eval_steps_per_span)[1]
        evaluate_fine = jax.jit(lambda p, k: rate(p, k, fine_link))
        out["rates_fine"] = [float(evaluate_fine(params, k)) for k in keys]
        out["rate_fine_mean"] = float(np.mean(out["rates_fine"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--powers", type=float, nargs="+", default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "qam+gauss",
            "qam+nn",
            "gs+gauss",
            "ae+mem",
            "qam+dbp",
            "gs+dbp",
            "ae+mem+dbp",
            "qam+ffdbp",
        ],
    )
    parser.add_argument("--order", type=int, default=64)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--eval-blocks", type=int, default=16)
    parser.add_argument("--symbols", type=int, default=2048)
    parser.add_argument("--spans", type=int, default=10)
    parser.add_argument("--steps-per-span", type=int, default=30)
    parser.add_argument("--eval-steps-per-span", type=int, default=120)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    parser.add_argument(
        "--gradient", choices=["store", "checkpoint", "reversible"], default="store"
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    global M
    M = args.order
    if args.quick:
        args.powers, args.iters, args.eval_blocks, args.spans = [0.0], 10, 2, 2
        args.symbols, args.steps_per_span, args.eval_steps_per_span = 256, 10, 20
    if args.dtype == "c128":
        jax.config.update("jax_enable_x64", True)
    args.out.mkdir(parents=True, exist_ok=True)
    tx, link = system(args)
    results = []
    if args.append and (args.out / "results.json").exists():
        results = json.loads((args.out / "results.json").read_text())["results"]
    for power in args.powers:
        args.power = power
        for m_i, name in enumerate(args.methods):
            r = train_and_evaluate(name, args, tx, link, seed=1000 + m_i)
            results.append(r)
            fine = f"  fine {r['rate_fine_mean']:.4f}" if "rate_fine_mean" in r else ""
            print(
                f"{power:+5.1f} dBm  {name:11s} rate {r['rate_mean']:.4f} ± {r['rate_std']:.4f}"
                f"{fine}  ({r['train_s']:.0f} s)",
                flush=True,
            )
            config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
            payload = {
                "config": config,
                "devices": [str(d) for d in jax.devices()],
                "hpcom": hpcom.__version__,
                "jax": jax.__version__,
                "results": results,
            }
            (args.out / "results.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
