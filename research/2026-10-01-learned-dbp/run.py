"""Learned DBP and long-memory decoders, trained end to end through a WDM link.

Follow-up of ``research/2026-10-01-nn-wdm`` (same system: 3 × 32 GBd, 50 GHz
grid, 256 points, 10 × 80 km SSMF, EDFA NF 5 dB; the centre channel is
decoded). New trainable receiver blocks:

* **learned DBP** (``hpcom.autodiff.learned_dbp``) — DBP whose every step has
  its own trainable dispersion filter (phase and gain on 33 knots), Kerr
  coefficient and 17-tap power filter, initialised to standard DBP;
* **long-memory decoder** — a dilated 1-D CNN over ±84 symbols (receptive
  field 169: the neighbours walk off by ≈174 symbols over the link) feeding
  the same Gaussian-metric + learned-correction demapper.

Methods (the receiver front end of each, then what is learned):

* ``dbp4`` — single-channel DBP, 4 steps/span, fixed; Gaussian demapper;
* ``ldbp1``, ``ldbp4`` — single-channel learned DBP, 1 or 4 steps/span;
* ``dbp4+gs+cnn`` — fixed DBP, learned points, CNN decoder;
* ``ldbp4+gs+cnn`` — learned DBP, learned points and CNN decoder, jointly;
* ``wdbp4`` — wideband DBP of all three channels (the receiver digitises the
  whole band), 4 steps/span, fixed; ``wldbp2`` — its learned version at
  2 steps/span;
* ``ffdbp`` — full-field DBP with the forward steps (ideal reference).

Run::

    python run.py --out results
    python run.py --quick --out smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
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

M = 256
CUT = 1
DILATIONS = (1, 4, 16)  # kernel 9: receptive field 1 + 8·(1 + 4 + 16) = 169 symbols


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


def mlp_init(key, sizes, dtype):
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        (jax.random.normal(k, (a, b), dtype) * math.sqrt(2.0 / (a + b)), jnp.zeros(b, dtype))
        for k, (a, b) in zip(keys, itertools.pairwise(sizes), strict=True)
    ]


def mlp(params, x):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def cnn_init(key, dtype, width=32, kernel=9):
    keys = jax.random.split(key, len(DILATIONS) + 1)
    layers, channels = [], 3
    for k in keys[:-1]:
        scale = math.sqrt(2.0 / (channels * kernel))
        layers.append(
            (
                jax.random.normal(k, (width, channels, kernel), dtype) * scale,
                jnp.zeros(width, dtype),
            )
        )
        channels = width
    head = mlp_init(keys[-1], [width, 128, M], dtype)
    head[-1] = (head[-1][0] * 0.01, head[-1][1])
    return {"conv": layers, "head": head}


def cnn(params, z):
    """Dilated circular 1-D convolutions over (Re z, Im z, |z|²) → per-symbol logits correction."""
    x = jnp.stack([z.real, z.imag, jnp.abs(z) ** 2])[None]  # (1, 3, n)
    for (w, b), d in zip(params["conv"], DILATIONS, strict=True):
        pad = d * (w.shape[-1] - 1) // 2
        x = jnp.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="wrap")
        x = jax.lax.conv_general_dilated(x, w, (1,), "VALID", rhs_dilation=(d,))
        x = jax.nn.relu(x + b[None, :, None])
    return mlp(params["head"], x[0].T)


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": 0}


def adam_step(params, grads, state, lrs, scale=1.0):
    """Gradient ascent; ``lrs`` holds a learning rate per top-level key, times ``scale``."""
    t = state["t"] + 1
    g = jax.tree_util.tree_map(jnp.conj, grads)
    m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, state["m"], g)
    v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * jnp.abs(b) ** 2, state["v"], g)
    new = {}
    for key in params:
        new[key] = jax.tree_util.tree_map(
            lambda p, a, b, lr=lrs[key] * scale: (
                p + lr * (a / (1 - 0.9**t)) / (jnp.sqrt(b / (1 - 0.999**t)) + 1e-8)
            ),
            params[key],
            m[key],
            v[key],
        )
    return new, {"m": m, "v": v, "t": t}


def front_end(name, tx, link):
    """(receiver, learned-DBP structure or None, initial learned-DBP parameters or None)."""
    if name == "ffdbp":
        stage = DigitalBackPropagation(link=link, steps_per_span=None, samples_per_symbol=None)
        return tx.matched_receiver(channels=(CUT,), compensation=stage), None, None
    if name.startswith(("wdbp", "wldbp")):
        steps = int(re.search(r"(\d+)$", name.split("+")[0]).group(1))
        stage = DigitalBackPropagation(link=link, steps_per_span=steps, samples_per_symbol=None)
        rx = tx.matched_receiver(channels=(CUT,), compensation=stage)
        n, dt = tx.n_symbols * tx.samples_per_symbol, tx.dt_s
    else:
        steps = int(re.search(r"(\d+)$", name.split("+")[0]).group(1))
        stage = DigitalBackPropagation(
            link=link, steps_per_span=steps, samples_per_symbol=2, bandwidth_hz=1.1 * 32e9
        )
        rx = tx.matched_receiver(channels=(CUT,), compensation=stage)
        n, dt = tx.n_symbols * 2, 1 / (2 * 32e9)
    if "ldbp" not in name:
        return rx, None, None
    structure, params = ad.learned_dbp_init(link, n, dt, steps_per_span=steps, intensity_taps=8)
    return rx, structure, params


def make_method(name, args, tx, link, key):
    rdtype, cdtype = jnp.float32, jnp.complex64
    qam = jnp.asarray(tx.resolved_constellation.points, cdtype)
    rx, structure, ldbp0 = front_end(name, tx, link)
    params, lrs = {}, {}
    if ldbp0 is not None:
        params["ldbp"] = jax.tree_util.tree_map(lambda v: jnp.asarray(v, rdtype), ldbp0)
        lrs["ldbp"] = args.lr_dbp
        params["kappa_scale"] = jnp.asarray(1.0, rdtype)  # one overall ξ, then per-step κ
        lrs["kappa_scale"] = 1e-3
    if "gs" in name:
        params["points"] = qam
        lrs["points"] = 1e-2
    if "cnn" in name:
        params["cnn"] = cnn_init(key, rdtype)
        lrs["cnn"] = args.lr
        params["log_scale"] = jnp.asarray(0.0, rdtype)
        lrs["log_scale"] = args.lr
    power = jnp.asarray(args.power, rdtype)

    def points_of(p):
        pts = p.get("points", qam)
        return pts / jnp.sqrt(jnp.mean(jnp.abs(pts) ** 2))

    def received(p, key, channel=link):
        k_data, k_noise = jax.random.split(key)
        pts = points_of(p)
        idx = jax.random.randint(k_data, (3, tx.n_symbols), 0, M)
        x = ad.modulate(pts[idx], tx, launch_power_dbm=power)
        y = ad.propagate_link(x, channel, tx.dt_s, key=k_noise)
        comp = None
        if "ldbp" in p:
            ldbp = p["ldbp"]._replace(nonlinear=p["ldbp"].nonlinear * p["kappa_scale"])
            comp = lambda s, dt, off: ad.learned_dbp(s, structure, ldbp)  # noqa: E731
        z = ad.demodulate(y, rx, tx.dt_s, reference=pts[idx[CUT : CUT + 1]], compensation=comp)[0]
        return pts, idx[CUT], z

    if "log_scale" in params:  # start at the Gaussian demapper
        pts0, idx0, z0 = jax.jit(received)(params, jax.random.key(7))
        params["log_scale"] = -jnp.log(jnp.mean(jnp.abs(z0 - pts0[idx0]) ** 2)).astype(rdtype)

    def rate(p, key, channel=link):
        pts, idx, z = received(p, key, channel)
        if "cnn" not in p:
            return ad.information_rates(z, idx, pts, tx.resolved_constellation.bit_labels)[0]
        logits = -jnp.exp(p["log_scale"]) * jnp.abs(z[:, None] - pts[None, :]) ** 2 + cnn(
            p["cnn"], z
        )
        picked = jnp.take_along_axis(logits, idx[:, None], axis=-1)[:, 0]
        return math.log2(M) - jnp.mean(jax.nn.logsumexp(logits, axis=-1) - picked) / math.log(2.0)

    return params, lrs, rate, points_of


def train_and_evaluate(name, args, tx, link, seed):
    k_init, k_train = jax.random.split(jax.random.key(seed))
    k_eval = jax.random.key(int(1e6 + 10 * args.power))  # shared by all methods: paired
    params, lrs, rate, points_of = make_method(name, args, tx, link, k_init)
    history, t0 = [], time.perf_counter()
    if params:
        value_and_grad = jax.jit(jax.value_and_grad(rate))
        state = adam_init(params)
        average = params
        for i, k in enumerate(jax.random.split(k_train, args.iters)):
            value, grads = value_and_grad(params, k)
            scale = 0.5 * (1 + math.cos(math.pi * i / args.iters)) if args.cosine else 1.0
            params, state = adam_step(params, grads, state, lrs, scale)
            if args.ema:  # Polyak average of the noisy iterates, used for evaluation
                average = jax.tree_util.tree_map(
                    lambda a, b: args.ema * a + (1 - args.ema) * b, average, params
                )
            if i % 25 == 0 or i == args.iters - 1:
                history.append((i, float(value)))
        if args.ema:
            params = average
    evaluate = jax.jit(rate)
    keys = jax.random.split(k_eval, args.eval_blocks)
    out = {
        "method": name,
        "power_dbm": args.power,
        "rates": [float(evaluate(params, k)) for k in keys],
        "history": history,
        "points": np.asarray(points_of(params)).astype(np.complex128).view(np.float64).tolist(),
        "train_s": time.perf_counter() - t0,
    }
    out["rate_mean"] = float(np.mean(out["rates"]))
    out["rate_std"] = float(np.std(out["rates"]) / math.sqrt(len(out["rates"])))
    if "ldbp" in params:
        p = params["ldbp"]
        out["ldbp"] = {
            "kappa_scale": float(params["kappa_scale"]),
            "nonlinear": np.asarray(p.nonlinear).tolist(),
            "intensity": np.asarray(p.intensity).tolist(),
            "phase": np.asarray(p.phase).tolist(),
            "log_gain": np.asarray(p.log_gain).tolist(),
        }
    if args.eval_steps_per_span:
        fine = system(args, steps_per_span=args.eval_steps_per_span)[1]
        evaluate_fine = jax.jit(lambda p, k: rate(p, k, fine))
        out["rates_fine"] = [float(evaluate_fine(params, k)) for k in keys]
        out["rate_fine_mean"] = float(np.mean(out["rates_fine"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--powers", type=float, nargs="+", default=[-4.0, -2.0, 0.0, 2.0])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "dbp4",
            "ldbp1",
            "ldbp4",
            "dbp4+gs+cnn",
            "ldbp4+gs+cnn",
            "wdbp4",
            "wldbp2",
            "ffdbp",
        ],
    )
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--eval-blocks", type=int, default=16)
    parser.add_argument("--symbols", type=int, default=2048)
    parser.add_argument("--spans", type=int, default=10)
    parser.add_argument("--steps-per-span", type=int, default=30)
    parser.add_argument("--eval-steps-per-span", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-dbp", type=float, default=1e-3)
    parser.add_argument("--cosine", action="store_true", help="cosine learning-rate decay to 0")
    parser.add_argument(
        "--ema", type=float, default=0.0, help="evaluate a Polyak average (e.g. 0.99)"
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.powers, args.iters, args.eval_blocks, args.spans = [0.0], 10, 2, 2
        args.symbols, args.steps_per_span, args.eval_steps_per_span = 256, 10, 20
    args.out.mkdir(parents=True, exist_ok=True)
    tx, link = system(args)
    results = []
    if args.append and (args.out / "results.json").exists():
        results = json.loads((args.out / "results.json").read_text())["results"]
    for power in args.powers:
        args.power = power
        for m_i, name in enumerate(args.methods):
            r = train_and_evaluate(name, args, tx, link, seed=2000 + m_i)
            results.append(r)
            fine = f"  fine {r['rate_fine_mean']:.4f}" if "rate_fine_mean" in r else ""
            print(
                f"{power:+5.1f} dBm  {name:13s} rate {r['rate_mean']:.4f} ± {r['rate_std']:.4f}"
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
