"""End-to-end learning through a dual-polarisation 5-channel WDM link (reversible adjoint).

System: 5 × 32 GBd channels on a 50 GHz grid, dual polarisation, 64 points,
10 × 80 km SSMF with EDFAs (NF 5 dB); the split-step Manakov simulation of
the whole field (50 steps/span, logarithmic) is in the training loop, with
gradients by the reversible adjoint (memory independent of the step count).
The centre channel (CUT = 2) is scored; the rate is the Gaussian-demapper MI
per polarisation, averaged over the two polarisations (bit/symbol/pol;
×2 for bit/4D). Comparable setup: Jovanovic et al., JLT 41(12) 2023 (DP
5 × 32 GBd, 64 points, 10 × 100 km; +0.21 bit/4D for a learned
constellation through the SSFM).

Methods (receiver front end last; what is learned):

* ``qam+cdc`` — QAM, CD compensation, Gaussian demapper (fixed baseline);
* ``gs+cdc`` — learned constellation shared by all channels and polarisations;
* ``gs+txcnn+cdc`` — learned constellation and a transmitter with memory: a
  residual dilated CNN pre-distorter on each channel's symbol sequence (both
  polarisations, receptive field 169 symbols), starting at identity;
* ``gs+wdbp2`` — learned constellation, fixed wideband DBP of all five
  channels at 2 steps/span; ``gs+wldbp2`` — the same with learned DBP
  (``hpcom.autodiff.learned_dbp``);
* ``gs+txcnn+wldbp2`` — constellation, pre-distorter and learned DBP jointly;
* ``ffdbp`` — QAM, full-field DBP with the forward steps (ideal reference).

Run::

    python run.py --cosine --ema 0.99 --out results
    python run.py --quick --out smoke
"""

from __future__ import annotations

import argparse
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

M = 64
N_CH = 5
CUT = 2
DILATIONS = (1, 4, 16)  # kernel 9: receptive field 1 + 8·(1 + 4 + 16) = 169 symbols


def system(args, steps_per_span=None):
    fiber = Fiber.ssmf(
        80.0, n_steps=steps_per_span or args.steps_per_span, polarization="dual"
    ).with_steps(distribution="logarithmic", log_factor=1 / 3)
    link = Link.from_spans(args.spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))
    tx = WDMTransmitter(
        constellation=ConstellationConfig.from_square_qam(M),
        symbol_rate_hz=32e9,
        n_symbols=args.symbols,
        samples_per_symbol=8,
        polarization="dual",
        wdm=WDMConfig(n_channels=N_CH, spacing_hz=50e9),
    )
    return tx, link


def txcnn_init(key, dtype, width=32, kernel=9):
    """Residual pre-distorter: (Re, Im) × 2 polarisations in, the same out (last layer ×0.01)."""
    keys = jax.random.split(key, len(DILATIONS) + 1)
    layers, channels = [], 4
    for k in keys[:-1]:
        scale = math.sqrt(2.0 / (channels * kernel))
        layers.append(
            (
                jax.random.normal(k, (width, channels, kernel), dtype) * scale,
                jnp.zeros(width, dtype),
            )
        )
        channels = width
    out = jax.random.normal(keys[-1], (4, width, 1), dtype) * math.sqrt(2.0 / width) * 0.01
    return {"conv": layers, "out": (out, jnp.zeros(4, dtype))}


def txcnn(params, s):
    """Pre-distort one channel's symbols ``s`` (2, n): s + CNN(s), circular in time."""
    x = jnp.concatenate([s.real, s.imag])[None]  # (1, 4, n)
    for (w, b), d in zip(params["conv"], DILATIONS, strict=True):
        pad = d * (w.shape[-1] - 1) // 2
        x = jnp.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="wrap")
        x = jax.lax.conv_general_dilated(x, w, (1,), "VALID", rhs_dilation=(d,))
        x = jax.nn.relu(x + b[None, :, None])
    w, b = params["out"]
    y = jax.lax.conv_general_dilated(x, w, (1,), "VALID")[0] + b[:, None]  # (4, n)
    return s + (y[:2] + 1j * y[2:]).astype(s.dtype)


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
    stage_name = name.split("+")[-1]
    if stage_name == "cdc":
        return (
            tx.matched_receiver(channels=(CUT,), cd_compensation_s2=link.total_dispersion_s2),
            None,
            None,
        )
    if stage_name == "ffdbp":
        stage = DigitalBackPropagation(link=link, steps_per_span=None, samples_per_symbol=None)
        return tx.matched_receiver(channels=(CUT,), compensation=stage), None, None
    steps = int(re.search(r"(\d+)$", stage_name).group(1))
    stage = DigitalBackPropagation(link=link, steps_per_span=steps, samples_per_symbol=None)
    rx = tx.matched_receiver(channels=(CUT,), compensation=stage)
    if not stage_name.startswith("wldbp"):
        return rx, None, None
    n, dt = tx.n_symbols * tx.samples_per_symbol, tx.dt_s
    structure, params = ad.learned_dbp_init(link, n, dt, steps_per_span=steps, intensity_taps=8)
    return rx, structure, params


def make_method(name, args, tx, link, key):
    rdtype, cdtype = jnp.float32, jnp.complex64
    qam = jnp.asarray(tx.resolved_constellation.points, cdtype)
    labels = tx.resolved_constellation.bit_labels
    rx, structure, ldbp0 = front_end(name, tx, link)
    params, lrs = {}, {}
    if ldbp0 is not None:
        params["ldbp"] = jax.tree_util.tree_map(lambda v: jnp.asarray(v, rdtype), ldbp0)
        lrs["ldbp"] = args.lr_dbp
        params["kappa_scale"] = jnp.asarray(1.0, rdtype)  # one overall ξ, then per-step κ
        lrs["kappa_scale"] = 1e-3
    if name.startswith("gs"):
        params["points"] = qam
        lrs["points"] = 1e-2
    if "txcnn" in name:
        params["txcnn"] = txcnn_init(key, rdtype)
        lrs["txcnn"] = args.lr
    power = jnp.asarray(args.power, rdtype)

    def points_of(p):
        pts = p.get("points", qam)
        return pts / jnp.sqrt(jnp.mean(jnp.abs(pts) ** 2))

    def received(p, key, channel=link, gradient="store"):
        k_data, k_noise = jax.random.split(key)
        pts = points_of(p)
        idx = jax.random.randint(k_data, (2, N_CH, tx.n_symbols), 0, M)  # (pol, channel, n)
        s = pts[idx]
        if "txcnn" in p:
            s = jax.vmap(lambda c: txcnn(p["txcnn"], c), in_axes=1, out_axes=1)(s)
            s = s / jnp.sqrt(jnp.mean(jnp.abs(s) ** 2, axis=(0, 2), keepdims=True))
        x = ad.modulate(s, tx, launch_power_dbm=power)
        y = ad.propagate_link(x, channel, tx.dt_s, key=k_noise, gradient=gradient)
        comp = None
        if "ldbp" in p:
            ldbp = p["ldbp"]._replace(nonlinear=p["ldbp"].nonlinear * p["kappa_scale"])
            comp = lambda f, dt, off: ad.learned_dbp(f, structure, ldbp)  # noqa: E731
        ref = pts[idx[:, CUT : CUT + 1]]  # the receiver decodes the constellation points
        z = ad.demodulate(y, rx, tx.dt_s, reference=ref, compensation=comp)[:, 0]  # (2, n)
        return pts, idx[:, CUT], z

    def rate(p, key, channel=link, gradient="store"):
        pts, idx, z = received(p, key, channel, gradient)
        per_pol = [ad.information_rates(z[q], idx[q], pts, labels)[0] for q in range(2)]
        return (per_pol[0] + per_pol[1]) / 2.0

    return params, lrs, rate, points_of


def train_and_evaluate(name, args, tx, link, seed):
    k_init, k_train = jax.random.split(jax.random.key(seed))
    k_eval = jax.random.key(int(1e6 + 10 * args.power))  # shared by all methods: paired
    params, lrs, rate, points_of = make_method(name, args, tx, link, k_init)
    history, t0 = [], time.perf_counter()
    if params:
        value_and_grad = jax.jit(jax.value_and_grad(lambda p, k: rate(p, k, link, args.gradient)))
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
    parser.add_argument("--powers", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "qam+cdc",
            "gs+cdc",
            "gs+txcnn+cdc",
            "gs+wdbp2",
            "gs+wldbp2",
            "gs+txcnn+wldbp2",
            "ffdbp",
        ],
    )
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--eval-blocks", type=int, default=16)
    parser.add_argument("--symbols", type=int, default=2048)
    parser.add_argument("--spans", type=int, default=10)
    parser.add_argument("--steps-per-span", type=int, default=50)
    parser.add_argument("--eval-steps-per-span", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-dbp", type=float, default=1e-3)
    parser.add_argument(
        "--gradient", choices=["store", "checkpoint", "reversible"], default="reversible"
    )
    parser.add_argument("--cosine", action="store_true", help="cosine learning-rate decay to 0")
    parser.add_argument(
        "--ema", type=float, default=0.0, help="evaluate a Polyak average (e.g. 0.99)"
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.powers, args.iters, args.eval_blocks, args.spans = [0.0], 4, 2, 2
        args.symbols, args.steps_per_span, args.eval_steps_per_span = 256, 6, 12
    args.out.mkdir(parents=True, exist_ok=True)
    tx, link = system(args)
    results = []
    if args.append and (args.out / "results.json").exists():
        results = json.loads((args.out / "results.json").read_text())["results"]
    for power in args.powers:
        args.power = power
        for m_i, name in enumerate(args.methods):
            r = train_and_evaluate(name, args, tx, link, seed=3000 + m_i)
            results.append(r)
            fine = f"  fine {r['rate_fine_mean']:.4f}" if "rate_fine_mean" in r else ""
            print(
                f"{power:+5.1f} dBm  {name:16s} rate {r['rate_mean']:.4f} ± {r['rate_std']:.4f}"
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
