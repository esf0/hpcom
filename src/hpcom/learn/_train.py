"""Training (Adam with cosine decay and a Polyak average) and paired evaluation."""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Iterator
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from hpcom._model import HpcomModel
from hpcom.autodiff._fiber import GradientMode
from hpcom.learn._blocks import Params
from hpcom.learn._system import EndToEnd, Precision
from hpcom.link import Link

__all__ = ["Evaluation", "Trained", "Trainer", "evaluate"]


@contextlib.contextmanager
def _precision(precision: Precision) -> Iterator[None]:
    if precision == "double":
        with jax.enable_x64(True):
            yield
    else:
        yield


class Trained(HpcomModel):
    """Output of :meth:`Trainer.fit`: the parameters and the training curve."""

    params: Any
    history: tuple[tuple[int, float], ...]
    seconds: float


class Trainer(HpcomModel):
    r"""Gradient ascent on :meth:`EndToEnd.rate_fn` with Adam.

    Every step draws a fresh block of data and noise. Learning rates are
    the blocks' own; ``schedule="cosine"`` decays them to zero, and with
    ``ema`` the returned parameters are the exponential moving (Polyak)
    average of the iterates, which removes the drift of parameters whose
    gradients are mostly noise. Gradients through the link use ``gradient``
    (``"reversible"``: memory independent of the number of steps).

    Attributes
    ----------
    steps
        Adam steps.
    schedule
        ``"cosine"`` decay to 0 or ``"constant"``.
    ema
        Polyak-average factor (``None``: the last iterate).
    gradient
        Gradient mode of :func:`hpcom.autodiff.propagate_link`.
    precision
        ``"single"`` (complex64, fast on GPUs) or ``"double"``.
    log_every
        Training-curve sampling.
    """

    steps: int = Field(default=2000, ge=0)
    schedule: Literal["cosine", "constant"] = "cosine"
    ema: float | None = Field(default=0.99, ge=0, lt=1)
    gradient: GradientMode = "reversible"
    precision: Precision = "single"
    log_every: int = Field(default=25, ge=1)

    def fit(self, system: EndToEnd, seed: int = 0, params: Params | None = None) -> Trained:
        """Train ``system`` from its initial parameters (or ``params``)."""
        t0 = time.perf_counter()
        with _precision(self.precision):
            k_init, k_train = jax.random.split(jax.random.key(seed))
            p = system.init(k_init, self.precision) if params is None else params
            rates = system.learning_rates()
            trainable = {
                k: v for k, v in p.items() if jax.tree_util.tree_leaves(v) and rates[k] > 0
            }
            history: list[tuple[int, float]] = []
            if not trainable or self.steps == 0:
                return Trained(params=p, history=(), seconds=time.perf_counter() - t0)
            rate = system.rate_fn(self.precision, self.gradient)
            fixed = {k: v for k, v in p.items() if k not in trainable}

            def objective(train: Params, key: Any) -> Any:
                return rate({**fixed, **train}, key)

            value_and_grad = jax.jit(jax.value_and_grad(objective))
            m = jax.tree_util.tree_map(jnp.zeros_like, trainable)
            v = jax.tree_util.tree_map(jnp.zeros_like, trainable)
            average = trainable
            for i, key in enumerate(jax.random.split(k_train, self.steps)):
                value, grads = value_and_grad(trainable, key)
                t = i + 1
                scale = (
                    0.5 * (1.0 + math.cos(math.pi * i / self.steps))
                    if self.schedule == "cosine"
                    else 1.0
                )
                g = jax.tree_util.tree_map(jnp.conj, grads)  # ascent direction for complex leaves
                m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
                v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * jnp.abs(b) ** 2, v, g)
                trainable = {
                    name: jax.tree_util.tree_map(
                        lambda p_, a, b, lr=rates[name] * scale, t=t: (
                            p_ + lr * (a / (1 - 0.9**t)) / (jnp.sqrt(b / (1 - 0.999**t)) + 1e-8)
                        ),
                        trainable[name],
                        m[name],
                        v[name],
                    )
                    for name in trainable
                }
                if self.ema is not None:
                    average = jax.tree_util.tree_map(
                        lambda a, b: self.ema * a + (1 - self.ema) * b, average, trainable
                    )
                if i % self.log_every == 0 or i == self.steps - 1:
                    history.append((i, float(value)))
            final = average if self.ema is not None else trainable
            return Trained(
                params={**fixed, **final},
                history=tuple(history),
                seconds=time.perf_counter() - t0,
            )


class Evaluation(HpcomModel):
    """Rates of fixed evaluation blocks: per block, their mean and standard error (bit/symbol)."""

    rates: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.rates))

    @property
    def stderr(self) -> float:
        return (
            float(np.std(self.rates) / math.sqrt(len(self.rates))) if len(self.rates) > 1 else 0.0
        )


def evaluate(
    system: EndToEnd,
    params: Params,
    *,
    blocks: int = 16,
    seed: int = 1_000_000,
    link: Link | None = None,
    precision: Precision = "single",
) -> Evaluation:
    """The rate on ``blocks`` blocks drawn from ``seed`` — the same blocks for every system (paired).

    ``link`` (default: the system's) may be a finer simulation of the same
    link; it must have the same structure.
    """
    with _precision(precision):
        rate = jax.jit(system.rate_fn(precision, "store", link))
        keys = jax.random.split(jax.random.key(seed), blocks)
        return Evaluation(rates=tuple(float(rate(params, k)) for k in keys))
