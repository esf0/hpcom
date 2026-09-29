r"""Differentiable propagation with JAX: gradients through fibers, links and DBP.

Pure JAX functions with the physics, schemes and step sequences of hpcom's
propagators, differentiable with :func:`jax.grad`, compilable with
:func:`jax.jit` and batchable with :func:`jax.vmap`:

* :func:`propagate` — one fiber span (scalar, Manakov, multicore; every
  fixed-step scheme; generalised-NLSE terms);
* :func:`amplify`, :func:`propagate_link` — EDFAs with reparameterised ASE
  from a JAX PRNG key, whole links;
* :func:`backpropagate`, :func:`compensate` — digital back-propagation with
  a trainable link model and nonlinearity scale (learned DBP).

The configuration objects (:class:`~hpcom.Fiber`, :class:`~hpcom.Link`, …)
fix the structure; the continuous parameters are pytrees —
:class:`FiberParameters` (in ps/km units, float32-safe),
:class:`AmplifierParameters` — whose leaves can be differentiated or
trained::

    import jax, jax.numpy as jnp
    from hpcom import Fiber, autodiff as ad

    jax.config.update("jax_enable_x64", True)
    fiber = Fiber.ssmf(80.0, n_steps=100)
    params = ad.FiberParameters.from_fiber(fiber)

    def loss(p):
        y = ad.propagate(x, fiber, dt_s, params=p)
        return jnp.sum(jnp.abs(y - target) ** 2)

    grads = jax.grad(loss)(params)  # d loss / d(α, β's, γ, …)

Memory for long links: ``gradient="checkpoint"`` (√steps) or
``"reversible"`` (constant — rebuilds the states by the exact inverse of the
split step); see :func:`propagate`. Requires ``pip install "hpcom[jax]"``;
see ``docs/autodiff.rst``.
"""

from __future__ import annotations

try:
    import jax as _jax  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        'hpcom.autodiff needs JAX: pip install "hpcom[jax]" (NVIDIA GPU: "hpcom[jax-cuda]")'
    ) from exc

from hpcom.autodiff._fiber import FiberParameters, GradientMode, propagate
from hpcom.autodiff._link import (
    AmplifierParameters,
    LinkParameters,
    amplify,
    backpropagate,
    compensate,
    link_parameters,
    propagate_link,
)

__all__ = [
    "AmplifierParameters",
    "FiberParameters",
    "GradientMode",
    "LinkParameters",
    "amplify",
    "backpropagate",
    "compensate",
    "link_parameters",
    "propagate",
    "propagate_link",
]
