r"""Sensitivity of a field-to-field map: the singular values of its Jacobian.

How much does a link amplify a small change of its input — and, transposed,
a gradient flowing back to the transmitter? For a map :math:`y = f(x)` of
complex fields the Jacobian :math:`J` is real-linear on
:math:`(\operatorname{Re}x, \operatorname{Im}x)` (the Kerr term depends on
:math:`x^*`); its singular values :math:`\sigma_1 \ge \sigma_2 \ge \dots`
bound the gain of every input perturbation, :math:`\|J\delta\|/\|\delta\|`,
and of every back-propagated cotangent, :math:`\|J^\mathsf{T}g\|/\|g\|`.

**Structure.** Every split step is symplectic in the real coordinates: the
Kerr step :math:`A \mapsto Ae^{i\kappa P}` is the exact flow of the
Hamiltonian :math:`\kappa P^2/2` (Manakov: :math:`P = |A_x|^2 + |A_y|^2`), and
the dispersion step is unitary. Loss and gain only scale the field. A link
whose amplifiers restore the launch power is therefore symplectic as a
whole, and its singular values come in reciprocal pairs
:math:`\sigma_i\,\sigma_{2n+1-i} = 1`: a direction in which gradients grow
always comes with one in which they shrink by the same factor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from hpcom.autodiff._fiber import Array

__all__ = ["JacobianSpectrum", "jacobian_singular_values"]


class JacobianSpectrum(NamedTuple):
    """Leading singular values (descending) and right singular vectors of a Jacobian.

    ``vectors[i]`` is the input perturbation (shape of the input field,
    unit norm) amplified by ``singular_values[i]``.
    """

    singular_values: Array
    vectors: Array


def jacobian_singular_values(
    fn: Callable[[Array], Array],
    x: Array,
    *,
    k: int = 8,
    iterations: int = 30,
    key: Array | None = None,
) -> JacobianSpectrum:
    r"""The ``k`` largest singular values of the Jacobian of ``fn`` at ``x``.

    Block subspace iteration on :math:`J^\mathsf{T}J` with forward- and
    reverse-mode products (:func:`jax.linearize`, :func:`jax.linear_transpose`;
    no Jacobian is formed), then a Rayleigh–Ritz step: ``iterations`` costs
    ``k`` JVPs and ``k`` VJPs each. The error of :math:`\sigma_i` falls as
    :math:`(\sigma_{k+1}/\sigma_i)^{2\cdot\mathrm{iterations}}`; raise ``k``
    above the number you need when the spectrum is flat.

    ``fn`` maps a complex field to a complex field (e.g. ``lambda a:
    propagate_link(a, link, dt_s)`` with a noiseless link).
    """
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.complexfloating):
        x = x.astype(jnp.result_type(x.dtype, jnp.complex64))
    shape, n = x.shape, x.size
    rdtype = jnp.real(x).dtype

    def as_real(z: Array) -> Array:
        z = z.reshape(-1)
        return jnp.concatenate([jnp.real(z), jnp.imag(z)]).astype(rdtype)

    def as_complex(v: Array) -> Array:
        return (v[:n] + 1j * v[n:]).reshape(shape).astype(x.dtype)

    def f_real(v: Array) -> Array:
        return as_real(jnp.asarray(fn(as_complex(v))))

    v0 = as_real(x)
    _, jvp = jax.linearize(f_real, v0)
    transpose = jax.linear_transpose(jvp, v0)
    forward = jax.vmap(jvp)
    backward = jax.vmap(lambda w: transpose(w)[0])

    def orthonormal(rows: Array) -> Array:
        q, _ = jnp.linalg.qr(rows.T)
        return q.T

    key = jax.random.key(0) if key is None else key
    start = orthonormal(jax.random.normal(key, (k, 2 * n), rdtype))
    basis = lax.fori_loop(0, iterations, lambda _, v: orthonormal(backward(forward(v))), start)
    # forward(basis) is (J Vᵀ)ᵀ: the left factor of its SVD rotates the basis onto J's right vectors
    rotation, s, _ = jnp.linalg.svd(forward(basis), full_matrices=False)
    vectors = jax.vmap(as_complex)(rotation.T @ basis)
    return JacobianSpectrum(singular_values=s, vectors=vectors)
