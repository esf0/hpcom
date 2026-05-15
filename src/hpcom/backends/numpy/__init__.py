"""Pure-NumPy propagator backends.

These are reference implementations: simple, dependency-free, fast on CPU,
and used as a baseline that does not need TensorFlow or a GPU.
"""

from hpcom.backends.numpy.linear import LinearNumpyPropagator

__all__ = ["LinearNumpyPropagator"]
