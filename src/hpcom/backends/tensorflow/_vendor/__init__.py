"""Vendored copy of `ssfm-gpu` (https://github.com/esf0/ssfm-gpu).

Bundled into hpcom so the v1 release does not depend on the external
``ssfm-gpu`` PyPI package. Original source preserved verbatim under MIT
licence; see ``LICENSE`` and the repository's ``THIRD_PARTY_NOTICES.md``.

Do not import directly from outside ``hpcom.backends.tensorflow``. The public
backend is :class:`hpcom.backends.tensorflow.TensorFlowPropagator`.
"""

from . import conversion, examples, propagation

__all__ = ["propagation", "examples", "conversion"]
