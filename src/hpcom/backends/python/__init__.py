"""Pure-Python (NumPy) DSP kernels — the reference ``"python"`` kernel set.

See :mod:`hpcom.backends.kernels` for the kernel interface and registry. A
future ``hpcom.backends.rust`` package mirrors :mod:`.kernels` function by
function.
"""

from hpcom.backends.python.kernels import ccdm_decode, ccdm_encode, nearest_indices

__all__ = ["ccdm_decode", "ccdm_encode", "nearest_indices"]
