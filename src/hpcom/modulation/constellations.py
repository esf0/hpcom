"""Square-QAM constellations with Gray coding."""

from __future__ import annotations

import math
from functools import cache
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from hpcom._model import readonly

#: Square-QAM orders supported by :func:`square_qam`.
SUPPORTED_QAM_ORDERS = (4, 16, 64, 256, 1024, 4096)

#: Bit-labelling conventions of :func:`square_qam`. Both are Gray codes of the
#: same points; they differ in the direction the quadrature rows are counted,
#: so one is the complex conjugate of the other (same bits → conjugate symbols).
#:
#: * ``"q_descending"`` (default) — the Q half of a label is the Gray code of
#:   the row counted **from the top** (highest Q first). This is the pre-v1
#:   hpcom (``master``) mapping: the same bits give the same field.
#: * ``"q_ascending"`` — the Q half is the Gray code of the row counted from
#:   the bottom.
SquareQAMLabeling = Literal["q_descending", "q_ascending"]
DEFAULT_QAM_LABELING: SquareQAMLabeling = "q_descending"


def bits_per_symbol(modulation_order: int) -> int:
    """Number of bits per symbol, ``log₂ M``."""
    if modulation_order < 2 or (modulation_order & (modulation_order - 1)) != 0:
        raise ValueError(f"modulation_order must be a power of 2 ≥ 2, got {modulation_order}")
    return int(modulation_order).bit_length() - 1


@cache
def square_qam(
    modulation_order: int,
    labeling: SquareQAMLabeling = DEFAULT_QAM_LABELING,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.uint8]]:
    """Gray-coded square M-QAM, normalised to unit average power.

    Parameters
    ----------
    modulation_order
        ``M``, one of :data:`SUPPORTED_QAM_ORDERS`.
    labeling
        :data:`SquareQAMLabeling`. ``"q_descending"`` (default) reproduces
        pre-v1 hpcom bit for bit; ``"q_ascending"`` maps the same bits to the
        complex-conjugate symbols. BER, SER, MI and GMI agree in distribution
        for the two (``tests/numerical/test_labeling.py``).

    Returns
    -------
    points
        ``(M,)`` complex symbols; row ``i`` of ``bit_labels`` is the label of
        ``points[i]``. Point ``i = q·√M + p`` has in-phase level index ``p``
        and quadrature level index ``q`` (levels ``−(√M−1), …, √M−1``, step 2).
        The points do not depend on ``labeling``.
    bit_labels
        ``(M, log₂M)`` uint8, MSB first. The first half of each label is the
        Gray code of the Q row (``√M−1−q`` for ``"q_descending"``, ``q`` for
        ``"q_ascending"``), the second half that of the I level ``p``, so
        nearest neighbours along either axis differ in exactly one bit.

    Both arrays are read-only (the result is cached and shared).
    """
    if modulation_order not in SUPPORTED_QAM_ORDERS:
        raise ValueError(
            f"square_qam supports orders {SUPPORTED_QAM_ORDERS}, got {modulation_order}"
        )
    if labeling not in get_args(SquareQAMLabeling):
        raise ValueError(f"labeling must be one of {get_args(SquareQAMLabeling)}, got {labeling!r}")
    side = math.isqrt(modulation_order)
    bits_per_axis = bits_per_symbol(side)
    levels = 2.0 * np.arange(side) - (side - 1)
    norm = np.sqrt(2.0 * (side**2 - 1) / 3.0)  # RMS of the unscaled grid
    gray = np.arange(side) ^ (np.arange(side) >> 1)
    axis_bits = (gray[:, None] >> np.arange(bits_per_axis - 1, -1, -1)) & 1  # (side, bpa)

    q_idx, i_idx = np.divmod(np.arange(modulation_order), side)
    points = (levels[i_idx] + 1j * levels[q_idx]) / norm
    q_row = side - 1 - q_idx if labeling == "q_descending" else q_idx
    labels = np.concatenate([axis_bits[q_row], axis_bits[i_idx]], axis=1).astype(np.uint8)
    return readonly(points, np.complex128), readonly(labels, np.uint8)
