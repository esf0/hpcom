r"""Optical amplifiers.

Provides :class:`EDFA`. New amplifier types (Raman, SOA, …) plug in by
subclassing :class:`Amplifier` and implementing :meth:`Amplifier.amplify`.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from pydantic import Field

from hpcom._model import HpcomModel
from hpcom._typing import ComplexArray
from hpcom.units import PLANCK_J_S, db_to_linear, wavelength_to_frequency_hz


class Amplifier(HpcomModel):
    """Base class for optical amplifiers.

    ``amplify`` accepts a field of any shape (time on the last axis; a
    dual-polarisation field is ``(..., 2, n)``) and returns the same shape.
    Stochastic noise is drawn from the ``rng`` argument so reproducibility is
    controlled by the caller (:class:`~hpcom.simulation.Simulation` threads
    its seed through :class:`~hpcom.link.Link`).
    """

    @abstractmethod
    def amplify(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        """Return the amplified field. Implementations may add stochastic noise."""

    @property
    @abstractmethod
    def gain_linear(self) -> float:
        """Power gain ``G`` (linear)."""

    def apply(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        """:class:`~hpcom._typing.LinkComponent` interface: alias for :meth:`amplify`."""
        return self.amplify(field, dt_s, rng=rng)

    def apply_inverse(self, field: ComplexArray, dt_s: float) -> ComplexArray:
        """Noiseless inverse gain ``field / √G`` (used by digital back-propagation)."""
        del dt_s
        out: ComplexArray = np.asarray(field, dtype=np.complex128) / np.sqrt(self.gain_linear)
        return out


class EDFA(Amplifier):
    r"""Erbium-doped fibre amplifier: power gain ``G`` plus ASE noise.

    The output field is

    .. math:: A_\text{out} = \sqrt{G}\,A_\text{in} + n,

    where ``n`` is white, circular complex Gaussian ASE, independent in every
    sample and every polarisation, with one-sided power spectral density
    **per polarisation** (Essiambre et al., "Capacity limits of optical fiber
    networks", *J. Lightwave Technol.* 28(4), 662–701, 2010; the same model
    as OptiCommPy's ``edfa``)

    .. math::

        S_\text{ASE} = n_\text{sp}\, h\nu\, (G - 1),
        \qquad
        n_\text{sp} = \frac{G\,F - 1}{2\,(G - 1)},

    so :math:`S_\text{ASE} = (G F - 1)\,h\nu/2 \approx F\,h\nu\,G/2` for
    :math:`G \gg 1`. ``F`` is the linear noise figure (``10^{NF_dB/10}``) and
    ``ν = c/λ`` the carrier frequency. The ``n_sp``–``F`` relation is the
    standard one for a phase-insensitive amplifier,
    :math:`F = [1 + 2 n_\text{sp}(G-1)]/G`. The ASE in *both* polarisations
    together is :math:`2 S_\text{ASE} \approx F h\nu G`.

    In the simulation, sampled at :math:`f_s = 1/\Delta t`, each complex
    sample of each polarisation receives noise of variance
    :math:`\sigma^2 = S_\text{ASE} f_s` (``σ²/2`` per quadrature). A
    single-polarisation field only receives its co-polarised ASE — the
    orthogonal ASE does not reach a single-polarisation coherent receiver.

    ``G = 1`` (``gain_db = 0``) adds no noise. Note that ``F = 1``
    (``noise_figure_db = 0``) is *below* the quantum limit of a
    phase-insensitive amplifier (``F ≥ 2 − 1/G``, i.e. ≈ 3 dB) but still
    yields ASE: ``n_sp = 1/2``.

    ``noise_figure_db=None`` is an **ideal noiseless amplifier** — exact gain
    :math:`\sqrt{G}\,A_\text{in}`, ``n_sp = 0``, no random numbers drawn.
    It is not a physical device (no finite ``F`` gives it); use it to isolate
    deterministic effects such as fibre nonlinearity or a DSP algorithm. It
    replaces the pre-v1 idiom ``noise_figure_db=-200``.

    .. note::
       Earlier conventions, per polarisation and relative to the level above
       at high gain: pre-v1 hpcom injected ``hν(G−1)F`` in the Manakov model
       (+3 dB) and ``2hν(G−1)F`` in the scalar model (+6 dB); hpcom
       0.2.0.dev injected ``hν(G−1)F`` in both (+3 dB). To reproduce such a
       result here, raise ``noise_figure_db`` by that amount.
    """

    gain_db: float = Field(ge=0)
    noise_figure_db: float | None = Field(default=5.0, ge=0)
    wavelength_m: float = Field(default=1550e-9, gt=0)

    @property
    def gain_linear(self) -> float:
        """Power gain ``G = 10^(gain_db/10)``."""
        return db_to_linear(self.gain_db)

    @property
    def is_noiseless(self) -> bool:
        """``True`` when no ASE is added: ``noise_figure_db=None`` or ``gain_db=0``."""
        return self.noise_figure_db is None or self.gain_linear == 1.0

    @property
    def noise_figure_linear(self) -> float:
        """Noise figure ``F = 10^(noise_figure_db/10)``; ``nan`` for the ideal noiseless amplifier."""
        if self.noise_figure_db is None:
            return float("nan")
        return db_to_linear(self.noise_figure_db)

    @property
    def spontaneous_emission_factor(self) -> float:
        """Population-inversion factor ``n_sp = (G F − 1) / (2 (G − 1))``.

        ``0`` for the ideal noiseless amplifier (``noise_figure_db=None``);
        undefined (returned as ``nan``) for ``G = 1``, where no ASE is emitted.
        """
        g = self.gain_linear
        if g == 1.0:
            return float("nan")
        if self.noise_figure_db is None:
            return 0.0
        return (g * self.noise_figure_linear - 1.0) / (2.0 * (g - 1.0))

    @property
    def ase_psd_per_polarization_w_per_hz(self) -> float:
        """One-sided ASE power spectral density per polarisation, ``n_sp h ν (G−1)``, in W/Hz."""
        g = self.gain_linear
        if self.is_noiseless:
            return 0.0
        h_nu = PLANCK_J_S * wavelength_to_frequency_hz(self.wavelength_m)
        return (g * self.noise_figure_linear - 1.0) * h_nu / 2.0

    def ase_variance_per_sample(self, dt_s: float) -> float:
        """Complex noise variance added to each sample of each polarisation, ``S_ASE / Δt``."""
        return self.ase_psd_per_polarization_w_per_hz / dt_s

    def amplify(
        self,
        field: ComplexArray,
        dt_s: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> ComplexArray:
        field = np.asarray(field, dtype=np.complex128)
        amplified: ComplexArray = field * np.sqrt(self.gain_linear)
        variance = self.ase_variance_per_sample(dt_s)
        if variance == 0.0:
            return amplified
        if rng is None:
            rng = np.random.default_rng()
        sigma = np.sqrt(variance / 2.0)
        noise = rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
        amplified += sigma * noise
        return amplified
