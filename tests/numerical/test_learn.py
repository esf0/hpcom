"""End-to-end learning interface (:mod:`hpcom.learn`).

* the Gaussian-demapper rate of QAM over a linear link with ASE equals the
  MI of QAM on the AWGN channel at the predicted SNR;
* learned blocks start exactly at their fixed counterparts (learned DBP =
  DBP, NN decoder ≈ Gaussian demapper);
* training increases the rate (geometric shaping on an AWGN-like link);
* the trainer is deterministic, the evaluation is paired, and every
  layout (single/dual polarisation, single carrier/WDM) runs;
* invalid systems are rejected and systems round-trip through JSON.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pydantic import ValidationError  # noqa: E402

from hpcom import (  # noqa: E402
    EDFA,
    ConstellationConfig,
    Fiber,
    Link,
    OFDMConfig,
    OFDMTransmitter,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
    learn,
    nli,
)
from hpcom.metrics import information_rates  # noqa: E402
from hpcom.modulation.dm import maxwell_boltzmann  # noqa: E402


def _ps16() -> ConstellationConfig:
    qam = ConstellationConfig.from_square_qam(16)
    return ConstellationConfig.from_points(
        qam.points, qam.bit_labels, maxwell_boltzmann(qam.points, 0.05)
    )


def _link(
    spans: int = 2, polarization: str = "single", gamma: float = 1.27, nf: float = 5.0
) -> Link:
    fiber = Fiber.ssmf(80.0, n_steps=8, polarization=polarization).replace(gamma_per_w_per_km=gamma)
    return Link.from_spans(spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=nf))


def _sc(**kwargs) -> SingleCarrierTransmitter:
    fields = {"modulation_order": 16, "symbol_rate_hz": 32e9, "n_symbols": 1024, **kwargs}
    return SingleCarrierTransmitter(**fields)


def _wdm(**kwargs) -> WDMTransmitter:
    fields = {
        "modulation_order": 16,
        "symbol_rate_hz": 32e9,
        "n_symbols": 512,
        "samples_per_symbol": 5,
        "wdm": WDMConfig(n_channels=3, spacing_hz=50e9),
        **kwargs,
    }
    return WDMTransmitter(**fields)


def test_gaussian_demapper_rate_equals_qam_mi_at_the_predicted_snr():
    """Linear link, ASE only: the rate equals the MI of 16-QAM on AWGN at P/P_ASE."""
    link = _link(spans=20, gamma=0.0)
    tx = _sc(launch_power_per_channel_dbm=-18.0, n_symbols=4096)
    system = learn.EndToEnd(transmitter=tx, link=link)
    params = system.init(jax.random.key(0))
    rate = learn.evaluate(system, params, blocks=8).mean
    snr_db = nli.predict_snr(link, tx, eta_per_w2=1e-30).snr_db
    rng = np.random.default_rng(1)
    cfg = tx.resolved_constellation
    idx = rng.integers(0, 16, 400_000)
    x = cfg.points[idx] / np.sqrt(np.mean(np.abs(cfg.points) ** 2))
    noise = (rng.standard_normal(idx.size) + 1j * rng.standard_normal(idx.size)) * np.sqrt(
        10 ** (-snr_db / 10) / 2
    )
    mi, _ = information_rates(
        x + noise, idx, cfg.points / np.sqrt(np.mean(np.abs(cfg.points) ** 2)), cfg.bit_labels
    )
    assert 1.5 < mi < 3.8  # a regime where the MI is sensitive to the SNR
    assert rate == pytest.approx(mi, abs=0.02)


@pytest.mark.parametrize("wideband", [False, True])
def test_learned_dbp_starts_as_dbp(wideband):
    link = _link(polarization="dual")
    tx = _wdm(polarization="dual", launch_power_per_channel_dbm=4.0)
    fixed = learn.EndToEnd(
        transmitter=tx, link=link, front_end=learn.DBP(steps_per_span=2, wideband=wideband)
    )
    learned = fixed.replace(front_end=learn.LearnedDBP(steps_per_span=2, wideband=wideband))
    key = jax.random.key(3)
    a = jax.jit(fixed.rate_fn())(fixed.init(key), key)
    b = jax.jit(learned.rate_fn())(learned.init(key), key)
    assert float(b) == pytest.approx(float(a), abs=2e-4)


def test_neural_decoder_starts_at_the_gaussian_demapper():
    link = _link(spans=10, nf=6.0)
    tx = _sc(launch_power_per_channel_dbm=-6.0)
    gauss = learn.EndToEnd(transmitter=tx, link=link)
    for decoder in (learn.NNDecoder(), learn.CNNDecoder()):
        neural = gauss.replace(decoder=decoder)
        a = learn.evaluate(gauss, gauss.init(jax.random.key(0)), blocks=4).mean
        b = learn.evaluate(neural, neural.init(jax.random.key(0)), blocks=4).mean
        assert b == pytest.approx(a, abs=0.03)


def test_training_learns_a_shaped_constellation():
    """AWGN-like link (γ = 0): learned points beat 64-QAM, as geometric shaping does."""
    link = _link(spans=20, gamma=0.0)
    tx = _sc(modulation_order=64, launch_power_per_channel_dbm=-11.0, n_symbols=2048)
    qam = learn.EndToEnd(transmitter=tx, link=link)
    gs = qam.replace(encoder=learn.Constellation(learning_rate=5e-3))
    trained = learn.Trainer(steps=300, gradient="store").fit(gs)
    base = learn.evaluate(qam, qam.init(jax.random.key(0)), blocks=8)
    shaped = learn.evaluate(gs, trained.params, blocks=8)
    gains = np.subtract(shaped.rates, base.rates)  # paired: same data and noise
    assert gains.mean() > 0.02 and gains.mean() > 4 * gains.std() / np.sqrt(gains.size)
    assert trained.history[0][0] == 0 and len(trained.history) > 5


def test_trainer_is_deterministic_and_evaluation_paired():
    system = learn.EndToEnd(transmitter=_sc(), link=_link(), encoder=learn.Constellation())
    trainer = learn.Trainer(steps=5, gradient="store")
    a, b = trainer.fit(system, seed=7), trainer.fit(system, seed=7)
    np.testing.assert_array_equal(
        np.asarray(a.params["encoder"]["points"]), np.asarray(b.params["encoder"]["points"])
    )
    assert (
        learn.evaluate(system, a.params, blocks=3).rates
        == learn.evaluate(system, a.params, blocks=3).rates
    )
    untouched = learn.Trainer(steps=0).fit(system)
    np.testing.assert_array_equal(
        np.asarray(untouched.params["encoder"]["points"]),
        np.asarray(system.init(jax.random.split(jax.random.key(0))[0])["encoder"]["points"]),
    )


@pytest.mark.parametrize("polarization", ["single", "dual"])
@pytest.mark.parametrize("carrier", ["single", "wdm"])
def test_every_layout_trains_through_the_reversible_adjoint(polarization, carrier):
    tx = (_sc if carrier == "single" else _wdm)(polarization=polarization, n_symbols=256)
    link = _link(polarization=polarization)
    system = learn.EndToEnd(
        transmitter=tx,
        link=link,
        encoder=learn.NNEncoder(),
        predistorter=learn.CNNPredistorter(),
        front_end=learn.LearnedDBP(steps_per_span=1, wideband=carrier == "wdm"),
        decoder=learn.CNNDecoder(),
    )
    trained = learn.Trainer(steps=2, gradient="reversible").fit(system)
    finer = _link(polarization=polarization).replace(
        elements=tuple(
            e.with_steps(n_steps=16) if isinstance(e, Fiber) else e for e in link.elements
        )
    )
    rate = learn.evaluate(system, trained.params, blocks=2, link=finer).mean
    assert 3.5 < rate <= 4.0 + 1e-6


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "transmitter": OFDMTransmitter(
                    modulation_order=16,
                    subcarrier_spacing_hz=1e9,
                    ofdm=OFDMConfig(n_subcarriers=64, cp_len=8),
                    n_frames=4,
                )
            },
            "pulse-shaped",
        ),
        ({"transmitter": _wdm(), "channels": (3,)}, "outside"),
        ({"transmitter": _sc(), "channels": (0,)}, "WDM"),
        (
            {
                "transmitter": _sc(
                    modulation_order=None,
                    constellation=_ps16(),
                )
            },
            "PMF",
        ),
    ],
)
def test_invalid_systems_are_rejected(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        learn.EndToEnd(link=_link(), **kwargs)


def test_single_channel_learned_dbp_scores_one_channel():
    system = learn.EndToEnd(
        transmitter=_wdm(),
        link=_link(),
        channels=(0, 1),
        front_end=learn.LearnedDBP(steps_per_span=1),
    )
    with pytest.raises(ValueError, match="exactly one"):
        system.init(jax.random.key(0))


def test_system_round_trips_through_json():
    system = learn.EndToEnd(
        transmitter=_wdm(polarization="dual"),
        link=_link(polarization="dual"),
        channels=(1,),
        encoder=learn.NNEncoder(hidden=32),
        predistorter=learn.CNNPredistorter(dilations=(1, 2)),
        front_end=learn.LearnedDBP(steps_per_span=3, wideband=True),
        decoder=learn.NNDecoder(window=5),
    )
    restored = learn.EndToEnd.model_validate_json(system.model_dump_json())
    assert restored == system
    assert type(restored.front_end) is learn.LearnedDBP
    assert system.scored_channels == (1,) and learn.EndToEnd(
        transmitter=_wdm(), link=_link()
    ).scored_channels == (1,)
