"""Space-division multiplexing: one signal per core over :class:`~hpcom.MulticoreFiber` links.

* transmitters lay out ``(core, polarisation, channel)`` streams, the
  multicore field layout, with independent data and the launch power per
  core;
* over uncoupled cores the chain is exactly the single-core chain, core by
  core;
* over coupled cores only a MIMO equaliser across cores recovers the
  signals; DBP runs on the multicore field;
* the differentiable chain equals the NumPy one.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom import (
    EDFA,
    DigitalBackPropagation,
    Fiber,
    Link,
    MulticoreFiber,
    OFDMConfig,
    OFDMTransmitter,
    Simulation,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
)
from hpcom.multicore import hexagonal_coupling, linear_array_coupling

BASE = Fiber.ssmf(80.0, n_steps=20, polarization="dual")


def _sdm_link(
    coupling_per_m: float, n_cores: int = 3, noise_figure_db: float | None = 5.0, spans: int = 3
) -> Link:
    kappa = (
        hexagonal_coupling(coupling_per_m)
        if n_cores == 7
        else linear_array_coupling(n_cores, coupling_per_m)
    )
    mcf = MulticoreFiber.from_fiber(BASE, kappa)
    return Link.from_spans(spans, mcf, EDFA(gain_db=mcf.loss_db, noise_figure_db=noise_figure_db))


def _tx(**kwargs) -> SingleCarrierTransmitter:
    fields = {
        "modulation_order": 16,
        "symbol_rate_hz": 32e9,
        "n_symbols": 2048,
        "polarization": "dual",
        "n_cores": 3,
        "seed": 1,
        **kwargs,
    }
    return SingleCarrierTransmitter(**fields)


def test_transmitter_lays_out_cores_first_with_independent_data_and_per_core_power():
    tx = _tx(launch_power_per_channel_dbm=3.0)
    sig = tx.generate()
    assert tx.stream_shape == (3, 2) and sig.field.shape == (3, 2, 2048 * 8)
    assert not np.array_equal(sig.bits[0], sig.bits[1])
    power_per_core = np.sum(np.mean(np.abs(sig.field) ** 2, axis=-1), axis=-1)
    np.testing.assert_allclose(power_per_core, 10**0.3 * 1e-3, rtol=0.05)
    assert tx.launch_power_total_w == pytest.approx(3 * 10**0.3 * 1e-3)
    assert _tx(launch_power_total_dbm=0.0).launch_power_per_channel_w == pytest.approx(1e-3 / 3)
    wdm = WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=512,
        n_cores=2,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    )
    assert wdm.stream_shape == (2, 3) and wdm.generate().field.shape == (2, 512 * 8)


def test_simulation_rejects_a_core_count_mismatch_and_receivers_check_the_layout():
    link = _sdm_link(1e-6)
    tx = _tx(n_cores=2)
    with pytest.raises(ValueError, match="n_cores=2"):
        Simulation(transmitter=tx, link=link, receiver=tx.matched_receiver())
    rx = _tx().matched_receiver()
    with pytest.raises(
        ValueError, match=r"3-core dual-polarisation receiver expects shape \(3, 2, n\)"
    ):
        rx.receive(np.zeros((2, 2048 * 8), complex), 1e-12)
    assert rx.compatibility_issues(_tx(n_cores=2)) == ["n_cores: transmitter 2, receiver 3"]


def test_uncoupled_cores_are_exactly_the_single_core_chain():
    """Noiseless, nonlinear: every core's received symbols equal its own single-core run."""
    link = _sdm_link(0.0, noise_figure_db=None)
    tx = _tx(launch_power_per_channel_dbm=4.0)
    rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=3).run()
    single_link = Link.from_spans(3, BASE, EDFA(gain_db=BASE.loss_db, noise_figure_db=None))
    one = _tx(n_cores=1, launch_power_per_channel_dbm=4.0)
    for core in range(3):
        y = single_link.propagate(result.transmitted.field[core], one.dt_s)
        sent = one.generate(symbols=result.transmitted.data_symbols()[core])
        expected = one.matched_receiver(cd_compensation_s2=link.total_dispersion_s2).receive(
            y, one.dt_s, reference=sent
        )
        np.testing.assert_allclose(result.received.symbols[core], expected.symbols, atol=1e-10)
    cores = {m.core for m in result.stream_metrics}
    assert cores == {0, 1, 2} and {m.polarization for m in result.stream_metrics} == {0, 1}


def test_coupled_cores_need_a_mimo_equaliser_across_cores():
    """7-core hexagon, κL ≈ 0.4 rad over the link: per-core receivers fail, a joint 14×14 MIMO recovers."""
    link = _sdm_link(1e-6, n_cores=7)
    tx = _tx(n_cores=7, n_symbols=4096)
    sim = Simulation(
        transmitter=tx,
        link=link,
        receiver=tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2),
        seed=4,
    )
    per_core = sim.run()
    separate = sim.evaluate(
        per_core, tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2, equalize="mimo")
    )
    joint = sim.evaluate(
        per_core,
        tx.matched_receiver(
            cd_compensation_s2=link.total_dispersion_s2,
            equalize="mimo",
            mimo_across_cores=True,
            mimo_taps=11,
        ),
    )
    assert per_core.metrics.snr_db < 10.0 and separate.metrics.snr_db < 10.0
    assert joint.metrics.snr_db > 20.0


def test_dbp_on_a_multicore_link_with_wdm_channels():
    """Full-field DBP of the multicore WDM field, with the forward steps, undoes the noiseless link.

    (Coarser steps would need the FWM-aware step rule for a 150 GHz band;
    ``tests/numerical/test_dbp.py`` covers step counts.)
    """
    link = _sdm_link(1e-7, n_cores=2, noise_figure_db=None)
    tx = WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=2048,
        polarization="dual",
        n_cores=2,
        launch_power_per_channel_dbm=6.0,
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
        seed=5,
    )
    sim = Simulation(
        transmitter=tx,
        link=link,
        receiver=tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2, channels=(1,)),
        seed=6,
    )
    cdc = sim.run()
    dbp = sim.evaluate(
        cdc,
        tx.matched_receiver(
            channels=(1,),
            compensation=DigitalBackPropagation(
                link=link, steps_per_span=None, samples_per_symbol=None
            ),
        ),
    )
    assert cdc.metrics.snr_db < 25.0 and dbp.metrics.snr_db > 100.0
    labels = {(m.core, m.polarization, m.channel) for m in dbp.stream_metrics}
    assert labels == {(c, p, 1) for c in range(2) for p in range(2)}


def test_ofdm_over_cores():
    link = _sdm_link(0.0, n_cores=2, noise_figure_db=None)
    tx = OFDMTransmitter(
        modulation_order=16,
        subcarrier_spacing_hz=500e6,
        ofdm=OFDMConfig(n_subcarriers=64, cp_len=8, oversampling=2),
        n_frames=16,
        polarization="dual",
        n_cores=2,
        launch_power_per_channel_dbm=-5.0,
        seed=7,
    )
    rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=8).run()
    assert result.received_field.shape[:2] == (2, 2)
    assert result.metrics.snr_db > 30.0


def test_differentiable_chain_equals_numpy_for_sdm():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from hpcom import autodiff as ad

    link = _sdm_link(1e-6, n_cores=3)
    tx = _tx(n_symbols=512)
    sig = tx.generate()
    with jax.enable_x64(True):
        field = ad.modulate(jnp.asarray(sig.data_symbols()), tx)
        np.testing.assert_allclose(
            np.asarray(field), sig.field, atol=1e-14 * np.max(np.abs(sig.field))
        )
        y = link.propagate(sig.field, tx.dt_s, rng=np.random.default_rng(9))
        np.testing.assert_allclose(
            np.asarray(
                ad.propagate_link(
                    jnp.asarray(sig.field), link.replace(elements=link.elements[:1]), tx.dt_s
                )
            ),
            link.elements[0].propagate(sig.field, tx.dt_s),
            atol=1e-12 * np.max(np.abs(sig.field)),
        )
        for rx in (
            tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2),
            tx.matched_receiver(
                cd_compensation_s2=link.total_dispersion_s2,
                equalize="mimo",
                mimo_across_cores=True,
                mimo_taps=7,
            ),
        ):
            ref = rx.reference(sig)[1]
            expected = rx.receive(y, tx.dt_s, reference=sig).symbols
            out = ad.demodulate(jnp.asarray(y), rx, tx.dt_s, reference=jnp.asarray(ref))
            np.testing.assert_allclose(
                np.asarray(out), expected, atol=1e-7 * np.max(np.abs(expected))
            )
