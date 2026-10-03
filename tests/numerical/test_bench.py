"""Benchmark scenarios (:mod:`hpcom.bench`): definitions, paired scoring, leaderboard, CLI."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("jax")

from hpcom import EDFA, Fiber, SingleCarrierTransmitter, bench, learn

TINY = bench.Scenario(
    name="tiny",
    description="test scenario",
    transmitter=SingleCarrierTransmitter(
        modulation_order=256, symbol_rate_hz=32e9, n_symbols=256, polarization="dual"
    ),
    fiber=Fiber.ssmf(80.0, polarization="dual"),
    spans=2,
    steps_per_span=6,
    evaluation_steps_per_span=12,
    powers_dbm=(6.0, 10.0),
    evaluation_blocks=2,
)


def test_standard_scenarios_build_their_links_and_transmitters():
    assert set(bench.SCENARIOS) >= {"sc-dp-10x80", "wdm5-dp-10x80", "jovanovic2023"}
    for s in bench.SCENARIOS.values():
        link, scoring = s.link(), s.link(evaluation=True)
        assert len(link.elements) == 2 * s.spans
        assert scoring.elements[0].n_steps == 4 * link.elements[0].n_steps
        assert isinstance(link.elements[1], EDFA)
        assert link.elements[1].gain_db == pytest.approx(link.elements[0].loss_db)
        assert s.transmitter_at(1.5).launch_power_per_channel_dbm == 1.5
        assert bench.Scenario.model_validate_json(s.model_dump_json()) == s
    jovanovic = bench.scenario("jovanovic2023")
    fiber = jovanovic.link().elements[0]
    assert (fiber.length_km, fiber.gamma_per_w_per_km) == (100.0, 1.3)
    assert jovanovic.transmitter.roll_off == 0.01
    with pytest.raises(KeyError, match="unknown scenario"):
        bench.scenario("nope")


@pytest.mark.parametrize(
    ("name", "samples_per_symbol", "ok"),
    [
        ("wdm5-dp-10x80", 16, True),
        ("wdm5-dp-10x80", 8, False),
        ("jovanovic2023", 8, False),
        ("sc-dp-10x80", 4, True),
        ("sc-dp-10x80", 2, False),
    ],
)
def test_scenarios_sample_at_twice_the_occupied_bandwidth(name, samples_per_symbol, ok):
    """Kerr products reach ±1.5 B: below 2 B samples/s they alias into the signal band."""
    s = bench.scenario(name)
    tx = s.transmitter.replace(samples_per_symbol=samples_per_symbol)
    if ok:
        assert s.replace(transmitter=tx).transmitter.samples_per_symbol == samples_per_symbol
    else:
        with pytest.raises(ValueError, match="alias"):
            s.replace(transmitter=tx)


def test_methods_are_named_block_sets():
    assert {"qam+cdc", "gs+cdc", "gs+wldbp2", "qam+ffdbp"} <= set(bench.METHODS)
    m = bench.METHODS["gs+wldbp2"]
    assert isinstance(m.front_end, learn.LearnedDBP) and m.front_end.wideband
    assert bench.Method.model_validate_json(m.model_dump_json()) == m
    assert bench.METHODS["qam+cdc"].blocks() == {}


def test_run_scores_on_paired_blocks_and_ranks_methods():
    trainer = learn.Trainer(steps=2, gradient="store")
    cdc = bench.run(TINY, bench.METHODS["qam+cdc"], trainer)
    again = bench.run(TINY, bench.METHODS["qam+cdc"], trainer)
    assert cdc == again.model_copy(
        update={
            "rows": tuple(
                r.model_copy(update={"train_seconds": c.train_seconds})
                for r, c in zip(again.rows, cdc.rows, strict=True)
            )
        }
    )
    dbp = bench.run(TINY, bench.METHODS["qam+dbp4"], trainer)
    assert [r.power_dbm for r in cdc.rows] == [6.0, 10.0]
    assert dbp.rows[-1].rate_scored > cdc.rows[-1].rate_scored  # DBP helps at 10 dBm
    table = bench.leaderboard([cdc, dbp])
    assert table.index("qam+dbp4") < table.index("qam+cdc")


def test_command_line_appends_results(tmp_path, monkeypatch):
    monkeypatch.setitem(bench.SCENARIOS, "tiny", TINY)
    out = tmp_path / "result.json"
    bench.main(
        [
            "--scenario",
            "tiny",
            "--method",
            "qam+cdc",
            "--steps",
            "1",
            "--powers",
            "4",
            "--out",
            str(out),
        ]
    )
    bench.main(
        [
            "--scenario",
            "tiny",
            "--method",
            "qam+dbp1",
            "--steps",
            "1",
            "--powers",
            "4",
            "--out",
            str(out),
        ]
    )
    saved = [bench.MethodResult.model_validate(r) for r in json.loads(out.read_text())]
    assert {r.method for r in saved} == {"qam+cdc", "qam+dbp1"}
    assert saved[0].best.power_dbm == 4.0
    # another power for qam+cdc is added to its rows; the repeated one is identical
    args = ["--scenario", "tiny", "--method", "qam+cdc", "--steps", "1", "--powers", "4", "0"]
    bench.main([*args, "--out", str(out)])
    merged = {
        r.method: r
        for r in (bench.MethodResult.model_validate(x) for x in json.loads(out.read_text()))
    }
    assert [r.power_dbm for r in merged["qam+cdc"].rows] == [0.0, 4.0]
    assert merged["qam+cdc"].rows[1].rate_scored == saved[0].rows[0].rate_scored
    assert merged["qam+dbp1"] == saved[1]


def test_leaderboard_marks_an_optimum_at_the_edge_of_the_sweep():
    def result(method, rates):
        rows = tuple(
            bench.PowerResult(
                power_dbm=p,
                rate=r,
                rate_stderr=0.0,
                rate_scored=r,
                rate_scored_stderr=0.0,
                train_seconds=1.0,
            )
            for p, r in zip((-1.0, 1.0, 3.0), rates, strict=True)
        )
        return bench.MethodResult(scenario="s", method=method, rows=rows)

    inside, edge = result("inside", (5.0, 6.0, 5.5)), result("edge", (5.0, 5.5, 5.8))
    assert not inside.best_at_edge and edge.best_at_edge
    table = bench.leaderboard([inside, edge])
    assert "+3.0 dBm (edge)" in table and "+1.0 dBm |" in table
    with pytest.raises(ValueError, match="same scenario"):
        inside.merged(edge)
