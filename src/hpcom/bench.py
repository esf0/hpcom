r"""Benchmark scenarios for learning and DSP over fiber: the same links, data and scoring for everyone.

End-to-end learning papers rarely share a system, so their gains cannot be
compared. A :class:`Scenario` fixes everything but the method: transmitter
(symbol rate, pulse, constellation size, channels, polarisations), link,
launch-power sweep, scored channels, the evaluation blocks (same data and
noise for every method — a paired comparison) and the scoring simulation,
four times finer than the training one, so that no method can profit from
the training simulation's own errors.

A method is a set of :mod:`hpcom.learn` blocks (encoder, pre-distorter,
front end, decoder); :func:`run` trains it at every power with a
:class:`~hpcom.learn.Trainer` and scores it; :func:`leaderboard` tabulates
results. From the command line (for compute nodes)::

    python -m hpcom.bench --scenario wdm5-dp-10x80 --method gs+cdc --out result.json

Scenarios: ``sc-dp-10x80`` (one 32 GBd dual-polarisation channel, 256
points), ``wdm5-dp-10x80`` (five channels on a 50 GHz grid), and
``jovanovic2023`` (the system of Jovanovic et al., JLT 2023: 64 points,
roll-off 0.01, 10 × 100 km). Rates are the decoder's, in bit/symbol per
polarisation, on the finer simulation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from hpcom import learn
from hpcom._model import HpcomModel, Polymorphic
from hpcom.amplifier import EDFA
from hpcom.fiber import Fiber
from hpcom.link import Link
from hpcom.numerics import Scheme
from hpcom.transmitter import SingleCarrierTransmitter, Transmitter
from hpcom.wdm import WDMConfig, WDMTransmitter

__all__ = [
    "METHODS",
    "SCENARIOS",
    "Method",
    "MethodResult",
    "PowerResult",
    "Scenario",
    "leaderboard",
    "run",
    "scenario",
]


class Scenario(HpcomModel):
    """A fixed system to compare methods on.

    Attributes
    ----------
    name, description, reference
        Identification; ``reference`` cites a paper the setup follows.
    transmitter
        Transmitter at an arbitrary power (the sweep sets it).
    fiber
        One span (its step count is replaced by ``steps_per_span``).
    spans, noise_figure_db
        Link: spans with an EDFA after each that restores the span loss.
    steps_per_span, evaluation_steps_per_span
        Split-step resolution of training and of scoring (logarithmic, k = 1/3).
    powers_dbm
        Launch powers per channel (both polarisations).
    channels
        Scored WDM channels (``None``: centre; single carrier: ignored).
    evaluation_blocks, evaluation_seed
        The scoring blocks, identical for every method at a given power.
    """

    name: str
    description: str
    reference: str = ""
    transmitter: Polymorphic[Transmitter]
    fiber: Fiber
    spans: int = Field(ge=1)
    noise_figure_db: float = 5.0
    steps_per_span: int = Field(default=50, ge=1)
    evaluation_steps_per_span: int = Field(default=200, ge=1)
    powers_dbm: tuple[float, ...]
    channels: tuple[int, ...] | None = None
    evaluation_blocks: int = Field(default=16, ge=1)
    evaluation_seed: int = 1_000_000

    def link(self, evaluation: bool = False) -> Link:
        steps = self.evaluation_steps_per_span if evaluation else self.steps_per_span
        span = self.fiber.with_steps(
            n_steps=steps, distribution="logarithmic", log_factor=1.0 / 3.0
        )
        return Link.from_spans(
            self.spans, span, EDFA(gain_db=span.loss_db, noise_figure_db=self.noise_figure_db)
        )

    def transmitter_at(self, power_dbm: float) -> Transmitter:
        return self.transmitter.replace(launch_power_per_channel_dbm=power_dbm)

    def seed_at(self, power_dbm: float) -> int:
        """Evaluation seed at a power: the same blocks for every method there."""
        return self.evaluation_seed + round(10 * power_dbm)


def _ssmf(length_km: float, **changes: Any) -> Fiber:
    return Fiber.ssmf(length_km, polarization="dual").replace(**changes)


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario(
            name="sc-dp-10x80",
            description="one 32 GBd dual-polarisation channel, 1024 points, 10 × 80 km SSMF, "
            "EDFA NF 5 dB",
            transmitter=SingleCarrierTransmitter(
                modulation_order=1024,
                symbol_rate_hz=32e9,
                n_symbols=2048,
                samples_per_symbol=4,
                polarization="dual",
            ),
            fiber=_ssmf(80.0),
            spans=10,
            powers_dbm=(-1.0, 1.0, 3.0, 5.0, 7.0, 9.0),
        ),
        Scenario(
            name="wdm5-dp-10x80",
            description="5 × 32 GBd dual-polarisation on a 50 GHz grid (centre scored), 256 points, "
            "10 × 80 km SSMF, EDFA NF 5 dB",
            transmitter=WDMTransmitter(
                modulation_order=256,
                symbol_rate_hz=32e9,
                n_symbols=2048,
                samples_per_symbol=8,
                polarization="dual",
                wdm=WDMConfig(n_channels=5, spacing_hz=50e9),
            ),
            fiber=_ssmf(80.0),
            spans=10,
            powers_dbm=(-5.0, -3.0, -1.0, 1.0, 3.0, 5.0),
        ),
        Scenario(
            name="jovanovic2023",
            description="5 × 32 GBd dual-polarisation on a 50 GHz grid, RRC 0.01, 64 points, "
            "10 × 100 km SSMF (D = 16.464 ps/(nm·km), γ = 1.3 /W/km), EDFA NF 5 dB",
            reference="O. Jovanovic, F. Da Ros, D. Zibar, M. P. Yankov, JLT 41(12), 2023",
            transmitter=WDMTransmitter(
                modulation_order=64,
                symbol_rate_hz=32e9,
                n_symbols=2000,
                samples_per_symbol=8,
                roll_off=0.01,
                polarization="dual",
                wdm=WDMConfig(n_channels=5, spacing_hz=50e9),
            ),
            fiber=Fiber.ssmf(
                100.0,
                dispersion_ps_per_nm_per_km=16.464,
                polarization="dual",
                wavelength_m=299_792_458.0 / 193.41e12,
            ).replace(gamma_per_w_per_km=1.3),
            spans=10,
            powers_dbm=(-1.5, 0.5, 2.5),
        ),
    )
}


def scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}") from None


class Method(HpcomModel):
    """A method: the :mod:`hpcom.learn` blocks (``None``: the block's default)."""

    name: str
    encoder: Polymorphic[learn.Encoder] | None = None
    predistorter: Polymorphic[learn.Predistorter] | None = None
    front_end: Polymorphic[learn.FrontEnd] | None = None
    decoder: Polymorphic[learn.Decoder] | None = None

    def blocks(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in ("encoder", "front_end", "decoder") if getattr(self, k)}
        if self.predistorter is not None:
            out["predistorter"] = self.predistorter
        return out


#: The Kerr step of one-step-per-span DBP: effective length, 0.15 of the span from its input
#: (the best position for 32 GBd over 80 km spans: +0.86 bit/symbol at +3 dBm and +1.6 at
#: +7 dBm over a whole-step Kerr length at mid-span in ``sc-dp-10x80``).
DBP1_SCHEME = Scheme(kerr_length="effective", nonlinear_position=0.15)

#: Standard baselines: ``qam+cdc``, ``gs+cdc``, ``qam+dbp1``, ``qam+dbp4``, ``gs+ldbp1``,
#: ``qam+wdbp2``, ``gs+wldbp2``, ``qam+ffdbp`` (ideal full-field DBP with the forward steps).
#: DBP uses effective-length Kerr steps (:class:`hpcom.learn.DBP`; one step per span:
#: :data:`DBP1_SCHEME`); full-field DBP the forward scheme, as the exact inverse.
METHODS: dict[str, Method] = {
    m.name: m
    for m in (
        Method(name="qam+cdc"),
        Method(name="gs+cdc", encoder=learn.Constellation()),
        Method(name="qam+dbp1", front_end=learn.DBP(steps_per_span=1, scheme=DBP1_SCHEME)),
        Method(name="qam+dbp4", front_end=learn.DBP(steps_per_span=4)),
        Method(
            name="gs+ldbp1",
            encoder=learn.Constellation(),
            front_end=learn.LearnedDBP(steps_per_span=1, scheme=DBP1_SCHEME),
        ),
        Method(name="qam+wdbp2", front_end=learn.DBP(steps_per_span=2, wideband=True)),
        Method(
            name="gs+wldbp2",
            encoder=learn.Constellation(),
            front_end=learn.LearnedDBP(steps_per_span=2, wideband=True),
        ),
        Method(
            name="qam+ffdbp",
            front_end=learn.DBP(steps_per_span=None, wideband=True, scheme=None),
        ),
    )
}


class PowerResult(HpcomModel):
    """One launch power: rates (bit/symbol/pol) on the training and on the scoring simulation."""

    power_dbm: float
    rate: float
    rate_stderr: float
    rate_scored: float
    rate_scored_stderr: float
    train_seconds: float


class MethodResult(HpcomModel):
    """A method on a scenario: a row per power and the best scored rate."""

    scenario: str
    method: str
    rows: tuple[PowerResult, ...]

    @property
    def best(self) -> PowerResult:
        return max(self.rows, key=lambda r: r.rate_scored)


def run(
    scenario: Scenario,
    method: Method,
    trainer: learn.Trainer | None = None,
    *,
    powers_dbm: Sequence[float] | None = None,
    seed: int = 0,
) -> MethodResult:
    """Train ``method`` at every power of ``scenario`` and score it (paired across methods)."""
    trainer = learn.Trainer() if trainer is None else trainer
    link, scoring = scenario.link(), scenario.link(evaluation=True)
    rows = []
    for p in scenario.powers_dbm if powers_dbm is None else powers_dbm:
        system = learn.EndToEnd(
            transmitter=scenario.transmitter_at(p),
            link=link,
            channels=scenario.channels,
            **method.blocks(),
        )
        trained = trainer.fit(system, seed=seed)
        blocks, seed_p = scenario.evaluation_blocks, scenario.seed_at(p)
        on_training = learn.evaluate(
            system, trained.params, blocks=blocks, seed=seed_p, precision=trainer.precision
        )
        scored = learn.evaluate(
            system,
            trained.params,
            blocks=blocks,
            seed=seed_p,
            link=scoring,
            precision=trainer.precision,
        )
        rows.append(
            PowerResult(
                power_dbm=p,
                rate=on_training.mean,
                rate_stderr=on_training.stderr,
                rate_scored=scored.mean,
                rate_scored_stderr=scored.stderr,
                train_seconds=trained.seconds,
            )
        )
    return MethodResult(scenario=scenario.name, method=method.name, rows=tuple(rows))


def leaderboard(results: Sequence[MethodResult]) -> str:
    """Markdown table per scenario: best scored rate (and its power), best first."""
    lines = []
    for name in sorted({r.scenario for r in results}):
        rows = sorted((r for r in results if r.scenario == name), key=lambda r: -r.best.rate_scored)
        lines += [
            f"**{name}** — best rate on the scoring simulation (bit/symbol/pol)",
            "",
            "| method | rate | ± | at power | training time per power |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            b = r.best
            t = sum(x.train_seconds for x in r.rows) / len(r.rows)
            lines.append(
                f"| {r.method} | {b.rate_scored:.3f} | {b.rate_scored_stderr:.3f} | "
                f"{b.power_dbm:+.1f} dBm | {t:.0f} s |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m hpcom.bench", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--method", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--powers", type=float, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = []
    if args.out.exists():
        results = [MethodResult.model_validate(r) for r in json.loads(args.out.read_text())]
    for name in args.method:
        result = run(
            scenario(args.scenario),
            METHODS[name],
            learn.Trainer(steps=args.steps),
            powers_dbm=args.powers,
        )
        results = [r for r in results if (r.scenario, r.method) != (result.scenario, result.method)]
        results.append(result)
        best = result.best
        print(
            f"{args.scenario} {name}: best {best.rate_scored:.4f} ± {best.rate_scored_stderr:.4f} "
            f"at {best.power_dbm:+.1f} dBm",
            flush=True,
        )
        args.out.write_text(json.dumps([r.model_dump(mode="json") for r in results], indent=1))
    print(leaderboard(results))


if __name__ == "__main__":
    main()
