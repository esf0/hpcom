# Score a new method on a benchmark scenario

**Task.** Define a method from `hpcom.learn` blocks and score it on a fixed
scenario against a standard method. The scenario fixes the transmitter,
the link, the evaluation blocks and the scoring simulation, so the results
are comparable with the leaderboard.

```python
from hpcom import bench, learn

scenario = bench.scenario("sc-dp-10x80")
mine = bench.Method(name="gs+dbp1", encoder=learn.Constellation(),
                    front_end=learn.DBP(steps_per_span=1, scheme=bench.DBP1_SCHEME))
trainer = learn.Trainer(steps=100)                  # the leaderboard uses 2000
results = [bench.run(scenario, method, trainer, powers_dbm=[3.0])
           for method in (bench.METHODS["qam+dbp1"], mine)]
for r in results:
    print(f"{r.method}: {r.best.rate_scored:.3f} bit/symbol/pol at {r.best.power_dbm:+.1f} dBm")
table = bench.leaderboard(results)                  # the same results as a Markdown table
```

Output (JAX on CPU, 65 s):

```text
qam+dbp1: 7.550 bit/symbol/pol at +3.0 dBm
gs+dbp1: 7.738 bit/symbol/pol at +3.0 dBm
```

**Notes**

- Scenarios: `sc-dp-10x80` (1 channel, 1024 points), `wdm5-dp-10x80`
  (5 channels, centre channel scored), `jovanovic2023` (the system of
  Jovanovic et al., JLT 2023).
- Standard methods: `bench.METHODS`. Without `powers_dbm`, `bench.run` uses
  the full power sweep of the scenario.
- From a shell, for example on a GPU node:
  `python -m hpcom.bench --scenario sc-dp-10x80 --method qam+cdc gs+cdc --out results.json`.
  A second run into the same file adds its powers.
- Current results: [benchmarks/leaderboard/results/leaderboard.md](../benchmarks/leaderboard/results/leaderboard.md).
