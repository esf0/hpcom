# `hpcom._legacy`

These modules are the pre-v1 API, preserved verbatim for one release to ease migration.

**They will be removed in v2.** New code should use the public hpcom API
(`Transmitter`, `Fiber`, `Amplifier`, `Receiver`, `Link`, `Simulation`,
`Result`). See `docs/spec/hpcom-v1.md`.

Importing anything from `hpcom._legacy` (or accessing legacy submodules via
`hpcom.<name>` re-exports) emits a `DeprecationWarning`.

External dependencies of legacy code (e.g. `ssfm-gpu`, `tensorflow`) are not
declared by the new `pyproject.toml`. If you need legacy modules to actually
run, install those deps yourself.
