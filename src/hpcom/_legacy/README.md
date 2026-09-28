# `hpcom._legacy`

The pre-v1 hpcom modules, preserved unchanged for one release to ease
migration. **They will be removed in v2.**

They are importable under their old names (`import hpcom.channel`,
`from hpcom.signal import generate_wdm`, …) or as `hpcom._legacy.<name>`;
the first import emits a `DeprecationWarning`. `import hpcom` alone never
loads them. Their dependencies are installed with

    pip install "hpcom[legacy]"

(TensorFlow, scikit-learn, matplotlib, ssfm-gpu). See `docs/migration.rst`
for the v1 equivalents; `tests/numerical/test_legacy_parity.py` shows that
the v1 transmitter reproduces the legacy WDM field.
