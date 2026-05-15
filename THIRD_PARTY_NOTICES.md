# Third-Party Notices

hpcom bundles the following third-party code:

## ssfm-gpu

- Source: https://github.com/esf0/ssfm-gpu
- Version vendored: 0.1.4
- Location in tree: `src/hpcom/backends/tensorflow/_vendor/`
- Licence: MIT (see `src/hpcom/backends/tensorflow/_vendor/LICENSE`)
- Copyright: © 2022 Egor Sedov

The vendored copy is preserved verbatim. The TensorFlow backend
(`hpcom.backends.tensorflow.TensorFlowPropagator`) wraps it behind hpcom's
backend-agnostic `Propagator` interface. The `_vendor` prefix marks the
sub-package as private; do not import from it directly outside of
`hpcom.backends.tensorflow`.
