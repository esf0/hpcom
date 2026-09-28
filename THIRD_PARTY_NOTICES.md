# Third-Party Notices

hpcom does not bundle third-party source code.

The split-step Fourier propagators (`hpcom.backends.numpy.ssfm`,
`hpcom.backends.tensorflow.propagator`) implement the same step sequence as
[ssfm-gpu](https://github.com/esf0/ssfm-gpu) (MIT licence, © 2022 Egor Sedov),
which earlier development versions vendored; their outputs are checked against
golden data produced by ssfm-gpu 0.1.4 (`tests/numerical/_data/tf_parity_v0.npz`).

Optional dependencies are installed separately and keep their own licences:
TensorFlow (Apache-2.0) for the `tf` extras, and TensorFlow, scikit-learn
(BSD-3-Clause), matplotlib (PSF-based) and ssfm-gpu (MIT) for the `legacy`
extra.
