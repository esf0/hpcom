Channel noise (SNR in dB, unit-power constellation; noise = noisy − noiseless, own gains):

| P [dBm] | SNR lin ASE | SNR noise | SNR noiseless distortion | SNR noisy | σ²_noise/σ²_lin | per-point var max/min (noise / lin ASE) | outer ring tan/rad | max abs excess kurtosis | lag-1 corr |
|---|---|---|---|---|---|---|---|---|---|
| -1.0 | 15.04 | 15.01 | 24.04 | 14.50 | 1.005 | 1.008 / 1.008 | 1.004 | 0.007 | 0.0003 |
| +0.0 | 16.02 | 15.99 | 22.02 | 15.02 | 1.008 | 1.006 / 1.006 | 1.002 | 0.007 | 0.0013 |
| +1.0 | 17.00 | 16.95 | 19.99 | 15.20 | 1.013 | 1.009 / 1.007 | 1.010 | 0.004 | 0.0016 |
| +2.0 | 17.98 | 17.89 | 17.95 | 14.91 | 1.020 | 1.014 / 1.008 | 1.014 | 0.009 | 0.0019 |
| +3.0 | 18.93 | 18.80 | 15.90 | 14.10 | 1.032 | 1.013 / 1.005 | 1.022 | 0.011 | 0.0029 |

BER (gain = BER reduction against SMD on the true noisy output):

| P [dBm] | SMD, true noisy | SMD, noiseless + AWGN(σ²_noise) | SMD, noiseless + AWGN(σ²_lin + σ²_ss) | triplet marginals, true (gain) | triplet Viterbi, true (gain) | triplet marginals, noiseless + AWGN | triplet, GMM fitted on noisy | triplet, 1 Gaussian |
|---|---|---|---|---|---|---|---|---|
| -1.0 | 6.628e-03 | 6.565e-03 | 9.076e-03 | 6.530e-03 (1.5%) | 6.528e-03 (1.5%) | 6.412e-03 | 6.678e-03 | 6.535e-03 |
| +0.0 | 4.399e-03 | 4.398e-03 | 7.937e-03 | 4.188e-03 (4.8%) | 4.196e-03 (4.6%) | 4.200e-03 | 4.268e-03 | 4.192e-03 |
| +1.0 | 3.962e-03 | 3.954e-03 | 9.545e-03 | 3.645e-03 (8.0%) | 3.648e-03 (7.9%) | 3.615e-03 | 3.728e-03 | 3.643e-03 |
| +2.0 | 5.351e-03 | 5.253e-03 | 1.548e-02 | 4.620e-03 (13.7%) | 4.625e-03 (13.6%) | 4.559e-03 | 4.715e-03 | 4.652e-03 |
| +3.0 | 9.508e-03 | 9.506e-03 | 2.889e-02 | 8.047e-03 (15.4%) | 8.035e-03 (15.5%) | 8.038e-03 | 8.159e-03 | 8.090e-03 |
