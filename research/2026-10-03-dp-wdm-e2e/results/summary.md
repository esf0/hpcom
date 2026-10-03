**training simulation** — bit/symbol/pol (paired gain over QAM + CDC ± s.e.; ×2 for bit/4D)

| method | -2 dBm | +0 dBm | +2 dBm | best |
|---|---|---|---|---|
| QAM, CDC (fixed) | 5.789 | 5.768 | 5.400 | 5.789 |
| learned constellation, CDC | 5.844 (+0.054±0.001) | 5.838 (+0.070±0.003) | 5.526 (+0.126±0.004) | 5.844 |
| learned constellation + Tx CNN pre-distorter, CDC | 5.844 (+0.055±0.002) | 5.839 (+0.071±0.002) | 5.520 (+0.120±0.005) | 5.844 |
| learned constellation, wideband DBP 2 steps/span (fixed) | 5.194 (-0.595±0.005) | 4.478 (-1.290±0.009) | 3.382 (-2.018±0.018) | 5.194 |
| learned constellation, learned wideband DBP 2 steps/span | 5.887 (+0.097±0.002) | 5.936 (+0.168±0.005) | 5.871 (+0.471±0.013) | 5.936 |
| constellation + Tx CNN + learned wideband DBP, jointly | 5.886 (+0.097±0.002) | 5.937 (+0.169±0.005) | 5.872 (+0.472±0.012) | 5.937 |
| QAM, full-field DBP, forward steps (ideal) | 5.891 (+0.102±0.002) | 5.984 (+0.216±0.006) | 5.999 (+0.598±0.015) | 5.999 |

**finer simulation** — bit/symbol/pol (paired gain over QAM + CDC ± s.e.; ×2 for bit/4D)

| method | -2 dBm | +0 dBm | +2 dBm | best |
|---|---|---|---|---|
| QAM, CDC (fixed) | 5.800 | 5.794 | 5.473 | 5.800 |
| learned constellation, CDC | 5.852 (+0.053±0.002) | 5.860 (+0.067±0.003) | 5.595 (+0.122±0.005) | 5.860 |
| learned constellation + Tx CNN pre-distorter, CDC | 5.854 (+0.054±0.002) | 5.860 (+0.067±0.002) | 5.589 (+0.117±0.005) | 5.860 |
| learned constellation, wideband DBP 2 steps/span (fixed) | 5.213 (-0.586±0.005) | 4.510 (-1.283±0.009) | 3.419 (-2.054±0.019) | 5.213 |
| learned constellation, learned wideband DBP 2 steps/span | 5.896 (+0.096±0.002) | 5.952 (+0.158±0.005) | 5.923 (+0.450±0.014) | 5.952 |
| constellation + Tx CNN + learned wideband DBP, jointly | 5.895 (+0.095±0.002) | 5.953 (+0.159±0.005) | 5.924 (+0.452±0.013) | 5.953 |
| QAM, full-field DBP, forward steps (ideal) | 5.883 (+0.083±0.002) | 5.973 (+0.179±0.006) | 5.992 (+0.519±0.015) | 5.992 |
