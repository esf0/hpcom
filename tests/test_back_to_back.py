import numpy as np
# import tensorflow as tf
# import matplotlib
# import matplotlib.pyplot as plt
#
from datetime import datetime
# from importlib import reload
#
# from prettytable import PrettyTable
# from scipy.fft import fftshift, ifftshift, fft, ifft

import hpcom
from hpcom.signal import create_wdm_parameters, generate_wdm, get_points_wdm, receiver_wdm, nonlinear_shift, rrcosfilter
from hpcom.channel import create_channel_parameters

from ssfm_gpu.propagation import propagate_schrodinger, dispersion_compensation


wdm = create_wdm_parameters(n_channels=1,
                            p_ave_dbm=3,
                            n_symbols=2 ** 16,
                            m_order=16,
                            roll_off=0.1,
                            upsampling=4,
                            downsampling_rate=1,
                            symb_freq=34e9,
                            channel_spacing=75e9,
                            n_polarisations=1,
                            seed='fixed')

channel = create_channel_parameters(n_spans=12,
                                    z_span=80,
                                    alpha_db=0.0,
                                    gamma=1.2,
                                    noise_figure_db=-200,  # -200 means there is no noise
                                    dispersion_parameter=16.8,
                                    dz=1)


signal_x, wdm_info = generate_wdm(wdm)
points_x_orig = wdm_info['points_x'][0]
ft_filter_values = wdm_info['ft_filter_values_x'][0] / wdm['upsampling']  # [0] index for only one WDM channel

points_x = get_points_wdm(hpcom.signal.filter_shaper(signal_x, ft_filter_values)[::wdm['downsampling_rate']], wdm) # downsample
shift_factor_x = np.dot(np.transpose(np.conjugate(points_x_orig)), points_x_orig) / np.dot(np.transpose(np.conjugate(points_x_orig)), points_x)
print(shift_factor_x, np.absolute(shift_factor_x))
print('Difference:', np.max(np.absolute(points_x_orig-points_x)))

# let's try to propagate without nonlinearity

start_time = datetime.now()
signal_x = propagate_schrodinger(channel, signal_x, wdm['sample_freq'])
print("propagation took", (datetime.now() - start_time).total_seconds() * 1000, "ms")