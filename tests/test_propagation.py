from hpcom.channel import create_channel_parameters
from hpcom.signal import create_wdm_parameters, generate_wdm

from ssfm_gpu.propagation import propagate_manakov, propagate_schrodinger
from ssfm_gpu.conversion import convert_forward, convert_inverse

import numpy as np

# create parameters
wdm = create_wdm_parameters(n_channels=1, p_ave_dbm=6, n_symbols=2 ** 16, m_order=16, roll_off=0.1, upsampling=16,
                            downsampling_rate=1, symb_freq=34e9, channel_spacing=75e9, n_polarisations=2)


channel_dimension = create_channel_parameters(n_spans=10,
                                              z_span=80,
                                              alpha_db=0.2,
                                              gamma=1.2,
                                              noise_figure_db=-200,
                                              dispersion_parameter=16.8,
                                              dz=1)



# Manakov - two polarisations
signal_x, signal_y, wdm_info = generate_wdm(wdm)

dt_dim = 1. / wdm['sample_freq']
t_dim = np.arange(len(signal_x)) * dt_dim

convert_result = convert_inverse((signal_x, signal_y), t_dim, channel_dimension['z_span'],
                                 channel_dimension['beta2'],
                                 channel_dimension['gamma'],
                                 t0=1. / wdm['symb_freq'], type='manakov')

q_x = convert_result['q1']
q_y = convert_result['q2']
t = convert_result['t']
dt = t[0] - t[1]
z_span_dimless = convert_result['z']

channel_dimensionless = channel_dimension.copy()
channel_dimensionless['beta2'] = -2.
channel_dimensionless['gamma'] = 2 * 9./8.
channel_dimensionless['z_span'] = z_span_dimless

signal_x_prop, signal_y_prop = propagate_manakov(channel_dimension, signal_x, signal_y,
                                                 sample_freq=int(wdm['symb_freq'] * wdm['upsampling']))


q_x_prop, q_y_prop = propagate_manakov(channel_dimensionless, q_x, q_y,
                                       sample_freq=1./dt)

convert_inv_result = convert_inverse((q_x_prop, q_y_prop), t, channel_dimensionless['z_span'],
                                     channel_dimension['beta2'],
                                     channel_dimension['gamma'],
                                     t0=1. / wdm['symb_freq'], type='manakov')

signal_from_q_x_prop = convert_result['Q1']
signal_from_q_y_prop = convert_result['Q2']

