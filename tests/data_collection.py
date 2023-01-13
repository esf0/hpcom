# import os
# import numpy as np
# import scipy as sp
# import tensorflow as tf
# import commpy
import pandas as pd

# import matplotlib
# import matplotlib.pyplot as plt

# import signal_generation as sg
# import waveform_optimiser as wf
# from ssfm_gpu.ssfm_gpu import create_channel_parameters
import hpcom.signal as sg
import hpcom.channel as ch

# data_dir = "/work/ec180/ec180/esedov/data_collection/"
data_dir = "/home/esf0/PycharmProjects/hpcom/tests/data/"
job_name = 'test_1'

save_flag = False

df = pd.DataFrame()

n_channels_list = [1]
# n_channels_list = [1, 3, 7]
# p_ave_dbm_list = np.arange(61) * 0.5 - 20
# p_ave_dbm_list = np.arange(31) * 1 - 20
p_ave_dbm_list = [0]
# p_ave_dbm_list = [1, 2, 3, 4, 5, 6, 7, 8]
# p_ave_dbm_list = [0, -1, -2, -3, -4, -5, -6, 7, 8, 9, 10]

# n_span_list = [6]
n_span_list = [10, 15, 20]
n_runs = 1

for n_channels in n_channels_list:
    for p_ave_dbm in p_ave_dbm_list:
        for n_span in n_span_list:
            for run in range(n_runs):
                print(f'run = {run} / n_channels = {n_channels} / p_dbm = {p_ave_dbm} / n_span = {n_span}')
                wdm_full = sg.create_wdm_parameters(n_channels=n_channels, p_ave_dbm=p_ave_dbm, n_symbols=2 ** 16,
                                                    m_order=16, roll_off=0.1, upsampling=16,
                                                    downsampling_rate=1, symb_freq=34e9, channel_spacing=75e9,
                                                    n_polarisations=2, seed='time')

                channel_full = ch.create_channel_parameters(n_spans=n_span,
                                                       z_span=80,
                                                       alpha_db=0.2,
                                                       gamma=1.2,
                                                       noise_figure_db=-200,
                                                       dispersion_parameter=16.8,
                                                       dz=1)

                result_channel = ch.full_line_model_wdm(channel_full, wdm_full, channels_type='middle')

                result_dict = {}

                # result_dict['wdm'] = wdm_full
                # result_dict['channel'] = channel_full

                result_dict['run'] = run
                result_dict['n_channels'] = n_channels
                result_dict['p_ave_dbm'] = p_ave_dbm
                result_dict['z_km'] = n_span * 80

                result_dict['points_x_orig'] = result_channel['points_x_orig']
                result_dict['points_x'] = result_channel['points_x']
                result_dict['points_x_shifted'] = result_channel['points_x_shifted']

                result_dict['points_y_orig'] = result_channel['points_y_orig']
                result_dict['points_y'] = result_channel['points_y']
                result_dict['points_y_shifted'] = result_channel['points_y_shifted']

                result_dict['ber_x'] = result_channel['ber_x']
                result_dict['ber_y'] = result_channel['ber_y']

                result_dict['q_x'] = result_channel['q_x']
                result_dict['q_y'] = result_channel['q_y']

                df = df.append(result_dict, ignore_index=True)

    if save_flag:
        df.to_pickle(data_dir + 'data_collected_' + job_name + '.pkl')

print('calculations done')

