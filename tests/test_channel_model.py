import tensorflow as tf

from hpcom.channel import create_channel_parameters, full_line_model_default, full_line_model_wdm, full_line_model
from hpcom.signal import create_wdm_parameters, generate_wdm

from ssfm_gpu.propagation import propagate_manakov, propagate_schrodinger
from ssfm_gpu.conversion import convert_forward, convert_inverse

import numpy as np

# create parameters
wdm = create_wdm_parameters(n_channels=1, p_ave_dbm=6, n_symbols=2 ** 16, m_order=16, roll_off=0.1, upsampling=16,
                            downsampling_rate=1, symb_freq=34e9, channel_spacing=75e9, n_polarisations=2)


channel = create_channel_parameters(n_spans=10,
                                    z_span=80,
                                    alpha_db=0.2,
                                    gamma=1.2,
                                    noise_figure_db=-200,
                                    dispersion_parameter=16.8,
                                    dz=1)



# Manakov - two polarisations
try:
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), tf.config.list_physical_devices('GPU'))
    print('[full_line_model]: verbose=0')
    full_line_model(channel, wdm, verbose=0)
    print('[full_line_model]: verbose=1')
    full_line_model(channel, wdm, verbose=1)
    print('[full_line_model]: verbose=2')
    full_line_model(channel, wdm, verbose=2)
    print('[full_line_model]: finished')

    print('[full_line_model_wdm]: verbose=0')
    full_line_model_wdm(channel, wdm, channels_type='middle', verbose=0)
    print('[full_line_model_wdm]: verbose=1')
    full_line_model_wdm(channel, wdm, channels_type='middle', verbose=1)
    print('[full_line_model_wdm]: verbose=2')
    full_line_model_wdm(channel, wdm, channels_type='middle', verbose=2)
    print('[full_line_model_wdm]: finished')
except Exception as e:
    print(e)
finally:
    print('test_channel_model.py finished')