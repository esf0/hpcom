from ssfm_gpu.propagation import create_channel_parameters

from hpcom.signal import create_wdm_parameters, generate_wdm


# create parameters
wdm = create_wdm_parameters(n_channels=1, p_ave_dbm=6, n_symbols=2 ** 16, m_order=16, roll_off=0.1, upsampling=16,
                            downsampling_rate=1, symb_freq=34e9, channel_spacing=75e9, n_polarisations=2)


channel = create_channel_parameters(n_spans=25,
                                    z_span=80,
                                    alpha_db=0.2,
                                    gamma=1.2,
                                    noise_figure_db=4.5,
                                    dispersion_parameter=16.8,
                                    dz=1)