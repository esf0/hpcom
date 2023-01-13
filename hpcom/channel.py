import numpy as np
import scipy as sp

from datetime import datetime

from .signal import create_wdm_parameters, generate_wdm, generate_wdm_optimise, receiver, receiver_wdm,\
    nonlinear_shift, dbm_to_mw
from .modulation import get_modulation_type_from_order, get_scale_coef_constellation, \
    get_nearest_constellation_points_unscaled
from .metrics import get_ber_by_points, get_ber_by_points_ultimate, get_energy, get_average_power, get_evm_ultimate

from ssfm_gpu.propagation import propagate_manakov, propagate_schrodinger, dispersion_compensation_manakov

# Channel parameters


def get_default_channel_parameters():
    """
    Get default optical channel parameters.

    Returns:
        Dictionary with default optical channel parameters
            - 'n_spans' -- Total number of spans
            - 'z_span' -- Length of each span in [km]
            - 'alpha_db' -- :math:`\\alpha_{dB}`
            - 'alpha' -- :math:`\\alpha`
            - 'gamma' -- :math:`\\gamma`
            - 'noise_figure_db' -- :math:`NF_{dB}`
            - 'noise_figure' -- :math:`NF`
            - 'gain' -- :math:`G`
            - 'dispersion_parameter' -- :math:`D`
            - 'beta2' -- :math:`\\beta_2`
            - 'beta3' -- :math:`\\beta_3`
            - 'h_planck' -- Planck constant
            - 'fc' -- Carrier frequency math:`f_{carrier}`
            - 'dz' -- Fixed spatial step in [km]
            - 'nz' -- Number of steps per each spatial span
            - 'noise_density' -- Noise density math:`h \\cdot f_{carrier} \\cdot (G - 1) \\cdot NF`

    """

    channel = {}
    channel['n_spans'] = 12  # Number of spans
    channel['z_span'] = 80  # Span Length [km]
    channel['alpha_db'] = 0.225  # Attenuation coefficient [dB km^-1]
    channel['alpha'] = channel['alpha_db'] / (10 * np.log10(np.exp(1)))
    channel['gamma'] = 1.2  # Non-linear Coefficient [W^-1 km^-1]. Default = 1.2
    channel['noise_figure_db'] = 4.5  # Noise Figure [dB]. Default = 4.5
    channel['noise_figure'] = 10 ** (channel['noise_figure_db'] / 10)
    channel['gain'] = np.exp(channel['alpha'] * channel['z_span']) # gain for one span
    channel['dispersion_parameter'] = 16.8 #  [ps nm^-1 km^-1]  dispersion parameter
    channel['beta2'] = -(1550e-9 ** 2) * (channel['dispersion_parameter'] * 1e-3) / (2 * np.pi * 3e8)  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    channel['beta3'] = 0
    channel['h_planck'] = 6.62607015e-34  # Planck's constant [J/s]
    channel['fc'] = 299792458 / 1550e-9  # carrier frequency
    channel['dz'] = 1.0  # length of the step for SSFM [km]
    channel['nz'] = int(channel['z_span'] / channel['dz'])  # number of steps per each span
    channel['noise_density'] = channel['h_planck'] * channel['fc'] * (channel['gain'] - 1) * channel['noise_figure']
    channel['seed'] = 'fixed'

    return channel


def create_channel_parameters(n_spans, z_span, alpha_db, gamma, noise_figure_db, dispersion_parameter, dz, seed='fixed'):

    alpha = alpha_db / (10 * np.log10(np.exp(1)))
    noise_figure = 10 ** (noise_figure_db / 10)
    gain = np.exp(alpha * z_span)  # gain for one span
    beta2 = -(1550e-9 ** 2) * (dispersion_parameter * 1e-3) / (2 * np.pi * 3e8)  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    beta3 = 0
    h_planck = 6.6256e-34  # Planck's constant [J/s]
    # nu = 299792458 / 1550e-9  # light frequency carrier [Hz]
    fc = 299792458 / 1550e-9  # carrier frequency
    nz = int(z_span / dz)  # number of steps per each span
    noise_density = h_planck * fc * (gain - 1) * noise_figure

    channel = {}
    channel['n_spans'] = n_spans  # Number of spans
    channel['z_span'] = z_span  # Span Length [km]
    channel['alpha_db'] = alpha_db  # Attenuation coefficient [dB km^-1]
    channel['alpha'] = alpha
    channel['gamma'] = gamma  # Non-linear Coefficient [W^-1 km^-1]. Default = 1.2
    channel['noise_figure_db'] = noise_figure_db  # Noise Figure [dB]. Default = 4.5
    channel['noise_figure'] = noise_figure
    channel['gain'] = gain  # gain for one span
    channel['dispersion_parameter'] = dispersion_parameter  # [ps nm^-1 km^-1]  dispersion parameter
    channel['beta2'] = beta2  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    channel['beta3'] = beta3
    channel['h_planck'] = h_planck  # Planck's constant [J/s]
    channel['fc'] = h_planck  # carrier frequency
    channel['dz'] = dz  # length of the step for SSFM [km]
    channel['nz'] = nz  # number of steps per each span
    channel['noise_density'] = noise_density
    channel['seed'] = seed

    return channel


def full_line_model_default():

    # Specify channel parameters

    n_spans = 12
    z_span = 80
    alpha_db = 0.2
    gamma = 1.2
    noise_figure_db = 4.5
    dispersion_parameter = 16.8
    dz = 1
    channel = create_channel_parameters(n_spans, z_span, alpha_db, gamma, noise_figure_db, dispersion_parameter, dz)

    # or you can use default parameters
    # channel = get_default_channel_parameters()

    # Specify signal parameters

    wdm = create_wdm_parameters()

    return full_line_model(channel, wdm)


def full_line_model(channel, wdm, bits_x=None, bits_y=None, points_x=None, points_y=None):

    sample_freq = int(wdm['symb_freq'] * wdm['upsampling'])
    dt = 1. / sample_freq

    signal_x, signal_y, wdm_info = generate_wdm(wdm)
    points_orig_x = wdm_info['points_x']
    points_orig_y = wdm_info['points_y']
    ft_filter_values = wdm_info['ft_filter_values']
    np_signal = len(signal_x)

    e_signal_x = get_energy(signal_x, dt * np_signal)
    e_signal_y = get_energy(signal_y, dt * np_signal)

    signal_x, signal_y = propagate_manakov(channel, signal_x, signal_y, sample_freq)

    e_signal_x_prop = get_energy(signal_x, dt * np_signal)
    e_signal_y_prop = get_energy(signal_y, dt * np_signal)

    # print("Signal energy before propagation (x / y):", e_signal_x, e_signal_y)
    # print("Signal energy after propagation (x / y):", e_signal_x_prop, e_signal_y_prop)
    # print("Signal energy difference (x / y):",
    #       np.absolute(e_signal_x - e_signal_x_prop),
    #       np.absolute(e_signal_y - e_signal_y_prop))

    samples_x, samples_y = receiver(signal_x, signal_y, ft_filter_values, wdm['downsampling_rate'])
    samples_x, samples_y = dispersion_compensation_manakov(channel, samples_x, samples_y, dt * wdm['downsampling_rate'])

    sample_step = int(wdm['upsampling'] / wdm['downsampling_rate'])
    points_x = samples_x[::sample_step].numpy()
    points_y = samples_y[::sample_step].numpy()

    nl_shift_x = nonlinear_shift(points_x, points_orig_x)
    points_x_shifted = points_x * nl_shift_x

    nl_shift_y = nonlinear_shift(points_y, points_orig_y)
    points_y_shifted = points_y * nl_shift_y

    mod_type = get_modulation_type_from_order(wdm['m_order'])
    scale_constellation = get_scale_coef_constellation(mod_type) / np.sqrt(wdm['p_ave'] / 2)

    points_x_found = get_nearest_constellation_points_unscaled(points_x_shifted, mod_type)
    points_y_found = get_nearest_constellation_points_unscaled(points_y_shifted, mod_type)

    ber_x = get_ber_by_points(points_orig_x * scale_constellation, points_x_found, mod_type)
    ber_y = get_ber_by_points(points_orig_y * scale_constellation, points_y_found, mod_type)

    # print("BER (x / y):", BER_est(wdm['m_order'], points_x_shifted, points_orig_x), BER_est(wdm['m_order'], points_y_shifted, points_orig_y))
    # print("BER (x / y):", ber_x, ber_y)

    result = {
        'points_x': points_x,
        'points_orig_x': points_orig_x,
        'points_x_shifted': points_x_shifted,
        'points_x_found': points_x_found,
        'points_y': points_y,
        'points_orig_y': points_orig_y,
        'points_y_shifted': points_y_shifted,
        'points_y_found': points_y_found
    }

    return result


# @execution_time
def full_line_model_wdm(channel, wdm, bits_x=None, bits_y=None, points_x=None, points_y=None, channels_type='all'):

    dt = 1. / wdm['sample_freq']

    signal_x, signal_y, wdm_info = generate_wdm(wdm)

    points_x_orig = wdm_info['points_x']
    points_y_orig = wdm_info['points_y']

    ft_filter_values = wdm_info['ft_filter_values_x']
    np_signal = len(signal_x)

    # e_signal_x = get_energy(signal_x, dt * np_signal)
    # e_signal_y = get_energy(signal_y, dt * np_signal)
    p_signal_x = get_average_power(signal_x, dt)
    p_signal_y = get_average_power(signal_y, dt)
    p_signal_correct = dbm_to_mw(wdm['p_ave_dbm']) / 1000 / 2 * wdm['n_channels']
    print("Average signal power (x / y): %1.7f / %1.7f (has to be close to %1.7f)" % (p_signal_x, p_signal_y, p_signal_correct))

    start_time = datetime.now()
    signal_x, signal_y = propagate_manakov(channel, signal_x, signal_y, wdm['sample_freq'])
    end_time = datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000
    print("propagation took", execution_time, "ms")

    # e_signal_x_prop = get_energy(signal_x, dt * np_signal)
    # e_signal_y_prop = get_energy(signal_y, dt * np_signal)

    # print("Signal energy before propagation (x / y):", e_signal_x, e_signal_y)
    # print("Signal energy after propagation (x / y):", e_signal_x_prop, e_signal_y_prop)
    # print("Signal energy difference (x / y):",
    #       np.absolute(e_signal_x - e_signal_x_prop),
    #       np.absolute(e_signal_y - e_signal_y_prop))

    signal_x, signal_y = dispersion_compensation_manakov(channel, signal_x, signal_y, 1. / wdm['sample_freq'])

    samples_x = receiver_wdm(signal_x, ft_filter_values, wdm)
    samples_y = receiver_wdm(signal_y, ft_filter_values, wdm)

    # for k in range(wdm['n_channels']):
    #     samples_x[k], samples_y[k] = dispersion_compensation(channel, samples_x[k], samples_y[k], wdm['downsampling_rate'] / wdm['sample_freq'])

    # print(np.shape(samples_x))

    sample_step = int(wdm['upsampling'] / wdm['downsampling_rate'])

    if channels_type == 'all':

        points_x = []
        points_y = []

        points_x_shifted = []
        points_y_shifted = []

        for k in range(wdm['n_channels']):
            samples_x_temp = samples_x[k]
            samples_y_temp = samples_y[k]
            # print(np.shape(samples_x_temp[::sample_step]))
            points_x.append(samples_x_temp[::sample_step].numpy())
            points_y.append(samples_y_temp[::sample_step].numpy())

            nl_shift_x = nonlinear_shift(points_x[k], points_x_orig[k])
            points_x_shifted.append(points_x[k] * nl_shift_x)

            nl_shift_y = nonlinear_shift(points_y[k], points_y_orig[k])
            points_y_shifted.append(points_y[k] * nl_shift_y)

        mod_type = get_modulation_type_from_order(wdm['m_order'])
        scale_constellation = get_scale_coef_constellation(mod_type) / np.sqrt(wdm['p_ave'] / 2)

        points_x_found = []
        points_y_found = []

        ber_x = []
        ber_y = []
        q_x = []
        q_y = []
        for k in range(wdm['n_channels']):
            start_time = datetime.now()
            points_x_found.append(get_nearest_constellation_points_unscaled(points_x_shifted[k], mod_type))
            print("search x took", (datetime.now() - start_time).total_seconds() * 1000, "ms")

            start_time = datetime.now()
            points_y_found.append(get_nearest_constellation_points_unscaled(points_y_shifted[k], mod_type))
            print("search y took", (datetime.now() - start_time).total_seconds() * 1000, "ms")

            start_time = datetime.now()
            ber_x.append(get_ber_by_points(points_x_orig[k] * scale_constellation, points_x_found[k], mod_type))
            print("ber x took", (datetime.now() - start_time).total_seconds() * 1000, "ms")

            start_time = datetime.now()
            ber_y.append(get_ber_by_points(points_y_orig[k] * scale_constellation, points_y_found[k], mod_type))
            print("ber y took", (datetime.now() - start_time).total_seconds() * 1000, "ms")

            q_x.append(np.sqrt(2) * sp.special.erfcinv(2 * ber_x[k][0]))
            q_y.append(np.sqrt(2) * sp.special.erfcinv(2 * ber_y[k][0]))

            print("BER (x / y):", ber_x[k], ber_y[k])
            print(r'Q^2-factor (x / y):', q_x[k], q_y[k])

    elif channels_type == 'middle':

        k = (wdm['n_channels'] - 1) // 2

        samples_x_temp = samples_x[k]
        samples_y_temp = samples_y[k]

        points_x = samples_x_temp[::sample_step].numpy()
        points_y = samples_y_temp[::sample_step].numpy()

        nl_shift_x = nonlinear_shift(points_x, points_x_orig[k])
        points_x_shifted = points_x * nl_shift_x

        nl_shift_y = nonlinear_shift(points_y, points_y_orig[k])
        points_y_shifted = points_y * nl_shift_y

        mod_type = get_modulation_type_from_order(wdm['m_order'])
        scale_constellation = get_scale_coef_constellation(mod_type) / np.sqrt(wdm['p_ave'] / 2)

        points_x_found = get_nearest_constellation_points_unscaled(points_x_shifted, mod_type)
        points_y_found = get_nearest_constellation_points_unscaled(points_y_shifted, mod_type)

        ber_x = get_ber_by_points(points_x_orig[k] * scale_constellation, points_x_found, mod_type)
        ber_y = get_ber_by_points(points_y_orig[k] * scale_constellation, points_y_found, mod_type)
        q_x = np.sqrt(2) * sp.special.erfcinv(2 * ber_x[0])
        q_y = np.sqrt(2) * sp.special.erfcinv(2 * ber_y[0])

    else:
        print('Error[full_line_model_wdm]: no such type of channels_type variable')

    result = {
        'points_x': points_x,
        'points_x_orig': points_x_orig,
        'points_x_shifted': points_x_shifted,
        'points_x_found': points_x_found,
        'points_y': points_y,
        'points_y_orig': points_y_orig,
        'points_y_shifted': points_y_shifted,
        'points_y_found': points_y_found,
        'ber_x': ber_x,
        'ber_y': ber_y,
        'q_x': q_x,
        'q_y': q_y
    }

    return result


def full_line_model_optimise(channel, wdm, points_orig_x, points_orig_y, ft_tx_filter, ft_rx_filter, return_type='ber_x'):

    signal_x, signal_y = generate_wdm_optimise(wdm, points_orig_x, points_orig_y, ft_tx_filter)

    dt = 1. / wdm['sample_freq']
    p_signal_x = get_average_power(signal_x, dt)
    p_signal_y = get_average_power(signal_y, dt)
    p_signal_correct = dbm_to_mw(wdm['p_ave_dbm']) / 1000 / 2 * wdm['n_channels']
    print("Average signal power (x / y): %1.7f / %1.7f (has to be close to %1.7f)" % (p_signal_x, p_signal_y, p_signal_correct))

    signal_x, signal_y = propagate_manakov(channel, signal_x, signal_y, wdm['sample_freq'])

    samples_x, samples_y = receiver(signal_x, signal_y, ft_rx_filter, wdm['downsampling_rate'])
    samples_x, samples_y = dispersion_compensation_manakov(channel, samples_x, samples_y, wdm['downsampling_rate'] / wdm['sample_freq'])

    sample_step = int(wdm['upsampling'] / wdm['downsampling_rate'])
    points_x = samples_x[::sample_step].numpy()
    points_y = samples_y[::sample_step].numpy()

    nl_shift_x = nonlinear_shift(points_x, points_orig_x)
    points_x_shifted = points_x * nl_shift_x

    nl_shift_y = nonlinear_shift(points_y, points_orig_y)
    points_y_shifted = points_y * nl_shift_y

    mod_type = get_modulation_type_from_order(wdm['m_order'])
    scale_constellation = get_scale_coef_constellation(mod_type) / np.sqrt(wdm['p_ave'] / 2)

    points_x_found = get_nearest_constellation_points_unscaled(points_x_shifted, mod_type)
    points_y_found = get_nearest_constellation_points_unscaled(points_y_shifted, mod_type)

    ber_x = get_ber_by_points(points_orig_x * scale_constellation, points_x_found, mod_type)
    ber_y = get_ber_by_points(points_orig_y * scale_constellation, points_y_found, mod_type)
    q_x = np.sqrt(2) * sp.special.erfcinv(2 * ber_x[0])
    q_y = np.sqrt(2) * sp.special.erfcinv(2 * ber_y[0])
    # print("BER (x / y):", BER_est(wdm['m_order'], points_x_shifted, points_orig_x),
    # BER_est(wdm['m_order'], points_y_shifted, points_orig_y))
    print("BER (x / y):", ber_x, ber_y)
    print(r'Q^2-factor (x / y):', q_x, q_y)
    print("EVM (x / y):",
          get_evm_ultimate(points_orig_x, points_x_shifted, mod_type),
          get_evm_ultimate(points_orig_y, points_y_shifted, mod_type))

    if return_type == 'ber_x':
        result = get_ber_by_points_ultimate(points_orig_x, points_x_found, mod_type)[0]
    elif return_type == 'ber_y':
        result = get_ber_by_points_ultimate(points_orig_y, points_y_found, mod_type)[0]
    elif return_type == 'evm_x':
        result = get_evm_ultimate(points_orig_y, points_y_shifted, mod_type)
    elif return_type == 'evm_y':
        result = get_evm_ultimate(points_orig_y, points_y_shifted, mod_type)
    else:
        result = {
            'points_x': points_x,
            'points_orig_x': points_orig_x,
            'points_x_shifted': points_x_shifted,
            'points_x_found': points_x_found,
            'points_y': points_y,
            'points_orig_y': points_orig_y,
            'points_y_shifted': points_y_shifted,
            'points_y_found': points_y_found,
            'ber_x': ber_x,
            'ber_y': ber_y,
            'q_x': q_x,
            'q_y': q_y
        }

    return result