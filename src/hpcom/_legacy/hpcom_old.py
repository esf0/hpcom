import numpy as np
import scipy as sp
from ssfm_gpu import ssfm_gpu

import tensorflow as tf

from datetime import datetime
from .decorators import execution_time

# from . import metrics
# from . import modulation
# from . import signal
from .signal import receiver, receiver_wdm, generate_wdm, generate_wdm_optimise
from .signal import dbm_to_mw
from .metrics import get_average_power, get_ber_by_points, get_evm_ultimate, get_ber_by_points_ultimate, get_energy
from .modulation import get_modulation_type_from_order, get_scale_coef_constellation, \
    get_nearest_constellation_points_unscaled


def nonlinear_shift(points, points_orig):

    return np.dot(np.transpose(np.conjugate(points_orig)), points_orig) / np.dot(np.transpose(np.conjugate(points_orig)), points)


def full_line_model(channel, wdm, bits_x=None, bits_y=None, points_x=None, points_y=None):
    # Single channel WDM model for testing

    sample_freq = int(wdm['symb_freq'] * wdm['upsampling'])
    dt = 1. / sample_freq

    signal_x, signal_y, wdm_info = generate_wdm(wdm)
    points_orig_x = wdm_info['points_x']
    points_orig_y = wdm_info['points_y']
    ft_filter_values = wdm_info['ft_filter_values']
    np_signal = len(signal_x)

    e_signal_x = get_energy(signal_x, dt * np_signal)
    e_signal_y = get_energy(signal_y, dt * np_signal)

    signal_x, signal_y = ssfm_gpu.propagate_manakov(channel, signal_x, signal_y, sample_freq)

    e_signal_x_prop = get_energy(signal_x, dt * np_signal)
    e_signal_y_prop = get_energy(signal_y, dt * np_signal)

    print("Signal energy before propagation (x / y):", e_signal_x, e_signal_y)
    print("Signal energy after propagation (x / y):", e_signal_x_prop, e_signal_y_prop)
    print("Signal energy difference (x / y):",
          np.absolute(e_signal_x - e_signal_x_prop),
          np.absolute(e_signal_y - e_signal_y_prop))

    samples_x, samples_y = receiver(signal_x, signal_y, ft_filter_values, wdm['downsampling_rate'])
    samples_x, samples_y = ssfm_gpu.dispersion_compensation_manakov(channel, samples_x, samples_y, dt * wdm['downsampling_rate'])

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
    print("BER (x / y):", ber_x, ber_y)

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


@execution_time
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
    signal_x, signal_y = ssfm_gpu.propagate_manakov(channel, signal_x, signal_y, wdm['sample_freq'])
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

    signal_x, signal_y = ssfm_gpu.dispersion_compensation_manakov(channel, signal_x, signal_y, 1. / wdm['sample_freq'])

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
            end_time = datetime.now()
            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds() * 1000
            print("search took", execution_time, "ms")

            start_time = datetime.now()
            points_y_found.append(get_nearest_constellation_points_unscaled(points_y_shifted[k], mod_type))
            end_time = datetime.now()
            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds() * 1000
            print("search took", execution_time, "ms")

            start_time = datetime.now()
            ber_x.append(get_ber_by_points(points_x_orig[k] * scale_constellation, points_x_found[k], mod_type))
            end_time = datetime.now()
            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds() * 1000
            print("ber took", execution_time, "ms")

            start_time = datetime.now()
            ber_y.append(get_ber_by_points(points_y_orig[k] * scale_constellation, points_y_found[k], mod_type))
            end_time = datetime.now()
            time_diff = (end_time - start_time)
            execution_time = time_diff.total_seconds() * 1000
            print("ber took", execution_time, "ms")

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

    signal_x, signal_y = ssfm_gpu.propagate_manakov(channel, signal_x, signal_y, wdm['sample_freq'])

    samples_x, samples_y = receiver(signal_x, signal_y, ft_rx_filter, wdm['downsampling_rate'])
    samples_x, samples_y = ssfm_gpu.dispersion_compensation_manakov(channel, samples_x, samples_y, wdm['downsampling_rate'] / wdm['sample_freq'])

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


