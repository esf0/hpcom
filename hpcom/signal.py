import random
import json

import numpy as np
import tensorflow as tf

from datetime import datetime

# from hpcom.modulation import get_n_bits, get_constellation_point, get_modulation_type_from_order, \
#     get_scale_coef_constellation
from .modulation import get_n_bits, get_constellation_point, get_modulation_type_from_order, \
    get_scale_coef_constellation


# Signal generation (transceiver side, Tx)

def gen_bit_sequence(n_bits, seed=0):
    random.seed(seed)
    bits = random.getrandbits(n_bits)
    data = "{0:b}".format(int(bits))
    if len(data) < n_bits:
        data = ''.join('0' for add_bit in range(n_bits - len(data))) + data

    return data


def gen_wdm_bit_sequence(num_symbols, mod_type, n_carriers=1, seed=0):
    n_bits = n_carriers * get_n_bits(mod_type) * num_symbols
    return gen_bit_sequence(n_bits, seed)


def rrcosfilter_base(nt, beta, t_symb, sample_rate):

    one_over_ts = 1.0 / t_symb
    dt = 1. / float(sample_rate)
    t = (np.arange(nt) - nt / 2.) * dt
    rrc = np.zeros(nt, dtype=np.float)

    # found ranges for conditions
    zero_pos = np.where(np.isclose(t, 0., atol=1e-16, rtol=1e-15))
    if beta != 0:
        nodes_pos = np.where(np.isclose(abs(t), 0.25 * t_symb / beta, atol=1e-16, rtol=1e-15))
        all_pos = np.where(~(np.isclose(abs(t), 0.25 * t_symb / beta, atol=1e-16, rtol=1e-15) | np.isclose(t, 0., atol=1e-16, rtol=1e-15)))

    else:
        all_pos = np.where(~np.isclose(t, 0., atol=1e-16, rtol=1e-15))

    if beta != 0 and np.shape(nodes_pos)[1] != 0:
        nodes_values = np.ones(len(t[nodes_pos]), dtype=float) * beta * one_over_ts / np.sqrt(2) * \
                       ((1. + 2. / np.pi) * np.sin(0.25 * np.pi / beta) + (1. - 2. / np.pi) * np.cos(0.25 * np.pi / beta))
        rrc[nodes_pos] = nodes_values

    if np.shape(zero_pos)[1] != 0:
        rrc[zero_pos] = one_over_ts * (1. + beta * (4. / np.pi - 1))

    all_values = (np.sin(np.pi * (1. - beta) * t[all_pos] * one_over_ts) +
                  4. * beta * t[all_pos] * one_over_ts * np.cos(np.pi * (1. + beta) * t[all_pos] * one_over_ts)) / \
                 (np.pi * t[all_pos] * (1. - np.power(4. * beta * t[all_pos] * one_over_ts, 2)))
    rrc[all_pos] = all_values

    return rrc


def rrcosfilter(nt, beta, t_symb, sample_rate):

    return rrcosfilter_base(nt, beta, t_symb, sample_rate) * t_symb


def create_wdm_parameters(n_channels, p_ave_dbm, n_symbols, m_order, roll_off, upsampling,
                          downsampling_rate, symb_freq,
                          channel_spacing, n_polarisations=2,
                          np_filter=0, seed='fixed'):

    wdm = {}
    wdm['n_channels'] = n_channels
    wdm['channel_spacing'] = channel_spacing
    wdm['n_polarisations'] = n_polarisations
    wdm['p_ave_dbm'] = p_ave_dbm
    wdm['n_symbols'] = n_symbols
    wdm['m_order'] = m_order
    wdm['modulation_type'] = get_modulation_type_from_order(m_order)
    wdm['n_bits_symbol'] = get_n_bits(wdm['modulation_type'])
    wdm['roll_off'] = roll_off
    wdm['upsampling'] = upsampling
    wdm['downsampling_rate'] = downsampling_rate
    wdm['symb_freq'] = symb_freq
    wdm['sample_freq'] = int(symb_freq * upsampling)
    wdm['np_filter'] = np_filter
    wdm['p_ave'] = (10 ** (wdm['p_ave_dbm'] / 10)) / 1000
    wdm['seed'] = seed
    wdm['scale_coef'] = get_scale_coef_constellation(wdm['modulation_type']) / \
                        np.sqrt(wdm['p_ave'] / wdm['n_polarisations'])

    return wdm


def get_default_wdm_parameters():

    wdm = {}
    wdm['n_channels'] = 1
    wdm['channel_spacing'] = 75e9  # GHz
    wdm['n_polarisations'] = 2
    wdm['p_ave_dbm'] = 0  # dBm
    wdm['n_symbols'] = 2 ** 15
    wdm['m_order'] = 16
    wdm['roll_off'] = 0.1
    wdm['upsampling'] = 8
    wdm['downsampling_rate'] = 1
    wdm['symb_freq'] = 64e9  # GHz
    wdm['sample_freq'] = int(wdm['symb_freq'] * wdm['upsampling'])
    wdm['np_filter'] = 2 ** 12
    wdm['p_ave'] = (10 ** (wdm['p_ave_dbm'] / 10)) / 1000  # mW
    wdm['modulation_type'] = get_modulation_type_from_order(wdm['m_order'])
    wdm['n_bits_symbol'] = get_n_bits(wdm['modulation_type'])
    wdm['seed'] = 'fixed'
    wdm['scale_coef'] = get_scale_coef_constellation(wdm['modulation_type']) / \
                        np.sqrt(wdm['p_ave'] / wdm['n_polarisations'])

    return wdm


def update_wdm_parameters(wdm):

    wdm['sample_freq'] = int(wdm['symb_freq'] * wdm['upsampling'])
    wdm['p_ave'] = (10 ** (wdm['p_ave_dbm'] / 10)) / 1000  # mW
    wdm['modulation_type'] = get_modulation_type_from_order(wdm['m_order'])
    wdm['n_bits_symbol'] = get_n_bits(wdm['modulation_type'])
    wdm['scale_coef'] = get_scale_coef_constellation(wdm['modulation_type']) / \
                        np.sqrt(wdm['p_ave'] / wdm['n_polarisations'])

    return wdm


def update_wdm_parameters_from_json(json_file):
    # Load the JSON file as a dictionary
    with open(json_file, 'r') as f:
        update_params = json.load(f)

    # Get the default channel parameters
    wdm = get_default_wdm_parameters()

    # Update the default parameters with the ones from the JSON
    wdm.update(update_params)

    return update_wdm_parameters(wdm)


def check_wdm_parameters(wdm):

    if not (wdm['n_polarisations'] == 1 or wdm['n_polarisations'] == 2):
        print('[check_wdm_parameters] Error: wrong number of polarisations')
        return -1

    if wdm['n_channels'] % 2 == 0:
        print('[check_wdm_parameters] Error: number of channels has to be odd')
        return -2

    if wdm['p_ave'] != (10 ** (wdm['p_ave_dbm'] / 10)) / 1000:
        print('[check_wdm_parameters] Error: wrong power conversion')
        return -3

    return 0


def generate_wdm_base(wdm, bits=None, points=None, seed=0, ft_filter_values=None):

    sample_freq = int(wdm['symb_freq'] * wdm['upsampling'])  # sampling frequency
    t_s = 1 / wdm['symb_freq']  # symbol spacing

    if wdm['seed'] == 'time':
        seed = datetime.now().timestamp()
    else:
        seed = seed

    if bits is None:
        # bits = np.random.randint(0, 2, n_bits, int)  # random bit stream
        bits = gen_wdm_bit_sequence(wdm['n_symbols'], wdm['modulation_type'],
                                    n_carriers=1, seed=seed)  # n_carriers is 1 cause it one carrier per channel
    else:
        if len(bits) != wdm['n_bits_symbol'] * wdm['n_symbols']:
            print('[generate_wdm_base] Error: length of input bits does not correspond to the parameters')

    # if points provided than we do not set the power (it will be defined by points)
    if points is None:
        points = get_constellation_point(bits, type=wdm['modulation_type'])
        mod_type = get_modulation_type_from_order(wdm['m_order'])
        scale_constellation = np.sqrt(wdm['p_ave']) / get_scale_coef_constellation(mod_type)
        points = points * scale_constellation  # normalise power and scale to power

    points_sequence = np.zeros(wdm['upsampling'] * wdm['n_symbols'], dtype='complex')
    points_sequence[::wdm['upsampling']] = points  # every 'upsampling' samples, the value of points is inserted into the sequence
    points_sequence = tf.cast(points_sequence, tf.complex128)

    np_sequence = len(points_sequence)

    if ft_filter_values is None:
        filter_values = rrcosfilter(np_sequence, wdm['roll_off'], t_s, sample_freq)
        filter_values = tf.cast(filter_values, tf.complex128)
        # print('filter_values_mean', np.mean(filter_values))
        ft_filter_values = tf.signal.fftshift(tf.signal.fft(filter_values))
        ft_filter_values = tf.cast(ft_filter_values, tf.complex128)

    signal = filter_shaper(points_sequence, ft_filter_values)

    additional = {
        'ft_filter_values': ft_filter_values,
        'bits': bits,
        'points': points
    }

    return tf.cast(signal, tf.complex128), additional


def generate_wdm(wdm, bits=None, points=None, ft_filter_values=None):

    # n_symbols - Number of Symbols transmitted
    # m_order - Modulation Level
    # roll_off
    # upsampling
    # downsampling_rate

    # Check input parameters
    if check_wdm_parameters(wdm) != 0:
        print('[generate_wdm] Error: wrong wdm parameters')
        return -1

    # start_time = datetime.now()

    symb_freq = int(wdm['symb_freq'])  # symbol frequency
    sample_freq = int(symb_freq * wdm['upsampling'])  # sampling frequency used for the discrete simulation of analog signals
    dt = 1. / sample_freq
    dw = wdm['channel_spacing']

    bits_x = []
    bits_y = []
    points_x = []
    points_y = []
    ft_filter_values_x = []
    ft_filter_values_y = []

    if points is not None:
        points_x = points[0]
        points_y = points[1]
    elif bits is not None:
        bits_x = bits[0]
        bits_y = bits[1]

    if wdm['n_polarisations'] == 2:
        wdm_process = wdm.copy()
        wdm_process['p_ave'] = wdm_process['p_ave'] / 2
    else:
        wdm_process = wdm

    for wdm_index in range(wdm['n_channels']):

        if ft_filter_values is None:
            ft_filter_values_to_x = None
            ft_filter_values_to_y = None
        else:
            ft_filter_values_to_x = ft_filter_values[0][wdm_index]
            ft_filter_values_to_y = ft_filter_values[1][wdm_index]

        w_channel = 2. * np.pi * dw * (wdm_index - (wdm['n_channels'] - 1) // 2)

        if wdm['n_polarisations'] == 1:
            if points is not None:
                signal_temp, additional = generate_wdm_base(wdm_process, points=points_x[wdm_index], seed=wdm_index,
                                                            ft_filter_values=ft_filter_values_to_x)
            elif bits is not None:
                signal_temp, additional = generate_wdm_base(wdm_process, bits=bits_x[wdm_index], seed=wdm_index,
                                                            ft_filter_values=ft_filter_values_to_x)
            else:
                signal_temp, additional = generate_wdm_base(wdm_process, seed=wdm_index,
                                                            ft_filter_values=ft_filter_values_to_x)

            if wdm_index == 0:
                signal = signal_temp
                np_signal = len(signal)
                t = np.array([dt * (k - np_signal / 2) for k in range(np_signal)])
                signal *= np.exp(1.0j * w_channel * t)
            else:
                signal += signal_temp * np.exp(1.0j * w_channel * t)

            if bits is None:
                bits_x.append(additional['bits'])
            if points is None:
                points_x.append(additional['points'])
            ft_filter_values_x.append(additional['ft_filter_values'])

        elif wdm['n_polarisations'] == 2:
            if points is not None:
                signal_x_temp, additional_x = generate_wdm_base(wdm_process, points=points_x[wdm_index], seed=wdm_index,
                                                                ft_filter_values=ft_filter_values_to_x)
                signal_y_temp, additional_y = generate_wdm_base(wdm_process, points=points_y[wdm_index],
                                                                seed=wdm_index + wdm['n_channels'],
                                                                ft_filter_values=ft_filter_values_to_y)
            elif bits is not None:
                signal_x_temp, additional_x = generate_wdm_base(wdm_process, bits=bits_x[wdm_index], seed=wdm_index,
                                                                ft_filter_values=ft_filter_values_to_x)
                signal_y_temp, additional_y = generate_wdm_base(wdm_process, bits=bits_y[wdm_index],
                                                                seed=wdm_index + wdm['n_channels'],
                                                                ft_filter_values=ft_filter_values_to_y)
            else:
                signal_x_temp, additional_x = generate_wdm_base(wdm_process, seed=wdm_index,
                                                                ft_filter_values=ft_filter_values_to_x)
                signal_y_temp, additional_y = generate_wdm_base(wdm_process, seed=wdm_index + wdm['n_channels'],
                                                                ft_filter_values=ft_filter_values_to_y)

            if wdm_index == 0:
                signal_x = signal_x_temp
                signal_y = signal_y_temp
                np_signal = len(signal_x)
                t = np.array([dt * (k - np_signal / 2) for k in range(np_signal)])

                signal_x *= np.exp(1.0j * w_channel * t)
                signal_y *= np.exp(1.0j * w_channel * t)
            else:
                signal_x += signal_x_temp * np.exp(1.0j * w_channel * t)
                signal_y += signal_y_temp * np.exp(1.0j * w_channel * t)

            if bits is None:
                bits_x.append(additional_x['bits'])
                bits_y.append(additional_y['bits'])
            if points is None:
                points_x.append(additional_x['points'])
                points_y.append(additional_y['points'])

            # To check filter we return it
            ft_filter_values_x.append(additional_x['ft_filter_values'])
            ft_filter_values_y.append(additional_y['ft_filter_values'])

    # end_time = datetime.now()
    # time_diff = (end_time - start_time)
    # execution_time = time_diff.total_seconds() * 1000
    # print("Signal generation took", execution_time, "ms")

    additional_all = {
        'ft_filter_values_x': ft_filter_values_x,
        'ft_filter_values_y': ft_filter_values_y,
        'bits_x': bits_x,
        'bits_y': bits_y,
        'points_x': points_x,
        'points_y': points_y
    }

    if wdm['n_polarisations'] == 1:
        return tf.cast(signal, tf.complex128), additional_all
    else:
        return tf.cast(signal_x, tf.complex128), tf.cast(signal_y, tf.complex128), additional_all


# TODO: delete the usage of it
def generate_wdm_optimise(wdm, points_x, points_y, ft_filter_values):

    # n_symbols - Number of Symbols transmitted
    # m_order - Modulation Level
    # roll_off
    # upsampling
    # downsampling_rate

    # start_time = datetime.now()

    points_sequence_x = np.zeros(wdm['upsampling'] * wdm['n_symbols'], dtype='complex')
    points_sequence_x[::wdm['upsampling']] = points_x  # every ups samples, the value of sQ is inserted into the sequence
    points_sequence_x = tf.cast(points_sequence_x, tf.complex128)

    points_sequence_y = np.zeros(wdm['upsampling'] * wdm['n_symbols'], dtype='complex')
    points_sequence_y[::wdm['upsampling']] = points_y  # every ups samples, the value of sQ is inserted into the sequence
    points_sequence_y = tf.cast(points_sequence_y, tf.complex128)

    if len(points_sequence_x) != len(ft_filter_values):
        print('[generate_wdm_optimise] Error: different shapes of filter and points')
        return -2

    p_ave_filt = np.mean(np.power(np.absolute(ft_filter_values), 2))
    ft_filter_values *= np.sqrt(wdm['upsampling'] / p_ave_filt)

    ft_filter_values = tf.cast(ft_filter_values, tf.complex128)
    signal_x = filter_shaper(points_sequence_x, ft_filter_values)
    signal_y = filter_shaper(points_sequence_y, ft_filter_values)

    # end_time = datetime.now()
    # time_diff = (end_time - start_time)
    # execution_time = time_diff.total_seconds() * 1000
    # print("Signal generation took", execution_time, "ms")

    return tf.cast(signal_x, tf.complex128), tf.cast(signal_y, tf.complex128)


# Demodulation of the signal (receiver side, Rx)

def nonlinear_shift(points, points_orig):

    return np.dot(np.transpose(np.conjugate(points_orig)), points_orig) / np.dot(np.transpose(np.conjugate(points_orig)), points)


def cut_spectrum(spectrum, freq, bandwidth):
    if len(freq) != len(spectrum):
        print("[cut_spectrum] Error: spectrum and frequency arrays have different length")
        return -1

    spectrum_cut = np.copy(spectrum)
    ind = np.where(np.logical_or(freq < -bandwidth / 2, freq > bandwidth / 2))
    spectrum_cut[ind] = 0.0

    return spectrum_cut


def filter_shaper(signal, ft_filter_val):

    spectrum = tf.signal.fftshift(tf.signal.fft(signal))
    # print('with ifftshift')
    return tf.signal.ifftshift(tf.signal.ifft(tf.signal.ifftshift(spectrum * ft_filter_val)))
    # print('no ifftshift')
    # return tf.signal.ifft(tf.signal.ifftshift(spectrum * ft_filter_val))

    # return tf_convolution(signal, filter_val)
    # return np.convolve(signal, filter_val)


def filter_shaper_spectral(spectrum, ft_filter_val):
    # if we alrady have spectrum of the signal we don't need to calculate it again
    # print('with ifftshift')
    return tf.signal.ifftshift(tf.signal.ifft(tf.signal.ifftshift(spectrum * ft_filter_val)))
    # print('no ifftshift')
    # return tf.signal.ifft(tf.signal.ifftshift(spectrum * filter_val))


# def matched_filter_wdm(signal, filter_val, frequences, channel_bandwidth, n_channel):
#
#     spectrum = cut_spectrum(tf.signal.fftshift(tf.signal.fft(signal)),
#                             frequences + n_channel * channel_bandwidth,
#                             channel_bandwidth)
#     return filter_shaper_spectral(spectrum, filter_val) / tf.cast(tf.reduce_sum(tf.math.pow(tf.math.abs(filter_val), 2)), tf.complex128)


def matched_filter_wdm(signal, ft_filter_values, wdm):
    signals_decoded = []

    nt = len(signal)
    dt = 1. / wdm['sample_freq']
    t_span = dt * nt
    t = np.arange(-nt / 2, nt / 2) * dt
    f = np.arange(-nt / 2, nt / 2) * (1. / t_span)

    for k in range(wdm['n_channels']):

        w_channel = -2. * np.pi * wdm['channel_spacing'] * (k - (wdm['n_channels'] - 1) // 2)
        signal_shifted = signal * np.exp(1.0j * w_channel * t)
        spectrum = cut_spectrum(tf.signal.fftshift(tf.signal.fft(signal_shifted)), f, wdm['channel_spacing'])
        signals_decoded.append(matched_filter_spectral(spectrum, ft_filter_values[k]))

    return signals_decoded


def matched_filter(signal, filter_val):
    return filter_shaper(signal, filter_val / tf.cast(tf.reduce_mean(tf.math.pow(tf.math.abs(filter_val), 2)), tf.complex128))


def matched_filter_spectral(spectrum, filter_val):
    return filter_shaper_spectral(spectrum, filter_val / tf.cast(tf.reduce_mean(tf.math.pow(tf.math.abs(filter_val), 2)), tf.complex128))


def receiver_wdm(signal, ft_filter_values, wdm):

    signals_decoded = matched_filter_wdm(signal, ft_filter_values, wdm)
    for k in range(wdm['n_channels']):
        signals_decoded[k] = signals_decoded[k][::wdm['downsampling_rate']]

    return signals_decoded


def receiver(signal_x, signal_y, ft_filter_values, downsampling_rate):

    signal_x = matched_filter(signal_x, ft_filter_values)
    signal_y = matched_filter(signal_y, ft_filter_values)

    signal_x = signal_x[::downsampling_rate]  # downsample
    signal_y = signal_y[::downsampling_rate]

    return signal_x, signal_y


def get_points_wdm(samples, wdm):

    sample_step = int(wdm['upsampling'] / wdm['downsampling_rate'])
    points = samples[::sample_step].numpy()

    return points


# OFDM functions

def create_ofdm_parameters(n_carriers, p_ave_dbm, n_symbols, m_order, symb_freq,
                          cp_len, n_guard, n_pilot,
                          n_polarisations=2, seed='fixed'):
    ofdm = {'n_carriers': n_carriers,
            'cp_len': cp_len,
            'n_guard': n_guard,
            'n_pilot': n_pilot,
            'n_polarisations': n_polarisations,
            'p_ave_dbm': p_ave_dbm,
            'n_symbols': n_symbols,
            'm_order': m_order,
            'symb_freq': symb_freq,
            'modulation_type': get_modulation_type_from_order(m_order),
            'n_bits_symbol': get_n_bits(get_modulation_type_from_order(m_order)),
            'seed': seed
            }

    ofdm['p_ave'] = (10 ** (ofdm['p_ave_dbm'] / 10)) / 1000

    return ofdm


def generate_ofdm_symbol(ofdm, bits=None, points=None, seed='time'):

    sample_freq = int(ofdm['symb_freq'] * ofdm['n_carriers'])  # sampling frequency
    t_s = 1 / ofdm['symb_freq']  # symbol spacing

    if ofdm['seed'] == 'time':
        seed = datetime.now().timestamp()
    else:
        seed = seed

    if bits is None:
        # bits = np.random.randint(0, 2, n_bits, int)  # random bit stream
        bits = gen_wdm_bit_sequence(1, ofdm['modulation_type'],
                                    n_carriers=ofdm['n_carriers'], seed=seed)
    else:
        if len(bits) != ofdm['n_bits_symbol'] * ofdm['n_carriers']:
            print('[generate_ofdm_signal] Error: length of input bits does not correspond to the parameters')

    # if points provided than we do not set the power (it will be defined by points)
    if points is None:
        points = get_constellation_point(bits, type=ofdm['modulation_type'])
        mod_type = get_modulation_type_from_order(ofdm['m_order'])
        scale_constellation = np.sqrt(ofdm['p_ave']) / get_scale_coef_constellation(mod_type)
        points = points * scale_constellation  # normalise power and scale to power

    if len(points) != ofdm['n_carriers']:
        print('[generate_ofdm_signal] Error: length of input points does not correspond to the parameters')

    # generate OFDM symbol
    # ofdm_symbol = np.fft.ifft(points, ofdm['n_carriers'])  # IFFT
    ofdm_symbol = np.fft.ifft(points)  # IFFT
    # add cyclic prefix
    if ofdm['cp_len'] > 0:
        ofdm_symbol = np.concatenate((ofdm_symbol[-ofdm['cp_len']:], ofdm_symbol))


    return ofdm_symbol


def generate_ofdm_signal(ofdm, bits=None, points=None, seed='time'):
    # generate OFDM signal

    if bits is None:
        # bits = np.random.randint(0, 2, n_bits, int)  # random bit stream
        bits = gen_wdm_bit_sequence(ofdm['n_symbols'], ofdm['modulation_type'],
                                    n_carriers=ofdm['n_carriers'], seed=seed)
    else:
        if len(bits) != ofdm['n_bits_symbol'] * ofdm['n_carriers'] * ofdm['n_symbols']:
            print('[generate_ofdm_signal] Error: length of input bits does not correspond to the parameters')

    # if points provided than we do not set the power (it will be defined by points)
    if points is None:
        points = get_constellation_point(bits, type=ofdm['modulation_type'])
        mod_type = get_modulation_type_from_order(ofdm['m_order'])
        scale_constellation = np.sqrt(ofdm['p_ave']) / get_scale_coef_constellation(mod_type)
        points = points * scale_constellation  # normalise power and scale to power

    ofdm_symbols = [generate_ofdm_symbol(ofdm, points=points[i * ofdm['n_carriers']:(i + 1) * ofdm['n_carriers']]) for i in range(ofdm['n_symbols'])]
    ofdm_signal = np.concatenate(ofdm_symbols)

    add = {
        'bits': bits,
        'points': points
    }

    return ofdm_signal, add


def decode_ofdm_symbol(ofdm_symbol, ofdm):

    # remove cyclic prefix
    if ofdm['cp_len'] > 0:
        ofdm_symbol = ofdm_symbol[ofdm['cp_len']:]
    # FFT
    points = np.fft.fft(ofdm_symbol, ofdm['n_carriers'])

    return points


def decode_ofdm_signal(ofdm_signal, ofdm):

    # split the signal into OFDM symbols
    ofdm_symbols = np.split(ofdm_signal, ofdm['n_symbols'])

    # decode OFDM symbols
    points = [decode_ofdm_symbol(ofdm_symbol, ofdm) for ofdm_symbol in ofdm_symbols]
    points = np.concatenate(points)

    return points



# Additional functions

def dispersion_to_beta2(dispersion, wavelenght_nm=1550):
    # dispersion in ps/nm/km, wavelenght_nm in nm
    return -(wavelenght_nm ** 2) * (dispersion * 10 ** 3) / (2. * np.pi * 3.0 * 10 ** 8)


def nd_to_mw(p, t_symb=100, beta2=21.5, gamma=1.27 * 10**(-3)):
    # t_symb in ps, beta2 in ps^2/km, gamma in mW^-1 * km^-1
    return p * beta2 / gamma * (t_symb) ** (-2)


def mw_to_nd(p, t_symb=100, beta2=21.5, gamma=1.27 * 10**(-3)):
    # t_symb in ps, beta2 in ps^2/km, gamma in mW^-1 * km^-1
    return p / (beta2 / gamma * (t_symb) ** (-2))


def mw_to_dbm(p):
    return 10 * np.log10(p)


def dbm_to_mw(p):
    return 10 ** (p / 10)


def nd_to_dbm(p, t_symb=100, beta2=21.5, gamma=1.27 * 10**(-3)):
    return mw_to_dbm(nd_to_mw(p, t_symb, beta2, gamma))


def dbm_to_nd(p, t_symb=100, beta2=21.5, gamma=1.27 * 10**(-3)):
    return mw_to_nd(dbm_to_mw(p), t_symb, beta2, gamma)


def z_nd_to_km(z_nd, t_symb=100, beta2=21.5):
    return z_nd * t_symb ** 2 / beta2


def z_km_to_nd(z_km, t_symb=100, beta2=21.5):
    return z_km / (t_symb ** 2 / beta2)
