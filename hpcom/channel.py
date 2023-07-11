import numpy as np
import scipy as sp
import json

from datetime import datetime

from .signal import create_wdm_parameters, generate_wdm, generate_wdm_optimise, receiver, receiver_wdm,\
    nonlinear_shift, dbm_to_mw, get_default_wdm_parameters, get_points_wdm
from .modulation import get_modulation_type_from_order, get_scale_coef_constellation, \
    get_nearest_constellation_points_unscaled
from .metrics import get_ber_by_points, get_ber_by_points_ultimate, get_energy, get_average_power, get_evm_ultimate, \
    get_evm

from ssfm_gpu.propagation import propagate_manakov, propagate_manakov_backward, \
    propagate_schrodinger, dispersion_compensation_manakov

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


def update_channel_parameters_from_json(json_file):
    # Load the JSON file as a dictionary
    with open(json_file, 'r') as f:
        update_params = json.load(f)

    # Get the default channel parameters
    channel = get_default_channel_parameters()

    # Update the default parameters with the ones from the JSON
    channel.update(update_params)

    return channel


def create_channel_parameters(n_spans, z_span, alpha_db, gamma, noise_figure_db, dispersion_parameter, dz, seed='fixed'):
    """
    This function creates a dictionary containing the parameters of a communication channel.

    Args:
        n_spans: The number of spans in the channel.
        z_span: The length of each span in kilometers.
        alpha_db: The attenuation coefficient in dB/km.
        gamma: The non-linear coefficient in W^-1/km.
        noise_figure_db: The noise figure in dB.
        dispersion_parameter: The dispersion parameter in ps/nm/km.
        dz: The length of the step for SSFM in kilometers.
        seed: Optional. The seed for the random number generator. Default is 'fixed'.

    Returns:
        A dict containing the following key-value pairs:
        - n_spans: The number of spans in the channel.
        - z_span: The length of each span in kilometers.
        - alpha_db: The attenuation coefficient in dB/km.
        - alpha: The attenuation coefficient in 1/km.
        - gamma: The non-linear coefficient in W^-1/km.
        - noise_figure_db: The noise figure in dB.
        - noise_figure: The noise figure as a decimal.
        - gain: The gain for one span.
        - dispersion_parameter: The dispersion parameter in ps/nm/km.
        - beta2: The chromatic dispersion coefficient in s^2/km.
        - beta3: The third-order dispersion coefficient in s^3/km.
        - h_planck: Planck's constant in J/s.
        - fc: The carrier frequency in Hz.
        - dz: The length of the step for SSFM in kilometers.
        - nz: The number of steps per each span.
        - noise_density: The noise density in W/Hz/km.
        - seed: The seed for the random number generator.

    Examples:
        >>> channel = create_channel_parameters(12, 80, 0.2, 1.2, 4.5, 16.8, 1)
        >>> print(channel['alpha'])
        0.00010251704760434522
    """

    alpha = alpha_db / (10 * np.log10(np.exp(1)))
    noise_figure = 10 ** (noise_figure_db / 10)
    gain = np.exp(alpha * z_span)  # gain for one span
    beta2 = -(1550e-9 ** 2) * (dispersion_parameter * 1e-3) / (2 * np.pi * 3e8)  # conversion to beta2 [s^2 km^−1], D [s km^-2] or [ps nm^-1 km^-1]
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
    channel['noise_density'] = noise_density
    channel['gain'] = gain  # gain for one span
    channel['dispersion_parameter'] = dispersion_parameter  # [ps nm^-1 km^-1]  dispersion parameter
    channel['beta2'] = beta2  # conversion to beta2 - Chromatic Dispersion Coefficient [s^2 km^−1]
    channel['beta3'] = beta3
    channel['h_planck'] = h_planck  # Planck's constant [J/s]
    channel['fc'] = h_planck  # carrier frequency
    channel['dz'] = dz  # length of the step for SSFM [km]
    channel['nz'] = nz  # number of steps per each span
    channel['seed'] = seed

    return channel


def full_line_model_default():
    """
    This function simulates the transmission of a single-channel WDM signal through a default communication channel.

    Returns:
        A dict containing the following key-value pairs:
        - points_x: The received signal points for the x polarization.
        - points_x_orig: The original signal points for the x polarization.
        - points_x_shifted: The received signal points for the x polarization after nonlinear shift compensation.
        - points_x_found: The nearest constellation points for the received signal points for the x polarization.
        - points_y: The received signal points for the y polarization.
        - points_y_orig: The original signal points for the y polarization.
        - points_y_shifted: The received signal points for the y polarization after nonlinear shift compensation.
        - points_y_found: The nearest constellation points for the received signal points for the y polarization.
        - ber_x: The bit error rate (BER) for the x polarization.
        - ber_y: The bit error rate (BER) for the y polarization.
        - q_x: The Q^2 factor for the x polarization.
        - q_y: The Q^2 factor for the y polarization.

    Examples:
        >>> result = full_line_model_default()
        >>> print(result['ber_x'])
        0.0
    """

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

    wdm = get_default_wdm_parameters()

    return full_line_model(channel, wdm)


def full_line_model(channel, wdm, bits=None, points=None, verbose=0, dbp=None):
    """
    Simulates a full optical transmission line, including generation of a wavelength division multiplexed (WDM)
    signal with one WDM channel, propagation through a specified channel, and detection at the receiver.

    Args:
        channel: object, channel through which the WDM signal is passed
        wdm: dict, contains information about the WDM signal
        bits_x: tuple, number of bits in the x (and y) component of the signal (optional)
        points_x: tuple, points of the x (and y) component of the signal (optional)
        verbose: int: level of system messages. 0 -- nothing, 1 -- +metrics, 2 -- +time, 3 -- +everything (optional)
        dbp: list[int]: list of steps-per-span for DBP (optional)

    Returns:
        dict: containing the points and BER and Q-value of the signal in the x and y component

        - 'points_x' -- an array of the points of the x component of the signal after processing
        - 'points_x_orig' -- an array of the original points of the x component of the signal
        - 'points_x_shifted' -- an array of the points of the x component of the signal after shifting
        - 'points_x_found' -- an array of the nearest constellation points of the x component of the signal
        - 'points_y' -- an array of the points of the y component of the signal after processing
        - 'points_y_orig' -- an array of the original points of the y component of the signal
        - 'points_y_shifted' -- an array of the points of the y component of the signal after shifting
        - 'points_y_found' -- an array of the nearest constellation points of the y component of the signal
        - 'ber_x' -- the bit error rate of the x component of the signal
        - 'ber_y' -- the bit error rate of the y component of the signal
        - 'q_x' -- the Q-value of the x component of the signal
        - 'q_y' -- the Q-value of the y component of the signal

    """

    if wdm['n_channels'] > 1:
        raise Exception('Wrong number of WDM channels for [full_line_model] use [full_line_model_wdm]')

    sample_freq = int(wdm['symb_freq'] * wdm['upsampling'])
    dt = 1. / sample_freq

    signal_x, signal_y, wdm_info = generate_wdm(wdm, bits=bits, points=points)
    # generate_wdm is for multichannel wdm
    # for only one channel we have to take [0] element in list
    # that will correspond to desired values
    points_x_orig = wdm_info['points_x'][0]
    points_y_orig = wdm_info['points_y'][0]
    ft_filter_values = wdm_info['ft_filter_values_x'][0]
    np_signal = len(signal_x)

    e_signal_x = get_energy(signal_x, dt * np_signal)
    e_signal_y = get_energy(signal_y, dt * np_signal)
    p_signal_x = get_average_power(signal_x, dt)
    p_signal_y = get_average_power(signal_y, dt)
    p_signal_correct = dbm_to_mw(wdm['p_ave_dbm']) / 1000 / wdm['n_polarisations'] * wdm['n_channels']
    print("Average signal power (x / y): "
          "%1.7f / %1.7f (has to be close to %1.7f)" % (
          p_signal_x, p_signal_y, p_signal_correct)) if verbose >= 3 else ...

    start_time = datetime.now()
    signal_x, signal_y = propagate_manakov(channel, signal_x, signal_y, wdm['sample_freq'])
    print("propagation took", (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

    e_signal_x_prop = get_energy(signal_x, dt * np_signal)
    e_signal_y_prop = get_energy(signal_y, dt * np_signal)

    if verbose >= 3:
        print("Signal energy before propagation (x / y):", e_signal_x, e_signal_y)
        print("Signal energy after propagation (x / y):", e_signal_x_prop, e_signal_y_prop)
        print("Signal energy difference (x / y):",
              np.absolute(e_signal_x - e_signal_x_prop),
              np.absolute(e_signal_y - e_signal_y_prop))


    samples_x, samples_y = receiver(signal_x, signal_y, ft_filter_values, wdm['downsampling_rate'])
    samples_x, samples_y = dispersion_compensation_manakov(channel, samples_x, samples_y, dt * wdm['downsampling_rate'])

    sample_step = int(wdm['upsampling'] / wdm['downsampling_rate'])
    points_x = samples_x[::sample_step].numpy()
    points_y = samples_y[::sample_step].numpy()

    nl_shift_x = nonlinear_shift(points_x, points_x_orig)
    points_x_shifted = points_x * nl_shift_x

    nl_shift_y = nonlinear_shift(points_y, points_y_orig)
    points_y_shifted = points_y * nl_shift_y

    mod_type = get_modulation_type_from_order(wdm['m_order'])
    scale_constellation = get_scale_coef_constellation(mod_type) / np.sqrt(wdm['p_ave'] / wdm['n_polarisations'])

    start_time = datetime.now()
    points_x_found = get_nearest_constellation_points_unscaled(points_x_shifted, mod_type)
    points_y_found = get_nearest_constellation_points_unscaled(points_y_shifted, mod_type)
    print("search x and y points took",
          (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

    start_time = datetime.now()
    ber_x = get_ber_by_points(points_x_orig * scale_constellation, points_x_found, mod_type)
    ber_y = get_ber_by_points(points_y_orig * scale_constellation, points_y_found, mod_type)
    print("ber for x and y took",
          (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

    q_x = np.sqrt(2) * sp.special.erfcinv(2 * ber_x[0])
    q_y = np.sqrt(2) * sp.special.erfcinv(2 * ber_y[0])

    # print("BER (x / y):", BER_est(wdm['m_order'], points_x_shifted, points_x_orig), BER_est(wdm['m_order'], points_y_shifted, points_y_orig))
    print("BER (x / y):", ber_x, ber_y) if verbose >= 1 else ...
    print("Q^2-factor (x / y):", q_x, q_y) if verbose >= 1 else ...

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

    # if we want to check dbp here is it
    if dbp is not None:
        points_x_dbp = []  # if we have list of dbp steps we will store results as list
        points_y_dbp = []
        points_x_dbp_shifted = []
        points_y_dbp_shifted = []
        points_x_dbp_found = []
        points_y_dbp_found = []
        ber_x_dbp = []
        ber_y_dbp = []
        q_x_dbp = []
        q_y_dbp = []

        for n_steps_per_span in dbp:
            channel_back = channel.copy()
            channel_back['z_span'] = -channel['z_span']
            channel_back['nz'] = n_steps_per_span
            channel_back['dz'] = channel_back['z_span'] / n_steps_per_span

            channel_back['noise_figure_db'] = -200  # Noise Figure [dB]. Default = 4.5
            channel_back['noise_figure'] = 0
            channel_back['noise_density'] = 0.

            # [propagate_manakov_backward] is the proper function for backpropagation
            # which properly handle attenuation of the signal
            # if you still wand to use [propagate_manakov] then you have to set alpha = -alpha for channel parameters
            start_time = datetime.now()
            signal_x_dbp, signal_y_dbp = propagate_manakov_backward(channel_back, signal_x, signal_y,
                                                                    wdm['sample_freq'])
            print(f'DBP {n_steps_per_span} propagation took', (datetime.now() - start_time).total_seconds() * 1000,
                  "ms") if verbose >= 2 else ...

            samples_x_dbp, samples_y_dbp = receiver(signal_x_dbp, signal_y_dbp, ft_filter_values, wdm['downsampling_rate'])
            points_x_dbp.append(samples_x_dbp[::sample_step].numpy())
            points_y_dbp.append(samples_y_dbp[::sample_step].numpy())

            points_x_dbp_shifted.append(points_x_dbp[-1] * nonlinear_shift(points_x_dbp[-1], points_x_orig))
            points_y_dbp_shifted.append(points_y_dbp[-1] * nonlinear_shift(points_y_dbp[-1], points_y_orig))

            start_time = datetime.now()
            points_x_dbp_found.append(get_nearest_constellation_points_unscaled(points_x_dbp_shifted[-1], mod_type))
            points_y_dbp_found.append(get_nearest_constellation_points_unscaled(points_y_dbp_shifted[-1], mod_type))
            print("search x and y points took",
                  (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

            start_time = datetime.now()
            ber_x_dbp.append(get_ber_by_points(points_x_orig * scale_constellation, points_x_dbp_found[-1], mod_type))
            ber_y_dbp.append(get_ber_by_points(points_y_orig * scale_constellation, points_y_dbp_found[-1], mod_type))
            print("ber for x and y took",
                  (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

            q_x_dbp.append(np.sqrt(2) * sp.special.erfcinv(2 * ber_x_dbp[-1][0]))
            q_y_dbp.append(np.sqrt(2) * sp.special.erfcinv(2 * ber_y_dbp[-1][0]))

        result['points_x_dbp'] = points_x_dbp
        result['points_y_dbp'] = points_y_dbp
        result['points_x_dbp_shifted'] = points_x_dbp_shifted
        result['points_y_dbp_shifted'] = points_y_dbp_shifted
        result['points_x_dbp_found'] = points_x_dbp_found
        result['points_y_dbp_found'] = points_y_dbp_found
        result['ber_x_dbp'] = ber_x_dbp
        result['ber_y_dbp'] = ber_y_dbp
        result['q_x_dbp'] = q_x_dbp
        result['q_y_dbp'] = q_y_dbp

    return result


# @execution_time
def full_line_model_wdm(channel, wdm, bits=None, points=None,
                        channels_type='all', verbose=0, dbp=False, optimise='not',
                        ft_filter_values_tx=None, ft_filter_values_rx=None):
    """
        Simulates a full optical transmission line, including generation of a wavelength division multiplexed (WDM)
        signal with multiple WDM channels, propagation through a specified channel, and detection at the receiver.

        Args:
            channel: object, channel through which the WDM signal is passed
            wdm: dict, contains information about the WDM signal
            bits: tuple, number of bits in the x (and y) component of the signal (optional)
            points: tuple, points of the x (and y) component of the signal (optional)
            channels_type: str: 'all' -- calculate metrics for all channels, 'middle' -- calculate only for middle (optional)
            verbose: int: level of system messages. 0 -- nothing, 1 -- +metrics, 2 -- +time, 3 -- +everything (optional)
            dbp: bool: flag to dbp algorithm. Only True if you calculate DBP for already received points (optional)
            optimise: 'not' -- no optimisation, 'ber_x' -- optimise BER for x, 'evm_x' -- optimise EVM for x (optional)
            ft_filter_values_tx:
            ft_filter_values_rx:

        Returns:
            dict: containing the points and BER and Q-value of the signal in the x and y component

            - 'points_x' -- an array of the points of the x component of the signal after processing
            - 'points_orig_x' -- an array of the original points of the x component of the signal
            - 'points_x_shifted' -- an array of the points of the x component of the signal after shifting
            - 'points_x_found' -- an array of the nearest constellation points of the x component of the signal
            - 'points_y' -- an array of the points of the y component of the signal after processing
            - 'points_orig_y' -- an array of the original points of the y component of the signal
            - 'points_y_shifted' -- an array of the points of the y component of the signal after shifting
            - 'points_y_found' -- an array of the nearest constellation points of the y component of the signal
            - 'ber_x' -- the bit error rate of the x component of the signal
            - 'ber_y' -- the bit error rate of the y component of the signal
            - 'q_x' -- the Q-value of the x component of the signal
            - 'q_y' -- the Q-value of the y component of the signal

        """

    dt = 1. / wdm['sample_freq']

    signal_x, signal_y, wdm_info = generate_wdm(wdm, bits=bits, points=points, ft_filter_values=ft_filter_values_tx)

    points_x_orig = wdm_info['points_x']
    points_y_orig = wdm_info['points_y']

    if ft_filter_values_rx is None:
        ft_filter_values_x = wdm_info['ft_filter_values_x']
        ft_filter_values_y = wdm_info['ft_filter_values_y']
    else:
        ft_filter_values_x = ft_filter_values_rx[0]
        ft_filter_values_y = ft_filter_values_rx[1]

    np_signal = len(signal_x)

    e_signal_x = get_energy(signal_x, dt * np_signal)
    e_signal_y = get_energy(signal_y, dt * np_signal)
    p_signal_x = get_average_power(signal_x, dt)
    p_signal_y = get_average_power(signal_y, dt)
    p_signal_correct = dbm_to_mw(wdm['p_ave_dbm']) / 1000 / wdm['n_polarisations'] * wdm['n_channels']
    print("Average signal power (x / y): "
          "%1.7f / %1.7f (has to be close to %1.7f)" % (p_signal_x, p_signal_y, p_signal_correct)) if verbose >= 3 else ...

    # TODO: reconsider dbp logic
    start_time = datetime.now()
    if not dbp:
        signal_x, signal_y = propagate_manakov(channel, signal_x, signal_y, wdm['sample_freq'])
    else:
        print('DBP')
        channel_back = channel.copy()
        channel_back['z_span'] = -channel['z_span']

        # [propagate_manakov_backward] is the proper function for backpropagation
        # which properly handle attenuation of the signal
        # if you still wand to use [propagate_manakov] then you have to set alpha = -alpha for channel parameters
        signal_x, signal_y = propagate_manakov_backward(channel_back, signal_x, signal_y, wdm['sample_freq'])
    print("propagation took", (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

    e_signal_x_prop = get_energy(signal_x, dt * np_signal)
    e_signal_y_prop = get_energy(signal_y, dt * np_signal)

    if verbose >= 3:
        print("Signal energy before propagation (x / y):", e_signal_x, e_signal_y)
        print("Signal energy after propagation (x / y):", e_signal_x_prop, e_signal_y_prop)
        print("Signal energy difference (x / y):",
              np.absolute(e_signal_x - e_signal_x_prop),
              np.absolute(e_signal_y - e_signal_y_prop))

    if not dbp:
        signal_x, signal_y = dispersion_compensation_manakov(channel, signal_x, signal_y, 1. / wdm['sample_freq'])

    samples_x = receiver_wdm(signal_x, ft_filter_values_x, wdm)
    samples_y = receiver_wdm(signal_y, ft_filter_values_y, wdm)

    # TODO: make CDC after receiver
    # for k in range(wdm['n_channels']):
    #     samples_x[k], samples_y[k] = dispersion_compensation(channel, samples_x[k], samples_y[k], wdm['downsampling_rate'] / wdm['sample_freq'])

    # print(np.shape(samples_x))

    sample_step = int(wdm['upsampling'] / wdm['downsampling_rate'])

    # TODO: rewrite this part. Use one for and inside check if channel_type is 'all' or 'middle' and store only one
    if channels_type == 'all':

        points_x = []
        points_y = []

        points_x_shifted = []
        points_y_shifted = []

        for k in range(wdm['n_channels']):

            points_x.append(get_points_wdm(samples_x[k], wdm))
            points_y.append(get_points_wdm(samples_y[k], wdm))

            nl_shift_x = nonlinear_shift(points_x[k], points_x_orig[k])
            points_x_shifted.append(points_x[k] * nl_shift_x)

            nl_shift_y = nonlinear_shift(points_y[k], points_y_orig[k])
            points_y_shifted.append(points_y[k] * nl_shift_y)

        mod_type = get_modulation_type_from_order(wdm['m_order'])
        scale_constellation = get_scale_coef_constellation(mod_type) / np.sqrt(wdm['p_ave'] / wdm['n_polarisations'])

        points_x_found = []
        points_y_found = []

        ber_x = []
        ber_y = []
        q_x = []
        q_y = []
        evm_x = []
        evm_y = []

        for k in range(wdm['n_channels']):
            print('WDM channel', k) if verbose >= 1 else ...

            start_time = datetime.now()
            points_x_found.append(get_nearest_constellation_points_unscaled(points_x_shifted[k], mod_type))
            points_y_found.append(get_nearest_constellation_points_unscaled(points_y_shifted[k], mod_type))
            print("search x and y points took",
                  (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

            start_time = datetime.now()
            ber_x.append(get_ber_by_points(points_x_orig[k] * scale_constellation, points_x_found[k], mod_type))
            ber_y.append(get_ber_by_points(points_y_orig[k] * scale_constellation, points_y_found[k], mod_type))
            print("ber for x and y took",
                  (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

            q_x.append(np.sqrt(2) * sp.special.erfcinv(2 * ber_x[k][0]))
            q_y.append(np.sqrt(2) * sp.special.erfcinv(2 * ber_y[k][0]))

            evm_x.append(get_evm(points_x_orig[k] * scale_constellation, points_x_shifted[k] * scale_constellation))
            evm_y.append(get_evm(points_y_orig[k] * scale_constellation, points_y_shifted[k] * scale_constellation))

            print("BER (x / y):", ber_x[k], ber_y[k]) if verbose >= 1 else ...
            print(r'Q^2-factor (x / y):', q_x[k], q_y[k]) if verbose >= 1 else ...

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
        scale_constellation = get_scale_coef_constellation(mod_type) / np.sqrt(wdm['p_ave'] / wdm['n_polarisations'])

        start_time = datetime.now()
        points_x_found = get_nearest_constellation_points_unscaled(points_x_shifted, mod_type)
        points_y_found = get_nearest_constellation_points_unscaled(points_y_shifted, mod_type)
        print("search x and y points took",
              (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

        start_time = datetime.now()
        ber_x = get_ber_by_points(points_x_orig[k] * scale_constellation, points_x_found, mod_type)
        ber_y = get_ber_by_points(points_y_orig[k] * scale_constellation, points_y_found, mod_type)
        print("ber for x and y took",
              (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

        q_x = np.sqrt(2) * sp.special.erfcinv(2 * ber_x[0])
        q_y = np.sqrt(2) * sp.special.erfcinv(2 * ber_y[0])

        evm_x = get_evm(points_x_orig[k] * scale_constellation, points_x_shifted[k] * scale_constellation)
        evm_y = get_evm(points_y_orig[k] * scale_constellation, points_y_shifted[k] * scale_constellation)

        print("BER (x / y):", ber_x, ber_y) if verbose >= 1 else ...
        print(r'Q^2-factor (x / y):', q_x, q_y) if verbose >= 1 else ...

    else:
        print('Error[full_line_model_wdm]: no such type of channels_type variable')


    if optimise == 'not':

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
            'q_y': q_y,
            'evm_x': evm_x,
            'evm_y': evm_y
        }

    elif optimise == 'ber_x':
        return ber_x
    elif optimise == 'ber_y':
        return ber_y
    elif optimise == 'evm_x':
        return evm_x
    elif optimise == 'evm_y':
        return evm_y
    else:
        print('Error[full_line_model_wdm]: no such type of optimise variable')
        return None

    return result


def dbp_model_wdm(channel, wdm, points, n_steps_per_span, n_samples_per_symbol, channels_type='all', verbose=0):

    channel_dbp = channel.copy()
    channel_dbp['nz'] = n_steps_per_span
    channel_dbp['dz'] = channel_dbp['z_span'] / n_steps_per_span

    wdm_dbp = wdm.copy()
    wdm_dbp['upsampling'] = n_samples_per_symbol
    wdm_dbp['downsampling_rate'] = 1
    wdm_dbp['sample_freq'] = int(wdm['symb_freq'] * wdm_dbp['upsampling'])

    return full_line_model_wdm(channel_dbp, wdm_dbp, bits=None, points=points, channels_type=channels_type, verbose=verbose, dbp=True)


def full_line_model_optimise_legacy(channel, wdm, points_orig_x, points_orig_y, ft_tx_filter, ft_rx_filter, return_type='ber_x'):
    """
    Simulates an optical communication system with optimized parameters.

    Args:
        channel (dict): A dictionary with channel parameters.
        wdm (dict): A dictionary with signal parameters.
        points_orig_x (ndarray): An array with original x points.
        points_orig_y (ndarray): An array with original y points.
        ft_tx_filter (ndarray): A filter to use in the transmitter.
        ft_rx_filter (ndarray): A filter to use in the receiver.
        return_type (str): The type of result to return, default is 'ber_x'.

    Returns:
        The BER (bit error rate) for the specified type, or a dictionary with the following fields:
            - points_x (ndarray): An array with the x points.
            - points_orig_x (ndarray): An array with the original x points.
            - points_x_shifted (ndarray): An array with the shifted x points.
            - points_x_found (ndarray): An array with the found x points.
            - points_y (ndarray): An array with the y points.
            - points_orig_y (ndarray): An array with the original y points.
            - points_y_shifted (ndarray): An array with the shifted y points.
            - points_y_found (ndarray): An array with the found y points.
            - ber_x (float): The BER for the x points.
            - ber_y (float): The BER for the y points.
            - q_x (float): The Q^2-factor for the x points.
            - q_y (float): The Q^2-factor for the y points.

    """

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


def full_line_model_wo_metrics(channel, wdm, bits=None, points=None, verbose=0):
    """
    Simulates a full optical transmission line, including generation of a wavelength division multiplexed (WDM)
    signal with one WDM channel, propagation through a specified channel, and detection at the receiver.

    Args:
        channel: object, channel through which the WDM signal is passed
        wdm: dict, contains information about the WDM signal
        bits_x: tuple, number of bits in the x (and y) component of the signal (optional)
        points_x: tuple, points of the x (and y) component of the signal (optional)
        verbose: int: level of system messages. 0 -- nothing, 1 -- +metrics, 2 -- +time, 3 -- +everything (optional)

    Returns:
        dict: containing the points and BER and Q-value of the signal in the x and y component

        - 'points_x' -- an array of the points of the x component of the signal after processing
        - 'points_x_orig' -- an array of the original points of the x component of the signal
        - 'points_x_shifted' -- an array of the points of the x component of the signal after shifting
        - 'points_x_found' -- an array of the nearest constellation points of the x component of the signal
        - 'points_y' -- an array of the points of the y component of the signal after processing
        - 'points_y_orig' -- an array of the original points of the y component of the signal
        - 'points_y_shifted' -- an array of the points of the y component of the signal after shifting
        - 'points_y_found' -- an array of the nearest constellation points of the y component of the signal
        - 'ber_x' -- the bit error rate of the x component of the signal
        - 'ber_y' -- the bit error rate of the y component of the signal
        - 'q_x' -- the Q-value of the x component of the signal
        - 'q_y' -- the Q-value of the y component of the signal

    """

    if wdm['n_channels'] > 1:
        raise Exception('Wrong number of WDM channels for [full_line_model] use [full_line_model_wdm]')

    sample_freq = int(wdm['symb_freq'] * wdm['upsampling'])
    dt = 1. / sample_freq

    signal_x, signal_y, wdm_info = generate_wdm(wdm, bits=bits, points=points)
    # generate_wdm is for multichannel wdm
    # for only one channel we have to take [0] element in list
    # that will correspond to desired values
    points_x_orig = wdm_info['points_x'][0]
    points_y_orig = wdm_info['points_y'][0]
    ft_filter_values = wdm_info['ft_filter_values_x'][0]
    np_signal = len(signal_x)

    e_signal_x = get_energy(signal_x, dt * np_signal)
    e_signal_y = get_energy(signal_y, dt * np_signal)
    p_signal_x = get_average_power(signal_x, dt)
    p_signal_y = get_average_power(signal_y, dt)
    p_signal_correct = dbm_to_mw(wdm['p_ave_dbm']) / 1000 / wdm['n_polarisations'] * wdm['n_channels']
    print("Average signal power (x / y): "
          "%1.7f / %1.7f (has to be close to %1.7f)" % (
          p_signal_x, p_signal_y, p_signal_correct)) if verbose >= 3 else ...

    start_time = datetime.now()
    signal_x, signal_y = propagate_manakov(channel, signal_x, signal_y, wdm['sample_freq'])
    print("propagation took", (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

    e_signal_x_prop = get_energy(signal_x, dt * np_signal)
    e_signal_y_prop = get_energy(signal_y, dt * np_signal)

    if verbose >= 3:
        print("Signal energy before propagation (x / y):", e_signal_x, e_signal_y)
        print("Signal energy after propagation (x / y):", e_signal_x_prop, e_signal_y_prop)
        print("Signal energy difference (x / y):",
              np.absolute(e_signal_x - e_signal_x_prop),
              np.absolute(e_signal_y - e_signal_y_prop))

    samples_x, samples_y = receiver(signal_x, signal_y, ft_filter_values, wdm['downsampling_rate'])
    samples_x, samples_y = dispersion_compensation_manakov(channel, samples_x, samples_y, dt * wdm['downsampling_rate'])

    sample_step = int(wdm['upsampling'] / wdm['downsampling_rate'])
    points_x = samples_x[::sample_step].numpy()
    points_y = samples_y[::sample_step].numpy()

    nl_shift_x = nonlinear_shift(points_x, points_x_orig)
    points_x_shifted = points_x * nl_shift_x

    nl_shift_y = nonlinear_shift(points_y, points_y_orig)
    points_y_shifted = points_y * nl_shift_y

    result = {
        'points_x': points_x,
        'points_x_orig': points_x_orig,
        'points_x_shifted': points_x_shifted,
        'points_y': points_y,
        'points_y_orig': points_y_orig,
        'points_y_shifted': points_y_shifted
    }

    return result


def full_line_model_back_to_back(channel, wdm, bits_x=None, bits_y=None, points_x=None, points_y=None, verbose=0):
    """
    Simulates a full optical transmission line, including generation of a wavelength division multiplexed (WDM)
    signal with one WDM channel, propagation through a specified channel, backward propagation,
    and detection at the receiver.

    Args:
        channel: object, channel through which the WDM signal is passed
        wdm: dict, contains information about the WDM signal
        bits_x: int, number of bits in the x component of the signal (optional)
        bits_y: int, number of bits in the y component of the signal (optional)
        points_x: array, points of the x component of the signal (optional)
        points_y: array, points of the y component of the signal (optional)
        verbose: int: level of system messages. 0 -- nothing, 1 -- +metrics, 2 -- +time, 3 -- +everything (optional)

    Returns:
        dict: containing the points and BER and Q-value of the signal in the x and y component

        - 'points_x' -- an array of the points of the x component of the signal after processing
        - 'points_x_orig' -- an array of the original points of the x component of the signal
        - 'points_x_shifted' -- an array of the points of the x component of the signal after shifting
        - 'points_x_found' -- an array of the nearest constellation points of the x component of the signal
        - 'points_y' -- an array of the points of the y component of the signal after processing
        - 'points_y_orig' -- an array of the original points of the y component of the signal
        - 'points_y_shifted' -- an array of the points of the y component of the signal after shifting
        - 'points_y_found' -- an array of the nearest constellation points of the y component of the signal
        - 'ber_x' -- the bit error rate of the x component of the signal
        - 'ber_y' -- the bit error rate of the y component of the signal
        - 'q_x' -- the Q-value of the x component of the signal
        - 'q_y' -- the Q-value of the y component of the signal

    """

    if wdm['n_channels'] > 1:
        raise Exception('Wrong number of WDM channels for [full_line_model] use [full_line_model_wdm]')

    sample_freq = int(wdm['symb_freq'] * wdm['upsampling'])
    dt = 1. / sample_freq

    signal_x, signal_y, wdm_info = generate_wdm(wdm)
    # generate_wdm is for multichannel wdm
    # for only one channel we have to take [0] element in list
    # that will correspond to desired values
    points_x_orig = wdm_info['points_x'][0]
    points_y_orig = wdm_info['points_y'][0]
    ft_filter_values = wdm_info['ft_filter_values_x'][0]
    np_signal = len(signal_x)

    e_signal_x = get_energy(signal_x, dt * np_signal)
    e_signal_y = get_energy(signal_y, dt * np_signal)
    p_signal_x = get_average_power(signal_x, dt)
    p_signal_y = get_average_power(signal_y, dt)
    p_signal_correct = dbm_to_mw(wdm['p_ave_dbm']) / 1000 / wdm['n_polarisations'] * wdm['n_channels']
    print("Average signal power (x / y): "
          "%1.7f / %1.7f (has to be close to %1.7f)" % (
          p_signal_x, p_signal_y, p_signal_correct)) if verbose >= 3 else ...

    start_time = datetime.now()
    signal_x, signal_y = propagate_manakov(channel, signal_x, signal_y, wdm['sample_freq'])
    print("forward propagation took", (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

    e_signal_x_prop = get_energy(signal_x, dt * np_signal)
    e_signal_y_prop = get_energy(signal_y, dt * np_signal)

    if verbose >= 3:
        print("Signal energy before propagation (x / y):", e_signal_x, e_signal_y)
        print("Signal energy after propagation (x / y):", e_signal_x_prop, e_signal_y_prop)
        print("Signal energy difference (x / y):",
              np.absolute(e_signal_x - e_signal_x_prop),
              np.absolute(e_signal_y - e_signal_y_prop))

    channel_back = channel.copy()
    channel_back['z_span'] = -channel['z_span']

    # [propagate_manakov_backward] is the proper function for backpropagation
    # which properly handle attenuation of the signal
    # if you still wand to use [propagate_manakov] then you have to set alpha = -alpha for channel parameters
    start_time = datetime.now()
    signal_x, signal_y = propagate_manakov_backward(channel_back, signal_x, signal_y, wdm['sample_freq'])
    print("backward propagation took", (datetime.now() - start_time).total_seconds() * 1000,
          "ms") if verbose >= 2 else ...

    e_signal_x_prop_back = get_energy(signal_x, dt * np_signal)
    e_signal_y_prop_back = get_energy(signal_y, dt * np_signal)

    if verbose >= 3:
        print("Signal energy before propagation (x / y):", e_signal_x, e_signal_y)
        print("Signal energy after propagation (x / y):", e_signal_x_prop, e_signal_y_prop)
        print("Signal energy after backward propagation (x / y):", e_signal_x_prop_back, e_signal_y_prop_back)
        print("Signal energy difference (x / y):",
              np.absolute(e_signal_x - e_signal_x_prop),
              np.absolute(e_signal_y - e_signal_y_prop),
              np.absolute(e_signal_x - e_signal_x_prop_back),
              np.absolute(e_signal_y - e_signal_y_prop_back)
              )

    samples_x, samples_y = receiver(signal_x, signal_y, ft_filter_values, wdm['downsampling_rate'])
    # for back-to-back we don't need dispersion compensation
    # samples_x, samples_y = dispersion_compensation_manakov(channel, samples_x, samples_y, dt * wdm['downsampling_rate'])

    sample_step = int(wdm['upsampling'] / wdm['downsampling_rate'])
    points_x = samples_x[::sample_step].numpy()
    points_y = samples_y[::sample_step].numpy()

    nl_shift_x = nonlinear_shift(points_x, points_x_orig)
    points_x_shifted = points_x * nl_shift_x

    nl_shift_y = nonlinear_shift(points_y, points_y_orig)
    points_y_shifted = points_y * nl_shift_y

    mod_type = get_modulation_type_from_order(wdm['m_order'])
    scale_constellation = get_scale_coef_constellation(mod_type) / np.sqrt(wdm['p_ave'] / wdm['n_polarisations'])

    start_time = datetime.now()
    points_x_found = get_nearest_constellation_points_unscaled(points_x_shifted, mod_type)
    points_y_found = get_nearest_constellation_points_unscaled(points_y_shifted, mod_type)
    print("search x and y points took",
          (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

    start_time = datetime.now()
    ber_x = get_ber_by_points(points_x_orig * scale_constellation, points_x_found, mod_type)
    ber_y = get_ber_by_points(points_y_orig * scale_constellation, points_y_found, mod_type)
    print("ber for x and y took",
          (datetime.now() - start_time).total_seconds() * 1000, "ms") if verbose >= 2 else ...

    q_x = np.sqrt(2) * sp.special.erfcinv(2 * ber_x[0])
    q_y = np.sqrt(2) * sp.special.erfcinv(2 * ber_y[0])

    # print("BER (x / y):", BER_est(wdm['m_order'], points_x_shifted, points_x_orig), BER_est(wdm['m_order'], points_y_shifted, points_y_orig))
    print("BER (x / y):", ber_x, ber_y) if verbose >= 1 else ...
    print("Q^2-factor (x / y):", q_x, q_y) if verbose >= 1 else ...

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