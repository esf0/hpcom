import numpy as np
import scipy as sp

# from . import modulation
# from hpcom.modulation import get_scale_coef, get_constellation, get_nearest_constellation_points_new, \
#     get_bits_from_constellation_points
from .modulation import get_scale_coef, get_constellation, get_nearest_constellation_points_new, \
    get_bits_from_constellation_points

def get_energy(signal, dt):
    """
    Computes the energy of a signal.

    Args:
        signal: An array containing the signal.
        dt: The time step.

    Returns:
        The energy of the signal.
    """
    return np.sum(np.power(np.absolute(signal), 2)) * dt


def get_average_power(signal, dt):
    """
    Calculates the average power of a signal.

    Args:
        signal: A 1-D numpy array representing the signal.
        dt: A float representing the time interval between samples.

    Returns:
        A float representing the average power of the signal.
    """
    return get_energy(signal, dt) / (len(signal) * dt)


def get_sq_of_average_power(points):
    """
    Calculates the square root of the average power of a complex signal.

    Args:
    points (array-like): The complex signal for which the square root of the average power is to be calculated.

    Returns:
    float: The square root of the average power of the signal.

    Example:
    >>> points = np.array([1+2j, 3-1j, 2+2j, 1-2j, 2-2j, -3-1j, -1+1j])
    >>> get_sq_of_average_power(points)
    2.2781627366017876
    """
    return np.sqrt(np.mean(np.power(np.absolute(points), 2)))


def get_points_error_rate(points_init, points):
    """
    Calculates the bit error rate between two sets of points.

    Args:
        points_init (numpy.ndarray): The original set of points.
        points (numpy.ndarray): The set of points to compare against the original.

    Returns:
        tuple: A tuple of the error rate and error count. The error rate is a float between 0 and 1, and represents
        the proportion of points that are different between the two sets. The error count is an integer representing
        the total number of points that are different between the two sets.

    Raises:
        ValueError: If the sizes of the two sets of points are different.

    """

    if len(points_init) != len(points):
        print('Error: different bits sequence sizes:', len(points_init), len(points))
        return 1.0

    n = len(points)

    error_count = np.count_nonzero(points - points_init)
    return error_count / n, error_count


def get_bits_error_rate(bits_init, bits):
    """
    Calculates the bit error rate (BER) between two binary sequences.

    Args:
        bits_init: The initial binary sequence.
        bits: The binary sequence to compare with the initial sequence.

    Returns:
        A tuple containing the BER and the number of bit errors.
    """

    if len(bits_init) != len(bits):
        print('Error: different bits sequence sizes:', len(bits_init), len(bits))
        return 1.0

    n = len(bits)

    error_count = 0
    for i in range(n):
        if bits_init[i] != bits[i]:
            error_count += 1

    return error_count / n, error_count


def get_ber_by_points(points_init, points, mod_type):
    # TODO: initial points have a very small shift so we have to take exact
    points_init_new = get_nearest_constellation_points_new(points_init, get_constellation(mod_type))

    bits_init = get_bits_from_constellation_points(points_init_new, mod_type)
    bits = get_bits_from_constellation_points(points, mod_type)
    return get_bits_error_rate(bits_init, bits)


def get_ber_by_points_unscaled(points_init, points, mod_type):

    scale = get_scale_coef(points, mod_type)
    scale_init = get_scale_coef(points_init, mod_type)
    return get_ber_by_points(points_init * scale_init, points * scale, mod_type)


def get_ber_by_points_ultimate(points_init, points, mod_type):
    scale = get_scale_coef(points, mod_type)
    # points_ultimate = get_nearest_constellation_points(points * scale, mod_type)
    points_ultimate = get_nearest_constellation_points_new(points * scale, get_constellation(mod_type))
    return get_ber_by_points_unscaled(points_init, points_ultimate, mod_type)


def get_evm(points_init, points):
    return np.sqrt(np.mean(np.power(np.absolute(points - points_init), 2)))


def get_evm_rms(points_init, points, p_ave):
    return get_evm(points_init, points) / np.sqrt(p_ave)


def get_evm_rms_new(points_init, points):
    return get_evm(points_init, points) / np.sqrt(np.mean(np.power(np.absolute(points_init), 2)))


def get_evm_ultimate(points_init, points, mod_type):

    scale = get_scale_coef(points, mod_type)
    scale_init = get_scale_coef(points_init, mod_type)
    return get_evm(points_init * scale_init, points * scale)


def get_ber_from_evm(points_init, points, m):
    evm_rms = get_evm_rms_new(points_init, points)
    ber = 2 * (1. - np.power(m, -0.5)) / (np.log2(m)) * sp.special.erfc(np.sqrt(1.5 / ((m - 1) * np.power(evm_rms, 2))))
    return ber