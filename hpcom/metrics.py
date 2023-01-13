import numpy as np
import scipy as sp

# from . import modulation
from hpcom.modulation import get_scale_coef, get_constellation, get_nearest_constellation_points_new, \
    get_bits_from_constellation_points


def get_energy(signal, dt):
    return np.sum(np.power(np.absolute(signal), 2)) * dt


def get_average_power(signal, dt):
    return get_energy(signal, dt) / (len(signal) * dt)


def get_sq_of_average_power(points):
    return np.sqrt(np.mean(np.power(np.absolute(points), 2)))


def get_points_error_rate(points_init, points):
    if len(points_init) != len(points):
        print('Error: different bits sequence sizes:', len(points_init), len(points))
        return 1.0

    n = len(points)

    error_count = np.count_nonzero(points - points_init)
    return error_count / n, error_count


def get_bits_error_rate(bits_init, bits):
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