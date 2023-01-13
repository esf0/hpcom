import numpy as np


def get_sq_of_average_power(points):
    return np.sqrt(np.mean(np.power(np.absolute(points), 2)))


def get_modulation_type_from_order(order):

    mod_type = {4: "qpsk",
                16: "16qam",
                64: "64qam",
                256: "256qam",
                1024: "1024qam"}

    return mod_type[order]


def get_constellation_point(bit_data, type="qpsk"):
    # 64QAM -> 6 bits
    # 0b000000 for example
    # 1 bit -> the sigh of the real part: 0 -> '-'; 1 -> '+'
    # 2 and 3 bits -> value to convert to int and multiply by 2 and add 1
    # 00 -> 0 -> 0 * 2 + 1 = 1; 01 -> 1 -> 1 * 2 + 1 = 3;
    # 10 -> 2 -> 2 * 2 + 1 = 5; 11 -> 3 -> 3 * 2 + 1 = 7;
    # 4 bit -> the sigh of the imag part: 0 -> '-'; 1 -> '+'
    # 5 and 6 bits -> same as for real

    n_mod_type = {"qpsk": 1, "16qam": 2, "64qam": 3, "256qam": 4, "1024qam": 5}
    n = n_mod_type[type]

    # if type == "qpsk":
    #     n = 1
    # elif type == "16qam":
    #     n = 2
    # elif type == "64qam":
    #     n = 3
    # elif type == "256qam":
    #     n = 4
    # elif type == "1024qam":
    #     n = 5
    # else:
    #     print("[get_constellation_point]: unknown constellation type")

    # if bit sequence has less number of bit than we need to the constellation type add bits to the beginning
    if len(bit_data) < 2 * n:
        # temp to not change initial data
        temp_bit_data = ''.join('0' for add_bit in range(2 * n - len(bit_data))) + bit_data
    elif len(bit_data) > 2 * n:
        # if length of the sequence has not integer number of subsequence
        # add '0' bits to the beginning
        if len(bit_data) % (2 * n) != 0:
            temp_bit_data = ''.join('0' for add_bit in range(2 * n - (len(bit_data) % (2 * n)))) + bit_data
        else:
            temp_bit_data = bit_data

        # use recursion for subsequences
        points = [get_constellation_point(temp_bit_data[k * 2 * n:(k + 1) * 2 * n], type=type)
                  for k in range(len(temp_bit_data) // (2 * n))]
        # print("[get_constellation_point]: more bits than needed")
        return np.array(points)
    else:
        temp_bit_data = bit_data

    # generate constellation according to the Gray code (only 1 bit changes for neighbours)
    point = 1 + 1j
    if temp_bit_data[0] == '0':
        point = complex(-1, point.imag)
    if temp_bit_data[n] == '0':
        point = complex(point.real, -1)

    if type != "qpsk":
        point = complex(point.real * (int(temp_bit_data[1:n], 2) * 2 + 1),
                        point.imag * (int(temp_bit_data[n + 1:], 2) * 2 + 1))

    return point


def get_bits_from_constellation_point(point, type="qpsk"):
    # 64QAM -> 6 bits
    # 0b000000 for example
    # 1 bit -> the sigh of the real part: 0 -> '-'; 1 -> '+'
    # 2 and 3 bits -> value to convert to int and multiply by 2 and add 1
    # 00 -> 0 -> 0 * 2 + 1 = 1; 01 -> 1 -> 1 * 2 + 1 = 3;
    # 10 -> 2 -> 2 * 2 + 1 = 5; 11 -> 3 -> 3 * 2 + 1 = 7;
    # 4 bit -> the sigh of the imag part: 0 -> '-'; 1 -> '+'
    # 5 and 6 bits -> same as for real

    n_mod_type = {"qpsk": 1, "16qam": 2, "64qam": 3, "256qam": 4, "1024qam": 5}
    n = n_mod_type[type]

    bit_data = ''
    data_real = ''
    data_imag = ''

    if type != "qpsk":
        data_real = "{0:b}".format(int((np.absolute(point.real) - 1) / 2))
        if len(data_real) < n - 1:
            data_real = ''.join('0' for add_bit in range(n - 1 - len(data_real))) + data_real

        data_imag = "{0:b}".format(int((np.absolute(point.imag) - 1) / 2))
        if len(data_imag) < n - 1:
            data_imag = ''.join('0' for add_bit in range(n - 1 - len(data_imag))) + data_imag

    # real part
    if np.sign(point.real) == 1:
        bit_data = bit_data + ''.join('1')
    else:
        bit_data = bit_data + ''.join('0')

    bit_data = bit_data + data_real

    # imag part
    if np.sign(point.imag) == 1:
        bit_data = bit_data + ''.join('1')
    else:
        bit_data = bit_data + ''.join('0')

    bit_data = bit_data + data_imag

    return bit_data


def get_n_bits(type):
    n_bits = {"qpsk": 2, "16qam": 4, "64qam": 6, "256qam": 8, "1024qam": 10}
    return n_bits[type]


def get_constellation(mod_type):
    n_points = 2 ** get_n_bits(mod_type)
    points = np.zeros(n_points, dtype=complex)
    for i in range(n_points):
        data = "{0:b}".format(int(i))
        points[i] = get_constellation_point(data, mod_type)

    return points


def get_bits_dict_for_constellation(type="qpsk"):

    constellation = get_constellation(type)
    dict = {}
    for point in constellation:
        bits = get_bits_from_constellation_point(point, type=type)
        dict[bits] = point

    dict_inv = {v: k for k, v in dict.items()}

    return dict, dict_inv


def get_bits_from_constellation_points(points, type="qpsk"):
    dict_bits, dict_points = get_bits_dict_for_constellation(type=type)
    # print(dict_bits)
    # print(dict_points)

    bit_sequence = ''
    for i in range(len(points)):
        # symbol_bits = get_bits_from_constellation_point(points[i], type)
        # print(points[i])
        symbol_bits = dict_points[points[i]]
        bit_sequence = bit_sequence + symbol_bits

    return bit_sequence


def get_scale_coef_constellation(mod_type):
    # return np.max(np.absolute(get_constellation(mod_type)))
    # return np.sqrt(np.mean(np.power(np.absolute(get_constellation(mod_type)), 2)))
    return get_sq_of_average_power(get_constellation(mod_type))


def get_scale_coef(points, mod_type):
    return get_scale_coef_constellation(mod_type) / get_sq_of_average_power(points)


def get_nearest_constellation_points_new(points, constellation):
    # constellation = get_constellation(mod_type)
    # diff = np.absolute(constellation - point * np.ones(len(constellation)))
    # ind = np.where(diff == np.amin(diff))
    # if len(constellation[ind]) > 1:
    #     return constellation[ind][0]
    # else:
    #     return constellation[ind]

    points = np.array(points)
    result = np.zeros(len(points), dtype=complex)

    n_tot = len(constellation)

    distance = [np.absolute(points - constellation[k]) for k in range(n_tot)]

    for p in range(n_tot):

        dist_check = [(distance[p] < distance[(p + k + 1) % n_tot]) for k in range(n_tot - 1)]

        log_res = np.logical_and(dist_check[0], dist_check[1])
        for k in range(2, n_tot - 1):
            log_res = np.logical_and(log_res, dist_check[k])
        pos = np.where(log_res)  # position for all point which is close to constellation[p]
        result[pos] = constellation[p]

    return result


def get_nearest_constellation_point(point, mod_type):
    constellation = get_constellation(mod_type)
    diff = np.absolute(constellation - point * np.ones(len(constellation)))
    ind = np.where(diff == np.amin(diff))
    if len(constellation[ind]) > 1:
        return constellation[ind][0]
    else:
        return constellation[ind]


def get_nearest_constellation_points(points, mod_type):
    points_found = np.zeros(len(points), dtype=complex)
    for i_p in range(len(points)):
        points_found[i_p] = get_nearest_constellation_point(points[i_p], mod_type)
    return points_found


def get_nearest_constellation_points_unscaled(points, mod_type):
    scale = get_scale_coef(points, mod_type)
    # return get_nearest_constellation_points(points * scale, mod_type)
    return get_nearest_constellation_points_new(points * scale, get_constellation(mod_type))
