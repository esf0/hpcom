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


def get_n_bits(type):
    n_bits = {"qpsk": 2, "16qam": 4, "64qam": 6, "256qam": 8, "1024qam": 10}
    return n_bits[type]


def generate_grey_code(n):
    # Base case
    if n <= 0:
        return ['0']
    elif n == 1:
        return ['0', '1']

    # Recursive case
    first_half = generate_grey_code(n - 1)
    second_half = first_half.copy()

    # Reverse the second half
    second_half.reverse()

    # Append 0 to the first half
    first_half = ['0' + code for code in first_half]

    # Append 1 to the second half
    second_half = ['1' + code for code in second_half]

    # Concatenate both halves
    result = first_half + second_half

    return result


def generate_grey_code_2d(n):
    if n % 2 != 0:
        raise ValueError('n must be even.')
    # Generate 1D Grey code sequence
    grey_code_1d = np.array(generate_grey_code(n // 2))

    # Generate 2D Grey code sequence by concatenating 1D sequences
    grey_code_2d = np.char.add(grey_code_1d[:, None], grey_code_1d)

    return grey_code_2d


# TODO: finish odd degrees of 2 (32, 128, 512, etc.)
def generate_grey_code_2d_rect(n):
    n_small = n // 2
    n_big = n - n_small
    # Generate 1D Grey code sequence
    grey_code_1d_big = np.array(generate_grey_code(n_big))
    grey_code_1d_small = np.array(generate_grey_code(n_small))

    # Generate 2D Grey code sequence by concatenating 1D sequences
    grey_code_2d = np.char.add(grey_code_1d_big[:, None], grey_code_1d_small)

    return grey_code_2d


def generate_constellation(n):
    """
    Generates a 2D constellation of size 2^n x 2^n using Grey code.
    Args:
        n: number of bits

    Returns:

    """
    if n % 2 != 0:
        raise ValueError('n must be even.')
    # Generate 2D Grey code sequence
    grey_code_2d = generate_grey_code_2d(n)

    # Number of points along one dimension
    num_points = 2**(n // 2)

    # Generate coordinates
    coords = np.arange(num_points)

    # Scale and shift coordinates to match the constellation
    coords = 2*coords - (num_points - 1)

    # Generate grid
    I, Q = np.meshgrid(coords, -coords)  # Q coordinates are negated to match the conventional QAM constellation

    # Convert to complex
    constellation = I + 1j * Q

    return constellation, grey_code_2d


def generate_constellation_dict(n):

    constellation, grey_code_2d = generate_constellation(n)
    result = {grey_code_2d[i, j]: constellation[i, j] for i in range(len(grey_code_2d)) for j in range(len(grey_code_2d[i]))}
    inv = {v: k for k, v in result.items()}
    return result, inv


def get_constellation_point_old(bit_data, type="qpsk", constellation=None):
    """
    Get constellation point for given bit sequence
    Args:
        bit_data:
        type:
        constellation:

    Returns:

    """

    n = get_n_bits(type)

    if constellation is None:
        constellation, _ = generate_constellation_dict(n)


    # if bit sequence has less number of bit than we need to the constellation type add bits to the beginning
    if len(bit_data) < n:
        # temp to not change initial data
        temp_bit_data = ''.join('0' for add_bit in range(n - len(bit_data))) + bit_data
    elif len(bit_data) > n:
        # if length of the sequence has not integer number of subsequence
        # add '0' bits to the beginning
        if len(bit_data) % n != 0:
            temp_bit_data = ''.join('0' for add_bit in range(n - (len(bit_data) % n))) + bit_data
        else:
            temp_bit_data = bit_data

        # use recursion for subsequences
        points = [get_constellation_point(temp_bit_data[k * n:(k + 1) * n], type=type, constellation=constellation)
                  for k in range(len(temp_bit_data) // n)]
        # print("[get_constellation_point]: more bits than needed")
        return np.array(points)
    else:
        temp_bit_data = bit_data

    point = constellation[temp_bit_data]

    return point


def get_constellation_point(bit_data, type="qpsk", constellation=None):
    """
    Get constellation point for given bit sequence
    Args:
        bit_data: Bit sequence as a string
        type: Type of constellation
        constellation: Precomputed constellation dictionary

    Returns:
        Constellation points
    """

    n = get_n_bits(type)

    if constellation is None:
        constellation, _ = generate_constellation_dict(n)

    # Ensure bit_data is a multiple of n by padding with zeros
    padded_bit_data = bit_data.zfill((len(bit_data) + n - 1) // n * n)

    # Iterate over chunks of n bits and look up the corresponding constellation point
    points = [constellation[padded_bit_data[i:i + n]] for i in range(0, len(padded_bit_data), n)]

    return np.array(points)



def get_constellation(mod_type):
    n_points = 2 ** get_n_bits(mod_type)
    points = np.zeros(n_points, dtype=complex)
    for i in range(n_points):
        data = "{0:b}".format(int(i))
        points[i] = get_constellation_point(data, mod_type)

    return points


def get_bits_from_constellation_points(points, type="qpsk"):
    """
    Get bit sequence for given constellation points
    Args:
        points: array of constellation points
        type: type of constellation

    Returns: bit sequence

    """

    n_bits = get_n_bits(type)
    _, dict_points = generate_constellation_dict(n_bits)

    bit_sequence = [dict_points[point] for point in points]
    return ''.join(bit_sequence)



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
