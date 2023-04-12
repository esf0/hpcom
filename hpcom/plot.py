import matplotlib.pyplot as plt
from .modulation import get_bits_dict_for_constellation


def plot_constellation_with_bits(type="qpsk"):

    dict, _ = get_bits_dict_for_constellation(type)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    for key in dict.keys():

        point = dict[key]
        axs.scatter(point.real, point.imag, s=12, c='xkcd:bright green', marker='*')
        axs.text(point.real, point.imag, key)
        axs.grid(True)