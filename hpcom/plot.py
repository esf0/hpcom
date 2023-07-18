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


def plot_constellation(const_dict):
    plt.figure(figsize=(16, 16))
    plt.grid(True)

    # Extract keys and values from the first dict (bit to point mapping)
    bits = list(const_dict.keys())
    points = list(const_dict.values())

    # Plot each constellation point
    for i in range(len(bits)):
        plt.plot(points[i].real, points[i].imag, 'bo')
        plt.text(points[i].real, points[i].imag, str(bits[i]), color='r')

    plt.title('Constellation with Corresponding Bits')
    plt.xlabel('Real Component')
    plt.ylabel('Imag Component')
    plt.axis('equal')
    plt.show()