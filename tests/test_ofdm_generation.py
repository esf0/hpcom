import matplotlib.pyplot as plt
import numpy as np


from hpcom.signal import create_ofdm_parameters, generate_ofdm_signal, decode_ofdm_signal, dbm_to_mw
from hpcom.metrics import get_average_power, get_energy

# create parameters
ofdm = create_ofdm_parameters(n_carriers=128, p_ave_dbm=1, n_symbols=2, m_order=16,
                              symb_freq=34e9,
                              cp_len=0, n_guard=0, n_pilot=0,
                              n_polarisations=1, seed='fixed')

ofdm_signal, add = generate_ofdm_signal(ofdm)
np_signal = len(ofdm_signal)
dt = 1. / ofdm['symb_freq'] / ofdm['n_carriers']

e_signal = get_energy(ofdm_signal, dt * np_signal)
p_signal_x = get_average_power(ofdm_signal, dt)
p_signal_correct = dbm_to_mw(ofdm['p_ave_dbm']) / 1000 / ofdm['n_polarisations'] * ofdm['n_carriers']
print("Average signal power (x / x): "
      "%1.7f / %1.7f (has to be close to %1.7f)" % (
      p_signal_x, p_signal_x, p_signal_correct))

# ofdm_symbols = np.split(ofdm_signal, ofdm['n_symbols'])
#
# print(np.shape(ofdm_symbols))
# print(ofdm_symbols[0])

# print(ofdm_signal.shape)

point_orig = np.array(add['points'])
points = np.array(decode_ofdm_signal(ofdm_signal, ofdm))
print(np.max(points - point_orig))


fig, axs = plt.subplots(1, 2, figsize=(15, 15))
axs[0].scatter(point_orig.real, point_orig.imag, s=12, c='r', marker='x')
axs[0].grid(True)

axs[1].scatter(points.real, points.imag, s=12, c='b', marker='x')
axs[1].grid(True)

plt.show()

fig, axs = plt.subplots(1, 1, figsize=(30,10))
axs.plot(np.absolute(ofdm_signal),
               color='xkcd:royal blue', linewidth=5,
               label='signal')
axs.set_xlabel('Time $t$')
axs.set_ylabel(r'$|q(t)|$')
axs.legend()
axs.grid(True)

plt.show()