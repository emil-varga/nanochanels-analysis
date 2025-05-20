import numpy as np
import matplotlib.pyplot as plt

rhos_data = np.loadtxt('Superfluid_density.txt')

KT_jump_cgs = 3.52e-9 # g/(cm^2 K)
KT_jump = KT_jump_cgs * (1e-3) / (1e-2)**2 # kg/(m^2 K)

Tlambda = 2.1768

Tmin = rhos_data[0,0]
Tmax = rhos_data[-1,0]
N = len(rhos_data)
Ts = np.linspace(Tmin, Tlambda, N)
rhos = rhos_data[:, 1]

fig2, ax2 = plt.subplots()
ax2.axhline(KT_jump, ls='--', color='k')
ax2.plot(Tlambda - Ts, rhos*300e-9/Ts, label='300 nm')
ax2.plot(Tlambda - Ts, rhos*50e-9/Ts, label='50 nm')
ax2.plot(Tlambda - Ts, rhos*3e-9/Ts, label='3 nm')
ax2.plot(Tlambda - Ts, rhos*0.7e-9/Ts, label='0.7 nm')
ax2.axvline(Tlambda - 2.136, ls=":", color='r')
ax2.legend(loc='best')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('$T_\\lambda - T$ (K)')
ax2.set_ylabel('$D \\rho_s/T$ (kg m$^{-2}$ K$^{-1}$)')
# ax2.set_xlim(xmin=1.8)

plt.show()