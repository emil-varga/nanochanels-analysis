import numpy as np
import matplotlib.pyplot as plt

import os
from glob import glob

from config import data_path
from svp import P_to_T

import lmfit
from tqdm import tqdm

def process_dir(data_dir, NT=500, Nf=2380, figax=None):
    data_files = glob(os.path.join(data_dir, '*.npy'))

    Ts = np.linspace(1.8, 2.18, NT)
    spec_x = np.zeros((NT, Nf))
    spec_y = np.zeros_like(spec_x)
    counts = np.zeros_like(Ts)

    for file in data_files:
        d = np.load(file, allow_pickle=True).item()
        f = d['freq (Hz)'][:Nf]
        x1 = d['x (V)'][:Nf]
        y1 = d['y (V)'][:Nf]

        P = d['Pressure (Torr)']
        T = P_to_T(P)
        ix = np.argmin(abs(Ts - T))

        spec_x[ix,:] += x1
        spec_y[ix,:] += y1
        counts[ix] += 1


    spec_x /= counts[:, np.newaxis]
    spec_y /= counts[:, np.newaxis]

    spec_r = np.sqrt(spec_x**2 + spec_y**2)

    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    img = ax.imshow(spec_r, origin='lower', extent=[f.min(), f.max(), Ts.min(), Ts.max()],
                    aspect='auto', interpolation='nearest',
                    # vmin=0, vmax=10)
                    norm='log', vmin=1e-1)
   
    return spec_x, spec_y, f, Ts, img


def double_lorenz(f, f1, f2, A1, A2, w1, w2, phi1, phi2, b):
    l1 = A1*np.exp(1j*phi1)*f1*w1/(f**2 - f1**2 - 1j*f*w1)
    l2 = A2*np.exp(1j*phi2)*f2*w2/(f**2 - f2**2 - 1j*f*w2)
    return l1 + l2 + b

def double_lorenz_split(f, f1, f2, A1, A2, w1, w2, phi1, phi2, b):
    l1 = A1*np.exp(1j*phi1)*f1*w1/(f**2 - f1**2 - 1j*f*w1)
    l2 = A2*np.exp(1j*phi2)*f2*w2/(f**2 - f2**2 - 1j*f*w2)
    return l1, l2

def fit_img_slices(f, x, y):
    model = lmfit.Model(double_lorenz, independent_vars=['f'])
    p0 = model.make_params(f1=1066, f2=1092, w1=10, w2=10,
                           phi1=np.pi/2, phi2=-np.pi/2, A1=10, A2=10,
                           b=0)
    f1s = []
    f2s = []
    for k in tqdm(range(x.shape[0])):
        try:
            fit = model.fit(f=f, data=x[k,:] + 1j*y[k,:], params=p0)
            if k == 450:
                fit.plot()
                fig, ax = plt.subplots(2, 1, sharex=True)
                ax[0].plot(f, x[k, :], 'o', ms=3)
                ax[1].plot(f, y[k, :], 'o', ms=3)
                print(fit.best_values)
                l1, l2 = double_lorenz_split(f, **fit.best_values)
                ax[0].plot(f, l1.real, '--', color='g')
                ax[0].plot(f, l2.real, '--', color='r')
                ax[1].plot(f, l1.imag, '--', color='g')
                ax[1].plot(f, l2.imag, '--', color='r')
                ax[0].plot(f, fit.best_fit.real, '-')
                ax[1].plot(f, fit.best_fit.imag, '-')
            # ax.plot(f, fit.best_fit.imag, '--', color='r')
            # ax.plot(f, fit.init_fit.real, '--', color='k')
            # ax.plot(f, fit.init_fit.imag, '--', color='r')
            
            # print(fit.fit_report())
            # break
            p0 = fit.params
            f1s.append(fit.params['f1'].value)
            f2s.append(fit.params['f2'].value)
        except ValueError as e:
            f1s.append(np.nan)
            f2s.append(np.nan)
            print(e)
            print(f"Skipping {k}")
    return np.array(f1s), np.array(f2s)

data_dir_onebasin = os.path.join(data_path, 'spectra_basin2_towardsTc_demod_65.43kHz')
# data_dir_onebasin = os.path.join(data_path, 'spectra_basin2_towardsTc_demod_40kHz')
data_dir_xtalk = os.path.join(data_path, 'spectra_basin2_towardsTc_demod_xtalk_65.43kHz')

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

x, y, f, T, img = process_dir(data_dir_xtalk, figax=(fig, ax[1]))
_ = process_dir(data_dir_onebasin, figax=(fig, ax[0]), NT=200)

f1s, f2s = fit_img_slices(f, x, y)

ax[1].plot(f1s, T, '--', color='tab:orange')
ax[1].plot(f2s, T, '--', color='tab:green')

ff, aa = plt.subplots(2, 1, sharex=True)
aa[0].plot(T, f1s, label='$f_1$')
aa[0].plot(T, f2s, label='$f_2$')
rho_s_c = f2s**2 - f1s**2
rho_s_c /= np.nanmax(rho_s_c[T < 2])
rho_s_b = f1s**2
rho_s_b /= np.nanmax(rho_s_b[T < 2])
aa[1].plot(T, rho_s_b, label=r'bulk $\rho_s$')
aa[1].plot(T, rho_s_c, '--', label=r'confined $\rho_s$')
aa[1].set_xlabel("$T$ (K)")
aa[0].set_ylabel("$f_0$ (Hz)")
aa[1].set_ylabel(r"$\rho_s$")
aa[0].legend(loc='best')
aa[1].legend(loc='best')

ax[0].set_title('one-basin')
ax[1].set_title('crosstalk')

cbar = fig.colorbar(img, ax=ax)
cbar.set_label("response amplitude (a.u.)")

fig.supxlabel('frequency (Hz)')
fig.supylabel('temperature (K)')

ff.tight_layout()
# fig.tight_layout()

fig2, ax2 = plt.subplots()
ax2.plot(T, rho_s_b - rho_s_c)
ax2.set_xlabel("$T$ (K)")
ax2.set_ylabel(r"$\rho_s^b - \rho_s^c$ (a.u.)")
ax2.axhline(0, color='r')
ax2.set_ylim(-0.025, 0.025)

ff.savefig('frequencies.pdf')
fig.savefig('spectrum.pdf')
fig2.savefig("rho_diff.pdf")
ff.savefig('frequencies.png')
fig.savefig('spectrum.png')
fig2.savefig("rho_diff.png")

plt.show()