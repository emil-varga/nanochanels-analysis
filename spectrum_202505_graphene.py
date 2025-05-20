import matplotlib as mpl
mpl.use('qtagg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import os
from glob import glob

from config import data_path_2025_05 as data_path
from svp import P_to_T

import lmfit
from tqdm import tqdm

def process_dir(data_dir, NT=300, figax=None, what='onebasin-x'):
    data_files = glob(os.path.join(data_dir, '*.npy'))

    Ts = np.linspace(1.5, 2.18, NT)
    
    #first pass, find the maximum frequecny range
    fmin = np.inf
    fmax = 0
    mindf = np.inf
    for file in data_files:
        d = np.load(file, allow_pickle=True).item()
        f = d['freq (Hz)']
        if min(f) < fmin:
            fmin = min(f)
        if max(f) > fmax:
            fmax = max(f)
        if f[1] - f[0] < mindf:
            mindf = f[1] - f[0]
    
    frequencies = np.arange(fmin, fmax, mindf)

    spec_x = np.zeros((NT, len(frequencies)))
    spec_y = np.zeros_like(spec_x)
    counts = np.zeros_like(spec_x)

    for file in data_files:
        d = np.load(file, allow_pickle=True).item()
        f = d['freq (Hz)']
        match what:
            case 'onebasin-x':
                x1 = d['x (V)']
                y1 = d['y (V)']
            case 'onebasin-y':
                x1 = d['x2 (V)']
                y1 = d['y2 (V)']
            case 'xtalk-x':
                x1 = d['x3 (V)']
                y1 = d['y3 (V)']
            case 'xtalk-y':
                x1 = d['x4 (V)']
                y1 = d['y4 (V)']
        
        cx = np.interp(frequencies, f, x1, left=0, right=0)
        cy = np.interp(frequencies, f, y1, left=0, right=0)

        P = d['Pressure (Torr)']
        T = P_to_T(P)
        ix = np.argmin(abs(Ts - T))

        spec_x[ix, :] += cx
        spec_y[ix, :] += cy
        counts[ix, :] += 1


    spec_x /= counts
    spec_y /= counts

    spec_r = np.sqrt(spec_x**2 + spec_y**2)

    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    img = ax.imshow(spec_r, origin='lower', extent=[frequencies.min(), frequencies.max(), Ts.min(), Ts.max()],
                    aspect='auto', interpolation='nearest',
                    # vmin=0, vmax=10)
                    norm='log', vmin=1e-1)
   
    return spec_x, spec_y, frequencies, Ts, img


def double_lorenz(f, f1, f2, A1, A2, w1, w2, phi1, phi2, b, s=+1):
    l1 = abs(A1)*np.exp(1j*phi1)*f1*w1/(f**2 - f1**2 - s*1j*f*w1)
    l2 = abs(A2)*np.exp(1j*phi2)*f2*w2/(f**2 - f2**2 - s*1j*f*w2)
    return l1 + l2 + b

def double_lorenz_split(f, f1, f2, A1, A2, w1, w2, phi1, phi2, b):
    l1 = A1*np.exp(1j*phi1)*f1*w1/(f**2 - f1**2 - 1j*f*w1)
    l2 = A2*np.exp(1j*phi2)*f2*w2/(f**2 - f2**2 - 1j*f*w2)
    return l1, l2

def fit_img_slices(f, x, y, phi10=np.pi/2, phi20=-np.pi/2):
    model = lmfit.Model(double_lorenz, independent_vars=['f'])
    p0 = model.make_params(f1=1122, f2=1141, w1=10, w2=10,
                           phi1=phi10, phi2=phi20, A1=10, A2=10,
                           b=0)
    f1s = []
    f2s = []
    A1s = []
    A2s = []
    for k in tqdm(range(x.shape[0])):
        try:
            fit = model.fit(f=f, data=x[k,:] + 1j*y[k,:], params=p0)
            if k == 0:
                raise ValueError
            if k == 10:
                # fit.plot()
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
            A1s.append(fit.params['A1'].value)
            A2s.append(fit.params['A2'].value)
        except ValueError as e:
            f1s.append(np.nan)
            f2s.append(np.nan)
            A1s.append(np.nan)
            A2s.append(np.nan)
            print(e)
            print(f"Skipping {k}")
            # raise
    return np.array(f1s), np.array(f2s), np.array(A1s), np.array(A2s)

def fit_img_slices_common(f, Dx, Dy, Cx, Cy):

    def spectra(f1, f2, w1, w2, dA1, dA2, dsxy, cA1, cA2, csxy,
                Dx_phi, Dy_phi, Cx_phi, Cy_phi):
        dA1x = dA1
        dA2x = dA2
        dA1y, dA2y = dsxy*dA1x, dsxy*dA2x

        cA1x = cA1
        cA2x = cA2
        cA1y, cA2y = csxy*cA1x, csxy*cA2x

        sDx = double_lorenz(f, f1, f2, dA1x, dA2x, w1, w2, Dx_phi, Dx_phi, 0, 1)
        sDy = double_lorenz(f, f1, f2, dA1y, dA2y, w1, w2, Dy_phi, Dy_phi, 0, 1)
        sCx = double_lorenz(f, f1, f2, cA1x, cA2x, w1, w2, Cx_phi, Cx_phi + np.pi, 0, 1)
        sCy = double_lorenz(f, f1, f2, cA1y, cA2y, w1, w2, Cy_phi, Cy_phi + np.pi, 0, 1)

        return sDx, sDy, sCx, sCy
    
    def objective(params, *measured_spectra):
        f1, f2, w1, w2, dA1, dA2, dsxy, cA1, cA2, csxy,\
        Dx_phi1, Dy_phi1, Cx_phi1, Cy_phi1 = (x.value for x in params.values())
        mDx, mDy, mCx, mCy = measured_spectra

        sDx, sDy, sCx, sCy = spectra(f1, f2, w1, w2, dA1, dA2, dsxy, cA1, cA2, csxy,
                                     Dx_phi1, Dy_phi1, Cx_phi1, Cy_phi1)

        R = sum(abs(mDx - sDx)**2) + sum(abs(mDy - sDy)**2) + sum(abs(mCx - sCx)**2) + sum(abs(mCy - sCy)**2)
        return np.sqrt(R)

    p0 = lmfit.create_params(f1=1103, f2=1123, w1=5, w2=5,
                             dA1=4, dA2=4, dsxy=1,
                             cA1=10, cA2=10, csxy=0.5,
                             Dx_phi=np.pi, Dy_phi=-np.pi/4,
                             Cx_phi=0, Cy_phi=0)
    f1s = []
    f2s = []
    A1s = []
    A2s = []
    ra = []
    for k in tqdm(range(Dx.shape[0])):
        try:
            if k < 20:
                continue
            fit = lmfit.minimize(objective, params=p0, args=(Dx[k,:], Dy[k,:], Cx[k,:], Cy[k,:]),
                                 method='lbfgsb')
            dA2 = fit.params['dA2'].value()
            cA2 = fit.params['cA2'].value()
            ra.append(cA2/dA2)
            p0 = fit.params
            if k%20 == 0:
                # fit.plot()
                sDx, sDy, sCx, sCy = spectra(**fit.params)
                fig, ax = plt.subplots(2, 2, sharex=True)
                ax[0,0].plot(f, Dx[k, :].real, ':o', ms=3)
                ax[0,0].plot(f, Dx[k, :].imag, ':x', ms=3)

                ax[1,0].plot(f, Dy[k, :].real, ':o', ms=3)
                ax[1,0].plot(f, Dy[k, :].imag, ':x', ms=3)

                ax[0,1].plot(f, Cx[k, :].real, ':o', ms=3)
                ax[0,1].plot(f, Cx[k, :].imag, ':x', ms=3)

                ax[1,1].plot(f, Cy[k, :].real, ':o', ms=3)
                ax[1,1].plot(f, Cy[k, :].imag, ':x', ms=3)

                ax[0,0].plot(f, sDx.real, '-', ms=3)
                ax[0,0].plot(f, sDx.imag, '-', ms=3)

                ax[1,0].plot(f, sDy.real, '-', ms=3)
                ax[1,0].plot(f, sDy.imag, '-', ms=3)

                ax[0,1].plot(f, sCx.real, '-', ms=3)
                ax[0,1].plot(f, sCx.imag, '-', ms=3)

                ax[1,1].plot(f, sCy.real, '-', ms=3)
                ax[1,1].plot(f, sCy.imag, '-', ms=3)

                fig.tight_layout()
        except:
            ra.append(np.nan)
            # raise
    return np.array(ra)

data_dir = os.path.join(data_path, 'direct_and_xtalk_20mVrms')

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

NT = 300

X_Dx, Y_Dx, f, T, img = process_dir(data_dir, figax=(fig, ax[0,0]), 
                                    what='onebasin-x', NT=NT)
X_Dy, Y_Dy, f, T, img = process_dir(data_dir, figax=(fig, ax[1,0]),
                                    what='onebasin-y', NT=NT)
X_Cx, Y_Cx, f, T, img = process_dir(data_dir, figax=(fig, ax[0,1]),
                                    what='xtalk-x', NT=NT)
X_Cy, Y_Cy, f, T, img = process_dir(data_dir, figax=(fig, ax[1,1]),
                                    what='xtalk-y', NT=NT)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
process_dir(data_dir, figax=(fig, ax[0]), what='onebasin-y', NT=NT)
process_dir(data_dir, figax=(fig, ax[1]), what='xtalk-x', NT=NT)

# Dx = X_Dx + 1j*Y_Dx
# Dy = X_Dy + 1j*Y_Dy
# Cx = X_Cx + 1j*Y_Cx
# Cy = X_Cy + 1j*Y_Cy

# fit_img_slices_common(f, Dx, Dy, Cx, Cy)

plt.show()
