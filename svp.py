import numpy as np
from scipy.interpolate import interp1d

svp = np.loadtxt('SVP.txt')

T_K = svp[:,0]
P_Pa = svp[:,1]

P_to_T = interp1d(P_Pa/133.322, T_K, bounds_error=False, fill_value='extrapolate')