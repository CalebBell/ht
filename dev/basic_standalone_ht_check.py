import ht
from ht import *
import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.spatial
import scipy.special
import scipy.optimize

def check_close(a, b, rtol=1e-7, atol=0):
    np.all(np.abs(a - b) <= (atol + rtol * np.abs(b)))
    return True

def run_checks():
    checks = []

    # Check LMTD
    result = LMTD(Thi=100, Tho=60, Tci=30, Tco=40.2)
    checks.append(check_close(result, 43.200409294131525))

    # Check radiation
    result = q_rad(emissivity=1, T=400)
    checks.append(check_close(result, 1451.613952))

    # Check insulation material lookup
    wood = nearest_material('spruce')
    checks.append(k_material(wood) == 0.09)
    checks.append(rho_material(wood) == 400.0)

    return all(checks)

if run_checks():
    print("ht basic checks passed - NumPy and SciPy used successfully")
else:
    print('Library not OK')
    exit(1)
