'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import types

import ht
from ht import R_to_k, k_to_R, k_to_thermal_resistivity, thermal_resistivity_to_k

__all__ = ['wraps_numpydoc', 'u']

try:
    from pint import _DEFAULT_REGISTRY as u

except ImportError: # pragma: no cover
    raise ImportError('The unit handling in fluids requires the installation '
                      'of the package pint, available on pypi or from '
                      'https://github.com/hgrecco/pint')
from fluids.units import wraps_numpydoc

"""
Functions which will need custom wrappers:
ht.get_tube_TEMA, ht.check_tubing_TEMA
"""

__funcs = {}


for name in dir(ht):
    if name == '__getattr__' or name == '__test__':
        continue
    obj = getattr(ht, name)
    if isinstance(obj, types.FunctionType) and obj not in [ht.get_tube_TEMA, ht.check_tubing_TEMA]:
        obj = wraps_numpydoc(u)(obj)
    elif isinstance(obj, str):
        continue
    if name == '__all__':
        continue
    __all__.append(name)
    __funcs.update({name: obj})

globals().update(__funcs)

wrapped_R_to_k = R_to_k
wrapped_k_to_R = k_to_R


def R_to_k(R, t, A=1*u.m**2):
    if R.dimensionality == (u.K/u.W).dimensionality:
        if A.to_base_units().magnitude != 1:
            raise ValueError('The conversion with R in units of K/W is only permissible if A = 1 length**2')
        R = R*u.m**2
    elif R.dimensionality != (u.K*u.m**2/u.W).dimensionality:
        raise ValueError('Units of R must be either K/W  if A = 1 length**2 or m^2*K/W otherwise')
    return wrapped_R_to_k(R, t, A)

# k_to_R(k=0.5*u.W/u.m/u.K, t=0.025*u.m)
# TODO define behavior

def R_value_to_k(R_value, SI=True):
    r = R_value.to('m*K/W')
    return thermal_resistivity_to_k(r)


def k_to_R_value(k, SI=True):
    r = k_to_thermal_resistivity(k)
    if SI:
        return r.to('m^2*K/(W*inch)')
    else:
        return r.to('ft^2*delta_degF*hour/(BTU*inch)')
