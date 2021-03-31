# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.'''

from __future__ import division
import types
import numpy as np
from numpy.testing import assert_allclose
from fluids.numerics import assert_close, assert_close1d, assert_close2d
import pytest
import fluids
import ht
from ht.units import *



def assert_pint_allclose(value, magnitude, units, rtol=1e-7):
    assert_close(value.to_base_units().magnitude, magnitude, rtol=rtol)
    assert dict(value.dimensionality) == units

def test_sample_cases():
    ans = effectiveness_NTU_method(mh=5.2*u.kg/u.s, mc=1.45*u.kg/u.s,
                                   Cph=1860.*u.J/u.K/u.kg, Cpc=1900*u.J/u.K/u.kg,
                                   subtype='crossflow, mixed Cmax', Tci=15*u.K,
                                   Tco=85*u.K, Thi=130*u.K)

    assert_pint_allclose(ans['Cmax'], 9672.0, {'[length]': 2.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})
    assert_pint_allclose(ans['Cmin'], 2755.0, {'[length]': 2.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})
    assert_pint_allclose(ans['Cr'], 0.2848428453267163, {'[length]': 2.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})
    assert_pint_allclose(ans['UA'], 3041.751170834494, {'[length]': 2.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})
    assert_pint_allclose(ans['Q'], 192850, {'[length]': 2.0, '[mass]': 1.0, '[time]': -3.0})
    assert_pint_allclose(ans['NTU'], 1.1040839095588, {})
    assert_pint_allclose(ans['effectiveness'], 0.6086956521739131, {})
    assert_pint_allclose(ans['Tci'], 15, {'[temperature]': 1.0})
    assert_pint_allclose(ans['Tco'], 85, {'[temperature]': 1.0})
    assert_pint_allclose(ans['Thi'], 130, {'[temperature]': 1.0})
    assert_pint_allclose(ans['Tho'], 110.06100082712986, {'[temperature]': 1.0})

    ans = P_NTU_method(m1=5.2*u.kg/u.s, m2=1.45*u.kg/u.s, Cp1=1860.*u.J/u.kg/u.K,
                       Cp2=1900*u.J/u.kg/u.K, subtype='E', Ntp=4, T2i=15*u.K,
                       T1i=130*u.K, UA=3041.75*u.W/u.K)

    assert_pint_allclose(ans['C1'], 9672.0, {'[length]': 2.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})
    assert_pint_allclose(ans['C2'], 2755.0, {'[length]': 2.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})
    assert_pint_allclose(ans['UA'], 3041.75, {'[length]': 2.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})
    assert_pint_allclose(ans['Q'], 192514.71424206023, {'[length]': 2.0, '[mass]': 1.0, '[time]': -3.0})
    assert_pint_allclose(ans['NTU1'], 0.3144902812241522, {})
    assert_pint_allclose(ans['NTU2'], 1.1040834845735028, {})
    assert_pint_allclose(ans['T2i'], 15, {'[temperature]': 1.0})
    assert_pint_allclose(ans['T2o'], 84.87829918042112, {'[temperature]': 1.0})
    assert_pint_allclose(ans['T1i'], 130, {'[temperature]': 1.0})
    assert_pint_allclose(ans['T1o'], 110.09566643485729, {'[temperature]': 1.0})
    assert_pint_allclose(ans['P1'], 0.1730811614360235, {})
    assert_pint_allclose(ans['P2'], 0.6076373841775751, {})
    assert_pint_allclose(ans['R1'], 3.5107078039927404, {})
    assert_pint_allclose(ans['R2'], 0.2848428453267163, {})


def test_custom_wraps():
    k = R_to_k(R=1*u.K/u.W, t=.01*u.m)
    assert_pint_allclose(k, 1E-2, {'[length]': 1.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})

    k = R_to_k(R=1*u.K/u.W*u.m**2, t=.01*u.m, A=5*u.m**2)
    assert_pint_allclose(k, .002, {'[length]': 1.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})

    with pytest.raises(Exception):
        R_to_k(R=1*u.K/u.W, t=.01*u.m, A=2*u.m**2)


    # R_value_to_k
    k = R_value_to_k(0.12*u.parse_expression('m^2*K/(W*inch)'))
    assert_pint_allclose(k, 0.2116666666666667, {'[length]': 1.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0})

    k = R_value_to_k(0.71*u.parse_expression('ft^2*delta_degF*hour/(BTU*inch)'))
    assert_pint_allclose(k, 0.20313790001601909, {'[length]': 1.0, '[mass]': 1.0, '[temperature]': -1.0, '[time]': -3.0}, rtol=1e-4)

    # k_to_R_value
    R_value = k_to_R_value(k=0.2116666666666667*u.W/u.m/u.K, SI=True)
    assert_pint_allclose(R_value.to_base_units(), 4.724409448818897, {'[length]': -1.0, '[mass]': -1.0, '[temperature]': 1.0, '[time]': 3.0}, rtol=1e-4)

    R_value = k_to_R_value(k=0.71*u.W/u.m/u.K, SI=False)
    assert_pint_allclose(R_value.to_base_units(), 1.4084507042253525, {'[length]': -1.0, '[mass]': -1.0, '[temperature]': 1.0, '[time]': 3.0}, rtol=1e-4)


def test_check_signatures():
    from fluids.units import check_args_order
    bad_names = set(['__getattr__'])
    for name in dir(ht):
        if name in bad_names:
            continue
        obj = getattr(ht, name)
        if isinstance(obj, types.FunctionType) and obj not in [ht.get_tube_TEMA, ht.check_tubing_TEMA]:
            check_args_order(obj)
