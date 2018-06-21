# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from ht import *
import numpy as np
from numpy.testing import assert_allclose
from ht.core import is_heating_temperature, is_heating_property
import pytest
from ht.core import WALL_FACTOR_VISCOSITY, WALL_FACTOR_PRANDTL, WALL_FACTOR_TEMPERATURE, WALL_FACTOR_DEFAULT


def test_core():
    dT = LMTD(100., 60., 30., 40.2)
    assert_allclose(dT, 43.200409294131525)
    dT = LMTD(100., 60., 30., 40.2, counterflow=False)
    assert_allclose(dT, 39.75251118049003)
    
    assert LMTD(100., 60., 20., 60) == 40
    assert LMTD(100., 60., 20., 60, counterflow=False) == 0
    '''Test code for limits
    from sympy import *
    Thi, Tho, Tci, Tco = symbols('Thi, Tho, Tci, Tco')
    Thi = 100
    Tho=60
    Tci=20
    dTF1 = Thi-Tco
    dTF2 = Tho-Tci
    expression = (dTF2 - dTF1)/log(dTF2/dTF1)
    limit(expression, Tco, 60)
    # evaluates to 40
    
    # Numerical check - goes to zero
    Thi = 100
    Tho=60
    Tci=20
    dTF1 = Thi-Tci
    dTF2 = Tho-Tco
    expression = (dTF2 - dTF1)/log(dTF2/dTF1)
    N(limit(expression, Tco, Rational('60')-Rational('1e-50000')))
    limit(expression, Tco, 60) 
    # evaluates to zero
    '''


def test_is_heating_temperature():
    assert is_heating_temperature(T=200, T_wall=500)
    
    assert not is_heating_temperature(T=400, T_wall=200)
    
    # not heating when 400 K
    assert not is_heating_temperature(T=400, T_wall=400)
    
    
def test_is_heating_property():
    T1, T2 = 280, 330
#    C1, C2 = Chemical('hexane', T=T1), Chemical('hexane', T=T2)
#    mu1, mu2 = C1.mu, C2.mu
#    Pr1, Pr2 = C1.Pr, C2.Pr
    mu1, mu2 = 0.0003595695325135477, 0.0002210964201834834
    Pr1, Pr2 = 6.2859707150337805, 4.810661011475006
    
    assert is_heating_property(prop=mu1, prop_wall=mu2)
    assert is_heating_property(prop=Pr1, prop_wall=Pr2)
    
    # Equal temperatures - not heating in that case
    T1, T2 = 280, 280
    mu1, mu2 = 0.0003595695325135477, 0.0003595695325135477
    Pr1, Pr2 = 6.2859707150337805, 6.2859707150337805
    assert not is_heating_property(prop=mu1, prop_wall=mu2)
    assert not is_heating_property(prop=Pr1, prop_wall=Pr2)
    
    # Lower wall temperatures - not heating in that case
    T1, T2 = 280, 260
    mu1, mu2 = 0.0003595695325135477, 0.0004531397378208441
    Pr1, Pr2 = 6.2859707150337805, 7.27333944072039
    assert not is_heating_property(prop=mu1, prop_wall=mu2)
    assert not is_heating_property(prop=Pr1, prop_wall=Pr2)

def test_wall_factor():
    # Only one value provided
    with pytest.raises(Exception):
        wall_factor(mu=1, property_option=WALL_FACTOR_VISCOSITY)
    
    with pytest.raises(Exception):
        wall_factor(mu_wall=1, property_option=WALL_FACTOR_VISCOSITY)
    
    with pytest.raises(Exception):
        wall_factor(Pr=1, property_option=WALL_FACTOR_PRANDTL)
    
    with pytest.raises(Exception):
        wall_factor(Pr_wall=1, property_option=WALL_FACTOR_PRANDTL)
    
    with pytest.raises(Exception):
        wall_factor(T=1, property_option=WALL_FACTOR_TEMPERATURE)
    
    with pytest.raises(Exception):
        wall_factor(T_wall=1, property_option=WALL_FACTOR_TEMPERATURE)


def test_fin_efficiency_Kern_Kraus():
    
    eta = fin_efficiency_Kern_Kraus(0.0254, 0.05715, 3.8E-4, 200, 58)
    assert_allclose(eta, 0.8412588620231153)
    '''Code for comparing against several formulas:
    def fin_efficiency_Kern_Kraus(Do, Dfin, fin_thickness, kfin, h):
        # Should now be about 50/50 function vs special function
        kf = kfin
        tf = fin_thickness
        re = Dfin/2.
        ro = Do/2.
        m = (2.0*h/(kf*tf))**0.5
    
        mre = m*re
        mro = m*ro
        x0 = i1(mre)
        x1 = k1(mre)
        num = x0*k1(mro) - x1*i1(mro)
        den = i0(mro)*x1 + x0*k0(mro)
        
    #     num = i1(m*re)*k1(m*ro) - k1(m*re)*i1(m*ro)
    #     den = i0(m*ro)*k1(m*re) + i1(m*re)*k0(m*ro)
    
    #     num = iv(1, m*re)*kn(1, m*ro) - kn(1, m*re)*iv(1, m*ro)
    #     den = iv(0, m*ro)*kn(1, m*re) + iv(1, m*re)*kn(0, m*ro)
    #     print(num/den)
        
    #     r2c = re
    #     r1 = ro
    #     num = kn(1, m*r1)*iv(1, m*r2c) - iv(1, m* r1)*kn(1, m*r2c)
    #     den = iv(0, m*r1)*kn(1, m*r2c) + kn(0, m*r1)*iv(1, m*r2c)
    #     print(num/den)
        
        
        eta = 2.0*ro/(m*(re*re - ro*ro))*num/den # r2c = r2, r1 = ro
        return eta
        # Confirmed with Introduction to Heat Transfer
        # To create a pade approximation of this, it would require f(m, re, ro). Not worth it.
    '''