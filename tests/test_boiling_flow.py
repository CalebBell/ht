# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import pytest


def test_Lazarek_Black():
    q = 1E7
    h1 = Lazarek_Black(m=10, D=0.3, mul=1E-3, kl=0.6, Hvap=2E6, q=q)
    Te = q/h1
    assert_allclose(h1, 51009.87001967105)
    h2 = Lazarek_Black(m=10, D=0.3, mul=1E-3, kl=0.6, Hvap=2E6, Te=Te)
    assert_allclose(h1, h2)
    
    with pytest.raises(Exception):
        Lazarek_Black(m=10, D=0.3, mul=1E-3, kl=0.6, Hvap=2E6)
    '''
    The code to derive the form with `Te` specified is
    as follows:
    
    >>> from sympy import *
    >>> Relo, Bgish, kl, D, h, Te = symbols('Relo, Bgish, kl, D, h, Te',
    ... positive=True, real=True)
    >>> solve(Eq(h, 30*Relo**Rational(857,1000)*(Bgish*h*Te)**Rational(714,
    ... 1000)*kl/D), h)
    [27000*30**(71/143)*Bgish**(357/143)*Relo**(857/286)*Te**(357/143)*kl**(500/143)/D**(500/143)]
    '''
    

def test_Li_Wu():
    q = 1E5
    h = Li_Wu(m=1, x=0.2, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, q=q)
    Te = 1E5/h
    assert_allclose(h, 5345.409399239493)
    h2 = Li_Wu(m=1, x=0.2, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, Te=Te)
    assert_allclose(h2, h)
    
    with pytest.raises(Exception):
         Li_Wu(m=1, x=0.2, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5)
    
    '''
    The code to derive the form with `Te` specified is
    as follows:
    
    >>> from sympy import *
    >>> h, A, Te, G, Hvap = symbols('h, A, Te, G, Hvap', positive=True, real=True)
    >>> solve(Eq(h, A*(h*Te/G/Hvap)**0.3), h)
    [A**(10/7)*Te**(3/7)/(G**(3/7)*Hvap**(3/7))]
    '''

def test_Sun_Mishima():
    h = Sun_Mishima(m=1, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, Te=10)
    assert_allclose(h, 507.6709168372167)
    
    q = 1E5
    h = Sun_Mishima(m=1, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, q=q)
    Te = q/h
    h2 = Sun_Mishima(m=1, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, Te=Te)
    assert_allclose(h, h2)
    assert_allclose(h2, 2538.4455424345983)
    
    with pytest.raises(Exception):
        Sun_Mishima(m=1, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5)

    '''
    The code to derive the form with `Te` specified is
    as follows:
    
    >>> from sympy import *
    >>> h, A, Te, G, Hvap = symbols('h, A, Te, G, Hvap', positive=True, real=True)
    >>> solve(Eq(h, A*(h*Te/G/Hvap)**0.54), h)
    [A**(50/23)*Te**(27/23)/(G**(27/23)*Hvap**(27/23))]
    '''

def test_Thome():
    h = Thome(m=1, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, q=1E5)
    assert_allclose(h, 1633.008836502032)
    
    h = Thome(m=10, x=0.5, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, q=1E5)
    assert_allclose(h, 3120.1787715124824)
    
    Te = 32.04944566414243
    h2 = Thome(m=10, x=0.5, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, Te=Te)
    assert_allclose(h, h2)
    
    with pytest.raises(Exception):
        Thome(m=1, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6)
    
    
def test_Yun_Heo_Kim():
    q = 1E4
    h1 = Yun_Heo_Kim(m=1, x=0.4, D=0.3, rhol=567., mul=156E-6, sigma=0.02, Hvap=9E5, q=q)
    Te = q/h1
    h2 = Yun_Heo_Kim(m=1, x=0.4, D=0.3, rhol=567., mul=156E-6, sigma=0.02, Hvap=9E5, Te=Te)
    assert_allclose(h1, h2)
    
    with pytest.raises(Exception):
        Yun_Heo_Kim(m=1, x=0.4, D=0.3, rhol=567., mul=156E-6, sigma=0.02, Hvap=9E5)

    '''
    The code to derive the form with `Te` specified is
    as follows:
    
    >>> from sympy import *
    >>> h, A = symbols('h, A', positive=True, real=True)
    >>> solve(Eq(h, A*(h)**0.1993), h)
    [A**(10000/8707)]
    '''

def test_Liu_Winterton():
    h = Liu_Winterton(m=1, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, Cpl=2300, P=1E6, Pc=22E6, MW=44.02, Te=7)
    assert_allclose(h, 4747.749477190532)

def test_Chen_Edelstein():
    # Odd numbers for the test case from Serth, but not actually compared to
    # anything.
    h = Chen_Edelstein(m=0.106, x=0.2, D=0.0212, rhol=567, rhog=18.09, mul=156E-6, mug=7.11E-6, kl=0.086, Cpl=2730, Hvap=2E5, sigma=0.02, dPsat=1E5, Te=3)
    assert_allclose(h, 3289.058731974052)


def test_Chen_Bennett():
    h = Chen_Bennett(m=0.106, x=0.2, D=0.0212, rhol=567, rhog=18.09, mul=156E-6, mug=7.11E-6, kl=0.086, Cpl=2730, Hvap=2E5, sigma=0.02, dPsat=1E5, Te=3)
    assert_allclose(h, 4938.275351219369)
