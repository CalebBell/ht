# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

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
    

def test_Li_Wu():
    q = 1E5
    h = Li_Wu(m=1, x=0.2, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, q=q)
    Te = 1E5/h
    assert_allclose(h, 5345.409399239493)
    h2 = Li_Wu(m=1, x=0.2, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, Te=Te)
    assert_allclose(h2, h)
    
    with pytest.raises(Exception):
         Li_Wu(m=1, x=0.2, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5)
    


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
