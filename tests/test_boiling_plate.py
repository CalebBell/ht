# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

def test_h_boiling_Amalfi():
    h = h_boiling_Amalfi(m=3E-5, x=.4, Dh=0.00172, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=7.11E-6, sigma=0.02, Hvap=9E5, q=1E5, A_channel_flow=0.0003)
    assert_allclose(h, 776.0781179096225)
    
    h = h_boiling_Amalfi(m=3E-5, x=.4, Dh=0.0172, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=7.11E-6, sigma=0.02, Hvap=9E5, q=1E5, A_channel_flow=0.0003)
    assert_allclose(h, 527.4075513650002)
    

def test_h_boiling_Lee_Kang_Kim():
    h = h_boiling_Lee_Kang_Kim(m=3E-5, x=.4, D_eq=0.002, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=9E-6, Hvap=9E5, q=1E5, A_channel_flow=0.0003)
    assert_allclose(h, 1229.6271295086806)
    
    h = h_boiling_Lee_Kang_Kim(m=3E-5, x=.1, D_eq=0.002, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=9E-6, Hvap=9E5, q=1E5, A_channel_flow=0.0003)
    assert_allclose(h, 4211.51881493242)
    
def test_h_boiling_Han_Lee_Kim():
    h = h_boiling_Han_Lee_Kim(m=3E-5, x=.4, Dh=0.002, rhol=567., rhog=18.09, kl=0.086, mul=156E-6,  Hvap=9E5, Cpl=2200, q=1E5, A_channel_flow=0.0003, wavelength=3.7E-3, chevron_angle=45)
    assert_allclose(h, 675.7322255419421)
    
    # Eldeeb said in the Ge1, Ge2 terms it should be b/Dh but the original and
    # four others agree it's wavelength
    
    # Solotych has pi*beta/180 which is the same just written differently
    # Garcia-Cascales documents the original correctly
    
    
def test_h_boiling_Huang_Sheer():
    h = h_boiling_Huang_Sheer(rhol=567., rhog=18.09, kl=0.086, mul=156E-6, Hvap=9E5, sigma=0.02, Cpl=2200, q=1E4, Tsat=279.15)
    assert_allclose(h, 4401.055635078054)
    
def test_h_boiling_Yan_Lin():
    h = h_boiling_Yan_Lin(m=3E-5, x=.4, Dh=0.002, rhol=567., rhog=18.09,  kl=0.086, Cpl=2200, mul=156E-6, Hvap=9E5, q=1E5, A_channel_flow=0.0003)
    assert_allclose(h, 318.7228565961241)