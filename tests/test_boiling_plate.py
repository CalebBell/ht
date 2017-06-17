# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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