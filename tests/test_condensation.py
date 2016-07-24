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
from ht.boiling_nucleic import _angles_Stephan_Abdelsalam
import numpy as np

from numpy.testing import assert_allclose
import pytest



### Condensation

def test_h_Nusselt_laminar():
    h = Nusselt_laminar(370, 350, 7.0, 585., 0.091, 158.9E-6, 776900, 0.1)
    assert_allclose(h, 1482.206403453679)
    h_angle = [Nusselt_laminar(Tsat=370, Tw=350, rhog=7.0, rhol=585., kl=0.091, mul=158.9E-6, Hvap=776900, L=0.1, angle=i) for i in np.linspace(0, 90, 8)]
    h_angle_values = [0.0, 1018.0084987903685, 1202.9623322809389, 1317.0917328126477, 1393.7567182107628, 1444.0629692910647, 1472.8272516024929, 1482.206403453679]
    assert_allclose(h_angle, h_angle_values)


def test_h_Boyko_Kruzhilin():
    h_xs = [Boyko_Kruzhilin(m=0.35, rhog=6.36, rhol=582.9, kl=0.098,
	    mul=159E-6, Cpl=2520., D=0.03, x=i) for i in np.linspace(0,1,11)]
    h_xs_values = [1190.3309510899785, 3776.3883678904836, 5206.2779830848758, 6320.5657791981021, 7265.9323628276288, 8101.7278671405438, 8859.0188940546595, 9556.4866502932564, 10206.402815353165, 10817.34162173243, 11395.573750069829]
    assert_allclose(h_xs, h_xs_values)


def test_Akers_Deans_Crosser():
    h = Akers_Deans_Crosser(m=0.35, rhog=6.36, rhol=582.9, kl=0.098,  mul=159E-6, Cpl=2520., D=0.03, x=0.85)
    assert_allclose(h, 7117.24177265201)
    h = Akers_Deans_Crosser(m=0.01, rhog=6.36, rhol=582.9, kl=0.098,  mul=159E-6, Cpl=2520., D=0.03, x=0.85)
    assert_allclose(h, 737.5654803081094)

def test_h_kinetic():
    h = h_kinetic(300, 1E5, 18.02, 2441674)
    assert_allclose(h, 30788845.562480535, rtol=1e-5)