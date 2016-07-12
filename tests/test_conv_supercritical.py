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


def test_Nu_McAdams():
    Nu = Nu_McAdams(1E5, 1.2)
    assert_allclose(Nu, 261.3838629346147)


def test_Nu_Shitsman():
     Nus = [Nu_Shitsman(1E5, 1.2, 1.6), Nu_Shitsman(1E5, 1.6, 1.2)]
     assert_allclose(Nus, [266.1171311047253]*2)


def test_Nu_Griem():
    Nu = Nu_Griem(1E5, 1.2)
    assert_allclose(Nu, 275.4818576600527)
    hs = [225.8951232812432, 240.77114359488607, 275.4818576600527]
    hs_calc = [Nu_Griem(1E5, 1.2, H) for H in [1.52E6, 1.6E6, 1.8E6]]
    assert_allclose(hs, hs_calc)


def test_Nu_Jackson():
    Nu = Nu_Jackson(1E5, 1.2)
    assert_allclose(Nu, 252.37231572974918)    
    
    Nu_calc = [Nu_Jackson(1E5, 1.2, rho_w=125.8, rho_b=249.0233, 
                          Cp_avg=2080.845, Cp_b=2048.621, T_b=650, T_w=700, 
                          T_pc=T) for T in [750, 675, 600]]
    Nu_exp = [206.91175020307264, 206.93567238866916, 206.97455183928113]
    assert_allclose(Nu_calc, Nu_exp)
