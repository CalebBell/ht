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


def test_Nu_Gupta():
    Nu = Nu_Gupta(1E5, 1.2)
    assert_allclose(Nu, 189.78727690467736)
    Nu = Nu_Gupta(1E5, 1.2, 330, 290., 8e-4, 9e-4)
    assert_allclose(Nu, 186.20135477175126)


def test_Nu_Swenson():
    Nu = Nu_Swenson(1E5, 1.2)
    assert_allclose(Nu, 211.51968418167206)
    Nu = Nu_Swenson(1E5, 1.2, 330, 290.)
    assert_allclose(Nu, 217.92827034803668)


def test_Nu_Xu():
    Nu = Nu_Xu(1E5, 1.2)
    assert_allclose(Nu, 293.9572513612297)
    Nu = Nu_Xu(1E5, 1.2, 330, 290., 8e-4, 9e-4)
    assert_allclose(Nu, 289.133054256742)


def test_Nu_Mokry():
    Nu = Nu_Mokry(1E5, 1.2)
    assert_allclose(Nu, 228.8178008454556)
    Nu = Nu_Mokry(1E5, 1.2, 330, 290.)
    assert_allclose(Nu, 246.1156319156992)


def test_Nu_Bringer_Smith():
    Nu = Nu_Bringer_Smith(1E5, 1.2)
    assert_allclose(Nu, 208.17631753279107)


def test_Nu_Ornatsky():
    Nu = Nu_Ornatsky(1E5, 1.2, 1.5, 330, 290.)
    assert_allclose(Nu, 276.63531150832307)
    Nu = Nu_Ornatsky(1E5, 1.2, 1.5)
    assert_allclose(Nu, 266.1171311047253)


def test_Nu_Gorban():
    Nu = Nu_Gorban(1E5, 1.2)
    assert_allclose(Nu, 182.5367282733999)


def test_Nu_Zhu():
    Nu = Nu_Zhu(1E5, 1.2, 330, 290., 0.63, 0.69)
    assert_allclose(Nu, 240.1459854494706)
    Nu = Nu_Zhu(1E5, 1.2)
    assert_allclose(Nu, 241.2087720246979)


def test_Nu_Bishop():
    Nu = Nu_Bishop(1E5, 1.2, 330, 290., .01, 1.2)
    assert_allclose(Nu, 265.3620050072533)
    Nu = Nu_Bishop(1E5, 1.2)
    assert_allclose(Nu, 246.09835634820243)


def test_Nu_Yamagata():
    Nu = Nu_Yamagata(1E5, 1.2)
    assert_allclose(Nu, 283.9383689412967)

    Nu_calc = [Nu_Yamagata(1E5, 1.2, 1.5, Cp_avg=2080.845, Cp_b=2048.621, 
                           T_b=650, T_w=700, T_pc=T) for T in [750, 675, 600]]
    Nu_exp = [283.9383689412967, 187.02304885276828, 292.3473428004679]
    assert_allclose(Nu_calc, Nu_exp)

    
def test_Nu_Kitoh():
    Nu = Nu_Kitoh(1E5, 1.2)
    assert_allclose(Nu, 302.5006546293724)
    
    Nu_calc = [Nu_Kitoh(1E5, 1.2, H, 1500, 5E6) for H in [1.4E6, 2E6, 3.5E6]]
    Nu_exp = [331.80234139591306, 174.8417387874232, 308.40146536866945]
    assert_allclose(Nu_calc, Nu_exp)


def test_Nu_Krasnoshchekov_Protopopov():
    Nu = Nu_Krasnoshchekov_Protopopov(1E5, 1.2, 330, 290., 0.62, 0.52, 8e-4, 9e-4)
    assert_allclose(Nu, 228.85296737400222)


def test_Nu_Petukhov():
    Nu = Nu_Petukhov(1E5, 1.2, 330, 290., 8e-4, 9e-4)
    assert_allclose(Nu, 254.8258598466738)


def test_Nu_Krasnoshchekov():
    Nu_calc = [Nu_Krasnoshchekov(1E5, 1.2, rho_w=125.8, rho_b=249.0233, 
                          Cp_avg=2080.845, Cp_b=2048.621, T_b=650, T_w=700, 
                          T_pc=T) for T in [750, 675]]
    Nu_exp = [192.52819597784372, 192.54822916468785]
    assert_allclose(Nu_calc, Nu_exp)
    
    Nu_3 = Nu_Krasnoshchekov(1E5, 1.2, rho_w=125.8, rho_b=249.0233, 
                      Cp_avg=2080.845, Cp_b=2048.621, T_b=400, T_w=200, T_pc=400)
    assert_allclose(Nu_3, 192.2579518680533)
    
    Nu = Nu_Krasnoshchekov(1E5, 1.2)
    assert_allclose(Nu, 234.82855185610364)