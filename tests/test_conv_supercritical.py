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
from fluids.numerics import assert_close, assert_close1d, assert_close2d
import pytest


def test_Nu_McAdams():
    Nu = Nu_McAdams(1E5, 1.2)
    assert_close(Nu, 261.3838629346147)


def test_Nu_Shitsman():
     Nus = [Nu_Shitsman(1E5, 1.2, 1.6), Nu_Shitsman(1E5, 1.6, 1.2)]
     assert_close1d(Nus, [266.1171311047253]*2)


def test_Nu_Griem():
    Nu = Nu_Griem(1E5, 1.2)
    assert_close(Nu, 275.4818576600527)
    hs = [225.8951232812432, 240.77114359488607, 275.4818576600527]
    hs_calc = [Nu_Griem(1E5, 1.2, H) for H in [1.52E6, 1.6E6, 1.8E6]]
    assert_close1d(hs, hs_calc)


def test_Nu_Jackson():
    Nu = Nu_Jackson(1E5, 1.2)
    assert_close(Nu, 252.37231572974918)

    Nu_calc = [Nu_Jackson(1E5, 1.2, rho_w=125.8, rho_b=249.0233,
                          Cp_avg=2080.845, Cp_b=2048.621, T_b=650, T_w=700,
                          T_pc=T) for T in [750, 675, 600]]
    Nu_exp = [206.91175020307264, 206.93567238866916, 206.97455183928113]
    assert_close1d(Nu_calc, Nu_exp)


def test_Nu_Gupta():
    Nu = Nu_Gupta(1E5, 1.2)
    assert_close(Nu, 189.78727690467736)
    Nu = Nu_Gupta(1E5, 1.2, 330, 290., 8e-4, 9e-4)
    assert_close(Nu, 186.20135477175126)


def test_Nu_Swenson():
    Nu = Nu_Swenson(1E5, 1.2)
    assert_close(Nu, 211.51968418167206)
    Nu = Nu_Swenson(1E5, 1.2, 330, 290.)
    assert_close(Nu, 217.92827034803668)


def test_Nu_Xu():
    Nu = Nu_Xu(1E5, 1.2)
    assert_close(Nu, 293.9572513612297)
    Nu = Nu_Xu(1E5, 1.2, 330, 290., 8e-4, 9e-4)
    assert_close(Nu, 289.133054256742)


def test_Nu_Mokry():
    Nu = Nu_Mokry(1E5, 1.2)
    assert_close(Nu, 228.8178008454556)
    Nu = Nu_Mokry(1E5, 1.2, 330, 290.)
    assert_close(Nu, 246.1156319156992)


def test_Nu_Bringer_Smith():
    Nu = Nu_Bringer_Smith(1E5, 1.2)
    assert_close(Nu, 208.17631753279107)


def test_Nu_Ornatsky():
    Nu = Nu_Ornatsky(1E5, 1.2, 1.5, 330, 290.)
    assert_close(Nu, 276.63531150832307)
    Nu = Nu_Ornatsky(1E5, 1.2, 1.5)
    assert_close(Nu, 266.1171311047253)


def test_Nu_Gorban():
    Nu = Nu_Gorban(1E5, 1.2)
    assert_close(Nu, 182.5367282733999)


def test_Nu_Zhu():
    Nu = Nu_Zhu(1E5, 1.2, 330, 290., 0.63, 0.69)
    assert_close(Nu, 240.1459854494706)
    Nu = Nu_Zhu(1E5, 1.2)
    assert_close(Nu, 241.2087720246979)


def test_Nu_Bishop():
    Nu = Nu_Bishop(1E5, 1.2, 330.0, 290., .01, 1.2)
    assert_close(Nu, 265.3620050072533)
    Nu = Nu_Bishop(1E5, 1.2)
    assert_close(Nu, 246.09835634820243)


def test_Nu_Yamagata():
    Nu = Nu_Yamagata(1E5, 1.2)
    assert_close(Nu, 283.9383689412967)

    Nu_calc = [Nu_Yamagata(1E5, 1.2, 1.5, Cp_avg=2080.845, Cp_b=2048.621,
                           T_b=650, T_w=700, T_pc=T) for T in [750.0, 675.0, 600.0]]
    Nu_exp = [283.9383689412967, 187.02304885276828, 292.3473428004679]
    assert_close1d(Nu_calc, Nu_exp)


def test_Nu_Kitoh():
    Nu = Nu_Kitoh(1E5, 1.2)
    assert_close(Nu, 302.5006546293724)

    Nu_calc = [Nu_Kitoh(1E5, 1.2, H, 1500, 5E6) for H in [1.4E6, 2E6, 3.5E6]]
    Nu_exp = [331.80234139591306, 174.8417387874232, 308.40146536866945]
    assert_close1d(Nu_calc, Nu_exp)


def test_Nu_Krasnoshchekov_Protopopov():
    Nu = Nu_Krasnoshchekov_Protopopov(1E5, 1.2, 330, 290., 0.62, 0.52, 8e-4, 9e-4)
    assert_close(Nu, 228.85296737400222)


def test_Nu_Petukhov():
    Nu = Nu_Petukhov(1E5, 1.2, 330.0, 290., 8e-4, 9e-4)
    assert_close(Nu, 254.8258598466738)


def test_Nu_Krasnoshchekov():
    Nu_calc = [Nu_Krasnoshchekov(1E5, 1.2, rho_w=125.8, rho_b=249.0233,
                          Cp_avg=2080.845, Cp_b=2048.621, T_b=650, T_w=700,
                          T_pc=T) for T in [750.0, 675.0]]
    Nu_exp = [192.52819597784372, 192.54822916468785]
    assert_close1d(Nu_calc, Nu_exp)

    Nu_3 = Nu_Krasnoshchekov(1E5, 1.2, rho_w=125.8, rho_b=249.0233,
                      Cp_avg=2080.845, Cp_b=2048.621, T_b=400.0, T_w=200.0, T_pc=400.0)
    assert_close(Nu_3, 192.2579518680533)

    Nu = Nu_Krasnoshchekov(1E5, 1.2)
    assert_close(Nu, 234.82855185610364)