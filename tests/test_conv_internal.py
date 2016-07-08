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


### conv_internal
def test_Nu_const():
    assert_allclose(laminar_T_const(), 3.66)
    assert_allclose(laminar_Q_const(), 48/11.)


def test_laminar_entry_region():
    Nu = laminar_entry_thermal_Hausen(100000, 1.1, 5, .5)
    assert_allclose(Nu, 39.01352358988535)

    Nu = laminar_entry_Seider_Tate(Re=100000, Pr=1.1, L=5, Di=.5)
    assert_allclose(Nu, 41.366029684589265)
    Nu_wall = laminar_entry_Seider_Tate(100000, 1.1, 5, .5, 1E-3, 1.2E-3)
    assert_allclose(Nu_wall, 40.32352264095969)

    Nu = laminar_entry_Baehr_Stephan(100000, 1.1, 5, .5)
    assert_allclose(Nu, 72.65402046550976)

def test_turbulent_complicated():
    Nu1 = turbulent_Dittus_Boelter(1E5, 1.2, True, False)
    Nu2 = turbulent_Dittus_Boelter(Re=1E5, Pr=1.2, heating=False, revised=False)
    Nu3 = turbulent_Dittus_Boelter(Re=1E5, Pr=1.2, heating=False)
    Nu4 = turbulent_Dittus_Boelter(Re=1E5, Pr=1.2)
    Nu_values = [261.3838629346147, 279.89829163640354, 242.9305927410295, 247.40036409449127]
    assert_allclose([Nu1, Nu2, Nu3, Nu4], Nu_values)

    Nu1 = turbulent_Sieder_Tate(Re=1E5, Pr=1.2)
    Nu2 = turbulent_Sieder_Tate(1E5, 1.2, 0.01, 0.067)
    assert_allclose([Nu1, Nu2], [286.9178136793052, 219.84016455766044])

    Nus = [turbulent_entry_Hausen(1E5, 1.2, 0.154, i) for i in np.linspace(0,1,11)]
    Nus_values = [np.inf, 507.39810608575436, 400.1002551153033, 356.83464396632377, 332.50684459222612, 316.60088883614151, 305.25121748064328, 296.67481510644825, 289.92566421612082, 284.45128111774227, 279.90553997822707]
    assert_allclose(Nus, Nus_values)

def test_turbulent_simple():
    Nu = turbulent_Colburn(1E5, 1.2)
    assert_allclose(Nu, 244.41147091200068)

    Nu = turbulent_Drexel_McAdams(1E5, 0.6)
    assert_allclose(Nu, 171.19055301724387)

    Nu = turbulent_von_Karman(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 255.7243541243272)

    Nu = turbulent_Prandtl(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 256.073339689557)

    Nu = turbulent_Friend_Metzner(1E5, 100., 0.0185)
    assert_allclose(Nu, 1738.3356262055322)

    Nu = turbulent_Petukhov_Kirillov_Popov(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 250.11935088905105)

    Nu = turbulent_Webb(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 239.10130376815872)

    Nu = turbulent_Sandall(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 229.0514352970239)

    Nu = turbulent_Gnielinski(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 254.62682749359632)

    Nu = turbulent_Gnielinski_smooth_1(1E5, 1.2)
    assert_allclose(Nu, 227.88800494373442)

    Nu = turbulent_Gnielinski_smooth_2(1E5, 7.)
    assert_allclose(Nu, 577.7692524513449)

    Nu = turbulent_Churchill_Zajic(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 260.5564907817961)

    Nu = turbulent_ESDU(1E5, 1.2)
    assert_allclose(Nu, 232.3017143430645)

def test_turbulent_rough():
    Nu = turbulent_Martinelli(1E5, 100., 0.0185)
    assert_allclose(Nu, 887.1710686396347)

    Nu = turbulent_Nunner(1E5, 0.7, 0.0185, 0.005)
    assert_allclose(Nu, 101.15841010919947)

    Nu = turbulent_Dipprey_Sabersky(1E5, 1.2, 0.0185, 1E-3)
    assert_allclose(Nu, 288.33365198566656)

    Nu = turbulent_Gowen_Smith(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 131.72530453824106)

    Nu = turbulent_Kawase_Ulbrecht(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 389.6262247333975)

    Nu = turbulent_Kawase_De(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 296.5019733271324)

    Nu = turbulent_Bhatti_Shah(1E5, 1.2, 0.0185, 1E-3)
    assert_allclose(Nu, 302.7037617414273)

# TODO meta function  Nu internal


def test_Morimoto_Hotta():
    Nu = Morimoto_Hotta(1E5, 5.7, .05, .5)
    assert_allclose(Nu, 634.4879473869859)