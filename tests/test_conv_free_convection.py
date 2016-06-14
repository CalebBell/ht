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




### Free convection immersed

def test_Nu_vertical_plate_Churchill():
    Nu = Nu_vertical_plate_Churchill(0.69, 2.63E9)
    assert_allclose(Nu, 147.16185223770603)


def test_Nu_sphere_Churchill():
    Nu_Res = [Nu_sphere_Churchill(.7, i) for i in np.logspace(0, 10, 11)]
    Nu_Res_values = [2.415066377224484, 2.7381040025746382, 3.3125553308635283, 4.3340933312726548, 6.1507272232235417, 9.3821675084055443, 15.145453144794978, 25.670869440317578, 47.271761310748289, 96.479305628419823, 204.74310854292045]
    assert_allclose(Nu_Res, Nu_Res_values)


def test_Nu_vertical_cylinder_Griffiths_Davis_Morgan():
    Nu_all = [[Nu_vertical_cylinder_Griffiths_Davis_Morgan(i, 1E9, j) for i in (0.999999, 1.000001)] for j in (True, False, None)]
    Nu_all_values = [[127.7046167347578, 127.7047079158867], [119.14469068641654, 119.14475025877677], [119.14469068641654, 127.7047079158867]]
    assert_allclose(Nu_all, Nu_all_values)


def test_Nu_vertical_cylinder_Jakob_Linke_Morgan():
    Nu_all = [[Nu_vertical_cylinder_Jakob_Linke_Morgan(i, 1E8, j) for i in (0.999999, 1.000001)] for j in (True, False, None)]
    Nu_all_values = [[59.87647599476619, 59.87651591243016], [55.499986124994805, 55.5000138749948], [55.499986124994805, 59.87651591243016]]
    assert_allclose(Nu_all, Nu_all_values)


def test_Nu_vertical_cylinder_Carne_Morgan():
    Nu_all = [[Nu_vertical_cylinder_Carne_Morgan(i, 2E8, j) for i in (0.999999, 1.000001)] for j in (True, False, None)]
    Nu_all_values = [[216.88764905616722, 216.88781389084312], [225.77302655456344, 225.77315298749372], [225.77302655456344, 216.88781389084312]]
    assert_allclose(Nu_all, Nu_all_values)

### Giving up ono conv_free immersed for now, TODO